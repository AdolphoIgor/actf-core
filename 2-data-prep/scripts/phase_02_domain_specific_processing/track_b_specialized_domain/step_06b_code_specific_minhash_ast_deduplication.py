import datetime
import re
import hashlib
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

try:
    from tree_sitter import Parser, Language
    import tree_sitter_python
    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False


class CodeASTDeduplicator:
    """
    Ray Data Stateful Actor for Track B: Code-Aware AST Deduplication.
    
    ===========================================================================
    ARCHITECTURAL DESIGN:
    ===========================================================================
    1. Boilerplate Stripping: Removes headers and comments to prevent 
       false-positive matches on shared licenses.
    2. Line-Level MinHash: Computes LSH signatures over non-empty lines 
       (threshold >= 0.80) to capture textual forks.
    3. AST Canonicalization: Parses AST using Tree-Sitter, masks user-defined 
       variables into 'VAR', and generates a 64-bit structural signature to 
       eliminate logic clones despite renaming.
    ===========================================================================
    """
    def __init__(self, line_threshold: float = 0.80, num_perm: int = 128):
        self.num_perm = num_perm
        self.line_threshold = line_threshold
        
        self.seen_ast_hashes = set()
        self.seen_doc_ids = set()
        
        if DATASKETCH_AVAILABLE:
            self.lsh = MinHashLSH(threshold=self.line_threshold, num_perm=self.num_perm)
            
        if TREESITTER_AVAILABLE:
            # BUG FIX 1: Update to Tree-Sitter >=0.21.0 API syntax
            # set_language() is deprecated; the Language must be passed into the Parser constructor.
            PY_LANGUAGE = Language(tree_sitter_python.language())
            self.parser = Parser(PY_LANGUAGE)

    def _strip_boilerplate(self, code: str) -> str:
        """Removes open-source headers, docstrings, and comments."""
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)
        code = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', code)
        code = re.sub(r'#.*', '', code)
        return re.sub(r'\n\s*\n', '\n', code).strip()

    def _get_line_minhash(self, clean_code: str) -> "MinHash":
        """Generates MinHash signature using normalized code lines as tokens."""
        m = MinHash(num_perm=self.num_perm)
        lines = [line.strip().encode('utf8') for line in clean_code.split('\n') if line.strip()]
        
        # BUG FIX 2: Prevent universal collisions for files that reduce to empty strings
        # after boilerplate stripping.
        if not lines:
            m.update(b"empty_code_block")
        else:
            for line in lines:
                m.update(line)
        return m

    def _canonicalize_ast_node(self, node, source_bytes: bytes) -> str:
        """
        Recursively traverses the AST. Replaces identifiers with a generic 'VAR'
        placeholder while preserving structural keywords and control flow.
        """
        if not node.children:
            if node.type == 'identifier':
                return "VAR"
            return source_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
            
        canonical_children = []
        for child in node.children:
            canonical_children.append(self._canonicalize_ast_node(child, source_bytes))
            
        return f"({node.type} " + " ".join(canonical_children) + ")"

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        
        doc_ids = batch["doc_id"].to_pylist() if is_arrow_table else batch["doc_id"]
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        texts = text_column.to_pylist()
        
        keep_mask = []
        timestamps = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for doc_id, text in zip(doc_ids, texts):
            # 1. Strip comments & boilerplate
            clean_code = self._strip_boilerplate(text)
            
            is_unique = True
            
            # 2. Line-Level MinHash LSH
            if DATASKETCH_AVAILABLE and is_unique:
                m = self._get_line_minhash(clean_code)
                if self.lsh.query(m):
                    is_unique = False
                else:
                    self.lsh.insert(doc_id, m)
                    
            # 3. Canonical AST Hashing
            if TREESITTER_AVAILABLE and is_unique and clean_code:
                source_bytes = clean_code.encode('utf8')
                tree = self.parser.parse(source_bytes)
                
                canonical_ast_str = self._canonicalize_ast_node(tree.root_node, source_bytes)
                ast_hash = hashlib.sha256(canonical_ast_str.encode('utf8')).hexdigest()
                
                if ast_hash in self.seen_ast_hashes:
                    is_unique = False
                else:
                    self.seen_ast_hashes.add(ast_hash)

            if is_unique:
                keep_mask.append(True)
                timestamps.append(current_utc)
            else:
                keep_mask.append(False)

        mask_array = pa.array(keep_mask, type=pa.bool_())

        if is_arrow_table:
            filtered_table = batch.filter(mask_array)
            if "ast_dedup_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["ast_dedup_at"])
            return filtered_table.append_column("ast_dedup_at", pa.array(timestamps))
        else:
            filtered_dict = {
                key: [v for i, v in enumerate(value) if keep_mask[i]]
                for key, value in batch.items()
            }
            filtered_dict["ast_dedup_at"] = timestamps
            return filtered_dict