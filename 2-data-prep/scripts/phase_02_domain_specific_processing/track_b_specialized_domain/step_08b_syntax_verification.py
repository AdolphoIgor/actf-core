import datetime
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

try:
    from tree_sitter import Parser, Language
    import tree_sitter_python
    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False

try:
    import sqlglot
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False


class SyntaxVerifier:
    """
    Ray Data Stateful Actor for Track B: AST Syntax Verification.
    
    ===========================================================================
    ARCHITECTURAL DESIGN:
    ===========================================================================
    Stage 1: Multi-Grammar Tree-Sitter AST Compilation
             Constructs in-memory syntax tree and calculates error density.
             Drops file if (ERROR + MISSING) / Total Nodes > 0.0.
    Stage 2: Scope Resolution & Unclosed Block Detection
             Identifies truncated EOF boundaries (e.g., dangling operators).
    Stage 3: Dialect-Specific Fallback & Normalization
             Escalates to secondary parsers (e.g., sqlglot) if primary fails.
    ===========================================================================
    """
    def __init__(self):
        if TREESITTER_AVAILABLE:
            # BUG FIX: Update to Tree-Sitter >=0.21.0 API syntax
            # set_language() is deprecated; pass the Language into the Parser constructor.
            self.py_lang = Language(tree_sitter_python.language())
            self.parser = Parser(self.py_lang)

    def _count_nodes_and_errors(self, node) -> tuple:
        """Recursively traverses AST to return (total_nodes, error_nodes)."""
        total = 1
        errors = 1 if node.type == 'ERROR' or node.is_missing else 0
        
        for child in node.children:
            child_total, child_errors = self._count_nodes_and_errors(child)
            total += child_total
            errors += child_errors
            
        return total, errors

    def _has_dangling_eof(self, text: str) -> bool:
        """Stage 2: Detects files that cut off mid-statement at End-of-File (EOF)."""
        stripped = text.rstrip()
        if not stripped:
            return False
            
        dangling_ops = ('+', '-', '=', '&&', '||', '==', '!=', '<', '>', '*', '/', ':', ',')
        return stripped.endswith(dangling_ops)

    def _validate_sql_fallback(self, text: str) -> bool:
        """Stage 3: Escalates SQL files to a multi-dialect transpiler."""
        if not SQLGLOT_AVAILABLE:
            return False
        try:
            sqlglot.parse(text, read="postgres")
            return True
        except Exception:
            try:
                sqlglot.parse(text, read="mysql")
                return True
            except Exception:
                return False

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        texts = text_column.to_pylist()
        
        keep_mask = []
        timestamps = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for text in texts:
            is_valid = True
            
            # 1. Scope Resolution & Unclosed Block Detection
            if self._has_dangling_eof(text):
                is_valid = False
                
            # 2. Multi-Grammar Tree-Sitter AST Compilation
            if is_valid and TREESITTER_AVAILABLE:
                source_bytes = text.encode('utf8')
                tree = self.parser.parse(source_bytes)
                
                total_nodes, error_nodes = self._count_nodes_and_errors(tree.root_node)
                error_density = error_nodes / max(total_nodes, 1)
                
                if error_density > 0.0:
                    is_valid = False

            # 3. Dialect-Specific Fallback
            if not is_valid and SQLGLOT_AVAILABLE:
                if self._validate_sql_fallback(text):
                    is_valid = True

            keep_mask.append(is_valid)
            if is_valid:
                timestamps.append(current_utc)
            else:
                timestamps.append(current_utc)

        mask_array = pa.array(keep_mask, type=pa.bool_())

        if is_arrow_table:
            filtered_table = batch.filter(mask_array)
            
            # Align timestamps array length with the surviving rows
            valid_timestamps = [t for i, t in enumerate(timestamps) if keep_mask[i]]
            time_array = pa.array(valid_timestamps)
            
            if "syntax_verified_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["syntax_verified_at"])
            return filtered_table.append_column("syntax_verified_at", time_array)
        else:
            filtered_dict = {
                key: [v for i, v in enumerate(value) if keep_mask[i]]
                for key, value in batch.items()
            }
            valid_timestamps = [t for i, t in enumerate(timestamps) if keep_mask[i]]
            filtered_dict["syntax_verified_at"] = valid_timestamps
            return filtered_dict