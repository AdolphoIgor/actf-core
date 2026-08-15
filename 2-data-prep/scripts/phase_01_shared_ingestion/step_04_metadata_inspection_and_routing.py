import os
import datetime
from typing import Dict, Any, Union, Optional
import pyarrow as pa
import pyarrow.compute as pc

# FastText C++ binary import for micro-pass classification
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False


class MetadataRouter:
    """
    Ray Data Stateful Actor for Metadata Inspection & Provenance Routing.
    
    ===========================================================================
    ARCHITECTURAL DESIGN: Early Metadata-Driven DAG Branching
    ===========================================================================
    Tier 1: Provenance & Whitelist Inspection
            Explicit technical origins (e.g., `github_repo`, `financial_pdf`, 
            `latex_research`) bypass prose heuristics and route directly 
            to Branch B (`branch_id = 1`).
            
    Tier 2: Vectorized Snippet Disambiguation
            Un-tagged web pages undergo zero-copy PyArrow C++ symbol density 
            profiling. If density >= 0.15, a fastText micro-pass checks 
            syntax validity. Valid code snippets route to Branch B; 
            general prose routes to Branch A.
    ===========================================================================
    """
    
    TECHNICAL_WHITELIST = {
        "github_repo",
        "financial_pdf",
        "financial_audit_pdfs",
        "latex_research",
        "stack_overflow",
        "code_corpus"
    }

    def __init__(self, model_path: str = "/opt/models/fasttext_code_id.bin"):
        # Load C++ fastText binary into memory once per actor
        self.fasttext_model = None
        if FASTTEXT_AVAILABLE and os.path.exists(model_path):
            # fastText suppresses C++ stdout warnings natively via load_model
            self.fasttext_model = fasttext.load_model(model_path)

    def _compute_vectorized_symbol_density(self, text_column: pa.Array) -> pa.Array:
        """
        Computes structural symbol density strictly inside PyArrow C++ shared memory.
        Symbols evaluated: {, }, ;, [, ], =, ->, <, >.
        """
        # BUG FIX: Strip whitespace before length calculation to prevent 
        # spaces/indentation from diluting the structural symbol density.
        text_no_spaces = pc.replace_substring_regex(text_column, r"\s+", "")
        total_len = pc.cast(pc.utf8_length(text_no_spaces), pa.float64())
        
        # Prevent division by zero safely using PyArrow native masks
        safe_len = pc.if_else(pc.equal(total_len, 0.0), 1.0, total_len)
        
        symbols = ["{", "}", ";", "[", "]", "=", ">", "<", "->"]
        symbol_counts = None
        
        for sym in symbols:
            cnt = pc.cast(pc.count_substring(text_column, sym), pa.float64())
            if symbol_counts is None:
                symbol_counts = cnt
            else:
                symbol_counts = pc.add(symbol_counts, cnt)
                
        return pc.divide(symbol_counts, safe_len)

    def _is_technical_provenance(self, batch_metadata: Optional[Dict[bytes, bytes]], row_source: Optional[str] = None) -> bool:
        """Checks if Arrow schema metadata or row provenance matches technical whitelist."""
        if row_source and row_source in self.TECHNICAL_WHITELIST:
            return True
            
        if batch_metadata:
            for key in (b"source_type", b"doc_type", b"compliance_track"):
                if key in batch_metadata:
                    val = batch_metadata[key].decode("utf-8")
                    if val in self.TECHNICAL_WHITELIST:
                        return True
        return False

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        
        # Tier 1: Check batch-level schema metadata
        schema_metadata = batch.schema.metadata if is_arrow_table and batch.schema.metadata else None
        is_batch_whitelisted = self._is_technical_provenance(schema_metadata)
        
        # Extract row-level source_type if present
        source_types = None
        if is_arrow_table and "source_type" in batch.column_names:
            source_types = batch["source_type"].to_pylist()
        elif isinstance(batch, dict) and "source_type" in batch:
            source_types = batch["source_type"]
            
        # Tier 2: Vectorized PyArrow C++ Symbol Density Calculation
        density_array = self._compute_vectorized_symbol_density(text_column)
        densities = density_array.to_numpy(zero_copy_only=False)
        raw_texts = text_column.to_pylist()
        
        branch_ids = [0] * len(raw_texts)
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for i, text in enumerate(raw_texts):
            row_source = source_types[i] if source_types else None
            
            # 1. Tier 1 Check: Whitelist Provenance
            if is_batch_whitelisted or self._is_technical_provenance(None, row_source):
                branch_ids[i] = 1  # Route to Branch B (Technical)
            else:
                density = densities[i]
                # 2. Tier 2 Check: Vectorized Snippet Disambiguation
                if density >= 0.15:
                    if self.fasttext_model:
                        # FastText expects single-line strings without newlines
                        safe_text = text.replace("\n", " ")[:1000]
                        labels, probs = self.fasttext_model.predict(safe_text)
                        
                        # Result = Raw Garbage / OCR Error -> Branch A
                        # Result = Valid Source Code Snippet -> Branch B
                        is_code = labels[0] == "__label__code" if labels else True
                        branch_ids[i] = 1 if is_code else 0
                    else:
                        # Heuristic fallback if C++ binary is missing
                        branch_ids[i] = 1
                else:
                    branch_ids[i] = 0  # Route to Branch A (Prose)

        branch_array = pa.array(branch_ids, type=pa.int8())
        time_array = pa.array([current_utc] * len(raw_texts))

        if is_arrow_table:
            # Append branch_id and routed_at metadata columns
            res_table = batch
            for col in ("branch_id", "routed_at"):
                if col in res_table.column_names:
                    res_table = res_table.drop_columns([col])
                    
            res_table = res_table.append_column("branch_id", branch_array)
            return res_table.append_column("routed_at", time_array)
        else:
            batch["branch_id"] = branch_ids
            batch["routed_at"] = [current_utc] * len(raw_texts)
            return batch