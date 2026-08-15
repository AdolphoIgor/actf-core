import os
import datetime
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

# Optional high-performance dependencies
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

try:
    import rure  # Rust regex crate via Python bindings
    RURE_AVAILABLE = True
except ImportError:
    import re as rure
    RURE_AVAILABLE = False


class CodeSyntaxDisambiguation:
    """
    Ray Data Stateful Actor for Track B: Code & Syntax Disambiguation.
    
    ===========================================================================
    ARCHITECTURAL DESIGN: Syntax-Aware Structural Parsing
    ===========================================================================
    Stage 1: Vectorized Structural Character Profiling
             Calculates symbol density via PyArrow C++ shared memory. 
             Rejects densities < 0.01 (raw un-parsed prose/license headers) 
             and > 0.45 (minified binaries/memory dumps).
    Stage 2: Code Snippet Extraction & fastText Classification
             Isolates embedded code fences using Rust Regex engines. 
             Validates syntax via fastText code-detection embeddings (>=0.70).
    Stage 3: Indentation & Layout Profiling
             Analyzes line-length variance. Prunes minified JS files with 
             extreme single-line lengths (> 2,000 characters).
    ===========================================================================
    """

    def __init__(self, model_path: str = "/opt/models/fasttext_code_id.bin"):
        self.fasttext_model = None
        if FASTTEXT_AVAILABLE and os.path.exists(model_path):
            self.fasttext_model = fasttext.load_model(model_path)
            
        # Compile Rust Regex state machine for code fences (<pre><code> or ```)
        fence_pattern = r"(?s)(?:```|<pre><code>)(.*?)(?:```|</code></pre>)"
        if RURE_AVAILABLE and hasattr(rure, 'Rure'):
            # BUG FIX 1: Rust engine strictly requires the pattern to be compiled as bytes
            self.fence_regex = rure.Rure(fence_pattern.encode('utf-8'))
            self._use_rure = True
        else:
            self.fence_regex = rure.compile(fence_pattern)
            self._use_rure = False

    def _compute_vectorized_symbol_density(self, text_column: pa.Array) -> pa.Array:
        """
        PyArrow C++ native string kernel to measure structural programming characters.
        Symbols: {, }, ;, [, ], =, ->, <, >.
        """
        # BUG FIX 3 & 4: Stripped spaces to prevent density dilution, 
        # replaced hallucinated pc.string_length with utf8_length, 
        # and replaced hallucinated pc.maximum with if_else.
        text_no_spaces = pc.replace_substring_regex(text_column, r"\s+", "")
        total_len = pc.cast(pc.utf8_length(text_no_spaces), pa.float64())
        safe_len = pc.if_else(pc.equal(total_len, 0.0), 1.0, total_len)
        
        symbols = ["{", "}", ";", "[", "]", "=", "->", "<", ">"]
        symbol_counts = None
        
        for sym in symbols:
            cnt = pc.cast(pc.count_substring(text_column, sym), pa.float64())
            if symbol_counts is None:
                symbol_counts = cnt
            else:
                symbol_counts = pc.add(symbol_counts, cnt)
                
        return pc.divide(symbol_counts, safe_len)

    def _extract_and_classify_snippets(self, text: str) -> bool:
        """
        Stage 2 & 3: Snippet Extraction and Layout Profiling.
        """
        # Stage 3: Minified JS / Compressed Asset check (> 2000 chars)
        lines = text.split("\n")
        if any(len(line) > 2000 for line in lines):
            return False  # Flagged as non-trainable asset noise and dropped
            
        # Stage 2: Regex Fence Extraction
        if self._use_rure:
            # BUG FIX 2: rure requires the search payload to be bytes.
            # Additionally, m.start and m.end yield byte offsets, not string indices.
            text_bytes = text.encode('utf-8')
            match_iter = self.fence_regex.find_iter(text_bytes)
            snippets = [text_bytes[m.start:m.end].decode('utf-8', errors='ignore') for m in match_iter]
        else:
            snippets = self.fence_regex.findall(text)
            
        if not snippets:
            # If no fences exist, validate the whole file as code if fastText is available
            snippets = [text]
            
        # fastText Classification Pass
        if self.fasttext_model:
            for snippet in snippets:
                safe_snippet = snippet.replace("\n", " ")[:1000]
                labels, probs = self.fasttext_model.predict(safe_snippet)
                
                # Fails syntax classification confidence threshold (< 0.70)
                if labels and labels[0] == "__label__noise" and probs[0] >= 0.70:
                    return False
                    
        return True

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        
        # Stage 1: Vectorized Structural Character Profiling
        density_array = self._compute_vectorized_symbol_density(text_column)
        
        # Mask out densities < 0.01 (raw un-parsed prose)
        # Mask out densities > 0.45 (minified JS / compressed assets)
        mask_too_low = pc.less(density_array, 0.01)
        mask_too_high = pc.greater(density_array, 0.45)
        fail_density_mask = pc.or_(mask_too_low, mask_too_high)
        
        cpp_keep_mask = pc.invert(fail_density_mask).to_pylist()
        raw_texts = text_column.to_pylist()
        
        final_keep_mask = []
        timestamps = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Execute Layout & fastText Snippet passes on survivors
        for i, text in enumerate(raw_texts):
            if not cpp_keep_mask[i]:
                final_keep_mask.append(False)
                continue
                
            if self._extract_and_classify_snippets(text):
                final_keep_mask.append(True)
                timestamps.append(current_utc)
            else:
                final_keep_mask.append(False)

        mask_array = pa.array(final_keep_mask, type=pa.bool_())

        if is_arrow_table:
            filtered_table = batch.filter(mask_array)
            if "code_disambiguated_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["code_disambiguated_at"])
            return filtered_table.append_column("code_disambiguated_at", pa.array(timestamps))
        else:
            filtered_dict = {}
            for key, value in batch.items():
                filtered_dict[key] = [v for i, v in enumerate(value) if final_keep_mask[i]]
            filtered_dict["code_disambiguated_at"] = timestamps
            return filtered_dict