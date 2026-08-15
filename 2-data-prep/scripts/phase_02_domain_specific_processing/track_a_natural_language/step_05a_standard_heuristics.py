import datetime
import collections
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc


class StandardProseHeuristics:
    """
    Ray Data Stateful Actor for Track A: Standard Prose Heuristics.
    
    ===========================================================================
    ARCHITECTURAL DESIGN: Macro-Linguistic Profiling
    ===========================================================================
    Evaluates four statistical decision boundaries in parallel[cite: 2]:
    1. Punctuation Ratio: Drops if > 0.3 or == 0.0[cite: 2].
    2. Symbol-to-Word Ratio: Drops if operational symbols > 0.10[cite: 2].
    3. Stop-Word Density: Drops if < 0.05 (signals lists or error dumps)[cite: 2].
    4. N-Gram Repetition: Drops if phrases loop excessively[cite: 2].
    
    Uses Apache Arrow RecordBatches to evaluate statistical thresholds in 
    parallel and generate a zero-copy Boolean Mask Array[cite: 2].
    ===========================================================================
    """

    def __init__(self):
        # Core vocabulary words for Stop-Word Density check[cite: 2]
        self.stop_words = [" the ", " and ", " is ", " of ", " to ", " a ", " in ", " that "]
        # Operational symbols for Symbol-to-Word Ratio check[cite: 2]
        self.symbols = ["#", "$", "%", "@", "^", "*"]
        # Standard punctuation marks
        self.punctuation = [".", ",", "?", "!", ";", ":"]

    def _has_excessive_ngram_repetition(self, text: str) -> bool:
        """
        Calculates the frequency of repeating 3-grams to detect bad web scraping 
        or page crashes[cite: 2].
        """
        words = text.lower().split()
        if len(words) < 15:
            return False
            
        n = 3
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams:
            return False
            
        counts = collections.Counter(ngrams)
        most_common_count = counts.most_common(1)[0][1]
        
        # If a single 3-gram makes up more than 15% of the document, it loops excessively[cite: 2].
        if most_common_count / len(ngrams) > 0.15:
            return True
            
        return False

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        
        # Convert to lowercase for uniform PyArrow C++ string counting
        text_lower = pc.ascii_lower(text_column)
        
        # Fast PyArrow C++ word count approximation (spaces + 1)
        space_counts = pc.count_substring(text_lower, " ")
        word_counts = pc.add(space_counts, 1)
        safe_word_counts = pc.maximum(word_counts, 1)
        safe_word_counts_float = pc.cast(safe_word_counts, pa.float64())

        # 1. C++ Vectorized Punctuation Ratio[cite: 2]
        punct_counts = None
        for p in self.punctuation:
            c = pc.count_substring(text_lower, p)
            punct_counts = c if punct_counts is None else pc.add(punct_counts, c)
        punct_ratio = pc.divide(pc.cast(punct_counts, pa.float64()), safe_word_counts_float)

        # 2. C++ Vectorized Symbol-to-Word Ratio[cite: 2]
        sym_counts = None
        for s in self.symbols:
            c = pc.count_substring(text_lower, s)
            sym_counts = c if sym_counts is None else pc.add(sym_counts, c)
        sym_ratio = pc.divide(pc.cast(sym_counts, pa.float64()), safe_word_counts_float)

        # 3. C++ Vectorized Stop-Word Density[cite: 2]
        stop_counts = None
        for sw in self.stop_words:
            c = pc.count_substring(text_lower, sw)
            stop_counts = c if stop_counts is None else pc.add(stop_counts, c)
        stop_ratio = pc.divide(pc.cast(stop_counts, pa.float64()), safe_word_counts_float)

        # Construct Zero-Copy Boolean Masks[cite: 2]
        mask_punct_high = pc.greater(punct_ratio, 0.3)
        mask_punct_zero = pc.equal(punct_ratio, 0.0)
        mask_sym_high = pc.greater(sym_ratio, 0.10)
        mask_stop_low = pc.less(stop_ratio, 0.05)

        # Combine failures (True means the document FAILS the check)
        fail_mask = pc.or_(mask_punct_high, mask_punct_zero)
        fail_mask = pc.or_(fail_mask, mask_sym_high)
        fail_mask = pc.or_(fail_mask, mask_stop_low)
        
        # Invert to keep passing records
        cpp_keep_mask = pc.invert(fail_mask).to_pylist()
        raw_texts = text_column.to_pylist()
        
        final_keep_mask = []
        timestamps = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 4. Apply N-Gram Repetition Filter to survivors[cite: 2]
        for i, text in enumerate(raw_texts):
            if not cpp_keep_mask[i]:
                final_keep_mask.append(False)
                continue
                
            if self._has_excessive_ngram_repetition(text):
                final_keep_mask.append(False)
            else:
                final_keep_mask.append(True)
                timestamps.append(current_utc)

        mask_array = pa.array(final_keep_mask, type=pa.bool_())

        if is_arrow_table:
            filtered_table = batch.filter(mask_array)
            if "heuristics_filtered_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["heuristics_filtered_at"])
            return filtered_table.append_column("heuristics_filtered_at", pa.array(timestamps))
        else:
            filtered_dict = {}
            for key, value in batch.items():
                filtered_dict[key] = [v for i, v in enumerate(value) if final_keep_mask[i]]
            filtered_dict["heuristics_filtered_at"] = timestamps
            return filtered_dict