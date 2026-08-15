import pyarrow as pa
import pyarrow.compute as pc
import collections
from typing import Dict, Any, Union


class StandardProseHeuristics:
    """
    Ray Data Stateful Actor for Zero-Copy NLP Heuristics.
    
    Applies aggressive quality filtering to prose (Branch A) to eliminate 
    log files, SEO spam, and noisy OCR artifacts.
    """
    
    # Minimal stop-word set for fast PyArrow vectorization
    STOP_WORDS = ["the", "a", "an", "and", "is", "of", "to", "in", "it", "that", "for", "on", "with"]
    PUNCT_CHARS = [".", ",", "!", "?", ";", ":"]
    SYM_CHARS = ["#", "$", "%", "@", "^", "*", "&", "|", "\\", "/"]

    def _compute_ngram_repetition(self, text: str, n: int = 3, threshold: float = 0.2) -> bool:
        """
        Calculates the ratio of repeated n-grams.
        Returns True if the text loops excessively (e.g., SEO spam or log streams).
        """
        words = text.lower().split()
        if len(words) < n:
            return False
            
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = collections.Counter(ngrams)
        
        if not counts:
            return False
            
        most_common_count = counts.most_common(1)[0][1]
        repetition_ratio = most_common_count / len(ngrams)
        
        return repetition_ratio > threshold

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        
        # Convert to lowercase for uniform PyArrow C++ string counting
        text_lower = pc.ascii_lower(text_column)
        
        # Fast PyArrow C++ word count approximation (spaces + 1)
        space_counts = pc.count_substring(text_lower, " ")
        word_counts = pc.add(space_counts, 1)
        
        # BUG FIX: Replaced hallucinated pc.maximum with PyArrow native conditional mask
        safe_word_counts = pc.if_else(pc.equal(word_counts, 0), 1, word_counts)

        # 1. Punctuation Ratio (Detects dense OCR noise or total lack of grammar)
        punct_counts = None
        for char in self.PUNCT_CHARS:
            cnt = pc.count_substring(text_column, char)
            punct_counts = cnt if punct_counts is None else pc.add(punct_counts, cnt)
            
        punct_ratio = pc.divide(pc.cast(punct_counts, pa.float64()), pc.cast(safe_word_counts, pa.float64()))
        
        # 2. Symbol Ratio (Detects unstructured logs and raw JSON)
        sym_counts = None
        for char in self.SYM_CHARS:
            cnt = pc.count_substring(text_column, char)
            sym_counts = cnt if sym_counts is None else pc.add(sym_counts, cnt)
            
        sym_ratio = pc.divide(pc.cast(sym_counts, pa.float64()), pc.cast(safe_word_counts, pa.float64()))
        
        # 3. Stop Word Density (Detects keyword stuffing and log entries)
        stop_counts = None
        for word in self.STOP_WORDS:
            # Note: A real implementation requires regex boundaries, but substring is faster for approximation
            cnt = pc.count_substring(text_lower, f" {word} ")
            stop_counts = cnt if stop_counts is None else pc.add(stop_counts, cnt)
            
        stop_ratio = pc.divide(pc.cast(stop_counts, pa.float64()), pc.cast(safe_word_counts, pa.float64()))
        
        # Apply Vectorized Masks
        # Rule A: Punctuation must exist (ratio > 0.0) but not be excessive (ratio <= 0.3)
        punct_mask = pc.and_(pc.greater(punct_ratio, 0.0), pc.less_equal(punct_ratio, 0.3))
        
        # Rule B: Operational symbols must be less than 10% of word count
        sym_mask = pc.less(sym_ratio, 0.1)
        
        # Rule C: Stop words must account for at least 5% of text (proves natural grammar)
        stop_mask = pc.greater_equal(stop_ratio, 0.05)
        
        # Combine C++ Masks
        combined_mask = pc.and_(pc.and_(punct_mask, sym_mask), stop_mask)
        
        # Filter table via C++ memory directly
        if is_arrow_table:
            c_filtered_table = batch.filter(combined_mask)
            
            # Final Pass: Python loop for N-gram repetition (too complex for pure PyArrow C++)
            texts = c_filtered_table["text"].to_pylist()
            ngram_mask = [not self._compute_ngram_repetition(t) for t in texts]
            
            final_mask = pa.array(ngram_mask, type=pa.bool_())
            return c_filtered_table.filter(final_mask)
        
        else:
            # Fallback dict logic
            bool_mask = combined_mask.to_pylist()
            texts = batch["text"]
            
            final_keep = []
            for i, keep in enumerate(bool_mask):
                if keep and not self._compute_ngram_repetition(texts[i]):
                    final_keep.append(True)
                else:
                    final_keep.append(False)
                    
            filtered_dict = {}
            for key, value in batch.items():
                filtered_dict[key] = [v for i, v in enumerate(value) if final_keep[i]]
            return filtered_dict