import datetime
import hashlib
from typing import Dict, Any, Union, Set, List
import pyarrow as pa
import pyarrow.compute as pc

try:
    import pylcs  # Fast C++ Longest Common Subsequence engine
    PYLCS_AVAILABLE = True
except ImportError:
    PYLCS_AVAILABLE = False


class CrossDatasetDecontaminator:
    """
    Ray Data Stateful Actor for Phase 3: Benchmark Decontamination.
    
    ===========================================================================
    ARCHITECTURAL DESIGN: Two-Tier Down-Selected Search Pipeline[cite: 2]
    ===========================================================================
    Part A: Fast N-Gram Hashing (The Pre-Filter)
            Splits texts into sliding 13-grams. Transforms them into 64-bit 
            integer hashes. Evaluates against a loaded benchmark hash set in 
            O(1) time. Zero matches = Cleared. 1+ matches = Route to Part B[cite: 2].
            
    Part B: Longest Common Subsequence (LCS) Validation
            Executes explicit string alignment using a fast C++ LCS library.
            If the LCS ratio > 0.20 OR an exact match exceeds 20 consecutive 
            tokens, it represents a structural leak and the document is dropped[cite: 2].
    ===========================================================================
    """
    def __init__(self, 
                 benchmark_texts: List[str] = None, 
                 ngram_size: int = 13, 
                 lcs_ratio_threshold: float = 0.20):
        self.ngram_size = ngram_size
        self.lcs_ratio_threshold = lcs_ratio_threshold
        
        # In production, benchmark_texts are pulled from s3a://model-gatekeeper/gold_standards/[cite: 2]
        self.benchmark_texts = benchmark_texts or []
        
        # Tier 1: In-memory hash set of benchmark N-grams
        self.benchmark_hashes: Set[str] = set()
        self._initialize_benchmark_hashes()

    def _hash_ngram(self, ngram: str) -> str:
        """Generates a fixed 64-bit integer hash equivalent representation[cite: 2]."""
        # Using sha256 truncated to 16 chars for 64-bit equivalent MVP
        return hashlib.sha256(ngram.encode("utf-8")).hexdigest()[:16]

    def _get_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        """Extracts unique sliding N-grams from a token list."""
        ngrams = set()
        if len(tokens) < n:
            return ngrams
            
        for i in range(len(tokens) - n + 1):
            ngram_str = " ".join(tokens[i:i+n])
            ngrams.add(self._hash_ngram(ngram_str))
        return ngrams

    def _initialize_benchmark_hashes(self):
        """Pre-computes Tier 1 hashes for all benchmark datasets."""
        for text in self.benchmark_texts:
            tokens = text.lower().split()
            self.benchmark_hashes.update(self._get_ngrams(tokens, self.ngram_size))

    def _check_tier2_lcs_violation(self, text: str) -> bool:
        """
        Tier 2: Explicit string alignment check against benchmark texts.
        Returns True if the document violates the LCS threshold (is contaminated)[cite: 2].
        """
        if not PYLCS_AVAILABLE:
            # Fallback: if we hit Tier 2 and lack C++ LCS, drop the doc to be safe
            return True
            
        for benchmark in self.benchmark_texts:
            # Calculate Longest Common Subsequence length
            lcs_length = pylcs.lcs2(text, benchmark)
            
            # Condition 1: LCS exceeds 20% of the training document length[cite: 2]
            if lcs_length > (len(text) * self.lcs_ratio_threshold):
                return True
                
            # Condition 2: Exact match of > 20 consecutive tokens[cite: 2]
            # (Approximated here by checking if LCS length covers ~100+ chars)
            if lcs_length > 100: 
                return True
                
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
            
            # If no benchmarks exist, pass everything automatically
            if self.benchmark_texts:
                tokens = text.lower().split()
                
                # Part A: Fast N-Gram Hashing (Pre-Filter)[cite: 2]
                doc_hashes = self._get_ngrams(tokens, self.ngram_size)
                
                if not self.benchmark_hashes.isdisjoint(doc_hashes):
                    # Part B: Longest Common Subsequence (LCS) Validation[cite: 2]
                    if self._check_tier2_lcs_violation(text):
                        is_valid = False

            keep_mask.append(is_valid)
            timestamps.append(current_utc)

        mask_array = pa.array(keep_mask, type=pa.bool_())

        if is_arrow_table:
            filtered_table = batch.filter(mask_array)
            
            valid_timestamps = [t for i, t in enumerate(timestamps) if keep_mask[i]]
            time_array = pa.array(valid_timestamps)
            
            if "decontaminated_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["decontaminated_at"])
                
            return filtered_table.append_column("decontaminated_at", time_array)
        else:
            filtered_dict = {
                key: [v for i, v in enumerate(value) if keep_mask[i]]
                for key, value in batch.items()
            }
            valid_timestamps = [t for i, t in enumerate(timestamps) if keep_mask[i]]
            filtered_dict["decontaminated_at"] = valid_timestamps
            return filtered_dict