import datetime
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

from datasketch import MinHash, MinHashLSH


class ProseFuzzyDeduplicator:
    """
    Ray Data Stateful Actor for Track A: MinHash Fuzzy Deduplication.
    
    ===========================================================================
    ARCHITECTURAL DESIGN:
    ===========================================================================
    1. Shingling: Extracts word-level 3-grams from prose documents.
    2. MinHashing: Computes 128 permutations to generate a signature vector.
    3. LSH Indexing: Stores signatures in a banded LSH index (threshold=0.85).
       If a document collides with an existing signature, it is pruned.
    ===========================================================================
    """
    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.num_perm = num_perm
        self.threshold = threshold
        self.seen_doc_ids = set()
        
        # In an enterprise cluster, this is backed by a distributed RocksDB layer
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

    def _get_minhash(self, text: str) -> "MinHash":
        m = MinHash(num_perm=self.num_perm)
        words = text.lower().split()
        
        # BUG FIX: Safely handle short texts (< 3 words). 
        # Prevents empty MinHashes which would universally collide and delete unrelated short prose.
        if len(words) < 3:
            m.update(text.encode("utf8"))
        else:
            for i in range(len(words) - 2):
                shingle = " ".join(words[i:i+3]).encode("utf8")
                m.update(shingle)
        return m

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        texts = text_column.to_pylist()
        
        doc_ids = batch["doc_id"].to_pylist() if is_arrow_table else batch["doc_id"]
        
        keep_mask = []
        timestamps = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for doc_id, text in zip(doc_ids, texts):
            m = self._get_minhash(text)
            
            # Query LSH for Jaccard similarity >= threshold
            result = self.lsh.query(m)
            
            if not result:
                # Unique document -> Insert into LSH and keep
                self.lsh.insert(doc_id, m)
                self.seen_doc_ids.add(doc_id)
                keep_mask.append(True)
                timestamps.append(current_utc)
            else:
                # Fuzzy Duplicate -> Prune
                keep_mask.append(False)

        mask_array = pa.array(keep_mask, type=pa.bool_())

        if is_arrow_table:
            filtered_table = batch.filter(mask_array)
            if "fuzzy_dedup_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["fuzzy_dedup_at"])
            return filtered_table.append_column("fuzzy_dedup_at", pa.array(timestamps))
        else:
            filtered_dict = {
                key: [v for i, v in enumerate(value) if keep_mask[i]]
                for key, value in batch.items()
            }
            filtered_dict["fuzzy_dedup_at"] = timestamps
            return filtered_dict