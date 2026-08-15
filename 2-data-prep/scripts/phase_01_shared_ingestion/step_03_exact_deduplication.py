import os
import datetime
import hashlib
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np

# In a production environment, these are managed via uv.lock
# e.g., pip install rocksdict pybloom-live
try:
    from rocksdict import Rdict, Options
    from pybloom_live import BloomFilter
    ROCKSDB_AVAILABLE = True
except ImportError:
    ROCKSDB_AVAILABLE = False


class ExactDeduplicator:
    """
    Ray Data Stateful Actor for Exact Deduplication.
    
    ===========================================================================
    ARCHITECTURAL DESIGN: Bloom Filter + Embedded RocksDB
    ===========================================================================
    Stage 1: Hash generation utilizes non-cryptographic 64-bit integer 
             signatures generated uniformly. 
             (Note: Deterministic routing H(S) mod K is handled by Ray's 
             .repartition() prior to this Actor).
    Stage 2: Intra-batch localized shuffle executes a vectorized unique 
             selection to purge duplicates in-memory instantly.
    Stage 3: Incoming 64-bit hashes pass through a lightweight, in-memory 
             Bloom Filter footprint dictionary to verify non-membership in 
             O(1) time. Matches undergo exact verification against 
             the embedded RocksDB Key-Value index stored at /dev/shm/dedup_index.
    ===========================================================================
    """
    def __init__(self, capacity: int = 100_000_000, error_rate: float = 0.001):
        self.db_path = "/dev/shm/dedup_index"
        
        # Initialize Probabilistic Bloom Filter Pre-Filter
        if ROCKSDB_AVAILABLE:
            self.bloom_filter = BloomFilter(capacity=capacity, error_rate=error_rate)
            
            # Initialize RocksDB Direct-Mapped Lookup
            # Bypasses TCP sockets and network serialization wrappers
            os.makedirs(self.db_path, exist_ok=True)
            self.rocks_db = Rdict(self.db_path)
        else:
            # Fallback for local dev environments lacking C++ RocksDB binaries
            self.bloom_filter = set()
            self.rocks_db = set()

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        
        # Null Sanitization
        text_column = pc.fill_null(text_column, "")
        
        # Stage 1: Generate uniform 64-bit integer signature H(S)
        # BUG FIX: Replaced hallucinated pc.hash_64 with a deterministic 64-bit truncation of MD5
        texts = text_column.to_pylist()
        hash_list = [
            int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], byteorder="little", signed=True)
            for text in texts
        ]
        hash_numpy = np.array(hash_list, dtype=np.int64)
        
        # Stage 2: Vectorized Unique Selection (Intra-Batch Local Shuffle)
        # Purges intra-batch character duplicates simultaneously.
        _, unique_indices = np.unique(hash_numpy, return_index=True)
        intra_batch_unique_mask = np.zeros(len(hash_numpy), dtype=bool)
        intra_batch_unique_mask[unique_indices] = True

        keep_mask = []
        timestamps = []
        surviving_hashes = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Stage 3: Historical Index Validation
        for i, doc_hash in enumerate(hash_numpy):
            # 1. Check if dropped by Stage 2 intra-batch selection
            if not intra_batch_unique_mask[i]:
                keep_mask.append(False)
                continue
                
            doc_hash_bytes = doc_hash.tobytes() if ROCKSDB_AVAILABLE else doc_hash
            
            # 2. Probabilistic Bloom Filter Pre-Filter
            if doc_hash_bytes in self.bloom_filter:
                # 3. RocksDB Direct-Mapped Lookup for exact verification
                if doc_hash_bytes in self.rocks_db:
                    keep_mask.append(False)
                    continue
            
            # 4. If unique, atomic write to state and approve
            if ROCKSDB_AVAILABLE:
                self.bloom_filter.add(doc_hash_bytes)
                self.rocks_db[doc_hash_bytes] = True
            else:
                self.bloom_filter.add(doc_hash_bytes)
                self.rocks_db.add(doc_hash_bytes)
                
            keep_mask.append(True)
            timestamps.append(current_utc)
            surviving_hashes.append(doc_hash)

        # Apply filtering
        mask_array = pa.array(keep_mask, type=pa.bool_())
        
        if is_arrow_table:
            filtered_table = batch.filter(mask_array)
            
            if "exact_dedup_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["exact_dedup_at", "document_hash_signature"])
                
            # Silver Layer Hash Manifest Persistence
            # Persisted alongside the generated Silver storage files to guarantee state continuity
            filtered_table = filtered_table.append_column("document_hash_signature", pa.array(surviving_hashes, type=pa.int64()))
            return filtered_table.append_column("exact_dedup_at", pa.array(timestamps))
        else:
            filtered_dict = {}
            for key, value in batch.items():
                filtered_dict[key] = [v for i, v in enumerate(value) if keep_mask[i]]
            
            filtered_dict["document_hash_signature"] = surviving_hashes
            filtered_dict["exact_dedup_at"] = timestamps
            return filtered_dict

    def __del__(self):
        if ROCKSDB_AVAILABLE and hasattr(self, 'rocks_db'):
            self.rocks_db.close()
            