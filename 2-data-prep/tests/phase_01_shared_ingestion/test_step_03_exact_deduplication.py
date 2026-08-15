import pytest
import pyarrow as pa
import numpy as np
from scripts.phase_01_shared_ingestion.step_03_exact_deduplication import ExactDeduplicator


def test_exact_deduplicator_intra_batch_vectorized_selection():
    """
    Validates Stage 2: Vectorized unique selection purges intra-batch 
    character duplicates simultaneously.
    """
    actor = ExactDeduplicator()
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_1", "doc_2", "doc_3", "doc_4"],
        "text": [
            "Unique document content alpha.",
            "Duplicate document content beta.",
            "Duplicate document content beta.",  # Intra-batch duplicate
            "Unique document content gamma."
        ]
    })
    
    result_table = actor(input_table)
    
    assert isinstance(result_table, pa.Table)
    assert result_table.num_rows == 3
    
    surviving_ids = result_table["doc_id"].to_pylist()
    assert surviving_ids == ["doc_1", "doc_2", "doc_4"]


def test_exact_deduplicator_bloom_filter_historical_state():
    """
    Validates Stage 3: Historical duplicate eviction against prior execution DAGs.
    """
    actor = ExactDeduplicator()
    
    batch_1 = pa.Table.from_pydict({
        "doc_id": ["doc_1"],
        "text": ["Document present in Batch 1."]
    })
    
    batch_2 = pa.Table.from_pydict({
        "doc_id": ["doc_2", "doc_3"],
        "text": [
            "Document present in Batch 1.",  # Cross-batch duplicate simulating historical run
            "New document in Batch 2."
        ]
    })
    
    result_batch_1 = actor(batch_1)
    assert result_batch_1.num_rows == 1
    
    # The Bloom filter and RocksDB should catch doc_2 during this pass
    result_batch_2 = actor(batch_2)
    assert result_batch_2.num_rows == 1
    assert result_batch_2["doc_id"].to_pylist() == ["doc_3"]


def test_exact_deduplicator_silver_manifest_persistence():
    """
    Validates Stage 3: Persists non-duplicated hashes alongside Silver dataset files.
    """
    actor = ExactDeduplicator()
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_1"],
        "text": ["Ensure manifest is generated."]
    })
    
    result_table = actor(input_table)
    
    # Verify the Silver Layer Hash Manifest Persistence columns
    assert "document_hash_signature" in result_table.column_names
    assert "exact_dedup_at" in result_table.column_names
    
    # Verify it generated a 64-bit integer signature
    hash_type = result_table.schema.field("document_hash_signature").type
    assert hash_type == pa.int64()