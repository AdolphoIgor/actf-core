import pytest
import pyarrow as pa
from scripts.phase_02_domain_specific_processing.track_a_natural_language.step_06a_minhash_fuzzy_deduplication import ProseFuzzyDeduplicator


def test_prose_fuzzy_deduplicator():
    """Validates that documents with >= 80% Jaccard similarity are clustered and pruned."""
    actor = ProseFuzzyDeduplicator(threshold=0.80)
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_1", "doc_2", "doc_3"],
        "text": [
            "The quick brown fox jumps over the lazy dog in the forest.",
            "The quick brown fox jumps over the lazy dog in the woods.",  # Fuzzy Duplicate
            "Data engineering involves building robust pipelines and systems." # Unique
        ]
    })
    
    result = actor(input_table)
    assert result.num_rows == 2