import pytest
import pyarrow as pa
from scripts.phase_03_reconvergence_and_tokenization.step_10_cross_dataset_decontamination import CrossDatasetDecontaminator


def test_decontamination_zero_overlap():
    """Validates Tier 1: Documents with 0 hash matches are immediately cleared[cite: 2]."""
    mock_benchmark = [
        "What is the capital of France? The capital of France is Paris. This is an exact test set question."
    ]
    
    actor = CrossDatasetDecontaminator(benchmark_texts=mock_benchmark, ngram_size=5)
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_clean"],
        "text": ["Machine learning is the study of computer algorithms that can improve automatically through experience."]
    })
    
    result = actor(input_table)
    
    assert result.num_rows == 1
    assert "decontaminated_at" in result.column_names


def test_decontamination_lcs_violation():
    """Validates Tier 2: Documents passing the hash check but failing LCS validation are dropped[cite: 2]."""
    mock_benchmark = [
        "If Mary has 14 apples and gives 3 to John, she has 11 apples left. Calculate the remaining stock."
    ]
    
    actor = CrossDatasetDecontaminator(benchmark_texts=mock_benchmark, ngram_size=5)
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_leak"],
        # Contains an exact overlap of the benchmark evaluating question
        "text": ["Today we learn basic math. If Mary has 14 apples and gives 3 to John, she has 11 apples left. This is clear."]
    })
    
    result = actor(input_table)
    
    # Fails the 13-gram hash (simulated as 5-gram here for test brevity), then fails the LCS ratio check
    assert result.num_rows == 0