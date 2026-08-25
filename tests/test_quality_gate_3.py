import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dags.scripts.quality_gate_3 import (
    CANARY_BENCHMARKS,
    extract_13grams,
    normalize_and_tokenize,
    run_gate_3_validation,
)


def test_tokenization_and_ngrams():
    """Validates alphanumeric tokenization and 13-gram rolling extraction."""
    text = "The quick brown fox jumps over the lazy dog to test the benchmark pipeline accurately."
    tokens = normalize_and_tokenize(text)

    assert len(tokens) == 15
    assert "the" in tokens
    assert "pipeline" in tokens

    ngrams = extract_13grams(tokens)
    assert len(ngrams) == 3  # 15 tokens -> 3 rolling 13-grams
    assert "the quick brown fox jumps over the lazy dog to test the benchmark" in ngrams


def test_quality_gate_3_passes_clean_data(monkeypatch, tmp_path):
    """Validates Gate 3 passes when dataset is clean and free of leakage."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    # Create a synthetic PyArrow table representing clean records
    table = pa.Table.from_pydict(
        {
            "doc_id": ["clean_001", "clean_002"],
            "text": [
                "This is a perfectly safe corporate compliance document.",
                "Algorithm parameters tuned for high frequency trading protocols.",
            ],
        }
    )
    pq.write_table(table, silver_dir / "part-0000.parquet")

    # Inject mocked path
    monkeypatch.setenv("SILVER_STORAGE_PATH", str(silver_dir))
    monkeypatch.setattr("dags.scripts.quality_gate_3.SILVER_PATH", str(silver_dir))

    # Should execute cleanly without throwing AssertError
    run_gate_3_validation()


def test_quality_gate_3_fails_on_benchmark_leakage(monkeypatch, tmp_path):
    """Validates Gate 3 throws a strict Assertion error on 13-gram benchmark leakage."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    # Fetch a 13-gram chunk directly from the canary config
    canary_text = CANARY_BENCHMARKS[0]["text"]

    table = pa.Table.from_pydict(
        {
            "doc_id": ["leak_001"],
            "text": [f"Here is some prefix text. {canary_text} And some suffix text."],
        }
    )
    pq.write_table(table, silver_dir / "part-0000.parquet")

    monkeypatch.setenv("SILVER_STORAGE_PATH", str(silver_dir))
    monkeypatch.setattr("dags.scripts.quality_gate_3.SILVER_PATH", str(silver_dir))

    with pytest.raises(AssertionError, match="Contamination detected!") as exc_info:
        run_gate_3_validation()

    assert "leak_001" in str(exc_info.value)


def test_quality_gate_3_missing_schema_columns(monkeypatch, tmp_path):
    """Validates Gate 3 fails if the dataset is missing expected target schemas."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    # Missing 'text' and 'doc_id'
    table = pa.Table.from_pydict({"random_id": ["123"], "random_payload": ["text"]})
    pq.write_table(table, silver_dir / "part-0000.parquet")

    monkeypatch.setenv("SILVER_STORAGE_PATH", str(silver_dir))
    monkeypatch.setattr("dags.scripts.quality_gate_3.SILVER_PATH", str(silver_dir))

    with pytest.raises(AssertionError, match="Dataset missing primary text payload column"):
        run_gate_3_validation()
