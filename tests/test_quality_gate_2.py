import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dags.scripts.quality_gate_2 import run_gate_2_validation


def test_quality_gate_2_passes_on_valid_silver_data(monkeypatch, tmp_path):
    """Validates Gate 2 passes when the Silver dataset exists, is populated, and has the correct schema."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    table = pa.Table.from_pydict(
        {
            "doc_id": ["doc_001", "doc_002"],
            "text": ["Curated prose text.", "def valid_code(): pass"],
            "branch_id": [0, 1],
        }
    )
    pq.write_table(table, silver_dir / "part-0000.parquet")

    monkeypatch.setenv("SILVER_STORAGE_PATH", str(silver_dir))
    monkeypatch.setattr("dags.scripts.quality_gate_2.SILVER_PATH", str(silver_dir))

    assert run_gate_2_validation(str(silver_dir)) is True


def test_quality_gate_2_fails_on_missing_directory(monkeypatch, tmp_path):
    """Validates Gate 2 throws an AssertionError if the Silver storage path does not exist."""
    missing_dir = tmp_path / "does_not_exist"

    monkeypatch.setenv("SILVER_STORAGE_PATH", str(missing_dir))
    monkeypatch.setattr("dags.scripts.quality_gate_2.SILVER_PATH", str(missing_dir))

    with pytest.raises(AssertionError, match="does not exist"):
        run_gate_2_validation(str(missing_dir))


def test_quality_gate_2_fails_on_empty_dataset(monkeypatch, tmp_path):
    """Validates Gate 2 throws an AssertionError if the Silver dataset contains zero records."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    schema = pa.schema([("doc_id", pa.string()), ("text", pa.string()), ("branch_id", pa.int32())])
    empty_table = pa.Table.from_batches([], schema=schema)
    pq.write_table(empty_table, silver_dir / "part-0000.parquet")

    monkeypatch.setenv("SILVER_STORAGE_PATH", str(silver_dir))
    monkeypatch.setattr("dags.scripts.quality_gate_2.SILVER_PATH", str(silver_dir))

    with pytest.raises(AssertionError, match="0 records|empty"):
        run_gate_2_validation(str(silver_dir))


def test_quality_gate_2_fails_on_invalid_schema(monkeypatch, tmp_path):
    """Validates Gate 2 throws an AssertionError if the required text or ID columns are missing."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    table = pa.Table.from_pydict({"random_metadata": ["meta1"], "timestamp": ["2026-06-03"]})
    pq.write_table(table, silver_dir / "part-0000.parquet")

    monkeypatch.setenv("SILVER_STORAGE_PATH", str(silver_dir))
    monkeypatch.setattr("dags.scripts.quality_gate_2.SILVER_PATH", str(silver_dir))

    with pytest.raises(AssertionError, match="missing|column"):
        run_gate_2_validation(str(silver_dir))
