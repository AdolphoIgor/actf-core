import pyarrow as pa
import pytest
from dags.scripts.quality_gate_2 import (
    audit_silver_parquet_files,
    audit_silver_table,
    run_gate_2_validation,
)


def test_audit_silver_table_valid():
    table = pa.table(
        {
            "id": pa.array(["doc_1", "doc_2", "doc_3"]),
            "text": pa.array(["Sample clean text A", "Sample clean text B", "Sample clean text C"]),
        }
    )
    audit_silver_table(table)


def test_audit_silver_table_empty():
    table = pa.table({"id": pa.array([]), "text": pa.array([])})
    with pytest.raises(AssertionError, match="Silver table is empty"):
        audit_silver_table(table)


def test_audit_silver_table_missing_columns():
    table = pa.table(
        {
            "id": pa.array(["doc_1", "doc_2"]),
            # Intentionally omitting "text" to test the strict schema assertion
        }
    )
    with pytest.raises(AssertionError, match="Required column 'text' is missing from Silver table"):
        audit_silver_table(table)


def test_audit_silver_table_null_ids():
    table = pa.table(
        {
            "id": pa.array(["doc_1", None, "doc_3"]),
            "text": pa.array(["Text 1", "Text 2", "Text 3"]),
        }
    )
    with pytest.raises(AssertionError, match="has 1 null entries"):
        audit_silver_table(table)


def test_audit_silver_parquet_files_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty_silver"
    empty_dir.mkdir()

    with pytest.raises(AssertionError, match="Failed to load Parquet dataset"):
        audit_silver_parquet_files(str(empty_dir))


def test_run_gate_2_validation_success(tmp_path):
    import pyarrow.parquet as pq

    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "id": pa.array(["1", "2"]),
            "text": pa.array(["Valid text A", "Valid text B"]),
        }
    )
    pq.write_table(table, silver_dir / "part-0.parquet")

    run_gate_2_validation(str(silver_dir))
