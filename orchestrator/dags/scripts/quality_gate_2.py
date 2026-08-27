import os

import pyarrow.dataset as ds
import pyarrow.parquet as pq

SILVER_STORAGE_PATH = os.environ.get("SILVER_STORAGE_PATH", "data/silver")


def audit_silver_table(table):
    """
    Validates Silver Parquet schema, null constraints, and text integrity.
    """
    column_names = table.column_names
    assert len(column_names) > 0, "Silver table contains zero columns."
    assert table.num_rows > 0, "Silver table is empty (0 rows)."

    # Required metadata & content fields
    expected_fields = ["id", "text"]
    for field in expected_fields:
        # BUG FIX 1: Strictly assert the column exists rather than silently skipping
        assert field in column_names, f"Required column '{field}' is missing from Silver table."

        col = table[field]
        null_count = col.null_count
        assert null_count == 0, f"Column '{field}' has {null_count} null entries in Silver storage."


def audit_silver_parquet_files(storage_path: str):
    """
    Asserts that all parquet files in Silver partition are readable and valid.
    """
    assert os.path.exists(storage_path), f"Silver storage path '{storage_path}' does not exist."

    # BUG FIX 2: Catch PyArrow's ValueError if the directory is completely empty
    try:
        dataset = ds.dataset(storage_path, format="parquet")
    except Exception as e:
        raise AssertionError(
            f"Failed to load Parquet dataset at '{storage_path}'. Directory might be empty or corrupted. Error: {e}"
        )

    files = dataset.files
    assert len(files) > 0, f"No Parquet files found in Silver path: {storage_path}"

    for file_path in files:
        assert os.path.getsize(file_path) > 0, f"Silver Parquet file is empty: {file_path}"
        parquet_file = pq.ParquetFile(file_path)
        assert parquet_file.metadata.num_rows > 0, f"Silver Parquet has 0 rows: {file_path}"


def run_gate_2_validation(storage_path: str = SILVER_STORAGE_PATH):
    """
    Main entry point for Quality Gate 2 (Silver Parquet Cleanliness Gate).
    """
    print(f"[QUALITY GATE 2] Validating Silver layer at: {storage_path}")
    audit_silver_parquet_files(storage_path)

    dataset = ds.dataset(storage_path, format="parquet")
    table = dataset.to_table()
    audit_silver_table(table)

    print(
        f"[QUALITY GATE 2] PASSED: Successfully validated {table.num_rows} rows across Silver partitions."
    )


if __name__ == "__main__":
    run_gate_2_validation()
