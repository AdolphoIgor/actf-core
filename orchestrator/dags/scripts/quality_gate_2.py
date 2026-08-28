import os

import pyarrow.dataset as ds
import pyarrow.parquet as pq

SILVER_STORAGE_PATH = os.environ.get("SILVER_STORAGE_PATH", "data/silver")
SILVER_PATH = SILVER_STORAGE_PATH  # Alias for backward-compatibility with legacy test runners


def audit_silver_table(table):
    """
    Validates Silver Parquet schema, null constraints, and text integrity.
    Supports either 'doc_id' or 'id' as the unique entity key alongside 'text'.
    """
    column_names = table.column_names
    assert len(column_names) > 0, "Silver table contains zero columns."
    assert table.num_rows > 0, "Silver table is empty (0 records)."

    # Validate tracking identifier ('doc_id' or 'id')
    has_id_field = "doc_id" in column_names or "id" in column_names
    assert has_id_field, (
        "Required tracking identifier ('doc_id' or 'id') is missing from Silver table columns."
    )

    id_field = "doc_id" if "doc_id" in column_names else "id"
    id_null_count = table[id_field].null_count
    assert id_null_count == 0, (
        f"Column '{id_field}' has {id_null_count} null entries in Silver storage."
    )

    # Validate mandatory text payload field
    assert "text" in column_names, "Required column 'text' is missing from Silver table columns."
    text_null_count = table["text"].null_count
    assert text_null_count == 0, (
        f"Column 'text' has {text_null_count} null entries in Silver storage."
    )


def audit_silver_parquet_files(storage_path: str):
    """
    Asserts that all parquet files in the Silver partition are readable and non-empty.
    """
    assert os.path.exists(storage_path), f"Silver storage path '{storage_path}' does not exist."

    try:
        dataset = ds.dataset(storage_path, format="parquet")
    except Exception as e:
        raise AssertionError(
            f"Failed to load Parquet dataset at '{storage_path}'. Directory might be empty or corrupted. Error: {e}"
        )

    files = dataset.files
    assert len(files) > 0, f"No Parquet files found in Silver path: {storage_path}"

    for file_path in files:
        assert os.path.getsize(file_path) > 0, (
            f"Silver Parquet file is empty (0 bytes): {file_path}"
        )
        parquet_file = pq.ParquetFile(file_path)
        assert parquet_file.metadata.num_rows > 0, (
            f"Silver Parquet file is empty (0 records): {file_path}"
        )


def run_gate_2_validation(storage_path: str = None, **context):
    """
    Main entry point for Quality Gate 2 (Silver Parquet Cleanliness Gate).
    Dynamically resolves the storage path from environment variables at execution time.
    """
    if storage_path is None:
        storage_path = os.environ.get(
            "SILVER_STORAGE_PATH", os.environ.get("SILVER_PATH", SILVER_STORAGE_PATH)
        )

    print(f"[QUALITY GATE 2] Validating Silver layer at: {storage_path}")
    audit_silver_parquet_files(storage_path)

    dataset = ds.dataset(storage_path, format="parquet")
    table = dataset.to_table()
    audit_silver_table(table)

    print(
        f"[QUALITY GATE 2] PASSED: Successfully validated {table.num_rows} rows across Silver partitions."
    )
    return True


# Alias for DAG operator task mapping
execute_silver_quality_gate = run_gate_2_validation

if __name__ == "__main__":
    run_gate_2_validation()
