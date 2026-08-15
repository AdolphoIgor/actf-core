import os

def execute_silver_quality_gate(bucket_name: str, prefix: str, **context):
    """
    Gate 2: Pre-Tokenization Quality Gate.
    Verifies that the silver data partition contains valid, non-empty, schema-compliant Parquet files.
    """
    # Defer heavy imports inside the callable to prevent DAG parsing errors
    import pyarrow.dataset as ds
    from pyarrow.fs import S3FileSystem

    print(f"Executing Gate 2 (Silver Quality Check) on s3://{bucket_name}/{prefix}")
    
    # Initialize S3 filesystem using MinIO local endpoints
    s3_fs = S3FileSystem(
        endpoint_override=os.environ.get("MINIO_ENDPOINT", "http://minio-storage:9000"),
        access_key=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        scheme="http",
        region="us-east-1"
    )

    dataset_path = f"{bucket_name}/{prefix}"
    
    try:
        dataset = ds.dataset(dataset_path, format="parquet", filesystem=s3_fs)
    except FileNotFoundError:
        raise RuntimeError(f"QUALITY GATE FAILED: No Parquet dataset found at {dataset_path}")

    # 1. Row Count Validation
    total_rows = dataset.count_rows()
    print(f"Detected {total_rows} total rows in Silver partition.")
    if total_rows == 0:
        raise ValueError("QUALITY GATE FAILED: Silver dataset is completely empty.")

    # 2. Schema Validation
    required_columns = {"text"}
    actual_columns = set(dataset.schema.names)
    
    if not required_columns.issubset(actual_columns):
        missing = required_columns - actual_columns
        raise ValueError(f"QUALITY GATE FAILED: Schema missing required columns: {missing}")

    # 3. Null Field Audit
    batches = dataset.to_batches(columns=["text"])
    null_count = sum(batch.column("text").null_count for batch in batches)
        
    if null_count > 0:
        raise ValueError(f"QUALITY GATE FAILED: Found {null_count} null records in the 'text' column.")

    print("Gate 2 Passed: Silver data is valid, non-null, and ready for tokenization.")
    return True