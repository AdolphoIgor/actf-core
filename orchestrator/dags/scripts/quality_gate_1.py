import os

from airflow.utils.state import TaskInstanceState

# Map upstream task IDs to their target Bronze S3 storage prefixes
STREAM_PREFIX_MAP = {
    "spark_parallel_bronze_extraction": "bronze/postgres_enterprise/compliance_audit/",
    "ray_parallel_unstructured_extraction": "bronze/local_filesystem/compliance_documents/",
}


def execute_bronze_quality_gate(bucket_name="company-ai-datalake", prefix=None, **context):
    """
    Quality Gate 1: Asserts storage contracts on Bronze S3 partitions.
    Supports both dynamic Airflow DAG execution and direct programmatic invocation.
    """
    # Deferred import prevents heavy client initialization during DAG parsing
    import boto3

    dag_run = context.get("dag_run")
    prefixes_to_check = []

    if prefix:
        prefixes_to_check.append(prefix)
    elif dag_run:
        # Dynamically collect prefixes for tasks that succeeded in this run
        for task_id, stream_prefix in STREAM_PREFIX_MAP.items():
            ti = dag_run.get_task_instance(task_id)
            if ti and ti.state == TaskInstanceState.SUCCESS:
                prefixes_to_check.append(stream_prefix)
    else:
        # Standalone default check across all registered streams
        prefixes_to_check = list(STREAM_PREFIX_MAP.values())

    if not prefixes_to_check:
        print("QUALITY GATE 1 SKIPPED: No upstream extraction streams executed in this run.")
        return True

    s3 = boto3.resource(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio-storage:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    )
    bucket = s3.Bucket(bucket_name)

    for p in prefixes_to_check:
        print(f"Commencing Ingestion Gate 1 check for active stream path: {p}")
        objects = list(bucket.objects.filter(Prefix=p))

        if not objects:
            raise ValueError(
                f"QUALITY GATE 1 FAILURE: Expected data but found 0 objects under active path {p}"
            )

        data_files_found = 0
        for obj in objects:
            filename = obj.key.split("/")[-1]

            # Filter out Spark/Hadoop metadata markers and hidden files
            if filename.startswith(("_", ".")) or filename == "_SUCCESS":
                continue

            data_files_found += 1

            if obj.size == 0:
                raise ValueError(
                    f"QUALITY GATE 1 FAILURE: Corrupted zero-byte file found at: {obj.key}"
                )

        if data_files_found == 0:
            raise ValueError(
                f"QUALITY GATE 1 FAILURE: Only metadata markers found under active path {p}"
            )

        print(f"QUALITY GATE 1 PASSED: Verified {data_files_found} data asset(s) under {p}")

    return True
