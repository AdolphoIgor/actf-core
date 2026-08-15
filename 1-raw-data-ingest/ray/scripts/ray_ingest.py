import argparse
import datetime
import os
import sys
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def parse_args():
    parser = argparse.ArgumentParser(description="Ray Unstructured Data Ingestion Engine")
    parser.add_argument("--last_sync", type=str, default=None, help="High watermark timestamp filter")
    args = parser.parse_args()
    
    # Reconcile Airflow string formatting: Convert 'None' string to Python None
    if args.last_sync in (None, "None", "", "null"):
        args.last_sync = None
    return args

def get_s3_client():
    endpoint = os.getenv("MINIO_ENDPOINT", "http://minio-storage:9000")
    key_id = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key,
        region_name=region
    )

def run_ingestion():
    args = parse_args()
    source_dir = "/opt/airflow/data/unstructured_samples"
    target_bucket = "company-ai-datalake"
    target_prefix = "bronze/local_filesystem/compliance_documents"
    
    print(f"Starting Ray Ingestion. Source: {source_dir}, Last Sync: {args.last_sync}")
    
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        
    s3_client = get_s3_client()
    processed_count = 0
    latest_timestamp = args.last_sync
    
    # Scan source files
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            try:
                file_mtime_epoch = os.path.getmtime(file_path)
            except OSError as e:
                raise RuntimeError(f"FILE STAT FAILURE: Unable to access metadata for '{file_path}': {e}") from e

            file_mtime_iso = datetime.datetime.fromtimestamp(
                file_mtime_epoch, tz=datetime.timezone.utc
            ).isoformat()
            
            # Filter files modified after last_sync (if last_sync is provided)
            if args.last_sync and file_mtime_iso <= args.last_sync:
                continue
                
            s3_key = f"{target_prefix}/{file_name}"
            
            # ========================================================================
            # PROTECTED FILE I/O & NETWORK TRANSPORT LAYER
            # ========================================================================
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                    
                s3_client.put_object(
                    Bucket=target_bucket, 
                    Key=s3_key, 
                    Body=file_bytes
                )
            except (IOError, OSError) as e:
                # Catches local filesystem permission/reading errors
                raise RuntimeError(
                    f"LOCAL FILE READ ERROR: Failed to read document from disk at '{file_path}': {e}"
                ) from e
            except (BotoCoreError, ClientError) as e:
                # Catches MinIO/S3 connection timeouts, socket drops, or credential failures
                raise RuntimeError(
                    f"STORAGE NETWORK ERROR: Failed to transfer '{file_name}' to S3 key '{s3_key}': {e}"
                ) from e
            except Exception as e:
                # Fallback safety net for unexpected runtime errors
                raise RuntimeError(
                    f"UNEXPECTED INGESTION FAILURE processing file '{file_path}': {e}"
                ) from e
                
            processed_count += 1
            if not latest_timestamp or file_mtime_iso > latest_timestamp:
                latest_timestamp = file_mtime_iso

    print(f"Ingestion cycle completed. Processed files count: {processed_count}")

    # ========================================================================
    # ZERO-WRITE GUARD LOGIC
    # ========================================================================
    if processed_count == 0:
        if args.last_sync is None:
            raise RuntimeError(
                f"INITIAL INGESTION FAILURE: Found 0 processable files in '{source_dir}'. "
                "Ensure source sample data is present."
            )
        else:
            print(f"NOTICE: 0 new files found since last sync ({args.last_sync}). No update needed.")
            print(f"RAY_MARKER_MAX_TS:{args.last_sync}")
            return

    # Print updated watermark marker for Airflow parsing on successful write
    print(f"RAY_MARKER_MAX_TS:{latest_timestamp}")

if __name__ == "__main__":
    run_ingestion()