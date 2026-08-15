import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_enterprise_bronze_ingest(last_processed_ts=None):
    print(f"Starting Spark Extraction. High-Watermark State Check: {last_processed_ts}")
    
    spark = SparkSession.builder \
        .appName("ACTF-RegTech-Bronze-Spark") \
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT", "http://minio-storage:9000")) \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

    jdbc_url = "jdbc:postgresql://postgres-db:5432/enterprise_db"
    
    # FIX: Corrected column reference from 'updated_at' to 'created_at' to match seed schema
    if last_processed_ts and last_processed_ts != "None":
        filter_clause = f"WHERE corporate_division NOT IN ('Human Resources', 'Facility Operations') AND created_at > '{last_processed_ts}'"
        write_mode = "append"
    else:
        filter_clause = "WHERE corporate_division NOT IN ('Human Resources', 'Facility Operations')"
        write_mode = "overwrite"

    base_properties = {
        "user": "db_extraction_user",
        "password": "secure_password_123",
        "driver": "org.postgresql.Driver"
    }

    bounds_query = f"""
        (SELECT MIN(id) as min_id, MAX(id) as max_id 
         FROM production_logs.compliance_audit_ledger 
         {filter_clause}) AS bounds
    """
    
    print("Evaluating database boundary indexes for dynamic partitioning...")
    try:
        bounds_df = spark.read.jdbc(url=jdbc_url, table=bounds_query, properties=base_properties)
        bounds_row = bounds_df.collect()[0]
        min_id = bounds_row["min_id"]
        max_id = bounds_row["max_id"]
    except Exception as e:
        print(f"Failed to query database index boundaries: {str(e)}")
        spark.stop()
        sys.exit(1)

    if min_id is None or max_id is None:
        print("Empty delta sequence. No matching rows found to ingest. Exiting safely.")
        spark.stop()
        return

    print(f"Resolved dynamic extraction boundaries: MIN(id)={min_id}, MAX(id)={max_id}")

    total_range = max_id - min_id
    calculated_partitions = max(1, min(8, total_range // 500))
    print(f"Allocating {calculated_partitions} parallel partition streams for this run.")

    connection_properties = {
        **base_properties,
        "partitionColumn": "id",
        "lowerBound": str(min_id),
        "upperBound": str(max_id),
        "numPartitions": str(calculated_partitions)
    }

    # FIX: Corrected ts_marker alias to use 'created_at'
    target_query = f"""
        (SELECT *, created_at::text as ts_marker FROM production_logs.compliance_audit_ledger 
         {filter_clause}) AS filtered_audit
    """
    
    df = spark.read.jdbc(url=jdbc_url, table=target_query, properties=connection_properties)
    
    max_ts = df.agg(F.max("ts_marker")).collect()[0][0]
    
    df_partitioned = df \
        .withColumn("year", F.date_format(F.col("created_at"), "yyyy")) \
        .withColumn("month", F.date_format(F.col("created_at"), "MM")) \
        .withColumn("day", F.date_format(F.col("created_at"), "dd")) \
        .coalesce(1)

    target_bronze_path = "s3a://company-ai-datalake/bronze/postgres_enterprise/compliance_audit/"
    
    print(f"Writing partitioned Parquet files to: {target_bronze_path}")
    df_partitioned.write \
        .mode(write_mode) \
        .format("parquet") \
        .partitionBy("year", "month", "day") \
        .save(target_bronze_path)

    print(f"XCOM_MARKER_MAX_TS:{max_ts}")
    spark.stop()

if __name__ == "__main__":
    passed_ts = sys.argv[1] if len(sys.argv) > 1 else None
    run_enterprise_bronze_ingest(passed_ts)