import os
import glob
import time
from datetime import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.exceptions import AirflowSkipException
from airflow.utils.trigger_rule import TriggerRule

# Clean import from the scripts directory
from scripts.quality_gate_1 import execute_bronze_quality_gate

default_args = {
    'owner': 'ai_ops',
    'depends_on_past': False,
    'start_date': datetime(2026, 6, 9),
    'retries': 0,
}

DAG_DOC_MD = """
# Hybrid Bronze Ingestion Pipeline (`dag_02_ingest_source_to_bronze`)

Extracts raw structured logs from PostgreSQL (via PySpark) and unstructured text documents from the landing zone (via Ray Data) into the Medallion Bronze Parquet store.
"""

def resolve_high_watermark(state_key):
    try:
        return Variable.get(state_key)
    except KeyError:
        return "None"

def check_postgres_delta(state_key):
    last_ts = resolve_high_watermark(state_key)
    pg_hook = PostgresHook(postgres_conn_id='airflow_db')
    
    if last_ts != "None":
        query = f"""
            SELECT COUNT(1) 
            FROM production_logs.compliance_audit_ledger 
            WHERE corporate_division NOT IN ('Human Resources', 'Facility Operations') 
              AND created_at > '{last_ts}';
        """
    else:
        query = """
            SELECT COUNT(1) 
            FROM production_logs.compliance_audit_ledger 
            WHERE corporate_division NOT IN ('Human Resources', 'Facility Operations');
        """
        
    count = pg_hook.get_first(query)[0]
    print(f"Pre-processing Postgres Check: Found {count} new records since high-watermark '{last_ts}'.")
    
    if count == 0:
        raise AirflowSkipException("Zero new database records found. Skipping Spark ingestion execution.")

def check_files_delta(state_key):
    last_sync = resolve_high_watermark(state_key)
    landing_zone_dir = "/opt/airflow/data/unstructured_samples"
    
    if not os.path.exists(landing_zone_dir):
        raise AirflowSkipException("Landing zone directory does not exist. Skipping Ray ingestion.")
        
    all_files = glob.glob(os.path.join(landing_zone_dir, "*.*"))
    new_files_count = 0
    
    for f in all_files:
        mtime = datetime.fromtimestamp(os.path.getmtime(f)).isoformat()
        if last_sync == "None" or mtime > last_sync:
            new_files_count += 1
            
    print(f"Pre-processing File Check: Found {new_files_count} new documents since high-watermark '{last_sync}'.")
    
    if new_files_count == 0:
        raise AirflowSkipException("Zero new files found in landing zone. Skipping Ray job submission.")

def capture_spark_state(task_instance, state_key):
    logs = task_instance.xcom_pull(task_ids='spark_parallel_bronze_extraction')
    if logs:
        for line in str(logs).split('\n'):
            if "XCOM_MARKER_MAX_TS:" in line:
                new_ts = line.split("XCOM_MARKER_MAX_TS:")[-1].strip()
                Variable.set(state_key, new_ts)
                print(f"Stored updated Spark sync marker: {new_ts}")

def submit_ray_job(state_key):
    import requests
    
    last_sync = resolve_high_watermark(state_key)
    ray_dashboard_url = "http://ray-head:8265"
    script_path = "/home/ray/workspace/1-raw-data-ingest/ray/scripts/ray_ingest.py"
    
    job_payload = {
        "entrypoint": f"python {script_path} --last_sync {last_sync}"
    }
    
    print(f"Submitting job to Ray Cluster: {job_payload}")
    resp = requests.post(f"{ray_dashboard_url}/api/jobs/", json=job_payload)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    
    while True:
        status_resp = requests.get(f"{ray_dashboard_url}/api/jobs/{job_id}")
        status = status_resp.json()["status"]
        print(f"Ray Job {job_id} Status: {status}")
        
        if status in ["SUCCEEDED", "STOPPED", "FAILED"]:
            break
        time.sleep(5)
        
    logs_resp = requests.get(f"{ray_dashboard_url}/api/jobs/{job_id}/logs")
    if logs_resp.status_code == 200:
        job_logs = logs_resp.json().get("logs", "")
        if job_logs: 
            print(f"\n=================== RAY JOB LOGS ({job_id}) ===================\n")
            print(job_logs)
            print("\n===============================================================\n")

    if status != "SUCCEEDED":
        raise Exception(f"Ray Job {job_id} failed with status: {status}. Review logs above for details.")
        
    for line in job_logs.split('\n'):
        if "RAY_MARKER_MAX_TS:" in line:
            new_ts = line.split("RAY_MARKER_MAX_TS:")[-1].strip()
            Variable.set(state_key, new_ts)
            print(f"Stored updated Ray sync marker: {new_ts}")

with DAG(
    dag_id='dag_02_ingest_source_to_bronze',
    default_args=default_args,
    description='Pipeline 2: Ingests dynamic structured DB rows and raw files to Bronze via Spark & Ray',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['regtech', 'bronze_ingest'],
    doc_md=DAG_DOC_MD
) as dag:

    sensor_postgres = PythonOperator(
        task_id='sense_postgres_delta',
        python_callable=check_postgres_delta,
        op_kwargs={'state_key': 'last_spark_compliance_ts'}
    )

    sensor_files = PythonOperator(
        task_id='sense_files_delta',
        python_callable=check_files_delta,
        op_kwargs={'state_key': 'last_ray_files_ts'}
    )

    spark_ts = resolve_high_watermark("last_spark_compliance_ts")

    extract_spark = SparkSubmitOperator(
        task_id='spark_parallel_bronze_extraction',
        application='/opt/airflow/1-raw-data-ingest/spark/scripts/spark_ingest.py',
        conn_id='spark_default',
        application_args=[spark_ts],
        packages='org.postgresql:postgresql:42.7.3',
        env_vars={
            'MINIO_ENDPOINT': 'http://minio-storage:9000',
            'AWS_ACCESS_KEY_ID': 'minioadmin',
            'AWS_SECRET_ACCESS_KEY': 'minioadmin'
        },
        conf={
            "spark.master": "spark://spark-master:7077",
            "spark.executor.memory": "1g",
            "spark.executor.cores": "1"
        }
    )

    update_spark_state = PythonOperator(
        task_id='commit_spark_high_watermark',
        python_callable=capture_spark_state,
        op_kwargs={'state_key': 'last_spark_compliance_ts'}
    )

    extract_ray = PythonOperator(
        task_id='ray_parallel_unstructured_extraction',
        python_callable=submit_ray_job,
        op_kwargs={'state_key': 'last_ray_files_ts'}
    )

    quality_gate_validation = PythonOperator(
        task_id='data_quality_gate_1_bronze_check',
        python_callable=execute_bronze_quality_gate,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        op_kwargs={
            'bucket_name': 'company-ai-datalake'
        }
    )

    sensor_postgres >> extract_spark >> update_spark_state
    sensor_files >> extract_ray
    [update_spark_state, extract_ray] >> quality_gate_validation