import os
import time
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Clean import from the Airflow scripts directory
from scripts.quality_gate_2 import execute_silver_quality_gate

default_args = {
    'owner': 'ai_ops',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def submit_ray_prep_job(script_name: str, input_path: str, output_path: str):
    import requests
    
    ray_dashboard_url = "http://ray-head:8265"
    script_full_path = f"/home/ray/workspace/2-data-prep/scripts/{script_name}"
    
    entrypoint_cmd = f"python {script_full_path} --input_path {input_path} --output_path {output_path}"
    job_payload = {"entrypoint": entrypoint_cmd}
    
    print(f"Submitting Data Prep job to Ray: {job_payload}")
    
    try:
        resp = requests.post(f"{ray_dashboard_url}/api/jobs/", json=job_payload, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = resp.text if 'resp' in locals() and resp is not None else str(e)
        raise Exception(f"Failed to submit Ray job. Response: {error_msg}")

    job_id = resp.json()["job_id"]
    
    current_sleep = 5
    max_sleep = 60

    while True:
        try:
            status_resp = requests.get(f"{ray_dashboard_url}/api/jobs/{job_id}", timeout=10)
            status_resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to fetch status for job {job_id}. Retrying... Error: {e}")
            time.sleep(current_sleep)
            continue

        status = status_resp.json()["status"]
        print(f"Ray Job {job_id} [{script_name}] Status: {status}")
        
        if status in ["SUCCEEDED", "STOPPED", "FAILED"]:
            break
            
        time.sleep(current_sleep)
        current_sleep = min(current_sleep + 5, max_sleep)
        
    try:
        logs_resp = requests.get(f"{ray_dashboard_url}/api/jobs/{job_id}/logs", timeout=15)
        job_logs = logs_resp.json().get("logs", "") if logs_resp.status_code == 200 else "Unable to retrieve logs."
    except requests.exceptions.RequestException:
        job_logs = "Request timeout while fetching logs."
    
    print(f"\n=================== RAY JOB LOGS ({job_id}) ===================\n")
    print(job_logs)
    print("\n===============================================================\n")

    if status != "SUCCEEDED":
        raise Exception(f"Ray Data Prep Job {job_id} ({script_name}) failed with status: {status}.")


with DAG(
    dag_id='dag_03_prep_bronze_to_silver',
    default_args=default_args,
    description='Silver Data Preparation: Normalization, Cleaning, and Quality Filtering',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
) as dag:

    step_normalization = PythonOperator(
        task_id='ray_step_normalization',
        python_callable=submit_ray_prep_job,
        op_kwargs={
            'script_name': 'phase_01_shared_ingestion/step_01_normalization.py',
            'input_path': 's3://company-ai-datalake/bronze/local_filesystem/compliance_documents/',
            'output_path': 's3://company-ai-datalake/silver/local_filesystem/compliance_documents/normalized/'
        },
    )

    data_quality_gate_2_silver_check = PythonOperator(
        task_id='data_quality_gate_2_silver_check',
        python_callable=execute_silver_quality_gate,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        op_kwargs={
            'bucket_name': 'company-ai-datalake',
            'prefix': 'silver/local_filesystem/compliance_documents/normalized/'
        },
    )

    step_normalization >> data_quality_gate_2_silver_check