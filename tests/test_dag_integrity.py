import os
import sys
import pytest
from airflow.models import DagBag

# Safeguard: Ensure the dags directory is explicitly on Python's search path
DAGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "orchestrator", "dags"))
if not os.path.exists(DAGS_DIR):
    DAGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dags"))

if DAGS_DIR not in sys.path:
    sys.path.insert(0, DAGS_DIR)


@pytest.fixture(scope="module")
def dagbag():
    """
    Loads all DAG files from the dags folder into an Airflow DagBag instance.
    """
    return DagBag(dag_folder=DAGS_DIR, include_examples=False)


def test_dag_import_errors(dagbag):
    """
    Asserts that no Python syntax errors, missing module imports, 
    or execution failures occurred while parsing DAG files.
    """
    import_errors = dagbag.import_errors
    assert len(import_errors) == 0, f"DAG Import Failures Detected: {import_errors}"


def test_required_dags_exist(dagbag):
    """
    Verifies that all core platform DAGs are loaded and recognized by the scheduler.
    """
    # BUG FIX: Use the actual internal dag_id, not the physical Python filename
    expected_dags = [
        "dag_00_simulation_seed_postgres",
        "dag_01_simulation_seed_files",
        "dag_02_ingest_source_to_bronze",
        "dag_03_prep_bronze_to_silver"
    ]
    
    for dag_id in expected_dags:
        assert dag_id in dagbag.dags, f"DAG '{dag_id}' was not found in DagBag."


def test_dag_configuration_standards(dagbag):
    """
    Enforces enterprise standards on production processing DAGs (catchup=False, max_active_runs=1).
    """
    for dag_id, dag in dagbag.dags.items():
        if dag_id.startswith(("2_", "3_")):
            assert dag.catchup is False, f"DAG '{dag_id}' must set catchup=False."
            assert dag.max_active_runs == 1, f"DAG '{dag_id}' must set max_active_runs=1."


def test_ingest_source_to_bronze_topology(dagbag):
    """
    Validates full 6-task pipeline graph for DAG 'dag_02_ingest_source_to_bronze'.
    """
    # BUG FIX: Reference the correct dag_id
    dag = dagbag.get_dag("dag_02_ingest_source_to_bronze")
    assert len(dag.tasks) == 6, f"Expected 6 tasks in dag_02_ingest_source_to_bronze, found {len(dag.tasks)}"

    task_spark = dag.get_task("spark_parallel_bronze_extraction")
    task_ray = dag.get_task("ray_parallel_unstructured_extraction")
    task_gate = dag.get_task("data_quality_gate_1_bronze_check")

    assert task_gate in task_spark.downstream_list or task_gate in dag.get_task("commit_spark_high_watermark").downstream_list
    assert task_gate in task_ray.downstream_list
    assert task_gate.trigger_rule == "none_failed_min_one_success"


def test_prep_bronze_to_silver_topology(dagbag):
    """
    Validates task graph dependencies for DAG 'dag_03_prep_bronze_to_silver'.
    """
    # BUG FIX: Reference the correct Phase 3 dag_id, not the Phase 2 one
    dag = dagbag.get_dag("dag_03_prep_bronze_to_silver")
    assert len(dag.tasks) == 2, f"Expected 2 tasks in dag_03_prep_bronze_to_silver, found {len(dag.tasks)}"

    task_norm = dag.get_task("ray_step_normalization")
    task_gate_2 = dag.get_task("data_quality_gate_2_silver_check")

    assert task_gate_2 in task_norm.downstream_list
    assert task_gate_2.trigger_rule == "all_success"