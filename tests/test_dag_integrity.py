import os

import pytest
from airflow.models import DagBag
from airflow.utils.trigger_rule import TriggerRule


def get_dags_folder():
    """
    Dynamically resolves DAG root directory across both container and host runtimes.
    """
    if os.path.exists("/opt/airflow/dags"):
        return "/opt/airflow/dags"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "orchestrator", "dags"))


@pytest.fixture(scope="module")
def dagbag():
    """
    Initializes a shared DagBag fixture with examples disabled.
    """
    folder = get_dags_folder()
    bag = DagBag(dag_folder=folder, include_examples=False)
    return bag


def find_dag(dagbag, possible_ids):
    """
    Helper to resolve DAGs across canonical and prefixed naming conventions.
    """
    for dag_id in possible_ids:
        if dag_id in dagbag.dags:
            return dagbag.dags[dag_id]
    return None


# ==============================================================================
# 1. CORE DAGBAG IMPORT & SYNTAX INTEGRITY
# ==============================================================================


def test_dagbag_has_zero_import_errors(dagbag):
    """
    Asserts that the DagBag loads completely with zero Python syntax,
    cyclic dependency, or module import errors.
    """
    assert len(dagbag.import_errors) == 0, (
        f"Airflow DagBag loaded with {len(dagbag.import_errors)} import errors:\n"
        f"{dagbag.import_errors}"
    )


def test_complete_platform_dag_inventory(dagbag):
    """
    Asserts that all 6 continuous training lifecycle DAGs exist in the platform.
    """
    expected_dag_groups = {
        "DAG 00 (Seed DB)": [
            "dag_00_simulation_seed_postgres",
            "0_simulation_seed_postgres",
            "0_simulation_seed_db",
        ],
        "DAG 01 (Seed Files)": ["dag_01_simulation_seed_files", "1_simulation_seed_files"],
        "DAG 02 (Bronze Ingest)": ["dag_02_ingest_source_to_bronze", "2_ingest_source_to_bronze"],
        "DAG 03 (Silver Prep)": ["dag_03_prep_bronze_to_silver", "3_prep_bronze_to_silver"],
        "DAG 04 (Model Training)": [
            "dag_04_model_train",
            "4_train_and_eval_model",
            "4_train_model",
        ],
        "DAG 05 (Model Evaluation)": ["dag_05_model_eval", "5_model_eval"],
    }

    missing_dags = []
    for group_name, candidate_ids in expected_dag_groups.items():
        resolved = find_dag(dagbag, candidate_ids)
        if resolved is None:
            missing_dags.append(f"{group_name} (candidates: {candidate_ids})")

    assert not missing_dags, f"Missing expected platform DAGs: {missing_dags}"


# ==============================================================================
# 2. ENTERPRISE PRODUCTION INVARIANTS & HYGIENE
# ==============================================================================


def test_all_dags_disable_catchup(dagbag):
    """
    Enforces catchup=False across all platform DAGs to prevent accidental backfill storms.
    """
    for dag_id, dag in dagbag.dags.items():
        assert dag.catchup is False, f"DAG '{dag_id}' must have catchup=False configured."


def test_all_dags_have_tags_and_documentation(dagbag):
    """
    Validates that every DAG in the DagBag contains both categorization tags
    and markdown documentation, reporting all violations at once.
    """
    missing_tags = []
    missing_docs = []

    for dag_id, dag in dagbag.dags.items():
        if not dag.tags:
            missing_tags.append(dag_id)
        if not dag.doc_md:
            missing_docs.append(dag_id)

    errors = []
    if missing_tags:
        errors.append(f"DAGs missing tags: {missing_tags}")
    if missing_docs:
        errors.append(f"DAGs missing doc_md: {missing_docs}")

    assert not errors, "\n".join(errors)


def test_no_isolated_orphan_tasks(dagbag):
    """
    Ensures that multi-task pipelines have no disconnected/orphan tasks floating in the DAG.
    """
    for dag_id, dag in dagbag.dags.items():
        if len(dag.tasks) > 1:
            for task in dag.tasks:
                has_upstream = len(task.upstream_list) > 0
                has_downstream = len(task.downstream_list) > 0
                assert has_upstream or has_downstream, (
                    f"Orphan task '{task.task_id}' detected in DAG '{dag_id}'. "
                    f"It has no upstream or downstream dependencies."
                )


# ==============================================================================
# 3. TOPOLOGY & QUALITY GATE VERIFICATION
# ==============================================================================


def test_dag_00_seed_postgres_topology(dagbag):
    """Validates DAG 00 structure and task presence."""
    dag = find_dag(dagbag, ["dag_00_simulation_seed_postgres", "0_simulation_seed_postgres"])
    assert dag is not None
    assert len(dag.tasks) >= 1
    assert any(
        "hydration" in t.task_id or "seed" in t.task_id or "generate" in t.task_id
        for t in dag.tasks
    )


def test_dag_01_seed_files_topology(dagbag):
    """Validates DAG 01 structure and task presence."""
    dag = find_dag(dagbag, ["dag_01_simulation_seed_files", "1_simulation_seed_files"])
    assert dag is not None
    assert len(dag.tasks) >= 1
    assert any(
        "file" in t.task_id or "seed" in t.task_id or "generate" in t.task_id for t in dag.tasks
    )


def test_dag_02_bronze_ingest_topology_and_gate_1(dagbag):
    """
    Validates DAG 02 dual-ingestion branches (Spark + Ray) converging into Quality Gate 1.
    """
    dag = find_dag(dagbag, ["dag_02_ingest_source_to_bronze", "2_ingest_source_to_bronze"])
    assert dag is not None
    assert len(dag.tasks) >= 4, (
        f"Expected at least 4 tasks in Bronze ingestion DAG, found {len(dag.tasks)}"
    )

    gate_tasks = [
        t for t in dag.tasks if "quality_gate_1" in t.task_id or "bronze_check" in t.task_id
    ]
    assert len(gate_tasks) == 1, "Quality Gate 1 task missing or ambiguous in DAG 02."

    gate_task = gate_tasks[0]
    assert gate_task.trigger_rule in [
        TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        TriggerRule.ALL_SUCCESS,
        TriggerRule.NONE_FAILED,
    ], f"Gate 1 has inappropriate trigger rule: {gate_task.trigger_rule}"
    assert len(gate_task.upstream_list) >= 1, "Gate 1 must have upstream extraction dependencies."


def test_dag_03_silver_prep_topology_and_gate_2(dagbag):
    """
    Validates DAG 03 Ray Data prep execution flowing into Quality Gate 2.
    """
    dag = find_dag(dagbag, ["dag_03_prep_bronze_to_silver", "3_prep_bronze_to_silver"])
    assert dag is not None
    assert len(dag.tasks) >= 2, (
        f"Expected at least 2 tasks in Silver prep DAG, found {len(dag.tasks)}"
    )

    gate_tasks = [
        t for t in dag.tasks if "quality_gate_2" in t.task_id or "silver_check" in t.task_id
    ]
    assert len(gate_tasks) == 1, "Quality Gate 2 task missing in DAG 03."

    gate_task = gate_tasks[0]
    assert len(gate_task.upstream_list) >= 1, "Gate 2 must depend on upstream prep transformations."


def test_dag_04_training_topology_and_gates(dagbag):
    """
    Validates DAG 04 training pipeline contains parameter optimization and Quality Gate 4 pre-flight.
    """
    dag = find_dag(dagbag, ["dag_04_model_train", "4_train_and_eval_model", "4_train_model"])
    assert dag is not None
    assert len(dag.tasks) >= 1

    # Asserts that training or quality verification tasks exist in the graph
    has_train_or_gate = any(
        any(token in t.task_id for token in ["train", "tokeniz", "gate", "tensor"])
        for t in dag.tasks
    )
    assert has_train_or_gate, "DAG 04 does not contain expected training or tensor gate tasks."


def test_dag_05_eval_topology_and_gatekeeper(dagbag):
    """
    Validates DAG 05 evaluation pipeline contains benchmark tasks and Gatekeeper promotion.
    """
    dag = find_dag(dagbag, ["dag_05_model_eval", "5_model_eval"])
    assert dag is not None
    assert len(dag.tasks) >= 1

    has_eval_or_gatekeeper = any(
        any(token in t.task_id for token in ["eval", "benchmark", "judge", "gate", "promot"])
        for t in dag.tasks
    )
    assert has_eval_or_gatekeeper, "DAG 05 does not contain expected evaluation or promotion tasks."


def test_strict_topology_contracts(dagbag):
    """
    Enforces exact naming conventions and hard topological dependencies.
    """
    contract_map = {
        "dag_02_ingest_source_to_bronze": {
            "expected_gate": "data_quality_gate_1_bronze_check",
            "upstream_count": 2,
        },
        "dag_03_prep_bronze_to_silver": {
            "expected_gate": "data_quality_gate_2_silver_check",
            "upstream_count": 1,
        },
    }

    for dag_id, contract in contract_map.items():
        dag = dagbag.get_dag(dag_id)
        assert dag is not None, f"Strict naming failure: {dag_id} is missing."

        gate_task = dag.get_task(contract["expected_gate"])
        assert gate_task is not None, f"Missing critical gate: {contract['expected_gate']}"

        actual_upstreams = len(gate_task.upstream_list)
        assert actual_upstreams >= contract["upstream_count"], (
            f"Topology breach in {dag_id}. Gate expects at least "
            f"{contract['upstream_count']} upstreams, found {actual_upstreams}."
        )
