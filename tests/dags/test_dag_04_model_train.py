import pytest
from airflow.models import DagBag


@pytest.fixture
def dagbag():
    return DagBag(dag_folder="orchestrator/dags", include_examples=False)


def test_dag_04_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="4_train_and_eval_model")
    assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"
    assert dag is not None
    assert len(dag.tasks) == 6


def test_dag_04_task_order(dagbag):
    dag = dagbag.get_dag(dag_id="4_train_and_eval_model")

    gate3_task = dag.get_task("data_quality_gate_3_leakage_check")
    tokenize_task = dag.get_task("step_11_12_tokenization_and_packing")
    gate4_task = dag.get_task("tensor_quality_gate_4_preflight_check")
    train_task = dag.get_task("step_13_14_training_and_staging")
    eval_task = dag.get_task("step_15_16_eval_and_gate_05_gatekeeper")
    promote_task = dag.get_task("step_17_mlflow_model_promotion")

    assert tokenize_task in gate3_task.downstream_list
    assert gate4_task in tokenize_task.downstream_list
    assert train_task in gate4_task.downstream_list
    assert eval_task in train_task.downstream_list
    assert promote_task in eval_task.downstream_list


def test_dag_04_default_args(dagbag):
    dag = dagbag.get_dag(dag_id="4_train_and_eval_model")
    assert dag.default_args["owner"] == "airflow"
    assert dag.default_args["retries"] == 0
    assert dag.catchup is False
    assert "training" in dag.tags
    assert "evaluation" in dag.tags
