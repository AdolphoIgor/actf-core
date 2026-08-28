import pytest
from airflow.models import DagBag


@pytest.fixture
def dagbag():
    return DagBag(dag_folder="orchestrator/dags", include_examples=False)


def test_dag_05_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="dag_05_model_eval")
    assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"
    assert dag is not None
    assert len(dag.tasks) == 4


def test_dag_05_task_dependencies(dagbag):
    dag = dagbag.get_dag(dag_id="dag_05_model_eval")

    benchmarks_task = dag.get_task("step_15_gold_benchmark_evaluation")
    judge_task = dag.get_task("step_16_llm_judge_scoring")
    gatekeeper_task = dag.get_task("gate_05_automated_gatekeeper")
    promotion_task = dag.get_task("step_17_mlflow_model_promotion")

    assert judge_task in benchmarks_task.downstream_list
    assert gatekeeper_task in judge_task.downstream_list
    assert promotion_task in gatekeeper_task.downstream_list


def test_dag_05_metadata_and_defaults(dagbag):
    dag = dagbag.get_dag(dag_id="dag_05_model_eval")
    assert dag.default_args["owner"] == "airflow"
    assert dag.default_args["retries"] == 0
    assert dag.catchup is False
    assert "evaluation" in dag.tags
    assert "gatekeeper" in dag.tags
    assert "promotion" in dag.tags
