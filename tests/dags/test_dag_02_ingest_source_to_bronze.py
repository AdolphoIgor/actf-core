import pytest
from unittest.mock import patch, MagicMock
from airflow.models import DagBag
from dags.dag_02_ingest_source_to_bronze import execute_bronze_quality_gate


def test_dag_02_loading_and_structure():
    """Validates that DAG 02 loads with zero syntax errors or import bugs."""
    dagbag = DagBag(dag_folder="orchestrator/dags", include_examples=False)
    
    # We reference the dag_id, which remains unchanged in the Airflow metadata
    dag = dagbag.get_dag("2_ingest_source_to_bronze") 
    
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) >= 2


@patch("dags.dag_02_ingest_source_to_bronze.S3FileSystem")
@patch("dags.dag_02_ingest_source_to_bronze.ds.dataset")
def test_bronze_gate_passes_valid_raw_data(mock_dataset, mock_s3):
    """Validates Gate 1 passing on healthy ingested raw Bronze files."""
    mock_ds_instance = MagicMock()
    mock_dataset.return_value = mock_ds_instance
    
    mock_ds_instance.schema.names = ["raw_payload", "ingested_at"]
    mock_ds_instance.count_rows.return_value = 500
    
    result = execute_bronze_quality_gate("company-ai-datalake", "bronze/raw_logs/")
    assert result is True


@patch("dags.dag_02_ingest_source_to_bronze.S3FileSystem")
@patch("dags.dag_02_ingest_source_to_bronze.ds.dataset")
def test_bronze_gate_fails_empty_ingestion(mock_dataset, mock_s3):
    """Validates Gate 1 failing when raw ingestion yields 0 records."""
    mock_ds_instance = MagicMock()
    mock_dataset.return_value = mock_ds_instance
    mock_ds_instance.count_rows.return_value = 0
    
    with pytest.raises(ValueError, match="Bronze dataset is completely empty"):
        execute_bronze_quality_gate("company-ai-datalake", "bronze/raw_logs/")