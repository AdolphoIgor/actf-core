import os
from unittest.mock import MagicMock, patch

import pytest
from airflow.models import DagBag
from dags.dag_02_ingest_source_to_bronze import execute_bronze_quality_gate


def get_dags_folder():
    if os.path.exists("/opt/airflow/dags"):
        return "/opt/airflow/dags"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dags"))


def test_dag_02_loading_and_structure():
    """Validates that DAG 02 loads with zero syntax errors or import bugs."""
    dagbag = DagBag(dag_folder=get_dags_folder(), include_examples=False)

    dag = dagbag.get_dag("2_ingest_source_to_bronze")
    if dag is None:
        dag = dagbag.get_dag("dag_02_ingest_source_to_bronze")

    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) >= 2


@patch("boto3.resource")
def test_bronze_gate_passes_valid_raw_data(mock_boto_resource):
    """Validates Gate 1 passing when healthy non-zero byte assets exist."""
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    mock_obj = MagicMock()
    mock_obj.key = "bronze/raw_logs/part-00000.parquet"
    mock_obj.size = 2048

    mock_bucket.objects.filter.return_value = [mock_obj]

    result = execute_bronze_quality_gate(
        bucket_name="company-ai-datalake", prefix="bronze/raw_logs/"
    )
    assert result is True


@patch("boto3.resource")
def test_bronze_gate_fails_empty_ingestion(mock_boto_resource):
    """Validates Gate 1 failing when raw ingestion yields zero objects."""
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    mock_bucket.objects.filter.return_value = []

    with pytest.raises(
        ValueError, match="QUALITY GATE 1 FAILURE: Expected data but found 0 objects"
    ):
        execute_bronze_quality_gate(bucket_name="company-ai-datalake", prefix="bronze/raw_logs/")


@patch("boto3.resource")
def test_bronze_gate_fails_corrupted_zero_byte_file(mock_boto_resource):
    """Validates Gate 1 failing when an empty zero-byte parquet file is detected."""
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    mock_obj = MagicMock()
    mock_obj.key = "bronze/raw_logs/corrupt.parquet"
    mock_obj.size = 0

    mock_bucket.objects.filter.return_value = [mock_obj]

    with pytest.raises(ValueError, match="Corrupted zero-byte file found"):
        execute_bronze_quality_gate(bucket_name="company-ai-datalake", prefix="bronze/raw_logs/")
