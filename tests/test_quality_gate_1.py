from unittest.mock import MagicMock, patch

import pytest
from airflow.utils.state import TaskInstanceState
from scripts.quality_gate_1 import execute_bronze_quality_gate


@patch("boto3.resource")
def test_quality_gate_1_passes_valid_stream(mock_boto_resource):
    """Validates Gate 1 passing when active stream prefix contains non-empty data files."""
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    valid_file = MagicMock()
    valid_file.key = "bronze/postgres_enterprise/compliance_audit/part-00000.parquet"
    valid_file.size = 4096

    mock_bucket.objects.filter.return_value = [valid_file]

    result = execute_bronze_quality_gate(
        bucket_name="company-ai-datalake", prefix="bronze/postgres_enterprise/compliance_audit/"
    )
    assert result is True


@patch("boto3.resource")
def test_quality_gate_1_fails_on_empty_bucket_prefix(mock_boto_resource):
    """Validates Gate 1 failing when an active stream prefix has zero objects."""
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    mock_bucket.objects.filter.return_value = []

    with pytest.raises(
        ValueError, match="QUALITY GATE 1 FAILURE: Expected data but found 0 objects"
    ):
        execute_bronze_quality_gate(
            bucket_name="company-ai-datalake", prefix="bronze/postgres_enterprise/compliance_audit/"
        )


@patch("boto3.resource")
def test_quality_gate_1_fails_on_zero_byte_file(mock_boto_resource):
    """Validates Gate 1 failing when a zero-byte corrupted file is present."""
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    corrupt_file = MagicMock()
    corrupt_file.key = "bronze/postgres_enterprise/compliance_audit/corrupt_empty.parquet"
    corrupt_file.size = 0

    mock_bucket.objects.filter.return_value = [corrupt_file]

    with pytest.raises(ValueError, match="Corrupted zero-byte file found"):
        execute_bronze_quality_gate(
            bucket_name="company-ai-datalake", prefix="bronze/postgres_enterprise/compliance_audit/"
        )


@patch("boto3.resource")
def test_quality_gate_1_ignores_metadata_markers(mock_boto_resource):
    """Validates that Gate 1 ignores _SUCCESS and hidden metadata markers."""
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    success_marker = MagicMock()
    success_marker.key = "bronze/postgres_enterprise/compliance_audit/_SUCCESS"
    success_marker.size = 0

    hidden_marker = MagicMock()
    hidden_marker.key = "bronze/postgres_enterprise/compliance_audit/.DS_Store"
    hidden_marker.size = 120

    mock_bucket.objects.filter.return_value = [success_marker, hidden_marker]

    with pytest.raises(ValueError, match="Only metadata markers found under active path"):
        execute_bronze_quality_gate(
            bucket_name="company-ai-datalake", prefix="bronze/postgres_enterprise/compliance_audit/"
        )


def test_quality_gate_1_skips_when_no_upstream_tasks_succeeded():
    """Validates Gate 1 graceful skip when context contains no successful upstream extractions."""
    mock_dag_run = MagicMock()

    # Simulate both extraction tasks being skipped
    mock_ti_spark = MagicMock()
    mock_ti_spark.state = TaskInstanceState.SKIPPED
    mock_ti_ray = MagicMock()
    mock_ti_ray.state = TaskInstanceState.SKIPPED

    def get_ti_side_effect(task_id):
        if task_id == "spark_parallel_bronze_extraction":
            return mock_ti_spark
        elif task_id == "ray_parallel_unstructured_extraction":
            return mock_ti_ray
        return None

    mock_dag_run.get_task_instance.side_effect = get_ti_side_effect

    result = execute_bronze_quality_gate(bucket_name="company-ai-datalake", dag_run=mock_dag_run)
    assert result is True
