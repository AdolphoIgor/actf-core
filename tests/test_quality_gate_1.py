import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# FIX: Import directly from the isolated scripts module
from scripts.quality_gate_1 import execute_bronze_quality_gate

def create_mock_s3_object(key: str, size: int):
    """Helper to construct mock S3 object metadata."""
    obj = MagicMock()
    obj.key = key
    obj.size = size
    return obj

# FIX: Patch boto3 exactly where it is being called
@patch("scripts.quality_gate_1.boto3.resource")
def test_quality_gate_1_ignores_metadata_markers(mock_boto_resource):
    """
    Ensures _SUCCESS files and hidden files (._*) are ignored while valid 
    data files pass validation.
    """
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    mock_bucket.objects.filter.return_value = [
        create_mock_s3_object("bronze/compliance_audit/_SUCCESS", 0),
        create_mock_s3_object("bronze/compliance_audit/._DS_Store", 4096),
        create_mock_s3_object("bronze/compliance_audit/part-0000.parquet", 1024)
    ]

    mock_ti = MagicMock()
    mock_ti.state = "success"
    mock_dag_run = MagicMock()
    mock_dag_run.get_task_instance.return_value = mock_ti
    
    context = {"dag_run": mock_dag_run}

    # Must pass without raising an exception
    execute_bronze_quality_gate(bucket_name="company-ai-datalake", **context)


@patch("scripts.quality_gate_1.boto3.resource")
def test_quality_gate_1_fails_on_zero_byte_file(mock_boto_resource):
    """
    Ensures that an empty 0-byte data file triggers an explicit Quality Gate failure.
    """
    mock_s3 = MagicMock()
    mock_bucket = MagicMock()
    mock_boto_resource.return_value = mock_s3
    mock_s3.Bucket.return_value = mock_bucket

    mock_bucket.objects.filter.return_value = [
        create_mock_s3_object("bronze/local_filesystem/compliance_documents/doc1.pdf", 0)
    ]

    mock_ti = MagicMock()
    mock_ti.state = "success"
    mock_dag_run = MagicMock()
    mock_dag_run.get_task_instance.return_value = mock_ti
    
    context = {"dag_run": mock_dag_run}

    with pytest.raises(ValueError, match="QUALITY GATE 1 FAILURE: Corrupted zero-byte file found"):
        execute_bronze_quality_gate(bucket_name="company-ai-datalake", **context)