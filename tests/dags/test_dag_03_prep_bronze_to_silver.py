import pytest
import pyarrow as pa
from unittest.mock import patch, MagicMock

# Import cleanly from the isolated scripts module
from scripts.quality_gate_2 import execute_silver_quality_gate


@patch("scripts.quality_gate_2.S3FileSystem")
@patch("scripts.quality_gate_2.ds.dataset")
def test_silver_gate_passes_valid_data(mock_dataset, mock_s3):
    """Validates that a clean, non-null dataset passes the gate."""
    mock_ds_instance = MagicMock()
    mock_dataset.return_value = mock_ds_instance
    
    mock_ds_instance.schema.names = ["text", "branch_id", "preserve_whitespace"]
    mock_ds_instance.count_rows.return_value = 100
    
    valid_batch = pa.RecordBatch.from_arrays([pa.array(["Valid text 1", "Valid text 2"])], names=["text"])
    mock_ds_instance.to_batches.return_value = [valid_batch]
    
    result = execute_silver_quality_gate("test-bucket", "silver/test/")
    assert result is True


@patch("scripts.quality_gate_2.S3FileSystem")
@patch("scripts.quality_gate_2.ds.dataset")
def test_silver_gate_fails_empty_dataset(mock_dataset, mock_s3):
    """Validates that a dataset with 0 rows triggers a failure."""
    mock_ds_instance = MagicMock()
    mock_dataset.return_value = mock_ds_instance
    mock_ds_instance.count_rows.return_value = 0
    
    with pytest.raises(ValueError, match="Silver dataset is completely empty"):
        execute_silver_quality_gate("test-bucket", "silver/test/")


@patch("scripts.quality_gate_2.S3FileSystem")
@patch("scripts.quality_gate_2.ds.dataset")
def test_silver_gate_fails_missing_schema(mock_dataset, mock_s3):
    """Validates that missing the required 'text' column triggers a failure."""
    mock_ds_instance = MagicMock()
    mock_dataset.return_value = mock_ds_instance
    mock_ds_instance.count_rows.return_value = 100
    mock_ds_instance.schema.names = ["wrong_column_name"]
    
    with pytest.raises(ValueError, match="Schema missing required columns"):
        execute_silver_quality_gate("test-bucket", "silver/test/")


@patch("scripts.quality_gate_2.S3FileSystem")
@patch("scripts.quality_gate_2.ds.dataset")
def test_silver_gate_fails_null_records(mock_dataset, mock_s3):
    """Validates that null values in the text column trigger a failure."""
    mock_ds_instance = MagicMock()
    mock_dataset.return_value = mock_ds_instance
    mock_ds_instance.schema.names = ["text"]
    mock_ds_instance.count_rows.return_value = 100
    
    invalid_batch = pa.RecordBatch.from_arrays([pa.array(["Valid text", None, "More text"])], names=["text"])
    mock_ds_instance.to_batches.return_value = [invalid_batch]
    
    with pytest.raises(ValueError, match="Found 1 null records"):
        execute_silver_quality_gate("test-bucket", "silver/test/")