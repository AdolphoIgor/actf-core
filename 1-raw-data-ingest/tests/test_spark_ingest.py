import pytest
from unittest.mock import patch, MagicMock

from scripts.spark_ingest import run_enterprise_bronze_ingest


def setup_mock_spark_session(mock_spark_class):
    """
    Helper function to build a deeply mocked SparkSession chain.
    """
    mock_spark = MagicMock()
    mock_builder = MagicMock()
    
    mock_spark_class.builder = mock_builder
    mock_builder.appName.return_value = mock_builder
    mock_builder.config.return_value = mock_builder
    mock_builder.getOrCreate.return_value = mock_spark
    
    return mock_spark


# NOTE: @patch decorators are applied bottom-to-top. 
# SparkSession is the first argument, F is the second argument.
@patch("scripts.spark_ingest.F")
@patch("scripts.spark_ingest.SparkSession")
def test_initial_ingest_overwrite_mode(mock_spark_class, mock_f, capsys):
    """
    Validates the initial historical load (last_processed_ts is None).
    Ensures 'overwrite' mode is used and the XCOM marker is printed.
    """
    mock_spark = setup_mock_spark_session(mock_spark_class)

    mock_bounds_df = MagicMock()
    mock_bounds_df.collect.return_value = [{"min_id": 1, "max_id": 5000}]
    
    mock_target_df = MagicMock()
    mock_target_df.agg.return_value.collect.return_value = [["2026-06-03 14:00:00"]]
    
    mock_target_df.withColumn.return_value = mock_target_df
    mock_target_df.coalesce.return_value = mock_target_df
    
    mock_writer = MagicMock()
    mock_target_df.write = mock_writer
    mock_writer.mode.return_value = mock_writer
    mock_writer.format.return_value = mock_writer
    mock_writer.partitionBy.return_value = mock_writer

    mock_spark.read.jdbc.side_effect = [mock_bounds_df, mock_target_df]

    run_enterprise_bronze_ingest(last_processed_ts=None)

    assert mock_spark.read.jdbc.call_count == 2
    mock_writer.mode.assert_called_with("overwrite")
    
    captured = capsys.readouterr()
    assert "XCOM_MARKER_MAX_TS:2026-06-03 14:00:00" in captured.out


@patch("scripts.spark_ingest.F")
@patch("scripts.spark_ingest.SparkSession")
def test_incremental_ingest_append_mode(mock_spark_class, mock_f, capsys):
    """
    Validates an incremental delta load.
    Ensures 'append' mode is used and the query filter contains the high-watermark.
    """
    mock_spark = setup_mock_spark_session(mock_spark_class)

    mock_bounds_df = MagicMock()
    mock_bounds_df.collect.return_value = [{"min_id": 5001, "max_id": 5500}]
    
    mock_target_df = MagicMock()
    mock_target_df.agg.return_value.collect.return_value = [["2026-06-04 09:30:00"]]
    mock_target_df.withColumn.return_value = mock_target_df
    mock_target_df.coalesce.return_value = mock_target_df
    
    mock_writer = MagicMock()
    mock_target_df.write = mock_writer
    mock_writer.mode.return_value = mock_writer
    mock_writer.format.return_value = mock_writer
    mock_writer.partitionBy.return_value = mock_writer

    mock_spark.read.jdbc.side_effect = [mock_bounds_df, mock_target_df]

    watermark = "2026-06-03 14:00:00"
    run_enterprise_bronze_ingest(last_processed_ts=watermark)

    mock_writer.mode.assert_called_with("append")
    
    bounds_call_kwargs = mock_spark.read.jdbc.call_args_list[0][1]
    assert watermark in bounds_call_kwargs["table"]


@patch("scripts.spark_ingest.F")
@patch("scripts.spark_ingest.SparkSession")
def test_empty_delta_sequence_early_exit(mock_spark_class, mock_f, capsys):
    """
    Validates that if the database returns empty bounds, the script exits safely.
    """
    mock_spark = setup_mock_spark_session(mock_spark_class)

    mock_bounds_df = MagicMock()
    mock_bounds_df.collect.return_value = [{"min_id": None, "max_id": None}]
    
    mock_spark.read.jdbc.return_value = mock_bounds_df

    run_enterprise_bronze_ingest()

    assert mock_spark.read.jdbc.call_count == 1
    mock_spark.stop.assert_called_once()
    
    captured = capsys.readouterr()
    assert "Empty delta sequence. No matching rows found" in captured.out


@patch("scripts.spark_ingest.F")
@patch("scripts.spark_ingest.SparkSession")
def test_database_connection_failure(mock_spark_class, mock_f, capsys):
    """
    Validates that a JDBC connection error triggers a hard sys.exit(1).
    """
    mock_spark = setup_mock_spark_session(mock_spark_class)

    mock_spark.read.jdbc.side_effect = Exception("FATAL: connection to database failed")

    with pytest.raises(SystemExit) as exit_state:
        run_enterprise_bronze_ingest()

    assert exit_state.value.code == 1
    mock_spark.stop.assert_called_once()
    
    captured = capsys.readouterr()
    assert "Failed to query database index boundaries" in captured.out