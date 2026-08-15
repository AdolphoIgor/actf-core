import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add script directory to sys.path for direct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ray", "scripts")))

from ray_ingest import parse_args, run_ingestion


def test_parse_args_none_string_conversion():
    """
    Tests that string 'None', 'null', or empty string CLI arguments 
    are converted to Python None.
    """
    with patch("argparse._sys.argv", ["ray_ingest.py", "--last_sync", "None"]):
        args = parse_args()
        assert args.last_sync is None

    with patch("argparse._sys.argv", ["ray_ingest.py", "--last_sync", "2026-08-04T12:00:00"]):
        args = parse_args()
        assert args.last_sync == "2026-08-04T12:00:00"


@patch("ray_ingest.get_s3_client")
@patch("os.path.exists", return_value=True)
@patch("os.walk")
def test_zero_write_guard_initial_load_raises_error(mock_walk, mock_exists, mock_s3):
    """
    RULE A: On initial load (last_sync is None), finding 0 files MUST raise a RuntimeError.
    """
    mock_walk.return_value = [("/opt/airflow/data/unstructured_samples", [], [])]
    
    with patch("argparse._sys.argv", ["ray_ingest.py", "--last_sync", "None"]):
        with pytest.raises(RuntimeError, match="INITIAL INGESTION FAILURE"):
            run_ingestion()


@patch("ray_ingest.get_s3_client")
@patch("os.path.exists", return_value=True)
@patch("os.walk")
def test_zero_write_guard_incremental_run_passes(mock_walk, mock_exists, mock_s3, capsys):
    """
    RULE B: On incremental run (last_sync is timestamp), finding 0 new files 
    must log a notice and exit cleanly (return status 0).
    """
    mock_walk.return_value = [("/opt/airflow/data/unstructured_samples", [], [])]
    
    with patch("argparse._sys.argv", ["ray_ingest.py", "--last_sync", "2026-08-04T10:00:00"]):
        # Should not raise an exception
        run_ingestion()
        
        captured = capsys.readouterr()
        assert "NOTICE: 0 new files found since last sync" in captured.out
        assert "RAY_MARKER_MAX_TS:2026-08-04T10:00:00" in captured.out