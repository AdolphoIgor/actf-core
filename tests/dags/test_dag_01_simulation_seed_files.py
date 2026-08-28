import json
from unittest.mock import MagicMock, patch

from dags.dag_01_simulation_seed_files import (
    FREE_TIER_MODEL_POOL,
    ResilientGeminiPool,
    apply_high_entropy_mutation,
    bootstrap_and_write_file,
    generate_unstructured_seed,
)


def test_file_seed_ast_mutation():
    """Validates that unstructured code blocks inside files receive AST-safe comment wrappers."""
    code_doc = "import os\nimport sys\n\ndef execute_query(): pass"
    mutated = apply_high_entropy_mutation(code_doc)

    assert mutated.startswith("# [")
    assert "def execute_query" in mutated


def test_bootstrap_and_write_file_delimiters(tmp_path):
    """Validates that records are written to disk with proper boundary delimiters."""
    target_file = tmp_path / "test_unstructured_batch.txt"
    seed_records = ["Document paragraph 1.", "Document paragraph 2."]

    bootstrap_and_write_file(seed_records, target_count=3, file_path=str(target_file))

    assert target_file.exists()
    content = target_file.read_text(encoding="utf-8")

    assert content.count("--- DOCUMENT BOUNDARY ---") == 3


# @patch("dags.dag_01_simulation_seed_files.genai.Client")
@patch("google.genai.Client")
def test_generate_unstructured_seed_mocked(mock_client_cls):
    """Validates unstructured seed generation via mocked Gemini response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        [
            "MEMORANDUM: Internal investigation into Order #9910.",
            "CHAT LOG: Trader-101 and Trader-202 discussing dark pool pricing.",
        ]
    )
    mock_client.models.generate_content.return_value = mock_response

    pool = ResilientGeminiPool(FREE_TIER_MODEL_POOL)
    records = generate_unstructured_seed(
        client=mock_client, seed_count=2, anomaly_focus="Dark Pool Front-Running", pool=pool
    )

    assert len(records) == 2
    assert "MEMORANDUM" in records[0]
