import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.step_17_mlflow_registry_promotion import (
    MLflowRegistryPromoter,
    RegistryPromotionError,
)


@pytest.fixture
def mock_mlflow_client():
    with patch("scripts.step_17_mlflow_registry_promotion.MlflowClient") as mock_cls:
        client_instance = MagicMock()
        mock_cls.return_value = client_instance
        yield client_instance


def test_compute_sha256(tmp_path):
    test_file = tmp_path / "weights.bin"
    test_content = b"tensor binary payload 12345"
    test_file.write_bytes(test_content)

    expected_hash = hashlib.sha256(test_content).hexdigest()
    actual_hash = MLflowRegistryPromoter._compute_sha256(test_file)

    assert actual_hash == expected_hash


def test_register_staged_version_success(mock_mlflow_client, tmp_path):
    weights_file = tmp_path / "model.safetensors"
    weights_file.write_bytes(b"dummy safetensors content")
    file_hash = hashlib.sha256(b"dummy safetensors content").hexdigest()

    receipt_file = tmp_path / "gate5_receipt.json"
    receipt_data = {
        "verdict": "PROMOTED",
        "tensor_sha256": file_hash,
        "timestamp_utc": "2026-08-23T12:00:00Z",
        "scorecard": {
            "ast_syntax_pass_rate": 1.0,
            "operational": {"ece": 0.02},
        },
    }
    receipt_file.write_text(json.dumps(receipt_data), encoding="utf-8")

    mock_version = MagicMock()
    mock_version.version = "3"
    mock_mlflow_client.create_model_version.return_value = mock_version

    promoter = MLflowRegistryPromoter(
        tracking_uri="http://localhost:5000",
        registered_model_name="test-model",
    )

    version = promoter.register_staged_version(
        run_id="run-abc-123",
        artifact_subpath="artifacts/model",
        weights_path=weights_file,
        receipt_path=receipt_file,
        provenance_metadata={"git_commit_sha": "commit_123", "dataset_root_hash": "root_456"},
    )

    assert version == "3"
    mock_mlflow_client.create_model_version.assert_called_once_with(
        name="test-model",
        source="runs:/run-abc-123/artifacts/model",
        run_id="run-abc-123",
        description="Automated build certified by Gate 5 at 2026-08-23T12:00:00Z.",
    )
    mock_mlflow_client.set_registered_model_alias.assert_called_with("test-model", "staged", "3")


def test_register_staged_version_fails_unpromoted_verdict(mock_mlflow_client, tmp_path):
    weights_file = tmp_path / "model.safetensors"
    weights_file.write_bytes(b"dummy content")

    receipt_file = tmp_path / "gate5_receipt.json"
    receipt_file.write_text(json.dumps({"verdict": "REJECTED"}), encoding="utf-8")

    promoter = MLflowRegistryPromoter(
        tracking_uri="http://localhost:5000",
        registered_model_name="test-model",
    )

    with pytest.raises(RegistryPromotionError, match="verdict is 'REJECTED'"):
        promoter.register_staged_version(
            run_id="run-123",
            artifact_subpath="artifacts/model",
            weights_path=weights_file,
            receipt_path=receipt_file,
            provenance_metadata={},
        )


def test_register_staged_version_fails_checksum_mismatch(mock_mlflow_client, tmp_path):
    weights_file = tmp_path / "model.safetensors"
    weights_file.write_bytes(b"dummy content")

    receipt_file = tmp_path / "gate5_receipt.json"
    receipt_file.write_text(
        json.dumps({"verdict": "PROMOTED", "tensor_sha256": "incorrect_hash"}),
        encoding="utf-8",
    )

    promoter = MLflowRegistryPromoter(
        tracking_uri="http://localhost:5000",
        registered_model_name="test-model",
    )

    with pytest.raises(RegistryPromotionError, match="Checksum mismatch"):
        promoter.register_staged_version(
            run_id="run-123",
            artifact_subpath="artifacts/model",
            weights_path=weights_file,
            receipt_path=receipt_file,
            provenance_metadata={},
        )


def test_promote_to_champion(mock_mlflow_client):
    prev_champ = MagicMock()
    prev_champ.version = "1"
    mock_mlflow_client.get_model_version_by_alias.return_value = prev_champ

    promoter = MLflowRegistryPromoter(
        tracking_uri="http://localhost:5000",
        registered_model_name="test-model",
    )

    result = promoter.promote_to_champion(candidate_version="2", archive_previous=True)

    assert result["status"] == "SUCCESS"
    assert result["active_champion_version"] == "2"
    assert result["previous_champion_version"] == "1"

    mock_mlflow_client.set_registered_model_alias.assert_any_call("test-model", "champion", "2")
    mock_mlflow_client.set_registered_model_alias.assert_any_call("test-model", "archived", "1")
    mock_mlflow_client.delete_registered_model_alias.assert_called_with("test-model", "staged")


def test_execute_emergency_rollback(mock_mlflow_client):
    archived_version = MagicMock()
    archived_version.version = "1"
    current_champion = MagicMock()
    current_champion.version = "2"

    def get_alias_side_effect(name, alias):
        if alias == "archived":
            return archived_version
        if alias == "champion":
            return current_champion
        raise Exception("Alias not found")

    mock_mlflow_client.get_model_version_by_alias.side_effect = get_alias_side_effect

    promoter = MLflowRegistryPromoter(
        tracking_uri="http://localhost:5000",
        registered_model_name="test-model",
    )

    result = promoter.execute_emergency_rollback()

    assert result["status"] == "ROLLED_BACK"
    assert result["reverted_to_version"] == "1"
    assert result["quarantined_version"] == "2"

    mock_mlflow_client.set_registered_model_alias.assert_any_call("test-model", "quarantined", "2")
    mock_mlflow_client.set_registered_model_alias.assert_any_call("test-model", "champion", "1")
