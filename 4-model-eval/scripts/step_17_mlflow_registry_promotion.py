import hashlib
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, TensorSpec


class RegistryPromotionError(Exception):
    """Raised when pre-promotion verification or registry mutation fails."""

    pass


class MLflowRegistryPromoter:
    """
    Manages receipt verification, immutable artifact registration,
    model alias assignments, and automated rollback orchestration.
    """

    def __init__(
        self,
        tracking_uri: str,
        registered_model_name: str,
        vocab_size: int = 32000,
    ):
        self.tracking_uri = tracking_uri
        self.model_name = registered_model_name
        self.vocab_size = vocab_size

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self._ensure_model_entity_exists()

    def _ensure_model_entity_exists(self):
        try:
            self.client.get_registered_model(self.model_name)
        except Exception:
            try:
                self.client.create_registered_model(
                    name=self.model_name,
                    description="Production LLM foundation and continuous fine-tuning registry.",
                )
            except Exception:
                pass

    @staticmethod
    def _compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _build_signature(self) -> ModelSignature:
        input_schema = Schema(
            [
                TensorSpec(type=np.dtype(np.int64), shape=(-1, -1), name="input_ids"),
                TensorSpec(type=np.dtype(np.int64), shape=(-1, -1), name="attention_mask"),
            ]
        )
        output_schema = Schema(
            [
                TensorSpec(
                    type=np.dtype(np.float32), shape=(-1, -1, self.vocab_size), name="logits"
                ),
            ]
        )
        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def register_staged_version(
        self,
        run_id: str,
        artifact_subpath: str,
        weights_path: Path,
        receipt_path: Path,
        provenance_metadata: dict[str, Any],
    ) -> str:
        """
        Registers an immutable model version from an active MLflow tracking run.
        Attaches Gate 5 receipt metadata and tags the version as @staged.
        """
        with open(receipt_path, encoding="utf-8") as f:
            receipt = json.load(f)

        if receipt.get("verdict") != "PROMOTED":
            raise RegistryPromotionError(
                f"Cannot register artifact: Gate 5 verdict is '{receipt.get('verdict')}'."
            )

        actual_hash = self._compute_sha256(weights_path)
        expected_hash = receipt.get("tensor_sha256")
        if expected_hash and actual_hash != expected_hash:
            raise RegistryPromotionError(
                f"Checksum mismatch: Actual {actual_hash} != Receipt {expected_hash}"
            )

        model_uri = f"runs:/{run_id}/{artifact_subpath}"
        model_version = self.client.create_model_version(
            name=self.model_name,
            source=model_uri,
            run_id=run_id,
            description=f"Automated build certified by Gate 5 at {receipt.get('timestamp_utc')}.",
        )
        version_str = str(model_version.version)

        tags = {
            "git_commit_sha": provenance_metadata.get("git_commit_sha", "unknown"),
            "dataset_root_hash": provenance_metadata.get("dataset_root_hash", "unknown"),
            "tensor_sha256": actual_hash,
            "gate5_verdict": receipt["verdict"],
            "gate5_timestamp": receipt.get("timestamp_utc", ""),
            "ast_syntax_rate": str(receipt.get("scorecard", {}).get("ast_syntax_pass_rate", 1.0)),
            "ece_score": str(receipt.get("scorecard", {}).get("operational", {}).get("ece", 0.0)),
            "lifecycle_state": "STAGED",
        }
        for k, v in tags.items():
            self.client.set_model_version_tag(self.model_name, version_str, k, str(v))

        self.client.set_registered_model_alias(self.model_name, "staged", version_str)
        return version_str

    def promote_to_champion(
        self,
        candidate_version: str,
        archive_previous: bool = True,
    ) -> dict[str, str]:
        """
        Atomically updates the @champion alias to the candidate version.
        Demotes the previous champion to @archived.
        """
        previous_champion_version = None
        try:
            current_champ = self.client.get_model_version_by_alias(self.model_name, "champion")
            previous_champion_version = str(current_champ.version)
        except Exception:
            pass

        self.client.set_registered_model_alias(self.model_name, "champion", candidate_version)
        self.client.set_model_version_tag(
            self.model_name, candidate_version, "lifecycle_state", "PRODUCTION"
        )

        try:
            self.client.delete_registered_model_alias(self.model_name, "staged")
        except Exception:
            pass

        if previous_champion_version and previous_champion_version != candidate_version:
            if archive_previous:
                self.client.set_registered_model_alias(
                    self.model_name, "archived", previous_champion_version
                )
                self.client.set_model_version_tag(
                    self.model_name, previous_champion_version, "lifecycle_state", "ARCHIVED"
                )

        return {
            "status": "SUCCESS",
            "model_name": self.model_name,
            "active_champion_version": candidate_version,
            "previous_champion_version": previous_champion_version or "none",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def execute_emergency_rollback(self) -> dict[str, str]:
        """
        Rolls back the @champion alias to the last certified @archived version.
        """
        try:
            archived_model = self.client.get_model_version_by_alias(self.model_name, "archived")
            target_version = str(archived_model.version)
        except Exception:
            raise RegistryPromotionError("Rollback failed: No '@archived' model alias found.")

        current_champ = self.client.get_model_version_by_alias(self.model_name, "champion")
        failed_version = str(current_champ.version)

        self.client.set_registered_model_alias(self.model_name, "quarantined", failed_version)
        self.client.set_model_version_tag(
            self.model_name, failed_version, "lifecycle_state", "QUARANTINED"
        )
        self.client.set_model_version_tag(
            self.model_name, failed_version, "quarantine_reason", "Emergency rollback triggered."
        )

        self.client.set_registered_model_alias(self.model_name, "champion", target_version)
        self.client.set_model_version_tag(
            self.model_name, target_version, "lifecycle_state", "PRODUCTION"
        )

        return {
            "status": "ROLLED_BACK",
            "reverted_to_version": target_version,
            "quarantined_version": failed_version,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
