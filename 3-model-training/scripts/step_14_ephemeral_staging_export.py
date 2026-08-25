import hashlib
import json
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from scripts.hardware_engine import get_inference_engine
except ImportError:
    from hardware_engine import get_inference_engine


@dataclass
class UploadTask:
    step: int
    local_checkpoint_path: Path
    local_inference_path: Path
    remote_destination_uri: str
    metadata: dict[str, Any]
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class EphemeralStagingExporter:
    def __init__(
        self,
        scratch_dir: str,
        remote_base_uri: str,
        max_local_snapshots: int = 2,
        remote_upload_fn: Callable[[Path, str], bool] | None = None,
    ):
        self.scratch_dir = Path(scratch_dir).resolve()
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.remote_base_uri = remote_base_uri
        self.max_local_snapshots = max_local_snapshots

        self.remote_upload_fn = remote_upload_fn or self._default_mock_upload

        self.task_queue: queue.Queue = queue.Queue()
        self.stop_signal = threading.Event()
        self.active_uploads: dict[int, Path] = {}
        self.completed_snapshots: list[int] = []
        self.lock = threading.Lock()

        self.worker_thread = threading.Thread(target=self._upload_consumer_loop, daemon=True)
        self.worker_thread.start()

    def stage_checkpoint_locally(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metadata: dict[str, Any],
    ) -> tuple[Path, Path]:
        raw_model = getattr(model, "module", model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)

        ckpt_filename = f"recovery_step_{step:07d}.pt"
        inf_filename = f"inference_step_{step:07d}.pt"

        final_ckpt_path = self.scratch_dir / ckpt_filename
        temp_ckpt_path = self.scratch_dir / f"{ckpt_filename}.tmp"

        final_inf_path = self.scratch_dir / inf_filename
        temp_inf_path = self.scratch_dir / f"{inf_filename}.tmp"

        # 1. Full State Recovery Payload
        recovery_payload = {
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        torch.save(recovery_payload, temp_ckpt_path)
        with open(temp_ckpt_path, "a+") as f:
            os.fsync(f.fileno())
        os.replace(temp_ckpt_path, final_ckpt_path)

        # 2. Stripped Inference Payload (Excludes optimizer states)
        inference_payload = {
            "step": step,
            "model_state_dict": {k: v.clone() for k, v in raw_model.state_dict().items()},
            "metadata": metadata,
        }
        torch.save(inference_payload, temp_inf_path)
        with open(temp_inf_path, "a+") as f:
            os.fsync(f.fileno())
        os.replace(temp_inf_path, final_inf_path)

        # 3. Enqueue Background Offload Task
        dest_uri = f"{self.remote_base_uri.rstrip('/')}/step_{step:07d}"
        task = UploadTask(
            step=step,
            local_checkpoint_path=final_ckpt_path,
            local_inference_path=final_inf_path,
            remote_destination_uri=dest_uri,
            metadata=metadata,
        )

        with self.lock:
            self.active_uploads[step] = final_ckpt_path

        self.task_queue.put(task)
        return final_ckpt_path, final_inf_path

    def _upload_consumer_loop(self):
        while not self.stop_signal.is_set():
            try:
                task: UploadTask = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_offload_task(task)
            except Exception as e:
                print(f"ERROR: Background offload failed for step {task.step}: {e}")
            finally:
                with self.lock:
                    self.active_uploads.pop(task.step, None)
                    self.completed_snapshots.append(task.step)
                    self._prune_local_scratch()
                self.task_queue.task_done()

    def _process_offload_task(self, task: UploadTask):
        ckpt_sha256 = self._compute_sha256(task.local_checkpoint_path)
        inf_sha256 = self._compute_sha256(task.local_inference_path)

        manifest_path = task.local_inference_path.with_suffix(".manifest.json")
        manifest_data = {
            "step": task.step,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(task.created_at)),
            "recovery_artifact": {
                "filename": task.local_checkpoint_path.name,
                "sha256": ckpt_sha256,
            },
            "inference_artifact": {
                "filename": task.local_inference_path.name,
                "sha256": inf_sha256,
            },
            "user_metadata": task.metadata,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        self.remote_upload_fn(
            task.local_checkpoint_path,
            f"{task.remote_destination_uri}/{task.local_checkpoint_path.name}",
        )
        self.remote_upload_fn(
            task.local_inference_path,
            f"{task.remote_destination_uri}/{task.local_inference_path.name}",
        )
        self.remote_upload_fn(manifest_path, f"{task.remote_destination_uri}/manifest.json")

    def _prune_local_scratch(self):
        while len(self.completed_snapshots) > self.max_local_snapshots:
            step_to_prune = self.completed_snapshots.pop(0)

            ckpt_file = self.scratch_dir / f"recovery_step_{step_to_prune:07d}.pt"
            inf_file = self.scratch_dir / f"inference_step_{step_to_prune:07d}.pt"
            manifest_file = self.scratch_dir / f"inference_step_{step_to_prune:07d}.manifest.json"

            for file_path in [ckpt_file, inf_file, manifest_file]:
                if file_path.exists() and step_to_prune not in self.active_uploads:
                    file_path.unlink()

    @staticmethod
    def _compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _default_mock_upload(local_path: Path, remote_uri: str) -> bool:
        time.sleep(0.01)
        return True

    def flush_and_shutdown(self):
        self.task_queue.join()
        self.stop_signal.set()
        self.worker_thread.join(timeout=5.0)


def validate_exported_inference_artifact(artifact_dir: str) -> bool:
    try:
        engine = get_inference_engine(artifact_dir)
        return engine is not None
    except Exception as e:
        print(f"Validation failed for exported artifact at {artifact_dir}: {e}")
        return False
