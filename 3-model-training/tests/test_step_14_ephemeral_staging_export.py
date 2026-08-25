import json
from pathlib import Path

import torch
import torch.nn as nn

from scripts.step_14_ephemeral_staging_export import (
    EphemeralStagingExporter,
)


class MockLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


def test_stage_checkpoint_locally_atomic_write(tmp_path):
    scratch_dir = tmp_path / "scratch"
    remote_dir = tmp_path / "remote"

    uploaded_files = []

    def mock_upload(local_path: Path, remote_uri: str) -> bool:
        uploaded_files.append((local_path.name, remote_uri))
        return True

    exporter = EphemeralStagingExporter(
        scratch_dir=str(scratch_dir),
        remote_base_uri=str(remote_dir),
        max_local_snapshots=2,
        remote_upload_fn=mock_upload,
    )

    model = MockLinearModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metadata = {"loss": 0.42, "epoch": 1}

    ckpt_path, inf_path = exporter.stage_checkpoint_locally(
        step=10,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        metadata=metadata,
    )

    assert ckpt_path.exists()
    assert inf_path.exists()
    assert not (scratch_dir / "recovery_step_0000010.pt.tmp").exists()

    ckpt_data = torch.load(ckpt_path)
    assert "optimizer_state_dict" in ckpt_data
    assert ckpt_data["step"] == 10

    inf_data = torch.load(inf_path)
    assert "optimizer_state_dict" not in inf_data
    assert "model_state_dict" in inf_data

    exporter.flush_and_shutdown()

    assert any("manifest.json" in dest for _, dest in uploaded_files)
    manifest_path = scratch_dir / "inference_step_0000010.manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
        assert manifest["step"] == 10
        assert manifest["user_metadata"]["loss"] == 0.42


def test_scratch_pruning_fifo_limit(tmp_path):
    scratch_dir = tmp_path / "scratch"
    remote_dir = tmp_path / "remote"

    exporter = EphemeralStagingExporter(
        scratch_dir=str(scratch_dir),
        remote_base_uri=str(remote_dir),
        max_local_snapshots=2,
    )

    model = MockLinearModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(1, 5):
        exporter.stage_checkpoint_locally(
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metadata={"step": step},
        )

    exporter.flush_and_shutdown()

    assert not (scratch_dir / "recovery_step_0000001.pt").exists()
    assert not (scratch_dir / "recovery_step_0000002.pt").exists()
    assert (scratch_dir / "recovery_step_0000003.pt").exists()
    assert (scratch_dir / "recovery_step_0000004.pt").exists()
