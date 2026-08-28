from pathlib import Path

import torch
import torch.nn as nn

from scripts.step_14_ephemeral_staging_export import EphemeralStagingExporter


class MockLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def forward(self, x):
        return self.layer(x)


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

    # Wait for the background queue to complete the mock upload
    exporter.flush_and_shutdown()

    assert ckpt_path.exists()
    assert inf_path.exists()
    assert not (scratch_dir / "recovery_step_0000010.pt.tmp").exists()

    ckpt_data = torch.load(ckpt_path, weights_only=False)
    assert "optimizer_state_dict" in ckpt_data
    assert "model_state_dict" in ckpt_data


def test_scratch_pruning_fifo_limit(tmp_path):
    scratch_dir = tmp_path / "scratch"
    remote_dir = tmp_path / "remote"

    def mock_upload(local_path: Path, remote_uri: str) -> bool:
        return True

    exporter = EphemeralStagingExporter(
        scratch_dir=str(scratch_dir),
        remote_base_uri=str(remote_dir),
        max_local_snapshots=2,
        remote_upload_fn=mock_upload,
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

    # Wait for the background queue to finish uploads and trigger pruning
    exporter.flush_and_shutdown()

    recovery_files = sorted(scratch_dir.glob("recovery_step_*.pt"))
    assert len(recovery_files) == 2

    remaining_steps = {f.name for f in recovery_files}
    assert "recovery_step_0000003.pt" in remaining_steps
    assert "recovery_step_0000004.pt" in remaining_steps
