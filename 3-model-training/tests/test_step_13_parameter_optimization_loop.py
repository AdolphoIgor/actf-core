import math
import warnings

import torch
import torch.nn as nn

from scripts.step_13_parameter_optimization_loop import (
    CosineWarmupLRScheduler,
    OptimizationConfig,
    ProductionOptimizationEngine,
    configure_decay_parameter_groups,
)


class MockCausalModel(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(x)
        h = self.norm(h)
        return self.head(h)


def test_parameter_group_decay_partitioning():
    model = MockCausalModel()
    groups = configure_decay_parameter_groups(model, weight_decay=0.1)

    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0

    for p in groups[0]["params"]:
        assert p.dim() >= 2

    for p in groups[1]["params"]:
        assert p.dim() < 2


def test_cosine_warmup_scheduler_trajectory():
    model = MockCausalModel()
    cfg = OptimizationConfig(
        max_learning_rate=1e-3,
        min_learning_rate=1e-5,
        warmup_steps=10,
        total_steps=100,
        device_override="cpu",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_learning_rate)
    scheduler = CosineWarmupLRScheduler(optimizer, cfg)

    assert scheduler.get_lr(0) == 0.0
    assert math.isclose(scheduler.get_lr(10), 1e-3, rel_tol=1e-5)
    assert 1e-5 < scheduler.get_lr(55) < 1e-3
    assert math.isclose(scheduler.get_lr(100), 1e-5, rel_tol=1e-5)
    assert scheduler.get_lr(150) == 1e-5


def test_optimization_step_execution_and_metrics():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        vocab_size = 128
        model = MockCausalModel(vocab_size=vocab_size)
        cfg = OptimizationConfig(
            max_learning_rate=1e-3,
            min_learning_rate=1e-5,
            warmup_steps=5,
            total_steps=20,
            max_grad_norm=1.0,
            dtype_override=torch.float32,
            device_override="cpu",
        )

        engine = ProductionOptimizationEngine(cfg=cfg, model=model, ignore_index=-100)

        micro_batches = []
        for _ in range(4):
            inp = torch.randint(0, vocab_size, (2, 16))
            tgt = inp.clone()
            tgt[:, :4] = -100
            micro_batches.append((inp, tgt))

        metrics = engine.run_optimization_step(micro_batches, is_distributed=False)

        assert "step_loss" in metrics
        assert "grad_norm" in metrics
        assert "learning_rate" in metrics
        assert "active_tokens" in metrics

        assert not math.isnan(metrics["step_loss"])
        assert metrics["grad_norm"] <= cfg.max_grad_norm + 1e-4
        assert metrics["active_tokens"] == 4 * 2 * (16 - 4)

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert not torch.isnan(p.grad).any()
