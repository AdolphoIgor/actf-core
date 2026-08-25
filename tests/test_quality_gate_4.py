import math

import pytest
import torch
import torch.nn as nn
from dags.scripts.quality_gate_4 import (
    IGNORE_INDEX,
    audit_autograd_gradient_flow,
    audit_forward_and_step0_loss,
    audit_initial_weights,
    audit_optimizer_parameter_groups,
)


class SimpleTransformerStub(nn.Module):
    """Lightweight neural module simulating a causal language model for fast unit testing."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Initialize weights with standard normal distribution for uniform logit dispersion
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits


def test_gate4_passes_on_healthy_model():
    vocab_size = 100
    model = SimpleTransformerStub(vocab_size=vocab_size)
    device = "cpu"
    dtype = torch.float32

    audit_initial_weights(model)

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    audit_optimizer_parameter_groups(model, optimizer)

    inputs = torch.randint(0, vocab_size, (2, 16))
    targets = inputs.clone()
    targets[:, :8] = IGNORE_INDEX

    loss = audit_forward_and_step0_loss(model, inputs, targets, vocab_size, device, dtype)
    assert abs(loss - math.log(vocab_size)) <= 0.60

    audit_autograd_gradient_flow(model, inputs, targets, vocab_size, device, dtype)


def test_gate4_fails_on_nan_weights():
    model = SimpleTransformerStub()
    with torch.no_grad():
        model.lm_head.weight[0, 0] = float("nan")

    with pytest.raises(AssertionError, match="NaN detected in initial parameter tensor"):
        audit_initial_weights(model)


def test_gate4_fails_on_overlapping_optimizer_groups():
    model = SimpleTransformerStub()
    all_params = list(model.parameters())

    # Intentionally duplicate a parameter into both groups
    optimizer = torch.optim.AdamW(
        [
            {"params": all_params, "weight_decay": 0.01},
            {"params": [all_params[0]], "weight_decay": 0.0},
        ]
    )

    with pytest.raises(AssertionError, match="appears in multiple optimizer groups"):
        audit_optimizer_parameter_groups(model, optimizer)


def test_gate4_fails_on_divergent_step0_loss():
    vocab_size = 100
    model = SimpleTransformerStub(vocab_size=vocab_size)

    # Artificially scale lm_head to produce extreme non-uniform logits
    with torch.no_grad():
        model.lm_head.weight.fill_(100.0)

    inputs = torch.randint(0, vocab_size, (2, 16))
    targets = inputs.clone()
    targets[:, :8] = IGNORE_INDEX

    with pytest.raises(
        AssertionError, match="diverges from theoretical ln|Softmax instability risk"
    ):
        audit_forward_and_step0_loss(model, inputs, targets, vocab_size, "cpu", torch.float32)


def test_gate4_fails_on_detached_gradient_flow():
    class DetachedStub(nn.Module):
        def __init__(self, vocab_size=100, hidden_dim=32):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden_dim)
            self.detached_layer = nn.Linear(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            h = self.emb(x)
            # Intentionally detach activations to break the autograd graph
            h_det = self.detached_layer(h).detach()
            return self.head(h_det)

    vocab_size = 100
    model = DetachedStub(vocab_size=vocab_size)
    inputs = torch.randint(0, vocab_size, (2, 16))
    targets = inputs.clone()
    targets[:, :8] = IGNORE_INDEX

    with pytest.raises(AssertionError, match="Parameters received no gradient"):
        audit_autograd_gradient_flow(model, inputs, targets, vocab_size, "cpu", torch.float32)
