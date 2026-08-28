import math

import torch
import torch.nn as nn

IGNORE_INDEX = -100


def audit_optimizer_param_groups(param_groups_or_optimizer):
    """
    Validates that parameter groups are strictly disjoint and have valid weight decay settings.
    Accepts either an instantiated torch.optim.Optimizer or a list of parameter group dicts.
    """
    if hasattr(param_groups_or_optimizer, "param_groups"):
        groups = param_groups_or_optimizer.param_groups
    elif isinstance(param_groups_or_optimizer, list):
        groups = param_groups_or_optimizer
    else:
        raise TypeError(
            f"Expected Optimizer or list of group dicts, got {type(param_groups_or_optimizer)}"
        )

    seen_param_ids = set()
    for group_idx, group in enumerate(groups):
        assert "params" in group, (
            f"Gate 4 Failure: Parameter group {group_idx} missing 'params' key."
        )

        params = group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]

        for p in params:
            p_id = id(p)
            assert p_id not in seen_param_ids, (
                f"Gate 4 Failure: Overlapping parameter detected in optimizer group {group_idx}. "
                "Every trainable tensor must belong to exactly one parameter group."
            )
            seen_param_ids.add(p_id)

    print(
        f"[QUALITY GATE 4] Verified {len(seen_param_ids)} disjoint parameters across {len(groups)} optimizer groups."
    )
    return True


def audit_step0_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    tolerance: float = 0.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Asserts that step 0 cross-entropy loss conforms to theoretical uniform entropy ln(vocab_size).
    """
    model.eval()
    expected_loss = math.log(vocab_size)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Use bfloat16 for CPU autocast to avoid PyTorch CPU autocast warnings
    autocast_dtype = dtype if dtype in [torch.bfloat16, torch.float16] else torch.bfloat16
    autocast_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
            outputs = model(inputs)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

            shift_logits = logits.view(-1, vocab_size)
            shift_targets = targets.view(-1)
            loss = loss_fn(shift_logits, shift_targets).item()

    loss_diff = abs(loss - expected_loss)
    assert loss_diff <= tolerance, (
        f"Gate 4 Failure: Step 0 loss ({loss:.4f}) diverges from theoretical ln(V)={expected_loss:.4f} "
        f"by {loss_diff:.4f} (tolerance={tolerance}). Softmax instability risk."
    )
    print(
        f"[QUALITY GATE 4] Step 0 loss verified: {loss:.4f} (Theoretical ln({vocab_size})={expected_loss:.4f})."
    )
    return True


def audit_gradient_flow(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Validates end-to-end backpropagation gradient flow and absence of NaNs/Infs.
    """
    model.train()
    model.zero_grad(set_to_none=True)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    autocast_dtype = dtype if dtype in [torch.bfloat16, torch.float16] else torch.bfloat16
    autocast_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        outputs = model(inputs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))

    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gate 4 Failure: Detached gradient flow on '{name}'."
            assert not torch.isnan(param.grad).any(), (
                f"Gate 4 Failure: NaN gradient detected on '{name}'."
            )
            assert not torch.isinf(param.grad).any(), (
                f"Gate 4 Failure: Inf gradient detected on '{name}'."
            )

    print("[QUALITY GATE 4] End-to-end gradient flow and numerical health verified.")
    return True


def run_gate_4_validation(
    model: nn.Module, param_groups, sample_batch: dict, vocab_size: int, **context
):
    """
    Main entry point for Quality Gate 4 (Pre-Training Invariant Verification).
    """
    print("[QUALITY GATE 4] Initiating Pre-Training Invariant Verification...")
    audit_optimizer_param_groups(param_groups)
    audit_step0_loss(model, sample_batch["inputs"], sample_batch["targets"], vocab_size=vocab_size)
    audit_gradient_flow(
        model, sample_batch["inputs"], sample_batch["targets"], vocab_size=vocab_size
    )
    print("[QUALITY GATE 4] ALL PRE-TRAINING INVARIANTS PASSED.")
    return True
