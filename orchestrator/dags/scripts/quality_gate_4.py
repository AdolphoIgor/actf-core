import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_VRAM_RATIO = float(os.environ.get("MAX_VRAM_USAGE_RATIO", "0.85"))
IGNORE_INDEX = -100


def audit_initial_weights(model: nn.Module):
    """Asserts absence of NaNs, Infs, and dead zero matrices in initialized weights."""
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN detected in initial parameter tensor: {name}"
        assert not torch.isinf(param).any(), f"Inf detected in initial parameter tensor: {name}"
        if param.dim() >= 2:
            assert not torch.all(param == 0), f"Dead weight matrix detected (all zeros): {name}"


def audit_tied_embeddings(model: nn.Module):
    """Asserts that tied input embeddings and output LM heads share the same data pointer."""
    tok_emb = (
        getattr(model, "tok_emb", None)
        or getattr(model, "wte", None)
        or getattr(model, "embed_tokens", None)
    )
    lm_head = getattr(model, "lm_head", None)

    if tok_emb is not None and lm_head is not None:
        emb_weight = tok_emb.weight if hasattr(tok_emb, "weight") else tok_emb
        head_weight = lm_head.weight if hasattr(lm_head, "weight") else lm_head
        if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
            assert emb_weight.data_ptr() == head_weight.data_ptr(), (
                f"Tied embedding pointer mismatch: tok_emb ({hex(emb_weight.data_ptr())}) "
                f"!= lm_head ({hex(head_weight.data_ptr())})"
            )


def audit_optimizer_parameter_groups(model: nn.Module, optimizer: torch.optim.Optimizer):
    """Asserts that decay and no-decay parameter sets are strictly disjoint."""
    seen_param_ptrs = set()
    total_grouped_params = 0

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            ptr = param.data_ptr()
            assert ptr not in seen_param_ptrs, (
                f"Parameter pointer {hex(ptr)} appears in multiple optimizer groups."
            )
            seen_param_ptrs.add(ptr)
            total_grouped_params += 1

    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_grouped_params == trainable_count, (
        f"Optimizer registered {total_grouped_params} parameters, but model has {trainable_count} trainable parameters."
    )


def audit_forward_and_step0_loss(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    sample_targets: torch.Tensor,
    vocab_size: int,
    device: str,
    dtype: torch.dtype,
) -> float:
    """Asserts output logit dynamic range, active token counts, and ln(V) initial loss calibration."""
    model.eval()
    sample_inputs = sample_inputs.to(device)
    sample_targets = sample_targets.to(device)

    active_tokens = (sample_targets != IGNORE_INDEX).sum().item()
    total_tokens = sample_targets.numel()
    assert active_tokens > 0, "Batch contains zero active loss targets (100% masked)."
    assert active_tokens < total_tokens, "SFT batch has zero masked tokens (Prompt unmasked)."

    with torch.no_grad():
        with torch.autocast(device_type=device if device == "cuda" else "cpu", dtype=dtype):
            outputs = model(sample_inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if isinstance(logits, tuple):
                logits = logits[0]

    max_logit = torch.max(torch.abs(logits)).item()
    assert not math.isnan(max_logit) and not math.isinf(max_logit), (
        "Forward pass generated NaN/Inf logits."
    )
    assert max_logit <= 25.0, (
        f"Initial max logit {max_logit:.2f} exceeds 25.0 (Softmax instability risk)."
    )

    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = sample_targets[..., 1:].contiguous().view(-1)

    loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="mean"
    ).item()
    expected_loss = math.log(vocab_size)
    delta_loss = abs(loss - expected_loss)

    assert delta_loss <= 0.60, (
        f"Step-0 Loss {loss:.4f} diverges from theoretical ln(V) = {expected_loss:.4f} "
        f"(Delta: {delta_loss:.4f} > 0.60 nats)."
    )
    return loss


def audit_autograd_gradient_flow(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    sample_targets: torch.Tensor,
    vocab_size: int,
    device: str,
    dtype: torch.dtype,
):
    """Executes a backward pass and asserts that 100% of trainable parameters receive valid gradients."""
    model.train()
    model.zero_grad(set_to_none=True)

    sample_inputs = sample_inputs.to(device)
    sample_targets = sample_targets.to(device)

    with torch.autocast(device_type=device if device == "cuda" else "cpu", dtype=dtype):
        outputs = model(sample_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if isinstance(logits, tuple):
            logits = logits[0]

        shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = sample_targets[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)

    loss.backward()

    missing_grads = []
    nan_grads = []
    grad_norms = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing_grads.append(name)
            continue
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            nan_grads.append(name)
            continue

        norm = torch.linalg.vector_norm(param.grad).item()
        grad_norms.append(norm**2)

    assert not missing_grads, f"Parameters received no gradient: {missing_grads[:3]}"
    assert not nan_grads, f"NaN/Inf gradients encountered on: {nan_grads[:3]}"

    global_norm = math.sqrt(sum(grad_norms))
    assert global_norm > 0.0, "Global gradient norm is zero."

    model.zero_grad(set_to_none=True)


def audit_vram_headroom(model: nn.Module, vocab_size: int, dtype: torch.dtype):
    """Profiles peak VRAM allocation on CUDA devices and asserts safety margins."""
    if not torch.cuda.is_available():
        print(
            "[QUALITY GATE 4] Non-CUDA environment detected. Skipping physical VRAM headroom audit."
        )
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    total_device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)

    mock_in = torch.randint(0, vocab_size, (2, 512), device="cuda")
    mock_target = mock_in.clone()
    mock_target[:, :256] = IGNORE_INDEX

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=dtype):
        outputs = model(mock_in)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), mock_target.view(-1), ignore_index=IGNORE_INDEX
        )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
    usage_ratio = peak_allocated_mb / total_device_memory

    assert usage_ratio <= MAX_VRAM_RATIO, (
        f"Peak memory {peak_allocated_mb:.1f} MB exceeds {MAX_VRAM_RATIO * 100:.0f}% of "
        f"total VRAM ({total_device_memory:.1f} MB). Risk of runtime OOM."
    )
    torch.cuda.empty_cache()


def run_gate_4_validation():
    print(f"[QUALITY GATE 4] Running Pre-Flight Tensor and Gradient Health Gate for: {MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Initialize model architecture
    try:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype)
    except Exception:
        # Fallback to local weight instantiation if cached weights exist
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)

    vocab_size = model.config.vocab_size

    # 1. Weight Distribution & Tied Pointers
    audit_initial_weights(model)
    audit_tied_embeddings(model)

    # 2. Optimizer Parameter Groups Disjointness
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=1e-4,
    )
    audit_optimizer_parameter_groups(model, optimizer)

    # 3. Synthetic Micro-Batch Preparation (Batch Size: 2, Seq Len: 64)
    batch_size, seq_len = 2, 64
    sample_inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    sample_targets = sample_inputs.clone()
    sample_targets[:, : seq_len // 2] = IGNORE_INDEX  # Mask prompt tokens

    # 4. Forward Logits & Step-0 Loss Verification
    loss = audit_forward_and_step0_loss(
        model, sample_inputs, sample_targets, vocab_size, device, dtype
    )

    # 5. Autograd Completeness & Gradient Flow Audit
    audit_autograd_gradient_flow(model, sample_inputs, sample_targets, vocab_size, device, dtype)

    # 6. VRAM Headroom Verification
    audit_vram_headroom(model, vocab_size, dtype)

    print(
        f"[QUALITY GATE 4] PASSED: Step-0 Loss calibrated ({loss:.4f}), gradients verified across all trainable parameters."
    )


if __name__ == "__main__":
    run_gate_4_validation()
