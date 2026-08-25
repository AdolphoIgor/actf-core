import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

try:
    from scripts.hardware_engine import get_training_setup
except ImportError:
    from hardware_engine import get_training_setup


@dataclass(frozen=True)
class OptimizationConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2000
    total_steps: int = 100000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 8
    device_override: str | None = None
    dtype_override: torch.dtype | None = None


def configure_decay_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    decay_ids = {id(p) for p in decay_params}
    no_decay_ids = {id(p) for p in no_decay_params}
    assert len(decay_ids.intersection(no_decay_ids)) == 0, "Parameter group collision detected."

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class CosineWarmupLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, cfg: OptimizationConfig):
        self.optimizer = optimizer
        self.max_lr = cfg.max_learning_rate
        self.min_lr = cfg.min_learning_rate
        self.warmup_steps = cfg.warmup_steps
        self.total_steps = cfg.total_steps
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * (float(step) / float(max(1, self.warmup_steps)))

        if step > self.total_steps:
            return self.min_lr

        decay_ratio = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def state_dict(self) -> dict[str, Any]:
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.current_step = state_dict["current_step"]


class ProductionOptimizationEngine:
    def __init__(
        self,
        cfg: OptimizationConfig,
        model: nn.Module | None = None,
        ignore_index: int = -100,
    ):
        self.cfg = cfg
        self.ignore_index = ignore_index

        if model is not None:
            self.device = torch.device(
                cfg.device_override
                if cfg.device_override
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.dtype = (
                cfg.dtype_override
                if cfg.dtype_override
                else (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
            )
            self.model = model.to(device=self.device, dtype=self.dtype)
        else:
            self.device, self.dtype, self.model = get_training_setup(
                model_id=cfg.model_name_or_path,
                device_override=cfg.device_override,
                dtype_override=cfg.dtype_override,
            )

        self.param_groups = configure_decay_parameter_groups(self.model, cfg.weight_decay)
        self.optimizer = torch.optim.AdamW(
            self.param_groups,
            lr=cfg.max_learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )
        self.scheduler = CosineWarmupLRScheduler(self.optimizer, cfg)

        self.use_fp16_scaler = (self.dtype == torch.float16) and (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_fp16_scaler)

    def run_optimization_step(
        self,
        micro_batches: list[tuple[torch.Tensor, torch.Tensor]],
        is_distributed: bool = False,
    ) -> dict[str, float]:
        t_start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        k_accum = len(micro_batches)
        assert k_accum > 0, "No micro-batches provided for optimization step."

        accum_loss = 0.0
        total_active_tokens = 0

        for k, (inputs, targets) in enumerate(micro_batches):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            is_last_micro_batch = k == k_accum - 1

            with torch.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=self.dtype,
            ):
                outputs = self.model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                if isinstance(logits, tuple):
                    logits = logits[0]

                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = targets[..., 1:].contiguous().view(-1)

                loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=self.ignore_index,
                    reduction="mean",
                )
                scaled_loss = loss / k_accum

            accum_loss += loss.item()
            total_active_tokens += (shift_labels != self.ignore_index).sum().item()

            if is_distributed and not is_last_micro_batch and hasattr(self.model, "no_sync"):
                with self.model.no_sync():
                    if self.use_fp16_scaler:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
            else:
                if self.use_fp16_scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

        if self.use_fp16_scaler:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.max_grad_norm
        )
        grad_norm_val = (
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        )

        assert not math.isnan(grad_norm_val) and not math.isinf(grad_norm_val), (
            "Non-finite gradient norm encountered during optimization step."
        )

        if self.use_fp16_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        current_lr = self.scheduler.step()
        step_duration = time.perf_counter() - t_start

        return {
            "step_loss": accum_loss / k_accum,
            "grad_norm": grad_norm_val,
            "learning_rate": current_lr,
            "active_tokens": float(total_active_tokens),
            "step_duration_sec": step_duration,
        }
