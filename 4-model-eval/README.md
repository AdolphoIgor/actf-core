
# Phase 4: Model Evaluation & Serving (`4-model-eval`)

This directory contains evaluation benchmarks, quality metrics, and high-throughput inference routines for validated model checkpoints.

---

## Architecture Overview

Phase 4 evaluates model outputs against compliance benchmarks. It supports both standard Hugging Face PyTorch CPU inference for local validation and C++ CUDA-accelerated engines (`vLLM` / `FlashAttention`) for high-throughput batch evaluation.

### Folder Structure

```

4-model-eval/
├── pyproject.toml
├── README.md
└── scripts/
└── hardware_engine.py  # Dual-engine fallback selector (vLLM vs HF Transformers)

```

---

## Dual Inference Engine (`scripts/hardware_engine.py`)

Because acceleration libraries like `vLLM` and `vllm-flash-attn` strictly require NVIDIA CUDA hardware (and fail to compile on CPU-only machines), evaluation scripts implement a runtime fallback mechanism.

### Hardware Detection Logic

```
           ┌──────────────────────────────┐
           │  torch.cuda.is_available()?  │
           └──────────────┬───────────────┘
                          │
           ┌──────────────┴──────────────┐
           │                             │
        [ YES ]                       [ NO ]
           │                             │
           ▼                             ▼
┌──────────────────────┐     ┌──────────────────────────┐
│    vLLM Engine       │     │ Hugging Face Transformers│
│ High-Throughput CUDA │     │ PyTorch CPU Fallback     │
└──────────────────────┘     └──────────────────────────┘

```

### Engine Selection Matrix

| Target Engine | Device | Acceleration Features | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **vLLM** | NVIDIA CUDA | PagedAttention, Continuous Batching | Cloud Production Batch Eval |
| **HF Transformers** | CPU | Native PyTorch execution | Local Pipeline Validation |

---

## Future Execution Workflow

When activating Phase 4:

1. **Local CPU Validation:**
   ```bash
   python scripts/hardware_engine.py --model_id "Qwen/Qwen2.5-0.5B-Instruct" --eval_dataset "s3://company-ai-datalake/gold/eval/"

```

2. **Cloud High-Throughput GPU Evaluation:**
Spin up the cluster with GPU capabilities:
```bash
docker compose --profile gpu up -d

```
