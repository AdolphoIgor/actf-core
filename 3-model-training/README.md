
# Phase 3: Model Training (`3-model-training`)

This directory houses the training, fine-tuning (LoRA / QLoRA), and checkpoint management workflows for the **Qwen2.5-0.5B-Instruct** model.

---

## Architecture Overview

Training routines are designed to run dynamically across both local resource-constrained CPU development environments and high-throughput Cloud GPU infrastructure.

### Folder Structure

```

3-model-training/
├── pyproject.toml
├── README.md
└── scripts/
└── hardware_engine.py  # Runtime GPU/CPU detection & engine selector

```

---

## Hardware Fallback Engine (`scripts/hardware_engine.py`)

To ensure seamless execution across local testing (e.g., CPU-only laptop) and Cloud production environments, utility scripts in this directory utilize dynamic PyTorch hardware detection.

### Engine Selection Matrix

| Environment | Detected Hardware | Selected Framework | Precision / Device |
| :--- | :--- | :--- | :--- |
| **Local Dev / Testing** | CPU Only | Hugging Face `transformers` + `peft` | `torch.float32` on `cpu` |
| **Cloud Cluster** | NVIDIA GPU (CUDA) | `transformers` + `accelerate` / DeepSpeed | `torch.bfloat16` or `float16` on `cuda` |

---

## Future Execution Workflow

When activating Phase 3:

1. **Local CPU Fine-Tuning (0.5B Model):**
    ```bash
    python scripts/hardware_engine.py --model_id "Qwen/Qwen2.5-0.5B-Instruct" --mode train
    ```

    *Uses standard PyTorch CPU allocation. Supported by 32GB system RAM.*

2. **Cloud GPU Multi-Node Training:**
    Ensure Docker Compose is booted with the `gpu` profile:
    ```bash
    docker compose --profile gpu up -d
    ```
---
