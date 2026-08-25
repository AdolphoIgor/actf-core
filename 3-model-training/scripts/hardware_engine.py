from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_training_setup(
    model_id: str,
    device_override: str | None = None,
    dtype_override: torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype, Any]:
    """
    Detects hardware capabilities and initializes a trainable Causal LM model.
    """
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype_override:
        dtype = dtype_override
    else:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
    except Exception:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype)

    return device, dtype, model


def get_inference_engine(model_id: str) -> Any | tuple[Any, Any]:
    """
    Initializes high-throughput vLLM engine if GPU is available;
    falls back to Hugging Face PyTorch CPU engine otherwise.
    """
    if torch.cuda.is_available():
        print("NVIDIA GPU detected. Initializing high-throughput vLLM engine...")
        from vllm import LLM

        return LLM(model=model_id)
    else:
        print("No GPU detected. Falling back to Hugging Face PyTorch CPU engine...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        return model, tokenizer
