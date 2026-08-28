import sys
from unittest.mock import MagicMock, patch

import torch

from scripts.hardware_engine import get_inference_engine, get_training_setup


def test_get_training_setup_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_model = MagicMock()
        mock_from_pretrained.return_value.to.return_value = mock_model

        device, dtype, model = get_training_setup("Qwen/Qwen2.5-0.5B-Instruct")

        assert device.type == "cpu"
        assert dtype == torch.float32
        assert model == mock_model


def test_get_training_setup_with_overrides():
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_model = MagicMock()
        mock_from_pretrained.return_value.to.return_value = mock_model

        device, dtype, model = get_training_setup(
            "Qwen/Qwen2.5-0.5B-Instruct",
            device_override="cpu",
            dtype_override=torch.float32,
        )

        assert device.type == "cpu"
        assert dtype == torch.float32
        assert model == mock_model


def test_get_inference_engine_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load,
    ):
        mock_tok.return_value = MagicMock()
        mock_model_load.return_value.to.return_value = MagicMock()

        model, tokenizer = get_inference_engine("Qwen/Qwen2.5-0.5B-Instruct")

        assert model is not None
        assert tokenizer is not None


def test_get_inference_engine_gpu_branch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with patch.dict(sys.modules, {"vllm": MagicMock()}):
        mock_vllm = sys.modules["vllm"]
        mock_engine_instance = MagicMock()
        mock_vllm.LLM.return_value = mock_engine_instance

        engine = get_inference_engine("Qwen/Qwen2.5-0.5B-Instruct")
        assert engine == mock_engine_instance
