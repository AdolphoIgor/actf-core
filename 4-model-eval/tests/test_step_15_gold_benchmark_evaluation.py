import torch
import torch.nn as nn

from scripts.step_15_gold_benchmark_evaluation import (
    GoldBenchmarkEvaluator,
)


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.vocab = {"<pad>": 0, "A": 1, "B": 2, "C": 3, "D": 4}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 50 + 1 for c in text]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join([chr((t - 1) % 50 + 65) for t in token_ids])


class MockLMHead(nn.Module):
    def __init__(self, vocab_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        return logits


def test_extract_final_number():
    evaluator = GoldBenchmarkEvaluator(
        model=MockLMHead(),
        tokenizer=MockTokenizer(),
        device="cpu",
        dtype=torch.float32,
    )

    assert evaluator._extract_final_number("Final answer is #### 42") == "42"
    assert evaluator._extract_final_number("Calculated balance: $1,250.50") == "1250.50"
    assert evaluator._extract_final_number("The result is -15.4") == "-15.4"
    assert evaluator._extract_final_number("No digits present") is None


def test_evaluate_multiple_choice():
    model = MockLMHead()
    tokenizer = MockTokenizer()
    evaluator = GoldBenchmarkEvaluator(
        model=model, tokenizer=tokenizer, device="cpu", dtype=torch.float32
    )

    samples = [{"prompt": "What is 2+2?", "choices": ["3", "4", "5"], "gold_idx": 0}]

    acc = evaluator.evaluate_multiple_choice(samples)
    assert 0.0 <= acc <= 1.0


def test_sandboxed_code_execution_pass():
    evaluator = GoldBenchmarkEvaluator(
        model=MockLMHead(),
        tokenizer=MockTokenizer(),
        device="cpu",
        dtype=torch.float32,
    )

    valid_code = "def check(candidate):\n    assert candidate(2, 3) == 5"
    full_program = "def add(a, b):\n    return a + b\n\n" + valid_code + "\n\ncheck(add)"

    assert evaluator._run_sandboxed_test(full_program, "add") is True


def test_sandboxed_code_execution_fail():
    evaluator = GoldBenchmarkEvaluator(
        model=MockLMHead(),
        tokenizer=MockTokenizer(),
        device="cpu",
        dtype=torch.float32,
    )

    failing_code = "def check(candidate):\n    assert candidate(2, 3) == 999"
    full_program = "def add(a, b):\n    return a + b\n\n" + failing_code + "\n\ncheck(add)"

    assert evaluator._run_sandboxed_test(full_program, "add") is False


def test_run_gold_battery_composite_score():
    evaluator = GoldBenchmarkEvaluator(
        model=MockLMHead(),
        tokenizer=MockTokenizer(),
        device="cpu",
        dtype=torch.float32,
    )

    mc_samples = [{"prompt": "Test Q", "choices": ["A", "B"], "gold_idx": 0}]
    math_samples = [{"prompt": "Calculate 5*5", "gold_answer": "#### 25"}]
    code_samples = [
        {
            "prompt": "def dummy():",
            "test_code": "def check(fn): assert True",
            "entry_point": "dummy",
        }
    ]
    schema_samples = [{"prompt": "Emit JSON", "required_keys": ["status"]}]

    results = evaluator.run_gold_battery(
        mc_samples=mc_samples,
        math_samples=math_samples,
        code_samples=code_samples,
        schema_samples=schema_samples,
    )

    assert "composite_score" in results
    assert "multiple_choice_acc" in results
    assert "math_exact_match" in results
    assert "code_pass1" in results
    assert "schema_validity" in results
    assert "eval_duration_sec" in results
    assert 0.0 <= results["composite_score"] <= 1.0
