import json
import math
import multiprocessing
import queue
import re
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scripts.hardware_engine import get_inference_engine
except ImportError:
    try:
        from hardware_engine import get_inference_engine
    except ImportError:
        get_inference_engine = None


def _code_execution_worker(full_code: str, entry_point: str, result_queue: multiprocessing.Queue):
    """Executes generated code in an isolated subprocess."""
    global_namespace = {}
    try:
        exec(full_code, global_namespace)
        result_queue.put({"status": "PASSED", "error": None})
    except Exception as e:
        result_queue.put({"status": "FAILED", "error": f"{type(e).__name__}: {str(e)}"})
    finally:
        # Guarantee the multiprocessing.Queue background thread flushes to the OS pipe
        # before the child process violently exits and destroys the buffer.
        time.sleep(0.1)


class GoldBenchmarkEvaluator:
    """
    Gold Benchmark Evaluation suite executing multi-task capability
    audits and regression verification on candidate model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        execution_timeout_sec: float = 3.0,
    ):
        self.model = model.to(device=device, dtype=dtype)
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.timeout = execution_timeout_sec
        self.model.eval()

    @torch.no_grad()
    def evaluate_multiple_choice(
        self,
        samples: list[dict[str, Any]],
        length_penalty: float = 1.0,
    ) -> float:
        """
        Computes accuracy over multiple-choice questions via log-likelihood ranking.
        Sample schema: {"prompt": str, "choices": List[str], "gold_idx": int}
        """
        correct_count = 0

        for sample in samples:
            prompt_text = sample["prompt"]
            choices = sample["choices"]
            gold_idx = sample["gold_idx"]

            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            scores = []

            for choice in choices:
                choice_ids = self.tokenizer.encode(f" {choice}", add_special_tokens=False)
                input_ids = torch.tensor(
                    [prompt_ids + choice_ids], dtype=torch.long, device=self.device
                )

                with torch.autocast(
                    device_type=self.device if self.device == "cuda" else "cpu",
                    dtype=self.dtype,
                ):
                    outputs = self.model(input_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if isinstance(logits, tuple):
                        logits = logits[0]

                shift_logits = logits[0, :-1, :].contiguous()
                shift_labels = input_ids[0, 1:].contiguous()
                log_probs = F.log_softmax(shift_logits, dim=-1)

                start_pos = len(prompt_ids) - 1
                end_pos = start_pos + len(choice_ids)
                target_indices = torch.arange(start_pos, end_pos, device=self.device)
                target_labels = shift_labels[target_indices]

                token_log_probs = log_probs[target_indices, target_labels]
                total_log_prob = token_log_probs.sum().item()

                normalized_score = total_log_prob / (len(choice_ids) ** length_penalty)
                scores.append(normalized_score)

            predicted_idx = int(torch.argmax(torch.tensor(scores)).item())
            if predicted_idx == gold_idx:
                correct_count += 1

        return correct_count / max(1, len(samples))

    @torch.no_grad()
    def evaluate_generative_math(
        self,
        samples: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> float:
        """
        Evaluates exact-match arithmetic from greedy CoT completions.
        Sample schema: {"prompt": str, "gold_answer": str}
        """
        correct_count = 0

        for sample in samples:
            prompt_text = sample["prompt"]
            gold_answer = sample["gold_answer"]

            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt_text, add_special_tokens=False)],
                dtype=torch.long,
                device=self.device,
            )

            curr_tokens = input_ids
            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=self.device if self.device == "cuda" else "cpu",
                    dtype=self.dtype,
                ):
                    outputs = self.model(curr_tokens)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if isinstance(logits, tuple):
                        logits = logits[0]

                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

                if next_token.item() == getattr(self.tokenizer, "eos_token_id", None):
                    break

            completion = self.tokenizer.decode(
                curr_tokens[0, input_ids.shape[1] :], skip_special_tokens=True
            )

            pred_num = self._extract_final_number(completion)
            gold_num = self._extract_final_number(gold_answer)

            if pred_num is not None and gold_num is not None:
                try:
                    if math.isclose(float(pred_num), float(gold_num), rel_tol=1e-5):
                        correct_count += 1
                except ValueError:
                    if pred_num.strip().lower() == gold_num.strip().lower():
                        correct_count += 1

        return correct_count / max(1, len(samples))

    @staticmethod
    def _extract_final_number(text: str) -> str | None:
        if "####" in text:
            ans = text.split("####")[-1].replace(",", "").replace("$", "").strip()
            match = re.search(r"^-?\d+(?:\.\d+)?", ans)
            if match:
                return match.group(0)

        cleaned = text.replace(",", "").replace("$", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        return numbers[-1] if numbers else None

    @torch.no_grad()
    def evaluate_code_synthesis(
        self,
        samples: list[dict[str, str]],
        max_new_tokens: int = 384,
    ) -> float:
        """
        Evaluates Python code synthesis via isolated subprocess execution.
        Sample schema: {"prompt": str, "test_code": str, "entry_point": str}
        """
        passed_count = 0

        for sample in samples:
            prompt_text = sample["prompt"]
            test_code = sample["test_code"]
            entry_point = sample["entry_point"]

            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt_text, add_special_tokens=False)],
                dtype=torch.long,
                device=self.device,
            )

            curr_tokens = input_ids
            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=self.device if self.device == "cuda" else "cpu",
                    dtype=self.dtype,
                ):
                    outputs = self.model(curr_tokens)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if isinstance(logits, tuple):
                        logits = logits[0]

                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

                if next_token.item() == getattr(self.tokenizer, "eos_token_id", None):
                    break

            completion = self.tokenizer.decode(
                curr_tokens[0, input_ids.shape[1] :], skip_special_tokens=True
            )

            if "```python" in completion:
                code_body = completion.split("```python")[1].split("```")[0].strip()
            elif "```" in completion:
                code_body = completion.split("```")[1].split("```")[0].strip()
            else:
                code_body = completion.strip()

            full_program = f"{prompt_text}\n{code_body}\n\n{test_code}\n\ncheck({entry_point})"

            if self._run_sandboxed_test(full_program, entry_point):
                passed_count += 1

        return passed_count / max(1, len(samples))

    def _run_sandboxed_test(self, full_code: str, entry_point: str) -> bool:
        ctx = multiprocessing.get_context("spawn")
        res_queue = ctx.Queue()
        process = ctx.Process(
            target=_code_execution_worker, args=(full_code, entry_point, res_queue)
        )
        process.start()

        passed = False
        try:
            # Block safely. Add 15.0s to account for massive PyTorch cold-start import overhead.
            res = res_queue.get(timeout=self.timeout + 15.0)
            if res.get("status") == "PASSED":
                passed = True
            elif res.get("status") == "FAILED":
                print(f"Sandbox Failed: {res.get('error')}")
        except queue.Empty:
            print("Sandbox Timed Out")
            passed = False
        except Exception as e:
            print(f"Sandbox Exception: {e}")
            passed = False
        finally:
            if process.is_alive():
                process.terminate()
            process.join()

        return passed

    @torch.no_grad()
    def evaluate_schema_integrity(
        self,
        samples: list[dict[str, Any]],
        max_new_tokens: int = 128,
    ) -> float:
        """
        Validates that generated JSON outputs parse cleanly via json.loads.
        Sample schema: {"prompt": str, "required_keys": List[str]}
        """
        valid_count = 0

        for sample in samples:
            input_ids = torch.tensor(
                [self.tokenizer.encode(sample["prompt"], add_special_tokens=False)],
                dtype=torch.long,
                device=self.device,
            )

            curr_tokens = input_ids
            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=self.device if self.device == "cuda" else "cpu",
                    dtype=self.dtype,
                ):
                    outputs = self.model(curr_tokens)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if isinstance(logits, tuple):
                        logits = logits[0]

                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

                if next_token.item() == getattr(self.tokenizer, "eos_token_id", None):
                    break

            completion = self.tokenizer.decode(
                curr_tokens[0, input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

            if "```json" in completion:
                completion = completion.split("```json")[1].split("```")[0].strip()
            elif "```" in completion:
                completion = completion.split("```")[1].split("```")[0].strip()

            try:
                data = json.loads(completion)
                if isinstance(data, dict):
                    req_keys = sample.get("required_keys", [])
                    if all(k in data for k in req_keys):
                        valid_count += 1
            except Exception:
                pass

        return valid_count / max(1, len(samples))

    def run_gold_battery(
        self,
        mc_samples: list[dict[str, Any]],
        math_samples: list[dict[str, str]],
        code_samples: list[dict[str, str]],
        schema_samples: list[dict[str, Any]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Executes the full test suite and returns a weighted composite scorecard."""
        w = weights or {"mc": 0.25, "math": 0.25, "code": 0.25, "schema": 0.25}
        t_start = time.perf_counter()

        score_mc = self.evaluate_multiple_choice(mc_samples)
        score_math = self.evaluate_generative_math(math_samples)
        score_code = self.evaluate_code_synthesis(code_samples)
        score_schema = self.evaluate_schema_integrity(schema_samples)

        composite_score = (
            w["mc"] * score_mc
            + w["math"] * score_math
            + w["code"] * score_code
            + w["schema"] * score_schema
        )

        duration = time.perf_counter() - t_start

        return {
            "composite_score": composite_score,
            "multiple_choice_acc": score_mc,
            "math_exact_match": score_math,
            "code_pass1": score_code,
            "schema_validity": score_schema,
            "eval_duration_sec": duration,
        }
