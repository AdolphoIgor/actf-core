# Model Evaluation and Gatekeeper Engine (`4-model-eval`)

The `4-model-eval` module governs the empirical capability auditing, qualitative evaluation, statistical non-inferiority certification, and automated model registry lifecycle of candidate foundation model checkpoints. It implements **Gate 5 (The Automated Gatekeeper)** to enforce zero-tolerance safety invariants and statistical quality baselines prior to production deployment.

---

## Evaluation & Promotion Lifecycle

```text
[ Staged Model Artifact ]
           │
           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4-MODEL-EVALUATION PIPELINE LIFECYCLE                                  │
├────────────────────────────────────────────────────────────────────────┤
│ Step 15: Gold Benchmark Evaluation                                     │
│   • Length-normalized multiple-choice log-likelihood ranking (MMLU).   │
│   • Exact-match generative arithmetic with Chain-of-Thought (GSM8K).   │
│   • Sandboxed subprocess execution for functional code (HumanEval).    │
│   • Deterministic JSON and Python AST schema validation.               │
│                                                                        │
│ Step 16: LLM-as-a-Judge Scoring                                        │
│   • Symmetric pairwise tournament evaluations (Forward and Reverse).   │
│   • Wilson 95% confidence interval estimation on effective win rates.  │
│   • Position bias detection and neutralization.                        │
│                                                                        │
│ Gate 5: Automated Gatekeeper Decision Engine                           │
│   • Tier 1: Zero-tolerance invariants (AST syntax, EOS, PII, KV-cache).│
│   • Tier 2: McNemar Chi-Square and Bootstrap non-inferiority testing.  │
│   • Tier 3: LLM Judge statistical certification (p_lower >= 0.50).     │
│   • Tier 4: Expected Calibration Error (ECE <= 0.06) and latency SLAs. │
│                                                                        │
│ Step 17: MLflow Model Registry Promotion                               │
│   • Cryptographic SHA-256 weight hash validation against receipts.     │
│   • Atomic alias cutovers (@champion, @challenger, @quarantined).      │
│   • Automated emergency rollback hooks to @archived versions.          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ Production: @champion ]       [ Quarantined: @quarantined ]

```

---

## Directory Layout

```text
4-model-eval/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── __init__.py
│   ├── gate_05_automated_gatekeeper.py
│   ├── hardware_engine.py
│   ├── step_15_gold_benchmark_evaluation.py
│   ├── step_16_llm_judge_scoring.py
│   └── step_17_mlflow_registry_promotion.py
└── tests/
    ├── __init__.py
    ├── test_gate_05_automated_gatekeeper.py
    ├── test_hardware_engine.py
    ├── test_step_15_gold_benchmark_evaluation.py
    ├── test_step_16_llm_judge_scoring.py
    └── test_step_17_mlflow_registry_promotion.py

```

---

## Script Architecture

### Hardware Engine (`scripts/hardware_engine.py`)

* `get_evaluation_setup(model_id)`: Resolves compute devices and initializes PyTorch Causal LMs with matching tokenizers for log-likelihood and AST evaluations.
* `get_inference_engine(model_id)`: Initializes `vLLM` on GPU clusters or falls back to Hugging Face on CPU.

### Step 15: Gold Benchmark Evaluation (`scripts/step_15_gold_benchmark_evaluation.py`)

* Runs deterministic capability audits across multiple tasks.
* Employs length-normalized log-likelihood scoring for multiple-choice benchmarks.
* Executes generated code in isolated subprocesses with timeout constraints.
* Generates a composite capability scorecard across all target domains.

### Step 16: LLM-as-a-Judge Scoring (`scripts/step_16_llm_judge_scoring.py`)

* Manages pairwise tournament scoring between candidate models and production baselines.
* Executes symmetric forward and reverse trials to eliminate position bias.
* Computes Wilson score confidence intervals to ensure non-inferiority.

### Gate 5: Automated Gatekeeper (`scripts/gate_05_automated_gatekeeper.py`)

* Evaluates candidates across a hierarchical four-tier framework.
* Applies paired McNemar Chi-Square tests and empirical bootstrap confidence intervals.
* Verifies probability calibration via Expected Calibration Error (ECE) and latency SLAs.
* Emits signed `GatekeeperReceipt` JSON artifacts required for registry promotion.

### Step 17: MLflow Registry Promotion (`scripts/step_17_mlflow_registry_promotion.py`)

* Interfaces with MLflow Model Registry using Model Aliases.
* Enforces SHA-256 weight hash validation against signed Gate 5 receipts.
* Atomically cuts over `@champion` aliases and demotes prior versions to `@archived`.
* Provides one-call emergency rollback capabilities.

---

## Testing

Run unit tests from the workspace root or the package directory:

```bash
# Run all evaluation and gatekeeper unit tests
pytest 4-model-eval/tests/ -v

# Run Gate 5 decision engine tests
pytest 4-model-eval/tests/test_gate_05_automated_gatekeeper.py -v

# Run MLflow registry promotion tests
pytest 4-model-eval/tests/test_step_17_mlflow_registry_promotion.py -v

```
