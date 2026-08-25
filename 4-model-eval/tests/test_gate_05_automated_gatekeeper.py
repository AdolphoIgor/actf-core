import json

import numpy as np

from scripts.gate_05_automated_gatekeeper import (
    AutomatedGatekeeperEngine,
)


def test_mcnemar_evaluation():
    engine = AutomatedGatekeeperEngine()

    base_correct = np.array([1, 1, 0, 0, 1, 1, 0, 1])
    cand_correct = np.array([1, 1, 1, 1, 1, 1, 0, 1])

    chi2, p_val, n_reg, n_imp = engine._evaluate_mcnemar(base_correct, cand_correct)

    assert n_reg == 0
    assert n_imp == 2
    assert 0.0 <= p_val <= 1.0


def test_bootstrap_ci():
    engine = AutomatedGatekeeperEngine(bootstrap_resamples=1000)

    base_arr = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
    cand_arr = np.array([0.85, 0.92, 0.75, 0.88, 0.98])

    mean_d, low_ci, upp_ci = engine._evaluate_bootstrap_ci(base_arr, cand_arr)

    assert mean_d > 0.0
    assert low_ci <= mean_d <= upp_ci


def test_wilson_ci():
    engine = AutomatedGatekeeperEngine()

    eff_p, low_ci, upp_ci = engine._compute_wilson_ci(wins=60, ties=10, total=100)

    assert eff_p == 0.65
    assert 0.50 < low_ci < eff_p
    assert eff_p < upp_ci < 1.0


def test_arbitrate_release_full_pass(tmp_path):
    engine = AutomatedGatekeeperEngine(bootstrap_resamples=500)

    benchmarks = {
        "mmlu": (np.ones(100), np.ones(100)),
        "gsm8k": (np.ones(100), np.ones(100)),
    }
    judge_results = (60, 30, 10, 100)

    receipt = engine.arbitrate_release(
        artifact_id="candidate_checkpoint_v1",
        ast_syntax_rate=1.0,
        eos_compliance_rate=0.995,
        kv_cache_delta=1e-5,
        pii_leaks=0,
        benchmarks=benchmarks,
        judge_results=judge_results,
        ece_score=0.03,
        itl_ms=12.5,
        max_itl_sla_ms=25.0,
        peak_vram_gb=14.0,
        vram_limit_gb=24.0,
    )

    assert receipt.verdict == "PROMOTED"
    assert receipt.tier1_invariants_passed is True
    assert receipt.tier2_statistics_passed is True
    assert receipt.tier3_judge_passed is True
    assert receipt.tier4_operational_passed is True
    assert len(receipt.rejection_reasons) == 0

    receipt_path = tmp_path / "receipts" / "gate5_receipt.json"
    engine.save_signed_receipt(receipt, receipt_path)
    assert receipt_path.exists()

    with open(receipt_path, encoding="utf-8") as f:
        data = json.load(f)
        assert data["verdict"] == "PROMOTED"


def test_arbitrate_release_tier1_failure():
    engine = AutomatedGatekeeperEngine(bootstrap_resamples=100)

    benchmarks = {"mmlu": (np.ones(50), np.ones(50))}
    judge_results = (60, 30, 10, 100)

    receipt = engine.arbitrate_release(
        artifact_id="candidate_checkpoint_v2",
        ast_syntax_rate=0.95,  # Failure: < 1.0
        eos_compliance_rate=0.995,
        kv_cache_delta=1e-5,
        pii_leaks=1,  # Failure: > 0
        benchmarks=benchmarks,
        judge_results=judge_results,
        ece_score=0.03,
        itl_ms=12.5,
        max_itl_sla_ms=25.0,
        peak_vram_gb=14.0,
        vram_limit_gb=24.0,
    )

    assert receipt.verdict == "QUARANTINED"
    assert receipt.tier1_invariants_passed is False
    assert any("Code AST Syntax Pass Rate" in r for r in receipt.rejection_reasons)
    assert any("PII Leaks Detected" in r for r in receipt.rejection_reasons)


def test_arbitrate_release_tier2_failure():
    engine = AutomatedGatekeeperEngine(bootstrap_resamples=100)

    # Base correct 100%, candidate drops to 0% on MMLU
    benchmarks = {
        "mmlu": (np.ones(50), np.zeros(50)),
    }
    judge_results = (60, 30, 10, 100)

    receipt = engine.arbitrate_release(
        artifact_id="candidate_checkpoint_v3",
        ast_syntax_rate=1.0,
        eos_compliance_rate=0.995,
        kv_cache_delta=1e-5,
        pii_leaks=0,
        benchmarks=benchmarks,
        judge_results=judge_results,
        ece_score=0.03,
        itl_ms=12.5,
        max_itl_sla_ms=25.0,
        peak_vram_gb=14.0,
        vram_limit_gb=24.0,
    )

    assert receipt.verdict == "QUARANTINED"
    assert receipt.tier2_statistics_passed is False
    assert any(
        "Significant regression" in r or "Non-inferiority breached" in r
        for r in receipt.rejection_reasons
    )


def test_arbitrate_release_tier3_failure():
    engine = AutomatedGatekeeperEngine(bootstrap_resamples=100)

    benchmarks = {"mmlu": (np.ones(50), np.ones(50))}
    # Low win rate: 30 wins, 65 losses, 5 ties
    judge_results = (30, 65, 5, 100)

    receipt = engine.arbitrate_release(
        artifact_id="candidate_checkpoint_v4",
        ast_syntax_rate=1.0,
        eos_compliance_rate=0.995,
        kv_cache_delta=1e-5,
        pii_leaks=0,
        benchmarks=benchmarks,
        judge_results=judge_results,
        ece_score=0.03,
        itl_ms=12.5,
        max_itl_sla_ms=25.0,
        peak_vram_gb=14.0,
        vram_limit_gb=24.0,
    )

    assert receipt.verdict == "QUARANTINED"
    assert receipt.tier3_judge_passed is False
    assert any("Judge Win Rate" in r or "Judge Wilson CI" in r for r in receipt.rejection_reasons)


def test_arbitrate_release_tier4_failure():
    engine = AutomatedGatekeeperEngine(bootstrap_resamples=100)

    benchmarks = {"mmlu": (np.ones(50), np.ones(50))}
    judge_results = (60, 30, 10, 100)

    receipt = engine.arbitrate_release(
        artifact_id="candidate_checkpoint_v5",
        ast_syntax_rate=1.0,
        eos_compliance_rate=0.995,
        kv_cache_delta=1e-5,
        pii_leaks=0,
        benchmarks=benchmarks,
        judge_results=judge_results,
        ece_score=0.12,  # Failure: > 0.06
        itl_ms=35.0,  # Failure: > 25.0 SLA
        max_itl_sla_ms=25.0,
        peak_vram_gb=30.0,  # Failure: > 24.0 limit
        vram_limit_gb=24.0,
    )

    assert receipt.verdict == "QUARANTINED"
    assert receipt.tier4_operational_passed is False
    assert any("Expected Calibration Error" in r for r in receipt.rejection_reasons)
    assert any("Inter-Token Latency" in r for r in receipt.rejection_reasons)
    assert any("Peak VRAM" in r for r in receipt.rejection_reasons)
