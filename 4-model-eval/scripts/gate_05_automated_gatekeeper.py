import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class GatekeeperReceipt:
    artifact_id: str
    verdict: str  # "PROMOTED" or "QUARANTINED"
    timestamp_utc: str
    tier1_invariants_passed: bool
    tier2_statistics_passed: bool
    tier3_judge_passed: bool
    tier4_operational_passed: bool
    rejection_reasons: list[str]
    scorecard: dict[str, Any]


class AutomatedGatekeeperEngine:
    """
    Automated Gate 5 production promotion arbiter executing multi-tier
    invariant checks, statistical non-inferiority audits, and signing release receipts.
    """

    def __init__(
        self,
        non_inferiority_margin: float = 0.005,
        significance_alpha: float = 0.05,
        max_allowable_ece: float = 0.06,
        min_judge_win_rate: float = 0.52,
        bootstrap_resamples: int = 10000,
    ):
        self.delta_margin = non_inferiority_margin
        self.alpha = significance_alpha
        self.max_ece = max_allowable_ece
        self.min_judge_win = min_judge_win_rate
        self.n_boot = bootstrap_resamples

    @staticmethod
    def _evaluate_mcnemar(
        base_correct: np.ndarray, cand_correct: np.ndarray
    ) -> tuple[float, float, int, int]:
        n_10 = int(np.sum((base_correct == 1) & (cand_correct == 0)))  # Regressions
        n_01 = int(np.sum((base_correct == 0) & (cand_correct == 1)))  # Improvements
        total = n_10 + n_01

        if total == 0:
            return 0.0, 1.0, 0, 0
        if total < 25:
            p_val = 2.0 * float(stats.binom.cdf(min(n_10, n_01), total, 0.5))
            return 0.0, min(1.0, p_val), n_10, n_01

        chi2 = float(((abs(n_01 - n_10) - 1.0) ** 2) / total)
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
        return chi2, p_val, n_10, n_01

    def _evaluate_bootstrap_ci(
        self, base_arr: np.ndarray, cand_arr: np.ndarray
    ) -> tuple[float, float, float]:
        deltas = cand_arr.astype(float) - base_arr.astype(float)
        mean_d = float(np.mean(deltas))

        rng = np.random.default_rng(seed=1337)
        idx = rng.integers(0, len(deltas), size=(self.n_boot, len(deltas)))
        boot_means = np.mean(deltas[idx], axis=1)

        low = float(np.percentile(boot_means, (self.alpha / 2.0) * 100))
        upp = float(np.percentile(boot_means, (1.0 - self.alpha / 2.0) * 100))
        return mean_d, low, upp

    @staticmethod
    def _compute_wilson_ci(
        wins: int, ties: int, total: int, z: float = 1.96
    ) -> tuple[float, float, float]:
        eff_wins = wins + 0.5 * ties
        p = eff_wins / max(1, total)
        denom = 1.0 + (z**2) / max(1, total)
        centre = p + (z**2) / (2.0 * max(1, total))
        spread = z * math.sqrt(
            (p * (1.0 - p) / max(1, total)) + (z**2) / (4.0 * (max(1, total) ** 2))
        )
        return p, (centre - spread) / denom, (centre + spread) / denom

    def arbitrate_release(
        self,
        artifact_id: str,
        ast_syntax_rate: float,
        eos_compliance_rate: float,
        kv_cache_delta: float,
        pii_leaks: int,
        benchmarks: dict[str, tuple[np.ndarray, np.ndarray]],
        judge_results: tuple[int, int, int, int],
        ece_score: float,
        itl_ms: float,
        max_itl_sla_ms: float,
        peak_vram_gb: float,
        vram_limit_gb: float,
    ) -> GatekeeperReceipt:
        rejections: list[str] = []
        scorecard: dict[str, Any] = {}

        # -------------------------------------------------------------
        # TIER 1: ZERO-TOLERANCE HARD INVARIANTS
        # -------------------------------------------------------------
        t1_passed = True
        if ast_syntax_rate < 1.0:
            t1_passed = False
            rejections.append(
                f"Tier 1 Failed: Code AST Syntax Pass Rate = {ast_syntax_rate * 100:.2f}% (Expected 100%)"
            )
        if eos_compliance_rate < 0.99:
            t1_passed = False
            rejections.append(
                f"Tier 1 Failed: EOS Compliance = {eos_compliance_rate * 100:.2f}% (Expected >= 99%)"
            )
        if kv_cache_delta >= 1e-3:
            t1_passed = False
            rejections.append(f"Tier 1 Failed: KV-Cache Logit Delta = {kv_cache_delta:.6e} >= 1e-3")
        if pii_leaks > 0:
            t1_passed = False
            rejections.append(f"Tier 1 Failed: PII Leaks Detected = {pii_leaks}")

        # -------------------------------------------------------------
        # TIER 2: BENCHMARK NON-INFERIORITY
        # -------------------------------------------------------------
        t2_passed = True
        bench_summary = {}
        for b_name, (base_arr, cand_arr) in benchmarks.items():
            chi2, p_val, n_reg, n_imp = self._evaluate_mcnemar(base_arr, cand_arr)
            mean_d, low_ci, upp_ci = self._evaluate_bootstrap_ci(base_arr, cand_arr)

            bench_summary[b_name] = {
                "delta": mean_d,
                "ci_lower": low_ci,
                "ci_upper": upp_ci,
                "mcnemar_p": p_val,
                "regressions": n_reg,
                "improvements": n_imp,
            }

            if n_reg > n_imp and p_val < self.alpha:
                t2_passed = False
                rejections.append(
                    f"Tier 2 Failed: Significant regression on {b_name} (p={p_val:.4f} < {self.alpha})"
                )
            if low_ci < -self.delta_margin:
                t2_passed = False
                rejections.append(
                    f"Tier 2 Failed: Non-inferiority breached on {b_name} (Lower CI {low_ci:.4f} < -{self.delta_margin})"
                )

        scorecard["benchmarks"] = bench_summary

        # -------------------------------------------------------------
        # TIER 3: LLM-AS-A-JUDGE TOURNAMENT ARBITRATION
        # -------------------------------------------------------------
        t3_passed = True
        c_wins, b_wins, ties, n_judge = judge_results
        eff_wr, wilson_low, wilson_upp = self._compute_wilson_ci(c_wins, ties, n_judge)

        scorecard["llm_judge"] = {
            "effective_win_rate": eff_wr,
            "wilson_ci_lower": wilson_low,
            "wilson_ci_upper": wilson_upp,
        }

        if wilson_low < 0.50:
            t3_passed = False
            rejections.append(f"Tier 3 Failed: Judge Wilson CI lower bound {wilson_low:.4f} < 0.50")
        if eff_wr < self.min_judge_win:
            t3_passed = False
            rejections.append(
                f"Tier 3 Failed: Judge Win Rate {eff_wr * 100:.1f}% < {self.min_judge_win * 100:.1f}% threshold"
            )

        # -------------------------------------------------------------
        # TIER 4: OPERATIONAL SLA & PROBABILISTIC CALIBRATION
        # -------------------------------------------------------------
        t4_passed = True
        if ece_score > self.max_ece:
            t4_passed = False
            rejections.append(
                f"Tier 4 Failed: Expected Calibration Error {ece_score:.4f} > {self.max_ece}"
            )
        if itl_ms > max_itl_sla_ms:
            t4_passed = False
            rejections.append(
                f"Tier 4 Failed: Inter-Token Latency {itl_ms:.2f}ms exceeds SLA limit {max_itl_sla_ms:.2f}ms"
            )
        if peak_vram_gb > vram_limit_gb:
            t4_passed = False
            rejections.append(
                f"Tier 4 Failed: Peak VRAM {peak_vram_gb:.1f}GB exceeds limit {vram_limit_gb:.1f}GB"
            )

        scorecard["operational"] = {
            "ece": ece_score,
            "itl_ms": itl_ms,
            "peak_vram_gb": peak_vram_gb,
        }
        scorecard["ast_syntax_pass_rate"] = ast_syntax_rate

        overall_pass = t1_passed and t2_passed and t3_passed and t4_passed
        verdict = "PROMOTED" if overall_pass else "QUARANTINED"

        return GatekeeperReceipt(
            artifact_id=artifact_id,
            verdict=verdict,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            tier1_invariants_passed=t1_passed,
            tier2_statistics_passed=t2_passed,
            tier3_judge_passed=t3_passed,
            tier4_operational_passed=t4_passed,
            rejection_reasons=rejections,
            scorecard=scorecard,
        )

    def save_signed_receipt(self, receipt: GatekeeperReceipt, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(receipt), f, indent=2)
        print(f"Gatekeeper Receipt saved -> Verdict: {receipt.verdict} at {output_path}")
