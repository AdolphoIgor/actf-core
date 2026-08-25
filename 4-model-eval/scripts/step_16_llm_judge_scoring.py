import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class PairwiseTrialResult:
    prompt_id: str
    decision_trial_1: str
    decision_trial_2: str
    resolved_winner: str  # "candidate", "baseline", "tie", or "inconsistent"
    is_position_biased: bool
    critique_1: str
    critique_2: str


@dataclass
class JudgeTournamentScorecard:
    total_evaluations: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    inconsistent_trials: int
    raw_win_rate: float
    effective_win_rate: float
    wilson_ci_lower: float
    wilson_ci_upper: float
    certified_promotion: bool


class ProductionLLMJudgeScorer:
    """
    Automated LLM Judge harness executing pairwise symmetric evaluations,
    bidirectional bias mitigation, and statistical capability certification.
    """

    def __init__(
        self,
        judge_generate_fn: Callable[[str], str],
        confidence_level_z: float = 1.96,
        min_win_rate_threshold: float = 0.52,
    ):
        self.judge_fn = judge_generate_fn
        self.z = confidence_level_z
        self.min_win_rate = min_win_rate_threshold

    def _build_judge_prompt(
        self,
        prompt: str,
        resp_1: str,
        resp_2: str,
        reference: str | None = None,
    ) -> str:
        ref_section = f"\n[Reference Ground Truth]:\n{reference}\n" if reference else ""
        return f"""You are an expert, impartial AI judge evaluating two candidate responses.
Analyze both responses based on:
1. Instruction Following and Negative Constraint Adherence.
2. Factuality, Logical Correctness, and Absence of Hallucinations.
3. Clarity, Scannability, and Organization.
4. Conciseness (penalize irrelevant padding or conversational filler).

[User Prompt]:
{prompt}
{ref_section}
[Candidate Response 1]:
{resp_1}

[Candidate Response 2]:
{resp_2}

Provide a concise, step-by-step critique analyzing the strengths and flaws of both responses.
Then declare the winner as strictly 'Candidate 1', 'Candidate 2', or 'Tie'.

You must respond in valid JSON matching this schema:
{{
  "critique": "<step-by-step reasoning>",
  "winner": "Candidate 1" | "Candidate 2" | "Tie"
}}
"""

    def _parse_judge_json(self, raw_output: str) -> tuple[str, str]:
        """Extracts critique and winner declaration from raw judge outputs."""
        try:
            cleaned = raw_output.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            data = json.loads(cleaned)
            winner = data.get("winner", "").strip()
            critique = data.get("critique", "").strip()

            if winner not in ["Candidate 1", "Candidate 2", "Tie"]:
                if "Candidate 1" in winner:
                    winner = "Candidate 1"
                elif "Candidate 2" in winner:
                    winner = "Candidate 2"
                else:
                    winner = "Tie"

            return winner, critique
        except Exception:
            if re.search(r"Candidate\s*1", raw_output, re.IGNORECASE):
                return "Candidate 1", "Regex parsed fallback"
            if re.search(r"Candidate\s*2", raw_output, re.IGNORECASE):
                return "Candidate 2", "Regex parsed fallback"
            return "Tie", "Parse failed fallback"

    def evaluate_pair(
        self,
        prompt_id: str,
        prompt: str,
        cand_resp: str,
        base_resp: str,
        reference: str | None = None,
    ) -> PairwiseTrialResult:
        """
        Executes symmetric forward and reverse trials to eliminate position bias.
        """
        # Trial 1: Candidate = Pos 1, Baseline = Pos 2
        p1 = self._build_judge_prompt(prompt, cand_resp, base_resp, reference)
        raw_1 = self.judge_fn(p1)
        dec_1, crit_1 = self._parse_judge_json(raw_1)

        # Trial 2: Baseline = Pos 1, Candidate = Pos 2
        p2 = self._build_judge_prompt(prompt, base_resp, cand_resp, reference)
        raw_2 = self.judge_fn(p2)
        dec_2, crit_2 = self._parse_judge_json(raw_2)

        # Symmetric Resolution Logic
        is_biased = False
        if dec_1 == "Candidate 1" and dec_2 == "Candidate 2":
            resolved = "candidate"
        elif dec_1 == "Candidate 2" and dec_2 == "Candidate 1":
            resolved = "baseline"
        elif dec_1 == "Tie" and dec_2 == "Tie":
            resolved = "tie"
        elif dec_1 == dec_2 and dec_1 in ["Candidate 1", "Candidate 2"]:
            is_biased = True
            resolved = "inconsistent"
        else:
            resolved = "tie"

        return PairwiseTrialResult(
            prompt_id=prompt_id,
            decision_trial_1=dec_1,
            decision_trial_2=dec_2,
            resolved_winner=resolved,
            is_position_biased=is_biased,
            critique_1=crit_1,
            critique_2=crit_2,
        )

    def compute_tournament_scorecard(
        self, trial_results: list[PairwiseTrialResult]
    ) -> JudgeTournamentScorecard:
        """
        Aggregates pairwise match results and calculates Wilson score confidence intervals.
        """
        n_total = len(trial_results)
        w_cand = sum(1 for r in trial_results if r.resolved_winner == "candidate")
        w_base = sum(1 for r in trial_results if r.resolved_winner == "baseline")
        ties = sum(1 for r in trial_results if r.resolved_winner == "tie")
        inconsistents = sum(1 for r in trial_results if r.resolved_winner == "inconsistent")

        # Inconsistent trials are treated as ties for win-rate calculations
        effective_wins = w_cand + 0.5 * (ties + inconsistents)
        effective_p = effective_wins / max(1, n_total)
        raw_win_rate = w_cand / max(1, n_total)

        z = self.z
        denominator = 1.0 + (z**2) / max(1, n_total)
        centre_adjusted_p = effective_p + (z**2) / (2.0 * max(1, n_total))
        adjusted_std = math.sqrt(
            (effective_p * (1.0 - effective_p) / max(1, n_total))
            + (z**2) / (4.0 * (max(1, n_total) ** 2))
        )

        ci_lower = (centre_adjusted_p - z * adjusted_std) / denominator
        ci_upper = (centre_adjusted_p + z * adjusted_std) / denominator

        is_certified = (ci_lower >= 0.50) and (effective_p >= self.min_win_rate)

        return JudgeTournamentScorecard(
            total_evaluations=n_total,
            candidate_wins=w_cand,
            baseline_wins=w_base,
            ties=ties,
            inconsistent_trials=inconsistents,
            raw_win_rate=raw_win_rate,
            effective_win_rate=effective_p,
            wilson_ci_lower=ci_lower,
            wilson_ci_upper=ci_upper,
            certified_promotion=is_certified,
        )
