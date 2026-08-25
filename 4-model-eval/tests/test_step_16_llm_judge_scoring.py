import json

from scripts.step_16_llm_judge_scoring import (
    PairwiseTrialResult,
    ProductionLLMJudgeScorer,
)


def test_parse_judge_json_clean_payload():
    scorer = ProductionLLMJudgeScorer(judge_generate_fn=lambda prompt: "")
    raw_json = json.dumps(
        {
            "critique": "Candidate 1 adhered strictly to all negative constraints.",
            "winner": "Candidate 1",
        }
    )

    winner, critique = scorer._parse_judge_json(raw_json)
    assert winner == "Candidate 1"
    assert "adhered strictly" in critique


def test_parse_judge_json_markdown_wrapped():
    scorer = ProductionLLMJudgeScorer(judge_generate_fn=lambda prompt: "")
    raw_markdown = """```json
    {
        "critique": "Candidate 2 provided more concise analysis.",
        "winner": "Candidate 2"
    }
    ```"""

    winner, critique = scorer._parse_judge_json(raw_markdown)
    assert winner == "Candidate 2"


def test_parse_judge_json_fallback_regex():
    scorer = ProductionLLMJudgeScorer(judge_generate_fn=lambda prompt: "")
    raw_malformed = "Based on my evaluation, Candidate 1 clearly wrote better structured code."

    winner, _ = scorer._parse_judge_json(raw_malformed)
    assert winner == "Candidate 1"


def test_evaluate_pair_consistent_candidate_win():
    # Trial 1: Pos 1 (Candidate) -> Candidate 1
    # Trial 2: Pos 2 (Candidate) -> Candidate 2
    mock_responses = iter(
        [
            json.dumps({"critique": "Pos 1 better", "winner": "Candidate 1"}),
            json.dumps({"critique": "Pos 2 better", "winner": "Candidate 2"}),
        ]
    )

    scorer = ProductionLLMJudgeScorer(judge_generate_fn=lambda prompt: next(mock_responses))
    result = scorer.evaluate_pair(
        prompt_id="p1",
        prompt="Write a compliance summary.",
        cand_resp="Candidate text",
        base_resp="Baseline text",
    )

    assert result.resolved_winner == "candidate"
    assert result.is_position_biased is False


def test_evaluate_pair_detects_position_bias():
    # Positional bias: Judge selects Position 1 in both trials
    mock_responses = iter(
        [
            json.dumps({"critique": "Pos 1 favored", "winner": "Candidate 1"}),
            json.dumps({"critique": "Pos 1 favored again", "winner": "Candidate 1"}),
        ]
    )

    scorer = ProductionLLMJudgeScorer(judge_generate_fn=lambda prompt: next(mock_responses))
    result = scorer.evaluate_pair(
        prompt_id="p2",
        prompt="Explain liquidity risk.",
        cand_resp="Candidate text",
        base_resp="Baseline text",
    )

    assert result.resolved_winner == "inconsistent"
    assert result.is_position_biased is True


def test_tournament_scorecard_certification():
    scorer = ProductionLLMJudgeScorer(
        judge_generate_fn=lambda prompt: "", min_win_rate_threshold=0.52
    )

    # 100 trials: 60 candidate wins, 30 baseline wins, 10 ties
    trial_results = []
    for i in range(60):
        trial_results.append(
            PairwiseTrialResult(f"p_{i}", "Candidate 1", "Candidate 2", "candidate", False, "", "")
        )
    for i in range(60, 90):
        trial_results.append(
            PairwiseTrialResult(f"p_{i}", "Candidate 2", "Candidate 1", "baseline", False, "", "")
        )
    for i in range(90, 100):
        trial_results.append(PairwiseTrialResult(f"p_{i}", "Tie", "Tie", "tie", False, "", ""))

    scorecard = scorer.compute_tournament_scorecard(trial_results)

    assert scorecard.total_evaluations == 100
    assert scorecard.candidate_wins == 60
    assert scorecard.baseline_wins == 30
    assert scorecard.ties == 10
    assert scorecard.effective_win_rate == 0.65
    assert scorecard.wilson_ci_lower > 0.50
    assert scorecard.certified_promotion is True


def test_tournament_scorecard_rejection_on_low_win_rate():
    scorer = ProductionLLMJudgeScorer(
        judge_generate_fn=lambda prompt: "", min_win_rate_threshold=0.52
    )

    # 100 trials: 40 candidate wins, 55 baseline wins, 5 ties
    trial_results = []
    for i in range(40):
        trial_results.append(
            PairwiseTrialResult(f"p_{i}", "Candidate 1", "Candidate 2", "candidate", False, "", "")
        )
    for i in range(40, 95):
        trial_results.append(
            PairwiseTrialResult(f"p_{i}", "Candidate 2", "Candidate 1", "baseline", False, "", "")
        )
    for i in range(95, 100):
        trial_results.append(PairwiseTrialResult(f"p_{i}", "Tie", "Tie", "tie", False, "", ""))

    scorecard = scorer.compute_tournament_scorecard(trial_results)

    assert scorecard.effective_win_rate < 0.50
    assert scorecard.certified_promotion is False
