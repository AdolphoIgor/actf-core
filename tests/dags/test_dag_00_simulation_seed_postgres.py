import json
import pytest
from unittest.mock import MagicMock, patch
from dags.dag_00_simulation_seed_postgres import (
    apply_high_entropy_mutation,
    bootstrap_records_with_high_entropy,
    ResilientGeminiPool,
    generate_free_tier_seed,
    FREE_TIER_MODEL_POOL,
)


def test_apply_high_entropy_mutation_python_code():
    """Validates that Python code gets AST-safe '#' comment header injection."""
    python_code = "def process_order(order_id):\n    return True"
    mutated = apply_high_entropy_mutation(python_code)
    
    assert mutated.startswith("# [")
    assert "def process_order" in mutated


def test_apply_high_entropy_mutation_sql_code():
    """Validates that SQL queries get AST-safe '/* */' block comment header injection."""
    sql_query = "SELECT user_id, amount FROM transactions WHERE status = 'FLAGGED';"
    mutated = apply_high_entropy_mutation(sql_query)
    
    assert mutated.startswith("/* [")
    assert "*/" in mutated
    assert "SELECT user_id" in mutated


def test_apply_high_entropy_mutation_prose():
    """Validates that plain natural language prose gets standard prefix headers."""
    prose = "Trader-4102 initiated an unapproved cross-border order for AAPL."
    mutated = apply_high_entropy_mutation(prose)
    
    assert "TRACE_ID:" in mutated
    assert "Trader-" in mutated
    assert "AAPL" not in mutated or any(t in mutated for t in ["MSFT", "GOOGL", "NVDA", "TSLA"])


def test_resilient_gemini_pool_rotation():
    """Validates failover rotation when a model exhausts its quota."""
    pool = ResilientGeminiPool(FREE_TIER_MODEL_POOL)
    
    first_model = pool.get_active_model()["name"]
    assert first_model == "gemini-3.5-flash-lite"
    
    pool.rotate_to_next_model("429 RESOURCE_EXHAUSTED")
    second_model = pool.get_active_model()["name"]
    assert second_model == "gemini-3.1-flash-lite"


def test_bootstrap_records_with_high_entropy():
    """Validates row replication up to target count with unique UUIDs and mutated amounts."""
    seed_records = [
        {"amount": 100000.00, "narrative": "Suspicious transfer detected.", "report": {"risk_tier": "HIGH"}}
    ]
    
    payloads = bootstrap_records_with_high_entropy(seed_records, target_count=5)
    
    assert len(payloads) == 5
    uuids = {r["audit_uuid"] for r in payloads}
    assert len(uuids) == 5


@patch("dags.dag_00_simulation_seed_postgres.genai.Client")
def test_generate_free_tier_seed_mocked(mock_client_cls):
    """Validates seed chunk extraction using a mocked Gemini API client."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {
            "division": "Investment Banking",
            "amount": 250000.00,
            "narrative": "Possible insider trading alert.",
            "report": "{\"framework\": \"SEC Rule 10b-5\", \"risk_tier\": \"HIGH\"}",
            "violation": True,
            "score": 8
        }
    ])
    mock_client.models.generate_content.return_value = mock_response
    
    pool = ResilientGeminiPool(FREE_TIER_MODEL_POOL)
    records = generate_free_tier_seed(
        client=mock_client,
        seed_count=1,
        date_start="2026-06-01",
        date_end="2026-06-05",
        anomaly_focus="Insider Trading",
        pool=pool
    )
    
    assert len(records) == 1
    assert records[0]["division"] == "Investment Banking"