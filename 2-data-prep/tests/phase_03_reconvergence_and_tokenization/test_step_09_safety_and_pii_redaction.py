import pytest
import pyarrow as pa
from unittest.mock import MagicMock
from scripts.phase_03_reconvergence_and_tokenization.step_09_safety_and_pii_redaction import SafetyAndPIIRedactor


def test_safety_toxicity_filtering():
    """Validates that text exceeding the toxicity threshold is pruned."""
    actor = SafetyAndPIIRedactor(toxicity_threshold=0.4)
    
    # Mocking Hugging Face toxicity pipeline
    actor.toxicity_pipeline = MagicMock()
    # Simulate: first is clean, second is highly toxic
    actor.toxicity_pipeline.side_effect = [
        [{'label': 'non-toxic', 'score': 0.99}],
        [{'label': 'toxic', 'score': 0.85}]
    ]
    
    # Mock Presidio to pass text through unchanged for this test
    actor.analyzer = MagicMock()
    actor.anonymizer = MagicMock()
    actor.anonymizer.anonymize.return_value = MagicMock(text="Clean text.")
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_clean", "doc_toxic"],
        "text": ["Clean text.", "Some highly toxic hateful content."]
    })
    
    result = actor(input_table)
    
    assert result.num_rows == 1
    assert result["doc_id"].to_pylist() == ["doc_clean"]


def test_safety_pii_redaction():
    """Validates that alphanumeric PII and NER entities are masked."""
    actor = SafetyAndPIIRedactor()
    
    # Disable toxicity check for this test
    actor.toxicity_pipeline = MagicMock()
    actor.toxicity_pipeline.return_value = [{'label': 'non-toxic', 'score': 0.99}]
    
    # Mock Presidio Anonymizer replacing an email
    actor.analyzer = MagicMock()
    actor.anonymizer = MagicMock()
    actor.anonymizer.anonymize.return_value = MagicMock(
        text="Please contact [EMAIL_ADDRESS] for support."
    )
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_pii"],
        "text": ["Please contact john.doe@enterprise.com for support."]
    })
    
    result = actor(input_table)
    
    # Verify the table wasn't dropped, but the text WAS mutated
    assert result.num_rows == 1
    assert result["text"].to_pylist() == ["Please contact [EMAIL_ADDRESS] for support."]
    assert "safety_audited_at" in result.column_names