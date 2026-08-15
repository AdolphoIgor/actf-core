import pytest
import pyarrow as pa
from unittest.mock import MagicMock
from scripts.phase_02_domain_specific_processing.track_a_natural_language.step_07a_natural_language_cqf import ProseQualityClassifier


def test_prose_cqf_filtering_and_scoring():
    """Validates that CQF drops low-quality text and appends the float64 score column."""
    actor = ProseQualityClassifier(model_path="mock_path", threshold=0.65)
    
    # Mocking fastText model predictions
    actor.fasttext_model = MagicMock()
    
    # Simulate: 
    # Doc 1 -> High Quality (0.92) -> KEEP
    # Doc 2 -> Low Quality  (0.80 LQ = 0.20 HQ) -> DROP
    actor.fasttext_model.predict.side_effect = [
        (["__label__hq"], [0.92]), 
        (["__label__lq"], [0.80])
    ]
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_hq", "doc_lq"],
        "text": [
            "This is a beautifully formatted, highly informative technical manual.",
            "Click here to buy cheap goods online SEO spam keyword keyword."
        ]
    })
    
    result = actor(input_table)
    
    # doc_lq should be dropped
    assert result.num_rows == 1
    assert result["doc_id"].to_pylist() == ["doc_hq"]
    
    # Ensure metadata columns were appended and retained their types
    assert "cqf_quality_score" in result.column_names
    assert "cqf_scored_at" in result.column_names
    
    score_type = result.schema.field("cqf_quality_score").type
    assert score_type == pa.float64()
    
    # Ensure the score matches the mocked HQ probability
    assert result["cqf_quality_score"].to_pylist() == [0.92]