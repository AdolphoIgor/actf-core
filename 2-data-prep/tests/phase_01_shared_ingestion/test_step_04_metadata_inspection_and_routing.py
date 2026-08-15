import pytest
import pyarrow as pa
from unittest.mock import MagicMock
from scripts.phase_01_shared_ingestion.step_04_metadata_inspection_and_routing import MetadataRouter


def test_metadata_router_explicit_whitelist_schema():
    """Validates Tier 1: Schema metadata whitelist routes directly to Branch B (1)[cite: 2]."""
    router = MetadataRouter(model_path="mock_path")
    router.fasttext_model = None  # Disable model inference for whitelist test
    
    schema = pa.schema(
        [("doc_id", pa.string()), ("text", pa.string())],
        metadata={b"source_type": b"github_repo"}
    )
    
    input_table = pa.Table.from_pydict(
        {
            "doc_id": ["doc_1"],
            "text": ["Simple prose sentence without code syntax."]
        },
        schema=schema
    )
    
    result_table = router(input_table)
    
    assert "branch_id" in result_table.column_names
    assert "routed_at" in result_table.column_names
    assert result_table["branch_id"].to_pylist() == [1]


def test_metadata_router_prose_web_crawl():
    """Validates Tier 2: Low symbol density web prose routes to Branch A (0)[cite: 2]."""
    router = MetadataRouter(model_path="mock_path")
    router.fasttext_model = None 
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_1"],
        "source_type": ["web_crawl"],
        "text": ["This is a standard web page paragraph explaining company policies and regulatory compliance."]
    })
    
    result_table = router(input_table)
    assert result_table["branch_id"].to_pylist() == [0]


def test_metadata_router_fasttext_inference():
    """Validates Tier 2: High density triggers fastText micro-pass classification[cite: 2]."""
    router = MetadataRouter(model_path="mock_path")
    
    # Mocking fastText model predict method
    router.fasttext_model = MagicMock()
    
    # Simulate fastText answering code for the first, noise for the second
    router.fasttext_model.predict.side_effect = [
        (["__label__code"], [0.99]), 
        (["__label__noise"], [0.85])
    ]
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_1", "doc_2"],
        "text": [
            "def process_data(x, y): return { 'result': [x[i] -> y[i] for i in range(len(x))]; }",
            "This has high symbols [ ] { } = -> < > but is OCR noise."
        ]
    })
    
    result_table = router(input_table)
    
    assert result_table["branch_id"].to_pylist() == [1, 0]
    assert router.fasttext_model.predict.call_count == 2

