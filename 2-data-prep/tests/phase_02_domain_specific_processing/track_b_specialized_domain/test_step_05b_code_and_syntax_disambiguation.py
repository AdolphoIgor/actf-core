import pytest
import pyarrow as pa
from unittest.mock import MagicMock
from scripts.phase_02_domain_specific_processing.track_b_specialized_domain.step_05b_code_and_syntax_disambiguation import CodeSyntaxDisambiguation


def test_code_syntax_valid_source_file():
    """Validates that a normal Python file passes structural symbol and layout constraints[cite: 2]."""
    actor = CodeSyntaxDisambiguation(model_path="mock_path")
    actor.fasttext_model = MagicMock()
    actor.fasttext_model.predict.return_value = (["__label__code"], [0.95])
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_1"],
        "text": ["def valid_code(x): \n    return { 'result': [x[i] -> y[i]] };\n"]
    })
    
    result = actor(input_table)
    assert result.num_rows == 1


def test_code_syntax_prose_or_license_header():
    """Validates Stage 1: Flags raw un-parsed prose or license headers (density < 0.01) and prunes them[cite: 2]."""
    actor = CodeSyntaxDisambiguation(model_path="mock_path")
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_license"],
        "text": ["Copyright 2026. Licensed under the Apache License, Version 2.0. You may not use this file except in compliance."]
    })
    
    result = actor(input_table)
    assert result.num_rows == 0


def test_code_syntax_minified_js_layout_profiling():
    """Validates Stage 3: Extreme single-line lengths (> 2,000 characters) are flagged as non-trainable asset noise[cite: 2]."""
    actor = CodeSyntaxDisambiguation(model_path="mock_path")
    
    # Simulate a minified JS string > 2000 chars with syntax symbols
    minified_line = "var a=1;{" * 500 + "}" * 500
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_minified"],
        "text": [minified_line]
    })
    
    result = actor(input_table)
    assert result.num_rows == 0


def test_code_syntax_corrupted_ocr_fasttext_check():
    """Validates Stage 2: OCR noise fails syntax classification confidence threshold (< 0.70)[cite: 2]."""
    actor = CodeSyntaxDisambiguation(model_path="mock_path")
    actor.fasttext_model = MagicMock()
    # Simulate fastText labeling the OCR as noise with high confidence
    actor.fasttext_model.predict.return_value = (["__label__noise"], [0.85])
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_ocr"],
        "text": ["p-ubl-ic v-oi-d; { } < > -> [ ]"] # High symbols, but garbled
    })
    
    result = actor(input_table)
    assert result.num_rows == 0