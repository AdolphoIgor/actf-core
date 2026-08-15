import pytest
import pyarrow as pa
from scripts.phase_01_shared_ingestion.step_02_boilerplate_stripping import strip_boilerplate_batch, strip_boilerplate_text


def test_strip_boilerplate_text_legal_stamps():
    """Validates that corporate confidentiality and draft watermarks are removed."""
    raw_text = "CONFIDENTIAL - INTERNAL USE ONLY\nHere is the actual project data.\nDraft Copy"
    result = strip_boilerplate_text(raw_text)
    
    assert "Here is the actual project data." in result
    assert "CONFIDENTIAL" not in result.upper()
    assert "Draft Copy" not in result


def test_strip_boilerplate_text_script_leakage():
    """Validates that <script> and <style> contents are fully destroyed."""
    raw_html = "<script>console.log('malicious code');</script><p>Clean text.</p><style>body {color: red;}</style>"
    result = strip_boilerplate_text(raw_html)
    
    assert "Clean text." in result
    assert "console.log" not in result
    assert "body {color" not in result


def test_strip_boilerplate_batch_pyarrow_table():
    """Validates 100% vectorized C++ boilerplate stripping and Code Disambiguation on PyArrow Tables."""
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_1", "doc_2"],
        "text": [
            "<h1>Header</h1><p>Useful prose.</p> Copyright (c) 2024 Acme Corp.",
            "public void run() { int x = 0; if (x == 0) { return; } };"
        ]
    })
    
    result_table = strip_boilerplate_batch(input_table)
    assert isinstance(result_table, pa.Table)
    
    cleaned_texts = result_table["text"].to_pylist()
    is_code = result_table["is_code_heavy"].to_pylist()
    
    # Validate Text Cleanup
    assert "<h1" not in cleaned_texts[0]
    assert "Useful prose." in cleaned_texts[0]
    
    # Validate Code Disambiguation (Heuristic Profiling)
    assert is_code[0] is False  # Natural language document
    assert is_code[1] is True   # High density of '{', '}', ';'


def test_strip_boilerplate_batch_dict_format():
    """Validates Ray Data dictionary batch format compatibility and math robustness."""
    input_batch = {
        # Include an empty string to test the PyArrow division-by-zero protection
        "text": ["<nav>Home</nav>Article content.", ""]
    }
    
    result_batch = strip_boilerplate_batch(input_batch)
    
    assert isinstance(result_batch, dict)
    assert "boilerplate_stripped_at" in result_batch
    assert "is_code_heavy" in result_batch
    
    assert "<nav>" not in result_batch["text"][0]
    assert result_batch["is_code_heavy"][0] is False
    assert result_batch["is_code_heavy"][1] is False # Empty string should not flag as code
