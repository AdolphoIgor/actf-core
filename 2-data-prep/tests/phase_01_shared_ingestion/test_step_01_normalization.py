import pytest
from scripts.phase_01_shared_ingestion.step_01_normalization import dehyphenate_text, normalize_batch


def test_dehyphenate_text_split_words():
    """
    Tests reassembly of layout-hyphenated words split across newlines by PDF/OCR engines.
    """
    raw_text = "The en-\nterprise software architecture"
    expected = "The enterprise software architecture"
    assert dehyphenate_text(raw_text) == expected


def test_dehyphenate_text_preserves_valid_hyphens():
    """
    Ensures standard hyphenated terms are not modified.
    """
    raw_text = "This is a high-quality product."
    assert dehyphenate_text(raw_text) == "This is a high-quality product."


def test_normalize_batch_unicode_nfkc():
    """
    Tests canonical decomposition/composition (NFD to NFC) and compatibility conversions.
    """
    # NFD representation of 'é' (\u0065\u0301) + ligature 'ﬁ' (\uFB01)
    raw_input = "caf\u0065\u0301 \uFB01le"
    batch = {"text": [raw_input]}
    
    result = normalize_batch(batch)
    
    # Converts ligature 'ﬁ' -> 'fi' and pre-composes 'é'
    assert result["text"][0] == "café file"


def test_normalize_batch_control_character_stripping():
    """
    Tests removal of zero-width spaces (\u200B) and non-printable control characters,
    while preserving standard whitespace (\n, \t).
    """
    # Zero-width space + Null byte (\x00) + valid text + newline
    raw_input = "Zero\u200B \x00Width\nSpace"
    batch = {"text": [raw_input]}
    
    result = normalize_batch(batch)
    
    assert result["text"][0] == "Zero Width\nSpace"


def test_normalize_batch_empty_and_null_strings():
    """
    Verifies graceful handling of empty or None-like strings.
    """
    batch = {"text": ["", None]}
    result = normalize_batch(batch)
    
    assert result["text"] == ["", ""]
    assert len(result["normalized_at"]) == 2


def test_normalize_batch_metadata_enrichment():
    """
    Verifies that the normalized_at timestamp column is added to every record.
    """
    batch = {"text": ["Sample document content"]}
    result = normalize_batch(batch)
    
    assert "normalized_at" in result
    assert isinstance(result["normalized_at"][0], str)
    assert len(result["normalized_at"][0]) > 0