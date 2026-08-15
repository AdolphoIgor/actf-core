import re
import datetime
import unicodedata
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

LAYOUT_HYPHEN_REGEX = re.compile(r'(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)')

# RE2-compliant Unicode syntax for PyArrow C++ Kernel execution
CONTROL_AND_PHANTOM_PATTERN = r"[\x00-\x08\x0B-\x1F\x7F\x{200B}\x{200C}\x{00AD}\x{FFFD}]"


def dehyphenate_text(text: str) -> str:
    """
    Reassembles layout-split words across line breaks caused by PDF/OCR margins.
    Preserves legitimate compound words (e.g., 'high-quality' remains intact).
    """
    if not text:
        return ""
    return LAYOUT_HYPHEN_REGEX.sub(r'\1\2', text)


def normalize_batch(batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
    """
    Ray Data batch transformation function.
    
    ===========================================================================
    ARCHITECTURAL DECISION: Hybrid C++ / C-Extension Normalization Pipeline
    ===========================================================================
    1. PyArrow RE2 C++ Kernel (pc.replace_substring_regex):
       Performs zero-copy, bulk regex sanitization across Arrow memory buffers 
       to strip control characters (\x00-\x1F) and phantom bytes (\u200B, \u00AD).
    
    2. Python stdlib C-Engine (unicodedata.normalize):
       PyArrow's C++ pc.utf8_normalize (driven by utf8proc) unfolds ligatures 
       (e.g., 'ﬁ' -> 'fi'), but fails to compose canonical combining accent 
       marks (e.g., 'e\u0301' remains decomposed as two characters instead of 
       composing into 'é'). Python's built-in unicodedata module is C-backed 
       and strictly adheres to the Unicode NFKC specification.
    
    3. Zero-Overhead Pass Synergy:
       Since PDF/OCR layout dehyphenation requires unpacking the Arrow string 
       buffer, running unicodedata.normalize inside the same iteration pass 
       incurs zero additional allocation cost while guaranteeing 100% spec 
       compliance.
    ===========================================================================
    """
    is_arrow_table = isinstance(batch, pa.Table)
    
    # 1. Standardize input column & handle null values
    text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
    text_column = pc.fill_null(text_column, "")
    
    # 2. PyArrow C++ Vectorized Pass: Strip non-printable & phantom characters
    sanitized_col = pc.replace_substring_regex(
        text_column, 
        pattern=CONTROL_AND_PHANTOM_PATTERN, 
        replacement=""
    )
    
    # 3. Python C-Engine Pass: Full Unicode NFKC Composition + Dehyphenation
    raw_strings = sanitized_col.to_pylist()
    processed_strings = [
        unicodedata.normalize("NFKC", dehyphenate_text(s)) if s else "" 
        for s in raw_strings
    ]
    
    timestamps = [datetime.datetime.now(datetime.timezone.utc).isoformat()] * len(processed_strings)

    # 4. Output Reconstruction (In-place or Table Append)
    if is_arrow_table:
        if "normalized_at" in batch.column_names:
            batch = batch.drop_columns(["normalized_at"])
        return batch.drop_columns(["text"]).append_column(
            "text", pa.array(processed_strings)
        ).append_column(
            "normalized_at", pa.array(timestamps)
        )
    else:
        batch["text"] = processed_strings
        batch["normalized_at"] = timestamps
        return batch