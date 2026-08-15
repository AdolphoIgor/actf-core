import re
import datetime
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

# =============================================================================
# RE2-Compliant Regex Patterns (Zero-Copy C++ Kernel Compatible)
# =============================================================================

SCRIPT_STYLE_PATTERN = r"(?is:<(?:script|style)\b[^>]*>.*?</(?:script|style)>)"
HTML_TAG_PATTERN = r"<[^>]+>"
PAGE_NUMBER_PATTERN = r"(?i:\bpage\s+\d+\s+of\s+\d+\b)"
COPYRIGHT_PATTERN = r"(?i:copyright\s+(?:\(c\)|©)?\s*\d{4}[^.\n]*\.)"
CONFIDENTIAL_PATTERN = r"(?i:\b(?:confidential - internal use only|confidential|draft copy)\b)"

COMBINED_BOILERPLATE_PATTERN = f"{SCRIPT_STYLE_PATTERN}|{HTML_TAG_PATTERN}|{PAGE_NUMBER_PATTERN}|{COPYRIGHT_PATTERN}|{CONFIDENTIAL_PATTERN}"


def strip_boilerplate_text(text: str) -> str:
    """
    Python fallback function for single-string boilerplate stripping.
    """
    if not text:
        return ""
    
    cleaned = re.sub(COMBINED_BOILERPLATE_PATTERN, "", text)
    cleaned = re.sub(r"\n(?:[ \t]*\n)+", "\n", cleaned)
    return cleaned.strip()


def strip_boilerplate_batch(batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
    """
    Ray Data batch transformation function.
    
    ===========================================================================
    ARCHITECTURAL DECISION: 100% C++ Vectorized Execution & Heuristic Profiling
    ===========================================================================
    1. Boilerplate Stripping: Executes multi-pattern regex replacement in C++.
    2. Code Disambiguation: Uses PyArrow math kernels to count structural 
       braces ({, }, ;) and compute a text-to-code density ratio. Documents 
       exceeding a 5% syntax threshold are flagged natively in C++ memory, 
       avoiding Python loop overhead entirely.
    ===========================================================================
    """
    is_arrow_table = isinstance(batch, pa.Table)
    
    text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
    text_column = pc.fill_null(text_column, "")
    
    # 1. C++ Kernel: Strip Boilerplate & Confidentiality Stamps
    cleaned_col = pc.replace_substring_regex(
        text_column,
        pattern=COMBINED_BOILERPLATE_PATTERN,
        replacement=""
    )
    
    # 2. C++ Kernel: Collapse extra whitespace/newlines and trim margins
    cleaned_col = pc.replace_substring_regex(cleaned_col, pattern=r"\n(?:[ \t]*\n)+", replacement="\n")
    cleaned_col = pc.utf8_trim_whitespace(cleaned_col)
    
    # 3. C++ Kernel: Brace Matching & Indentation Profiling (Code Density)
    text_len = pc.utf8_length(cleaned_col)
    safe_len = pc.if_else(pc.equal(text_len, 0), 1, text_len) # Prevent Division by Zero
    
    count_open = pc.count_substring(cleaned_col, "{")
    count_close = pc.count_substring(cleaned_col, "}")
    count_semi = pc.count_substring(cleaned_col, ";")
    
    total_syntax_chars = pc.add(pc.add(count_open, count_close), count_semi)
    
    code_density = pc.divide(
        pc.cast(total_syntax_chars, pa.float32()), 
        pc.cast(safe_len, pa.float32())
    )
    
    # Flag rows where >5% of the document consists of structural code characters
    is_code_heavy = pc.greater(code_density, 0.05)
    
    timestamps = [datetime.datetime.now(datetime.timezone.utc).isoformat()] * len(cleaned_col)

    if is_arrow_table:
        cols_to_drop = ["text"]
        if "boilerplate_stripped_at" in batch.column_names:
            cols_to_drop.append("boilerplate_stripped_at")
        if "is_code_heavy" in batch.column_names:
            cols_to_drop.append("is_code_heavy")
            
        return batch.drop_columns(cols_to_drop).append_column(
            "text", cleaned_col
        ).append_column(
            "is_code_heavy", is_code_heavy
        ).append_column(
            "boilerplate_stripped_at", pa.array(timestamps)
        )
    else:
        batch["text"] = cleaned_col.to_pylist()
        batch["is_code_heavy"] = is_code_heavy.to_pylist()
        batch["boilerplate_stripped_at"] = timestamps
        return batch
