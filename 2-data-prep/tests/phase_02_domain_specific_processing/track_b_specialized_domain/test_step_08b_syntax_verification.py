import pytest
import pyarrow as pa
from scripts.phase_02_domain_specific_processing.track_b_specialized_domain.step_08b_syntax_verification import SyntaxVerifier


def test_syntax_valid_source_file():
    """Validates Stage 1: Fully valid source files with 0 ERROR nodes are retained."""
    actor = SyntaxVerifier()
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_valid"],
        "text": [
            "def calculate_trajectory(velocity, time):\n"
            "    return velocity * time\n"
        ]
    })
    
    result = actor(input_table)
    assert result.num_rows == 1
    assert "syntax_verified_at" in result.column_names


def test_syntax_corrupted_code_block():
    """Validates Stage 1: Presence of ERROR nodes in AST causes the file to be pruned."""
    actor = SyntaxVerifier()
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_corrupted"],
        # Missing colon in function definition
        "text": ["def broken_syntax()\n    return True\n"]
    })
    
    result = actor(input_table)
    assert result.num_rows == 0


def test_syntax_truncated_eof():
    """Validates Stage 2: Files ending in dangling binary operators are immediately dropped."""
    actor = SyntaxVerifier()
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_truncated"],
        # Cut off mid-statement at EOF
        "text": ["def incomplete_function():\n    x = "]
    })
    
    result = actor(input_table)
    assert result.num_rows == 0


def test_syntax_dialect_mismatch_fallback():
    """Validates Stage 3: Dialect fallback engine saves files that fail the primary parser."""
    actor = SyntaxVerifier()
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_sql"],
        # Valid SQL, but will fail the primary Python Tree-Sitter parser
        "text": ["SELECT user_id, count(*) FROM sessions GROUP BY user_id;"]
    })
    
    result = actor(input_table)
    
    # Assuming SQLGLOT_AVAILABLE is True, it should escalate and pass
    assert result.num_rows == 1