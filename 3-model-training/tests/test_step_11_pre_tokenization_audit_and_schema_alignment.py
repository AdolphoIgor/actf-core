import pyarrow as pa
from scripts.Sstep_11_pre_tokenization_audit_and_schema_alignment import PreTokenizationAuditor


def test_pre_tokenization_null_audit():
    """Validates Stage 1: Zero-length string records and nulls are dropped."""
    actor = PreTokenizationAuditor()

    input_table = pa.Table.from_pydict(
        {"text": ["Valid text", "", None, "More valid text"], "branch_id": [0, 0, 1, 1]}
    )

    result = actor(input_table)

    assert result.num_rows == 2
    assert result["text"].to_pylist() == ["Valid text", "More valid text"]


def test_pre_tokenization_context_chunking():
    """Validates Stage 2: Documents exceeding max_context_length are split."""
    # Set an artificially low max length to test the chunking logic
    actor = PreTokenizationAuditor(max_context_length=50)

    long_text = "This is the first paragraph.\n\nThis is the second paragraph."

    input_table = pa.Table.from_pydict({"text": [long_text], "branch_id": [0]})

    result = actor(input_table)

    # The document should be split into 2 distinct rows
    assert result.num_rows == 2
    texts = result["text"].to_pylist()
    assert "first paragraph" in texts[0]
    assert "second paragraph" in texts[1]


def test_pre_tokenization_policy_injection():
    """Validates Stage 3: preserve_whitespace policy is correctly assigned based on branch_id."""
    actor = PreTokenizationAuditor()

    input_table = pa.Table.from_pydict(
        {"text": ["General prose.", "def code_block(): return True"], "branch_id": [0, 1]}
    )

    result = actor(input_table)

    assert "preserve_whitespace" in result.column_names
    policies = result["preserve_whitespace"].to_pylist()

    # Prose (0) -> False, Code (1) -> True
    assert policies[0] is False
    assert policies[1] is True
