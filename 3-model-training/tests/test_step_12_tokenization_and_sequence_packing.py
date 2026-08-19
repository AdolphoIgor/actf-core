from unittest.mock import MagicMock

import pyarrow as pa

from scripts.step_12_tokenization_and_sequence_packing import TokenizerAndPacker


def test_tokenizer_and_packer_sequence_packing():
    """Validates that variable-length documents are densely packed into fixed context windows[cite: 2]."""
    # Set a tiny max_seq_length to test the matrix chunking logic
    actor = TokenizerAndPacker(max_seq_length=10)
    actor.tokenizer = MagicMock()

    # Simulate the Rust backend returning 15 tokens across two documents
    actor.tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    }

    input_table = pa.Table.from_pydict(
        {"text": ["Document 1", "Document 2"], "preserve_whitespace": [False, True]}
    )

    result = actor(input_table)

    # 15 total tokens / 10 max_seq = 1 full row yielded, 5 tokens left in buffer
    assert result.num_rows == 1
    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names

    # Verify the first row is perfectly packed to 10 tokens
    input_ids = result["input_ids"].to_pylist()
    assert len(input_ids[0]) == 10
    assert input_ids[0] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Verify the attention mask is fully activated (no padding)
    attention_masks = result["attention_mask"].to_pylist()
    assert attention_masks[0] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Verify the remainder is safely held in the Actor's stateful buffer
    assert len(actor.token_buffer) == 5
    assert actor.token_buffer == [11, 12, 13, 14, 15]
