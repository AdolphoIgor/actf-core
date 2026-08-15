import pytest
import pyarrow as pa
from scripts.phase_02_domain_specific_processing.track_b_specialized_domain.step_06b_code_specific_minhash_ast_deduplication import CodeASTDeduplicator


def test_code_ast_deduplicator():
    """Validates that logic clones with different variable names and comments are pruned."""
    actor = CodeASTDeduplicator()
    
    code_a = """
    # Apache 2.0 License Header
    def calculate_sum(a, b):
        \"\"\"Returns the sum of two numbers\"\"\"
        return a + b
    """
    
    # Exact same structure and logic, different variables, different comments
    code_b = """
    # MIT License Header
    def compute_total(x, y):
        # Adds variables
        return x + y
    """
    
    # Unique structure
    code_c = """
    def print_message(msg):
        print(msg)
    """
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["code_1", "code_2", "code_3"],
        "text": [code_a, code_b, code_c]
    })
    
    result = actor(input_table)
    
    # code_2 should be dropped because its canonical AST matches code_1
    assert result.num_rows == 2
    assert result["doc_id"].to_pylist() == ["code_1", "code_3"]
