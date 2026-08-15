import pyarrow as pa
import pyarrow.compute as pc
from typing import Dict, Any, Union, List


class PreTokenizationAuditor:
    """
    Ray Data Stateful Actor for Phase 3: Pre-Tokenization Audit & Schema Alignment.
    
    ===========================================================================
    ARCHITECTURAL DESIGN:
    ===========================================================================
    Stage 1: Null Record Audit
             Purges zero-length string records or null fields resulting from 
             aggressive upstream PII redaction or decontamination scrubbing.
    Stage 2: UTF-8 & Context Length Boundary
             Replaces invalid UTF-8 bytes with U+FFFD. Splits documents 
             exceeding 128,000 characters at paragraph boundaries.
    Stage 3: Tokenizer Policy Injection
             Injects `preserve_whitespace = True` for Track B (Technical) 
             and `False` for Track A (Prose) to guide the downstream Rust BPE.
    ===========================================================================
    """
    def __init__(self, max_context_length: int = 128000):
        self.max_context_length = max_context_length

    def _sanitize_and_chunk(self, text: str) -> List[str]:
        """
        Sanitizes invalid UTF-8 bytes and chunks documents exceeding the 
        maximum context length at logical paragraph boundaries.
        """
        # Sanitize invalid bytes with the Unicode replacement character
        clean_text = text.encode("utf-8", "replace").decode("utf-8")
        
        if len(clean_text) <= self.max_context_length:
            return [clean_text]
            
        # Segment at paragraph boundaries to prevent tokenization OOM
        paragraphs = clean_text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0
        
        for p in paragraphs:
            if current_len + len(p) > self.max_context_length and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_len = len(p)
            else:
                current_chunk.append(p)
                current_len += len(p) + 2
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        
        # Extract branch_id for policy injection, default to 0 (Prose) if missing
        if is_arrow_table and "branch_id" in batch.column_names:
            branch_column = batch["branch_id"]
        elif not is_arrow_table and "branch_id" in batch:
            branch_column = pa.array(batch["branch_id"])
        else:
            branch_column = pa.array([0] * len(text_column), type=pa.int8())
            
        # Stage 1: Zero-copy Null & Empty Record Pruning
        is_null_mask = pc.is_null(text_column)
        
        # BUG FIX: Replaced hallucinated pc.string_length with pc.utf8_length.
        # Added pc.fill_null to safely prevent Null propagation through the length evaluation.
        safe_text_column = pc.fill_null(text_column, "")
        is_empty_mask = pc.equal(pc.utf8_length(safe_text_column), 0)
        
        drop_mask = pc.or_(is_null_mask, is_empty_mask)
        
        texts = text_column.to_pylist()
        branch_ids = branch_column.to_pylist()
        drop_list = drop_mask.to_pylist()
        
        final_texts = []
        final_branch_ids = []
        policies = []

        # Stage 2 & 3: UTF-8 Sanitization, Chunking, and Policy Injection
        for i, text in enumerate(texts):
            if drop_list[i]:
                continue
                
            chunks = self._sanitize_and_chunk(text)
            branch_id = branch_ids[i]
            
            # Policy Injection: branch_id 1 (Technical) preserves whitespace
            preserve_ws = True if branch_id == 1 else False
            
            for chunk in chunks:
                if chunk.strip():
                    final_texts.append(chunk)
                    final_branch_ids.append(branch_id)
                    policies.append(preserve_ws)
        
        # Because chunking changes the total row count (1-to-many), 
        # we construct a new tabular output schema.
        res_texts = pa.array(final_texts, type=pa.string())
        res_branches = pa.array(final_branch_ids, type=pa.int8())
        res_policies = pa.array(policies, type=pa.bool_())
        
        if is_arrow_table:
            out_dict = {
                "text": res_texts,
                "branch_id": res_branches,
                "preserve_whitespace": res_policies
            }
            return pa.Table.from_pydict(out_dict)
        else:
            return {
                "text": final_texts,
                "branch_id": final_branch_ids,
                "preserve_whitespace": policies
            }