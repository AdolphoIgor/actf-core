from typing import Dict, Any, Union
import pyarrow as pa

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TokenizerAndPacker:
    """
    Ray Data Stateful Actor for Phase 3: Tokenization & Sequence Packing.
    
    ===========================================================================
    ARCHITECTURAL DESIGN:
    ===========================================================================
    1. Actor Pools: Initializes a persistent pool of stateful worker processes, 
       locking a single instance of the tokenizer in CPU memory to eliminate 
       re-initialization overhead[cite: 2].
    2. Integer Conversion: Streams text blocks through the Rust-backed tokenizer, 
       bypassing the Python GIL to map text into numerical `input_ids`[cite: 2].
    3. Array Packing: Concatenates incoming tokens into a continuous buffer and 
       slices them into fixed-length arrays (e.g., 4096 tokens)[cite: 2]. 
       This prevents GPUs from wasting cycles on empty padding tokens[cite: 2].
    ===========================================================================
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", max_seq_length: int = 4096):
        self.max_seq_length = max_seq_length
        self.token_buffer = []
        
        if TRANSFORMERS_AVAILABLE:
            # Enforce use_fast=True to utilize the Rust tokenizers backend
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        else:
            self.tokenizer = None

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        texts = batch["text"].to_pylist() if is_arrow_table else batch["text"]
        
        packed_input_ids = []
        packed_attention_masks = []
        
        if not self.tokenizer or not texts:
            # Return empty schema if no tokenizer is available
            empty_dict = {"input_ids": [], "attention_mask": []}
            return pa.Table.from_pydict(empty_dict) if is_arrow_table else empty_dict

        # Batch encode bypasses the GIL using Rust multi-threading
        encodings = self.tokenizer(texts, add_special_tokens=True, truncation=False)
        
        # Extend the continuous stateful buffer
        for ids in encodings["input_ids"]:
            self.token_buffer.extend(ids)
            
        # Pack into dense max_seq_length matrices
        while len(self.token_buffer) >= self.max_seq_length:
            chunk = self.token_buffer[:self.max_seq_length]
            self.token_buffer = self.token_buffer[self.max_seq_length:]
            
            packed_input_ids.append(chunk)
            # Attention mask is all 1s because the sequence is densely packed without padding
            packed_attention_masks.append([1] * self.max_seq_length)

        out_dict = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_masks
        }
        
        if is_arrow_table:
            # These optimized integer arrays are serialized directly as flat Apache Arrow 
            # tables, ready for Parquet serialization into the Gold bucket[cite: 2].
            return pa.Table.from_pydict(out_dict)
        else:
            return out_dict