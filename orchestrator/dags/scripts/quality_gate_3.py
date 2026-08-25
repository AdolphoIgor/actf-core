import hashlib
import os
import re
from collections import defaultdict

import pyarrow.dataset as ds

# Environment paths
SILVER_PATH = os.environ.get("SILVER_STORAGE_PATH", "data/silver")

# 13-Gram Canary Benchmark Catalog
CANARY_BENCHMARKS = [
    {
        "id": "mmlu_econ_001",
        "text": "Which of the following describes the condition when market supply perfectly matches consumer demand at an equilibrium price level?",
    },
    {
        "id": "gsm8k_math_001",
        "text": "A compliance fund starts with 5000000 dollars and allocates 20 percent to equity pairs and 30 percent to fixed income instruments.",
    },
    {
        "id": "humaneval_code_001",
        "text": "def verify_transaction_sequence(orders: list[int], threshold: float) -> bool: return all(o > threshold for o in orders)",
    },
]


def normalize_and_tokenize(text: str) -> list[str]:
    """Lowercases and extracts alphanumeric token sequences."""
    if not text:
        return []
    return re.sub(r"[^\w\s]", " ", str(text).lower()).split()


def extract_13grams(tokens: list[str]) -> set[str]:
    """Extracts rolling 13-grams from a tokenized list."""
    if len(tokens) < 13:
        return set()
    return {" ".join(tokens[i : i + 13]) for i in range(len(tokens) - 12)}


def run_gate_3_validation():
    print(
        f"[QUALITY GATE 3] Scanning curated dataset for benchmark contamination at: {SILVER_PATH}"
    )

    assert os.path.exists(SILVER_PATH), f"Target dataset path does not exist: {SILVER_PATH}"

    # Use Dataset API to handle partitioned Parquet directories
    dataset = ds.dataset(SILVER_PATH, format="parquet")

    # Structural Field Integrity Scan
    schema_names = dataset.schema.names
    text_col = "text" if "text" in schema_names else "raw_audit_narrative"
    id_col = "doc_id" if "doc_id" in schema_names else "audit_uuid"

    assert text_col in schema_names, (
        "Dataset missing primary text payload column ('text' or 'raw_audit_narrative')."
    )
    assert id_col in schema_names, "Dataset missing document ID column ('doc_id' or 'audit_uuid')."

    # 1. Build Reference 13-Gram Benchmark Index
    benchmark_index = defaultdict(list)
    for bench in CANARY_BENCHMARKS:
        tokens = normalize_and_tokenize(bench["text"])
        for ng in extract_13grams(tokens):
            h = hashlib.sha256(ng.encode("utf-8")).hexdigest()
            benchmark_index[h].append(bench["id"])

    # 2. Memory-Safe Benchmark Decontamination Stream
    contaminated_docs = []
    total_records = 0

    # Stream in memory-efficient batches
    for batch in dataset.to_batches(columns=[id_col, text_col]):
        id_list = batch[id_col].to_pylist()
        text_list = batch[text_col].to_pylist()

        for doc_id, text in zip(id_list, text_list):
            total_records += 1
            tokens = normalize_and_tokenize(text)

            for ng in extract_13grams(tokens):
                h = hashlib.sha256(ng.encode("utf-8")).hexdigest()
                if h in benchmark_index:
                    contaminated_docs.append((doc_id, benchmark_index[h]))

    assert len(contaminated_docs) == 0, (
        f"Contamination detected! Found {len(contaminated_docs)} documents matching canonical benchmark 13-grams: {contaminated_docs[:5]}"
    )

    print(
        f"[QUALITY GATE 3] PASSED: Scanned {total_records} records with 0 benchmark contamination hits."
    )


if __name__ == "__main__":
    run_gate_3_validation()
