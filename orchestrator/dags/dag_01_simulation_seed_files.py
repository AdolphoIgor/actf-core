import json
import os
import random
import re
import time
import uuid
from datetime import UTC, datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ai_ops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 0,
}

DAG_DOC_MD = """
# Unstructured Compliance File Engine (`dag_01_simulation_seed_files`)

Generates synthetic unstructured compliance documents (memos, chat logs, SEC inquiries).
To optimize object storage and prevent small-file I/O bottlenecks, all documents generated in a single run are packed into one timestamped `.txt` file separated by boundary markers.

### Key Technical Features
* **Resilient Multi-Model Pool:** Auto-rotates Gemini models upon hitting Free-Tier quotas.
* **LSH-Aware Token Mutation:** Mutates entities and timestamps in raw text to defeat downstream Ray Data deduplication.
* **Aggregated I/O Persistence:** Writes a single packed text file per execution to `/opt/airflow/data/unstructured_samples` (e.g., `2026-06-03-14-30-00.txt`).
"""

FREE_TIER_MODEL_POOL = [
    {"name": "gemini-3.5-flash-lite", "sleep": 4.5},
    {"name": "gemini-3.1-flash-lite", "sleep": 4.5},
    {"name": "gemini-3.6-flash", "sleep": 12.5},
    {"name": "gemini-3.5-flash", "sleep": 12.5},
    {"name": "gemini-3-flash", "sleep": 12.5},
]

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "JPM",
    "GS",
    "BAC",
    "NVDA",
    "TSLA",
    "WFC",
    "C",
    "MS",
    "BLK",
]
LOG_PREFIXES = [
    "SYSTEM_AUDIT_TRACE",
    "SEC_MONITOR_ALERT",
    "COMPLIANCE_ENGINE_v4",
    "FINRA_AUTO_FLAG",
    "GITHUB_REPO_SYNC",
    "DB_QUERY_LOG",
]


class ResilientGeminiPool:
    def __init__(self, models_config):
        self.models = models_config
        self.current_index = 0

    def get_active_model(self):
        if self.current_index >= len(self.models):
            raise RuntimeError("CRITICAL: All models in the free tier pool have been exhausted.")
        return self.models[self.current_index]

    def rotate_to_next_model(self, error_reason):
        failed_model = self.models[self.current_index]["name"]
        print(f"[MODEL EXHAUSTED] Model '{failed_model}' failed with error: {error_reason}")
        self.current_index += 1
        if self.current_index < len(self.models):
            new_model = self.models[self.current_index]["name"]
            print(f"[FAILOVER SUCCESS] Rotated active LLM engine to: '{new_model}'")
        else:
            print("[FAILOVER FAILED] No remaining models available in pool.")


def apply_high_entropy_mutation(document_text: str) -> str:
    """
    Mutates text entities to ensure Jaccard similarity drops below LSH deduplication thresholds.
    Intelligently handles source code so as not to break AST parsers.
    """
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "TSLA", "NVDA"]:
        if ticker in document_text:
            document_text = document_text.replace(ticker, random.choice(TICKERS))

    document_text = re.sub(r"Trader-\d+", f"Trader-{random.randint(1000, 9999)}", document_text)
    document_text = re.sub(
        r"Operator ID:\s*\d+", f"Operator ID: {random.randint(1000, 9999)}", document_text
    )
    document_text = re.sub(
        r"Account #\d+", f"Account #{random.randint(100000, 999999)}", document_text
    )
    document_text = re.sub(r"ORD-\d+", f"ORD-{random.randint(100000, 999999)}", document_text)
    document_text = re.sub(
        r"0x[0-9A-Fa-f]{6,8}", f"0x{random.randint(0x100000, 0xFFFFFF):x}", document_text
    )

    ip_replacement = f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    document_text = re.sub(r" \d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3} ", ip_replacement, document_text)

    prefix = random.choice(LOG_PREFIXES)
    trace_id = str(uuid.uuid4())[:8].upper()
    timestamp_tag = (
        f"[{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S')}.{random.randint(100, 999)}Z]"
    )

    # AST-SAFE HEADER INJECTION
    if "def " in document_text or "import " in document_text:
        return f"# [{prefix}] {timestamp_tag} TRACE_ID:{trace_id}\n{document_text}"
    elif "SELECT " in document_text.upper() or "INSERT " in document_text.upper():
        return f"/* [{prefix}] {timestamp_tag} TRACE_ID:{trace_id} */\n{document_text}"
    elif "{" in document_text and "}" in document_text:
        return f"/* [{prefix}] {timestamp_tag} TRACE_ID:{trace_id} */\n{document_text}"
    else:
        return f"[{prefix}]\n{timestamp_tag} TRACE_ID:{trace_id}\n\n{document_text}"


def generate_unstructured_seed(client, seed_count, anomaly_focus, pool):
    from google.genai import types

    system_instruction = (
        "You are a Senior Wall Street Regulatory Compliance Auditor and Data Engine. "
        "Your sole output must be a clean, valid JSON array containing plain strings. "
        "Each string must be a highly realistic, unstructured compliance document (e.g., internal memo, email chain, chat transcript, or code snippet). "
        "Do not include markdown wrapping, backticks (such as ```json), or introduction/conclusion prose. "
        "Output pure, raw JSON array text only."
    )

    chunk_size = 50
    seed_records = []
    generated = 0

    while generated < seed_count:
        current_chunk_size = min(chunk_size, seed_count - generated)
        active_config = pool.get_active_model()
        target_model = active_config["name"]
        sleep_delay = active_config["sleep"]

        prompt = f"""
        Generate a JSON array containing exactly {current_chunk_size} unique, highly realistic unstructured strings.
        
        TARGET DATA DISTRIBUTION (Designed to stress-test Ray Data ML Pipelines):

        1. 15% ULTRA-HIGH-QUALITY REGULATORY PROSE:
           - Complex, multi-layered violations containing deep step-by-step compliance reasoning, explicit order book timestamps, and specific SEC/FINRA rule breakdowns.

        2. 10% SYNTHETIC NEAR-DUPLICATE CLUSTERS (To Test LSH Deduplication):
           - Pairs of near-identical automated alert logs triggered seconds apart.

        3. 30% STANDARD REGTECH ANOMALIES:
           - Valid compliance communications matching: {anomaly_focus}.

        4. 15% WEB-SCRAPING & OCR NOISE (For Track A Rejection Simulation):
           - Strings MUST contain garbage web-scraping artifacts: malformed HTML/CSS snippets, minified JavaScript mixed with broken English, heavy punctuation, and OCR errors. 
           - Example: `<div class="error">Sys err 0x99; var x={{{{}}}}; </div> User complains about login.`

        5. 20% VALID SOURCE CODE & SQL (For Track B AST Parsers):
           - Strings MUST contain SYNTACTICALLY FLAWLESS, production-grade source code.
           - Examples: Python algorithmic trading scripts, complex Postgres SQL fraud-detection queries, or valid JSON configurations. Do NOT wrap in markdown. It must parse cleanly in Tree-Sitter or SQLGlot.

        6. 10% PII NOISE (To Test Presidio NER Redaction):
           - Financial events contaminated with unescaped newlines, operator phone numbers (+1-555-0199), email signatures, and raw hex memory dumps.

        ANOMALY FOCUS:
        Ensure the core topics heavily revolve around: {anomaly_focus}.
        """

        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.88,
                ),
            )

            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()

            chunk = json.loads(raw_text)
            seed_records.extend(chunk)
            generated += len(chunk)
            print(
                f"Fetched seed chunk ({len(chunk)} documents) via '{target_model}'. Progress: {generated}/{seed_count}"
            )

            time.sleep(sleep_delay)

        except Exception as e:
            err_str = str(e)
            if any(
                token in err_str.lower()
                for token in ["429", "resource_exhausted", "not_found", "quota", "limit"]
            ):
                pool.rotate_to_next_model(err_str)
            else:
                print(
                    f"Transient error on model '{target_model}': {err_str}. Retrying in 5 seconds..."
                )
                time.sleep(5.0)

    return seed_records


def bootstrap_and_write_file(seed_records: list, target_count: int, file_path: str):
    print(
        f"Bootstrapping {len(seed_records)} seed documents up to {target_count} records into a single file..."
    )
    seed_len = len(seed_records)
    records_written = 0

    document_delimiter = "\n\n--- DOCUMENT BOUNDARY ---\n\n"

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            for i in range(target_count):
                base_text = str(seed_records[i % seed_len])
                mutated_text = apply_high_entropy_mutation(base_text)

                f.write(mutated_text)
                f.write(document_delimiter)

                records_written += 1
                if records_written % 1000 == 0:
                    print(
                        f"File System Sync Checkpoint: {records_written}/{target_count} records appended."
                    )

    except Exception as e:
        print(f"Error writing to file {file_path}: {str(e)}")

    print(
        f"High-entropy bootstrap complete. Successfully appended {records_written} documents to {os.path.basename(file_path)}."
    )


def execute_live_file_generation(**context):
    from google import genai

    load_type = context["params"]["load_type"]

    if load_type == "initial":
        target_count = 5000
        seed_target = 1000
    else:
        target_count = 1000
        seed_target = 200

    print(
        f"Initializing Synthetic Unstructured Generation. Target: {target_count} aggregated documents via mode: {load_type}"
    )

    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError(
            "CRITICAL: GEMINI_API_KEY environment variable is missing from Airflow runtime."
        )

    client = genai.Client()
    pool = ResilientGeminiPool(FREE_TIER_MODEL_POOL)

    output_dir = "/opt/airflow/data/unstructured_samples"
    os.makedirs(output_dir, exist_ok=True)

    timestamp_str = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S")
    target_file_path = os.path.join(output_dir, f"{timestamp_str}.txt")
    print(f"Targeting single output file for this run: {target_file_path}")

    foci = [
        "Insider Trading / MNPI leakage communications",
        "Spoofing the order book / Market Wash Trading chat logs",
        "AML Layering via shell routing inquiries",
        "High-frequency quote stuffing system alerts",
    ]

    inserted_total = 0

    while inserted_total < target_count:
        remaining_needed = target_count - inserted_total
        current_seed_target = min(seed_target, remaining_needed)

        try:
            seed_records = generate_unstructured_seed(
                client=client,
                seed_count=current_seed_target,
                anomaly_focus=random.choice(foci),
                pool=pool,
            )
        except Exception as e:
            print(f"Terminating generation loop: {str(e)}")
            break

        bootstrap_and_write_file(seed_records, remaining_needed, target_file_path)

        inserted_total += remaining_needed

    print(
        f"File seeding run finished. Aggregated {inserted_total} documents into {os.path.basename(target_file_path)}."
    )


with DAG(
    dag_id="dag_01_simulation_seed_files",
    default_args=default_args,
    description="Generates live unstructured compliance text files using Gemini API",
    schedule=None,
    catchup=False,
    tags=["demo", "data_generator", "unstructured"],
    doc_md=DAG_DOC_MD,
    params={
        "load_type": Param(
            default="incremental",
            type="string",
            enum=["initial", "incremental"],
            description="Determines the target volume of generated unstructured documents.",
        )
    },
) as dag:
    generate_files = PythonOperator(
        task_id="execute_llm_file_hydration", python_callable=execute_live_file_generation
    )
