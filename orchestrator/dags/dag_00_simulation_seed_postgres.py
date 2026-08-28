# /opt/airflow/dags/dag_03_simulation_seed_postgres.py
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
from airflow.providers.postgres.hooks.postgres import PostgresHook

default_args = {
    "owner": "ai_ops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 0,
}

DAG_DOC_MD = """
# Synthetic Compliance Data Engine (`dag_00_simulation_seed_postgres`)

The **`dag_00_simulation_seed_postgres`** DAG hydrates PostgreSQL with unstructured regulatory compliance audit logs. The pipeline generates financial transaction narratives, corrupted logs, and domain noise designed to stress-test downstream Ray Data cleaning, LSH deduplication, and LLM fine-tuning pipelines.
"""

FREE_TIER_MODEL_POOL = [
    {"name": "gemini-3.5-flash-lite", "sleep": 4.5},
    {"name": "gemini-3.1-flash-lite", "sleep": 4.5},
    {"name": "gemini-3.6-flash", "sleep": 12.5},
    {"name": "gemini-3.5-flash", "sleep": 12.5},
    {"name": "gemini-3-flash", "sleep": 12.5},
]

# ENTROPY POOLS FOR LSH DEFEAT
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


def apply_high_entropy_mutation(narrative: str) -> str:
    """
    Mutates entities, order IDs, timestamps, and log headers in narrative text.
    Intelligently handles source code so as not to break AST parsers.
    """
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "TSLA", "NVDA"]:
        if ticker in narrative:
            # Exclude source ticker so random.choice never produces a no-op
            candidates = [t for t in TICKERS if t != ticker]
            narrative = narrative.replace(ticker, random.choice(candidates))

    narrative = re.sub(r"Trader-\d+", f"Trader-{random.randint(1000, 9999)}", narrative)
    narrative = re.sub(
        r"Operator ID:\s*\d+", f"Operator ID: {random.randint(1000, 9999)}", narrative
    )
    narrative = re.sub(r"Account #\d+", f"Account #{random.randint(100000, 999999)}", narrative)
    narrative = re.sub(r"ORD-\d+", f"ORD-{random.randint(100000, 999999)}", narrative)
    narrative = re.sub(
        r"0x[0-9A-Fa-f]{6,8}", f"0x{random.randint(0x100000, 0xFFFFFF):x}", narrative
    )

    ip_replacement = f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    narrative = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", ip_replacement, narrative)

    prefix = random.choice(LOG_PREFIXES)
    trace_id = str(uuid.uuid4())[:8].upper()
    timestamp_tag = (
        f"[{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S')}.{random.randint(100, 999)}Z]"
    )

    if "def " in narrative or "import " in narrative:
        mutated_narrative = f"# [{prefix}] {timestamp_tag} TRACE_ID:{trace_id}\n{narrative}"
    elif "SELECT " in narrative.upper() or "INSERT " in narrative.upper():
        mutated_narrative = f"/* [{prefix}] {timestamp_tag} TRACE_ID:{trace_id} */\n{narrative}"
    elif "{" in narrative and "}" in narrative:
        mutated_narrative = f"/* [{prefix}] {timestamp_tag} TRACE_ID:{trace_id} */\n{narrative}"
    else:
        mutated_narrative = f"[{prefix}] {timestamp_tag} TRACE_ID:{trace_id} - {narrative}"

    return mutated_narrative


def generate_free_tier_seed(client, seed_count, date_start, date_end, anomaly_focus, pool):
    from google.genai import types

    system_instruction = (
        "You are a Senior Wall Street Regulatory Compliance Auditor and Data Engine. "
        "Your sole output must be a clean, valid JSON array containing objects matching the requested schema. "
        "Do not include markdown wrapping, backticks (such as ```json), or introduction/conclusion prose. "
        "Output pure, raw JSON string text only. Ensure all code snippets are properly JSON-escaped."
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
        Generate a JSON array containing exactly {current_chunk_size} unique, highly realistic objects representing Internal Investment Banking Audit Logs and System Artifacts.

        TARGET DATA DISTRIBUTION (Designed to stress-test Ray Data ML Pipelines):

        1. 15% ULTRA-HIGH-QUALITY REGULATORY PROSE:
           - Complex, multi-layered violations containing deep step-by-step compliance reasoning, explicit order book timestamps, and specific SEC/FINRA rule breakdowns.

        2. 10% SYNTHETIC NEAR-DUPLICATE CLUSTERS (To Test LSH Deduplication):
           - Pairs of near-identical automated alert logs triggered seconds apart.

        3. 30% STANDARD REGTECH ANOMALIES:
           - Valid compliance logs matching: {anomaly_focus}.

        4. 15% WEB-SCRAPING & OCR NOISE (For Track A Rejection Simulation):
           - The 'narrative' MUST contain garbage web-scraping artifacts: malformed HTML/CSS snippets, minified JavaScript mixed with broken English, heavy punctuation, and OCR errors. 
           - Example: `<div class="error">Sys err 0x99; var x={{}}; </div> User complains about login.`

        5. 20% VALID SOURCE CODE & SQL (For Track B AST Parsers):
           - The 'narrative' MUST contain SYNTACTICALLY FLAWLESS, production-grade source code.
           - Examples: Python algorithmic trading scripts, complex Postgres SQL fraud-detection queries, or valid JSON configurations. Do NOT wrap in markdown. It must parse cleanly in Tree-Sitter or SQLGlot. Set 'source_type' to 'code_corpus' or 'github_repo' in your mind.

        6. 10% PII NOISE (To Test Presidio NER Redaction):
           - Financial events contaminated with unescaped newlines, operator phone numbers (+1-555-0199), email signatures, and raw hex memory dumps.

        SCHEMA SPECIFICATION:
        Each object in the array must have these exact keys:
        - "division": Select from ["Investment Banking", "Wealth Management", "Institutional Trading", "Global Cleared Derivatives", "Human Resources", "Facility Operations"].
        - "amount": Decimal number between 50000.00 and 25000000.00.
        - "narrative": A detailed, unstructured string. For code/SQL, place the raw code here perfectly JSON-escaped.
        - "report": A single-line minified JSON string stringified inside this field with keys: {{"framework": "...", "risk_tier": "LOW|MEDIUM|HIGH|CRITICAL", "action_required": "..."}}
        - "violation": Boolean (true if risk_tier is HIGH or CRITICAL).
        - "score": Integer from 1 to 10.

        CONSTRAINTS:
        - Timestamps must fall strictly between {date_start} and {date_end}.
        - Focus core anomaly generation on: {anomaly_focus}.
        """

        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.85,
                ),
            )

            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()

            chunk = json.loads(raw_text)
            seed_records.extend(chunk)
            generated += len(chunk)
            print(
                f"Fetched seed chunk ({len(chunk)} records) via '{target_model}'. Progress: {generated}/{seed_count}"
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


def bootstrap_records_with_high_entropy(seed_records: list, target_count: int) -> list:
    print(
        f"Bootstrapping {len(seed_records)} seed records up to target count of {target_count} rows with high-entropy mutation..."
    )
    final_payload = []
    seed_len = len(seed_records)

    for i in range(target_count):
        base_record = seed_records[i % seed_len].copy()

        base_record["audit_uuid"] = str(uuid.uuid4())

        variance = random.uniform(0.95, 1.05)
        raw_amount = float(base_record.get("amount", 100000.00))
        base_record["amount"] = round(raw_amount * variance, 2)

        base_record["narrative"] = apply_high_entropy_mutation(str(base_record["narrative"]))

        final_payload.append(base_record)

    print(
        f"High-entropy bootstrap complete. Created {len(final_payload)} unique database payloads."
    )
    return final_payload


def execute_live_database_injection(**context):
    from google import genai

    load_type = context["params"]["load_type"]

    if load_type == "initial":
        target_count = 50000
        seed_target = 1000
        date_start, date_end = "2026-06-01", "2026-06-05"
        timestamp_str = "2026-06-03 11:15:00+00"
        year = 2026
    else:
        target_count = 2000
        seed_target = 200
        now = datetime.now(UTC)
        today = now.strftime("%Y-%m-%d")
        date_start, date_end = today, today
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S%z")
        year = now.year

    print(
        f"Initializing Synthetic Generation. Target: {target_count} records via mode: {load_type}"
    )

    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError(
            "CRITICAL: GEMINI_API_KEY environment variable is missing from Airflow runtime."
        )

    client = genai.Client()
    pool = ResilientGeminiPool(FREE_TIER_MODEL_POOL)

    pg_hook = PostgresHook(postgres_conn_id="airflow_db")
    connection = pg_hook.get_conn()
    cursor = connection.cursor()

    cursor.execute("CREATE SCHEMA IF NOT EXISTS production_logs;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS production_logs.compliance_audit_ledger (
            id BIGSERIAL PRIMARY KEY,
            audit_uuid VARCHAR(50) UNIQUE,
            created_at TIMESTAMP WITH TIME ZONE,
            created_year INT,
            corporate_division VARCHAR(30),
            transaction_amount NUMERIC(15, 2),
            raw_audit_narrative TEXT,
            compliance_json_report TEXT,
            has_violation BOOLEAN,
            risk_score INT
        );
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_compliance_audit_id ON production_logs.compliance_audit_ledger(id);"
    )

    cursor.execute("""
        SELECT setval(
            pg_get_serial_sequence('production_logs.compliance_audit_ledger', 'id'),
            COALESCE((SELECT MAX(id) FROM production_logs.compliance_audit_ledger), 1),
            true
        );
    """)
    connection.commit()

    foci = [
        "Insider Trading / MNPI leakage",
        "Spoofing the order book / Market Wash Trading",
        "AML Layering via shell routing",
        "High-frequency quote stuffing anomalies",
    ]

    backup_sql_path = "/opt/airflow/data/postgres_synthetic_seed.sql"
    inserted_total = 0

    with open(backup_sql_path, "a") as sql_file:
        sql_file.write(
            f"\n\n-- BATCH RUN: {datetime.now(UTC).isoformat()} | MODE: {load_type} --\n"
        )

        while inserted_total < target_count:
            remaining_needed = target_count - inserted_total
            current_seed_target = min(seed_target, remaining_needed)

            try:
                seed_records = generate_free_tier_seed(
                    client=client,
                    seed_count=current_seed_target,
                    date_start=date_start,
                    date_end=date_end,
                    anomaly_focus=random.choice(foci),
                    pool=pool,
                )
            except Exception as e:
                print(f"Terminating generation loop: {str(e)}")
                break

            final_payload = bootstrap_records_with_high_entropy(seed_records, remaining_needed)

            for record in final_payload:
                try:
                    audit_uuid = record["audit_uuid"]
                    clean_narrative = str(record["narrative"]).replace("'", "''")

                    if isinstance(record["report"], dict):
                        clean_report = json.dumps(record["report"]).replace("'", "''")
                    else:
                        clean_report = str(record["report"]).replace("'", "''")

                    is_violation = "TRUE" if record["violation"] else "FALSE"

                    insert_statement = f"""INSERT INTO production_logs.compliance_audit_ledger (audit_uuid, created_at, created_year, corporate_division, transaction_amount, raw_audit_narrative, compliance_json_report, has_violation, risk_score) 
                    VALUES ('{audit_uuid}', '{timestamp_str}', {year}, '{record["division"]}', {record["amount"]}, '{clean_narrative}', '{clean_report}', {is_violation}, {record["score"]});\n"""

                    cursor.execute(insert_statement)
                    sql_file.write(insert_statement)
                    inserted_total += 1

                    if inserted_total % 1000 == 0:
                        connection.commit()
                        print(
                            f"Database sync checkpoint: {inserted_total}/{target_count} records written."
                        )

                except Exception as e:
                    connection.rollback()
                    print(f"Error injecting record UUID {record.get('audit_uuid')}: {str(e)}")

            connection.commit()

    cursor.close()
    connection.close()
    print(f"Seeding run finished. Successfully injected {inserted_total} total records.")


with DAG(
    dag_id="dag_00_simulation_seed_postgres",
    default_args=default_args,
    description="Generates live compliance payloads and syntactic noise into Postgres using Gemini API",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["demo", "data_generator"],
    doc_md=DAG_DOC_MD,
    params={
        "load_type": Param(
            default="incremental",
            type="string",
            enum=["initial", "incremental"],
            description="Determines historical timeline markers and delivery dates applied to rows.",
        )
    },
) as dag:
    generate_data = PythonOperator(
        task_id="execute_llm_database_hydration", python_callable=execute_live_database_injection
    )
