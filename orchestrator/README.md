# Orchestration & Quality Gate Platform (`orchestrator/README.md`)

> **Architecture Disclaimer & Local Demonstration Harness**
> * **Local Demonstration Mode (Current Setup):**
> To enable instant, zero-cost execution on a developer workstation, the platform is orchestrated locally via `docker-compose.yaml`. Standalone containers serve as persistent infrastructure components:
> * **Orchestrator:** `actf-core-airflow-webserver` & `actf-core-airflow-scheduler`
> * **Databases & Storage:** `actf-core-postgres` & `actf-core-minio` (S3 API compatibility layer)
> * **Compute Services:** `spark-master` / `spark-worker` & `ray-head` / `ray-worker`
> 
> 
> * **Production Enterprise Pattern (Cloud Blueprint):**
> The pipeline logic, state handlers, and quality assertions are completely decoupled from the local Docker environment. In a cloud production environment:
> 1. **Local Containers $\rightarrow$ Managed/Kubernetes Services:** Standalone containers are substituted with ephemeral Kubernetes workloads (`SparkKubernetesOperator`, `KubeRay`) or managed cloud clusters (GCP Dataproc, AWS EMR / Glue).
> 2. **Local MinIO $\rightarrow$ Enterprise S3 / GCS Data Lake:** The `s3a://` endpoints target enterprise bucket storage.
> 3. **Just-In-Time Provisioning:** The "Sensor-First" pattern evaluates data deltas before invoking cluster creation APIs, ensuring compute infrastructure is provisioned on-demand and terminated immediately upon task completion.
> 
> 
> 
> 

---

## Table of Contents

1. [Environment & Airflow Runtime](https://www.google.com/search?q=%231-environment--airflow-runtime)
2. [DAG Execution Catalog](https://www.google.com/search?q=%232-dag-execution-catalog)
3. [DAG Deep Dives](https://www.google.com/search?q=%233-dag-deep-dives)
* [3.1 DAG 0: Synthetic Structured Seed (`dag_00_simulation_seed_postgres`)](https://www.google.com/search?q=%2331-dag-0-synthetic-structured-seed-dag_00_simulation_seed_postgres)
* [3.2 DAG 1: Synthetic Unstructured Seed (`dag_01_simulation_seed_files`)](https://www.google.com/search?q=%2332-dag-1-synthetic-unstructured-seed-dag_01_simulation_seed_files)
* [3.3 DAG 2: Hybrid Bronze Ingestion (`dag_02_ingest_source_to_bronze`)](https://www.google.com/search?q=%2333-dag-2-hybrid-bronze-ingestion-dag_02_ingest_source_to_bronze)
* [3.4 DAG 3: Silver Data Preparation (`dag_03_simulation_seed_postgres`)](https://www.google.com/search?q=%2334-dag-3-silver-data-preparation-dag_03_simulation_seed_postgres)
* [3.5 DAG 4: Model Training & Evaluation (`4_train_and_eval_model`)](https://www.google.com/search?q=%2335-dag-4-model-training--evaluation-4_train_and_eval_model)


4. [Data Quality Gates Specification (`dags/scripts/`)](https://www.google.com/search?q=%234-data-quality-gates-specification-dagsscripts)

---

## 1. Environment & Airflow Runtime

### 1.1 API Key Configuration

To enable LLM-based synthetic data generation, a valid Gemini API key is required. The key must be defined inside the `.env` file located at the repository root:

```env
GEMINI_API_KEY=AIzaSyYourActualKeyHere

```

> **Configuration Constraint:** Quotation marks (`"` or `'`) and trailing spaces surrounding the key string must be omitted inside the `.env` file to prevent formatting errors within containerized environments.

### 1.2 Container Environment Reload

Environment variables are injected into Docker containers during service instantiation. If modifications are applied to `.env`, runtime services must be recreated:

```bash
docker compose up -d --force-recreate airflow-webserver airflow-scheduler

```

Runtime injection of the environment key inside the webserver container can be verified via:

```bash
docker exec -it actf-core-airflow-webserver python -c "import os; print(os.getenv('GEMINI_API_KEY')[:10])"

```

---

## 2. DAG Execution Catalog

The orchestration platform executes five core DAGs sequentially, interspersed with automated data quality gates.

| DAG ID | Schedule / Trigger | Input Source | Target Destination | Associated Quality Gate |
| --- | --- | --- | --- | --- |
| **`dag_00_simulation_seed_postgres`** | Manual / Ad-hoc | Gemini Free-Tier API | PostgreSQL `production_logs.compliance_audit_ledger` | N/A |
| **`dag_01_simulation_seed_files`** | Manual / Ad-hoc | Gemini Free-Tier API | Local Landing Zone (`unstructured_samples/`) | N/A |
| **`dag_02_ingest_source_to_bronze`** | Workflow Trigger | Postgres & Landing Zone | Raw Parquet / S3 (`data/bronze/`) | `quality_gate_1.py` |
| **`dag_03_simulation_seed_postgres`** | Workflow Trigger | Bronze Data Lake | Cleaned Parquet (`data/silver/`) | `quality_gate_2.py` |
| **`4_train_and_eval_model`** | Workflow Trigger | Silver Cleaned Store | Qwen 0.5B Model Checkpoint | `quality_gate_3.py` & `quality_gate_4.py` |

---

## 3. DAG Deep Dives

---

### 3.1 DAG 0: Synthetic Structured Seed (`dag_00_simulation_seed_postgres`)

#### Overview & Architecture Highlights

The **`dag_00_simulation_seed_postgres`** DAG hydrates PostgreSQL with unstructured regulatory compliance audit logs. The pipeline produces complex financial transaction narratives, corrupted logs, and domain noise designed to stress-test downstream Ray Data cleaning, LSH deduplication, and LLM fine-tuning pipelines.

#### Architectural Solutions

##### 1. Resilient Multi-Model Failover Pool

API rate limits (`RPM`) and daily request limits (`RPD`) present significant bottlenecks for automated synthetic pipelines.

* **The Solution:** A fallback pool (`ResilientGeminiPool`) manages a prioritized queue of Gemini models (`gemini-3.5-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.6-flash` $\rightarrow$ `gemini-3.5-flash`).
* **Dynamic Throttling:** Each model entry defines an enforced inter-call sleep duration (e.g., 4.5s for 15 RPM models; 12.5s for 5 RPM models).
* **Automatic Failover:** When HTTP `429` or `RESOURCE_EXHAUSTED` errors are encountered, the active model is blacklisted and execution rotates seamlessly to the next available API endpoint.

##### 2. LSH-Aware High-Entropy Token Mutation

Naive replication of LLM-generated seed records results in high token overlap. Downstream Locality-Sensitive Hashing (LSH) deduplication algorithms using Jaccard Similarity:

$$J(A, B) = \frac{\vert{}A \cap B\vert{}}{\vert{}A \cup B\vert{}}$$

would evaluate cloned records at $J(A, B) \ge 0.95$, causing 95%+ of the generated dataset to be purged during Ray Data preprocessing.

* **The Solution:** An in-memory mutator (`bootstrap_records_with_high_entropy`) applies regex-based NLP mutations to every bootstrapped row before database insertion.
* **Entropy Injection:** Tickers, order IDs (`ORD-XXXXXX`), trader IDs, IP addresses, memory hex dumps, and log headers (`[SYSTEM_AUDIT_TRACE]`, `[SEC_MONITOR_ALERT]`) are randomized per row. Jaccard similarity is driven below $0.65$.

##### 3. Native PostgreSQL Sequence Synchronization

Primary key allocation is fully delegated to PostgreSQL `BIGSERIAL`. During pre-flight checks, `setval()` is executed against `production_logs.compliance_audit_ledger_id_seq` to ensure the sequence high-watermark matches `MAX(id)` prior to ingestion, eliminating primary key constraint violations (`compliance_audit_ledger_pkey`).

#### Step-by-Step DAG Execution & Terminal Debugging

##### Incremental Execution (Default Run)

By default, an incremental run processes a target load of 2,000 records using a 200-record LLM seed pool:

```bash
docker exec -it actf-core-airflow-webserver airflow tasks test 0_simulation_seed_postgres execute_llm_database_hydration 2026-01-01

```

##### Initial Load Execution (50,000 Records)

To execute a full historical initial load (50,000 target records bootstrapped from 1,000 LLM seed records), parameter overrides are passed via the `-t` flag:

```bash
docker exec -it actf-core-airflow-webserver airflow tasks test -t '{"load_type": "initial"}' 0_simulation_seed_postgres execute_llm_database_hydration 2026-01-01

```

##### Debugging via IDE (Remote `debugpy` Listener)

To attach a step-by-step debugger (such as VS Code or PyCharm) on port `5678`:

```bash
docker exec -it actf-core-airflow-webserver python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m airflow tasks test -t '{"load_type": "initial"}' 0_simulation_seed_postgres execute_llm_database_hydration 2026-01-01

```

#### Database Verification Queries

##### Check Total Row Count and High Watermark

```bash
docker exec -it actf-core-postgres psql -U db_extraction_user -d enterprise_db -c "SELECT COUNT(*) AS total_rows, COALESCE(MAX(id), 0) AS max_id FROM production_logs.compliance_audit_ledger;"

```

##### Inspect Mutated Narrative Records

```bash
docker exec -it actf-core-postgres psql -U db_extraction_user -d enterprise_db -c "SELECT id, audit_uuid, corporate_division, transaction_amount, LEFT(raw_audit_narrative, 80) AS narrative_sample FROM production_logs.compliance_audit_ledger ORDER BY id DESC LIMIT 5;"

```

##### Verify Postgres Sequence Alignment

```bash
docker exec -it actf-core-postgres psql -U db_extraction_user -d enterprise_db -c "SELECT last_value FROM production_logs.compliance_audit_ledger_id_seq;"

```

---

### 3.2 DAG 1: Synthetic Unstructured Seed (`dag_01_simulation_seed_files`)

#### Overview & Architecture Highlights

The **`dag_01_simulation_seed_files`** DAG generates synthetic unstructured compliance documents (memos, email chains, chat transcripts, SEC inquiries) as raw text.

#### Architectural Solutions

##### 1. Small-File Problem Mitigation

Generating thousands of tiny individual `.txt` files causes metadata server exhaustion and slow directory scans in object stores (HDFS/S3).

* **The Solution:** All documents generated during a single DAG execution are packed into a single consolidated file named by execution timestamp (`yyyy-mm-dd-hh-MM-ss.txt`). Records inside the file are demarcated using explicit boundary tags (`\n\n--- DOCUMENT BOUNDARY ---\n\n`).

##### 2. Multi-Loop Append Safety

Because model failovers may occur mid-run, the bootstrapper opens the target timestamped file in append mode (`"a"`). If an API error forces model rotation, previously generated document batches are preserved without data loss or file truncation.

---

### 3.3 DAG 2: Hybrid Bronze Ingestion (`dag_02_ingest_source_to_bronze`)

#### Overview & Execution Strategy

Extracts structured transactional audit records from PostgreSQL (via PySpark JDBC) and unstructured text files from the local landing zone (via Ray Data distributed remote tasks), persisting raw immutable payloads into the Medallion Bronze data lake (`company-ai-datalake/bronze/`).

#### Architectural Solutions

##### 1. Ephemeral Compute & Sensor-First Pattern

To avoid the "Zombie Cluster" anti-pattern—where heavy Spark and Ray compute nodes run continuously alongside the Airflow webserver, wasting resources and introducing memory blast-radius risks—the DAG is designed around an **Ephemeral Compute Lifecycle**:

* **Sensor Check:** A lightweight task evaluates the high-watermark delta before requesting compute assets. If zero new records/files exist, the DAG terminates immediately via `AirflowSkipException`.
* **Dynamic Provisioning & Teardown:** In production environments, cluster operators (e.g., `SparkKubernetesOperator`, `KubeRay`) dynamically provision compute nodes sized specifically for the detected workload volume. Upon job completion, resources are destroyed via `TriggerRule.ALL_DONE` to guarantee cluster termination even if workload tasks fail.

##### 2. Exact-Once High-Watermark State Tracking

To enforce idempotency ("once done is done"):

* **Structured Path (Spark):** Filters PostgreSQL queries using `created_at > '{last_spark_compliance_ts}'`. Upon completion, the driver emits `XCOM_MARKER_MAX_TS`, which is saved to Airflow Variables.
* **Unstructured Path (Ray):** Passes `--last_sync` to the Ray Job entrypoint. Ray filters files by file modification time (`mtime > last_sync`). Airflow fetches execution logs via the Ray REST API (`/api/jobs/{job_id}/logs`), parses `RAY_MARKER_MAX_TS`, and commits the updated timestamp to Airflow Variables.

---

### 3.4 DAG 3: Silver Data Preparation (`dag_02_ingest_source_to_bronze`)

* **Objective:** Executes multi-stage NLP cleaning, PII scrubbing, domain noise filtering, and LSH deduplication across Bronze Parquet and text inputs.
* **Execution Strategy:** Distributed Ray Data transformations utilizing a 12-stage cleaning pipeline.
* **Post-Condition:** Triggers `quality_gate_2.py` to enforce quality retention thresholds prior to tokenization.

---

### 3.5 DAG 4: Model Training & Evaluation (`4_train_and_eval_model`)

* **Objective:** Tokenizes Silver compliance narratives, packs sequence tensors, and executes fine-tuning on the Qwen 0.5B model architecture.
* **Execution Strategy:** PyTorch / HuggingFace Trainer backed by Ray Train.
* **Post-Condition:** Triggers `quality_gate_3.py` (split ratio validation) and `quality_gate_4.py` (model drift barrier assertion).

---

## 4. Data Quality Gates Specification (`dags/scripts/`)

Data quality gates act as automated, blocking assertion barriers between pipeline stages. If an assertion evaluates to false, the Airflow task raises an exception, halting downstream execution.

```text
[DAG 0 & 1: Seeds] ──► [Postgres / Files] ──► [DAG 2: Ingest] ──► (Quality Gate 1) ──► [Bronze Lake]
                                                                                              │
[Model Checkpoint] ◄── (Quality Gate 4) ◄── [DAG 4: Train] ◄── (Quality Gate 3) ◄── [DAG 3: Prep] ◄── (Quality Gate 2)

```

* **`quality_gate_1.py` (Ingestion Schema & Nullability Assertion):**
Asserts that primary keys (`audit_uuid`), timestamps (`created_at`), and corporate divisions are strictly non-null upon extraction. Additionally verifies partition footprint constraints and detects zero-byte asset corruption.
* **`quality_gate_2.py` (Cleaning & PII Threshold Validation):**
Validates that regex scrubbers successfully removed PII patterns (phone numbers, email signatures) and verifies that post-deduplication volume retention falls between expected statistical bounds (70%–85%).
* **`quality_gate_3.py` (Train/Eval Split Ratio Verification):**
Verifies that dataset partitioning strictly maintains an 80/20 train-to-validation split without record leakage between splits.
* **`quality_gate_4.py` (Model Drift & Accuracy Barrier Checks):**
Executes inference assertions against fine-tuned Qwen 0.5B checkpoints, verifying that validation loss remains below strict threshold criteria to prevent deployment of degraded models.