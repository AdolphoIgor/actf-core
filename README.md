# ACTF Core: Automated Continuous Training Framework

> **An Enterprise Distributed Continuous Training (CT) and Model Governance Platform**

---

## 1. Executive Summary and Platform Architecture

**ACTF Core** is an enterprise-grade reference architecture for automated continuous pre-training, domain adaptation, and supervised fine-tuning (SFT) of foundation models in regulated environments (RegTech, FinTech, Healthcare).

The platform bridges relational data lakehouses, distributed compute clusters, in-memory autograd engines, and automated statistical evaluation firewalls into an immutable, deterministic training loop.

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               ACTF CORE END-TO-END SYSTEM TOPOLOGY                               │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

   [ External Sources ]         [ Relational ETL ]           [ Bronze Storage ]       [ Prep & LSH ]
  ┌───────────────────┐        ┌──────────────────┐        ┌──────────────────┐     ┌────────────────┐
  │ PostgreSQL / Files│ ─────► │ Apache Spark 3.5 │ ─────► │ MinIO S3 (Bronze)│ ──► │  Ray Data 2.40 │
  └───────────────────┘        └──────────────────┘        └──────────────────┘     └────────────────┘
                                                                    │                       │
                                                      [ Quality Gate 1: Schema ]            ▼
                                                                                   ┌─────────────────┐
                                                                                   │ MinIO S3 (Silver│
                                                                                   └─────────────────┘
                                                                                            │
                                                                             [ Quality Gate 2: Clean ]
                                                                                            │
   [ MLflow Registry ]         [ Gate 5 Gatekeeper ]        [ Distributed Train ]           ▼
  ┌───────────────────┐        ┌───────────────────┐        ┌──────────────────┐   ┌─────────────────┐
  │ @champion Alias   │ ◄───── │ Statistical Test  │ ◄───── │ PyTorch / AdamW  │ ◄─│ Step 11-12 Pack │
  │ @challenger Canary│        │ McNemar / Wilson  │        │ Ephemeral Export │   └─────────────────┘
  └───────────────────┘        └───────────────────┘        └──────────────────┘            ▲
                                         ▲                           ▲                      │
                                         │                           │          [ Gate 3: Leakage ]
                                  [ Step 15-16 Eval ]       [ Gate 4: Tensor Health]
                                         │
┌────────────────────────────────────────┴─────────────────────────────────────────────────────────┐
│                   APACHE AIRFLOW 2.9+ ORCHESTRATION & STORAGE LAYER (DAGs 00-05)                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

```

### Core Capabilities

* **Decoupled Distributed Compute:** Separates high-throughput relational extraction (Apache Spark) from unstructured distributed tokenization, deduplication, and neural training (Ray Cluster + PyTorch).
* **5-Tier Quality Firewall:** Enforces storage contracts, data leakage boundaries, tensor graph stability, and statistical non-inferiority across Gates 1 through 5 before allocating compute or traffic.
* **In-Memory Tensor Pre-Flight (Gate 4):** Validates Step-0 cross-entropy calibration ($\mathcal{L}_0 \approx \ln V$), parameter finite bounds, tied embedding pointers, and autograd gradient flow prior to multi-node training.
* **Non-Blocking Ephemeral Staging (Steps 13-14):** Decouples GPU execution from network storage latency by staging full recovery states and stripped inference bundles locally to NVMe scratch with background offloading.
* **Statistical Capability Certification (Gate 5 & Step 17):** Arbitrates candidate promotions using paired McNemar tests, empirical bootstrap confidence intervals, and symmetric LLM-as-a-Judge tournaments before cutting over MLflow Model Registry aliases (`@champion`).

---

## 2. Platform Technology Matrix

| Layer | Technology | Operational Role |
| --- | --- | --- |
| **Orchestration** | Apache Airflow 2.9+ | Master workflow DAG execution with isolated task engines and state persistence. |
| **Metadata & State** | PostgreSQL 16 | Relational backend for Airflow metadata and synthetic transaction simulation. |
| **Data Lakehouse** | MinIO (S3 API) | Medallion object storage configured for `bronze`, `silver`, and `gold` partitions. |
| **Dataset Tracking** | DVC / Parquet | Cryptographic Merkle-root dataset versioning for training lineage. |
| **Structured Processing** | Apache Spark 3.5 | Parallel relational extraction, schema validation, and Bronze partition writing. |
| **Unstructured Processing** | Ray Cluster 2.40 | Distributed text normalization, MinHash LSH deduplication, and sequence packing. |
| **Optimization Engine** | PyTorch 2.2+ / FlashAttention | Mixed-precision training (BF16/FP16), AdamW parameter optimization, and norm clipping. |
| **Inference Runtime** | vLLM / Hugging Face | High-throughput GPU inference engine with automated CPU fallback. |
| **Experiment Tracking** | MLflow 2.12+ | Metric logging, parameter tracking, and Model Registry alias lifecycle management. |
| **Evaluation Suite** | SciPy / Custom Harness | Sandboxed code execution (Pass@1), MMLU log-likelihood, and statistical test engines. |
| **Package Management** | Astral `uv` | Deterministic dependency resolution across all workspace modules. |

---

## 3. Five-Tier Quality Firewall Architecture

ACTF Core treats quality enforcement as a sequence of deterministic gates that prevent compute waste and deployment regressions:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               ACTF CORE QUALITY FIREWALL MATRIX                                  │
├────────┬─────────────────────────────┬────────────────────────────────┬──────────────────────────┤
│ Gate   │ Verification Target         │ Inspection Mechanism           │ Failure Action           │
├────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ Gate 1 │ Bronze Storage Contracts    │ Schema matching & byte bounds  │ Quarantine raw payload   │
├────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ Gate 2 │ Silver Preparation Cleanliness│ Missingness, text density, LSH │ Halt prep DAG            │
├────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ Gate 3 │ Split Contamination Check   │ N-gram & embedding overlap     │ Reject dataset split     │
├────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ Gate 4 │ Pre-Flight Tensor Health    │ Step-0 loss ln(V), grad graph  │ Abort training job       │
├────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ Gate 5 │ Production Model Gatekeeper │ McNemar, Bootstrap, Wilson CI  │ Lock artifact to @archive│
└────────┴─────────────────────────────┴────────────────────────────────┴──────────────────────────┘

```

### Detailed Gate Specifications

* **Gate 1 (Bronze Storage Gate):** Asserts non-zero file sizes, strictly valid Parquet footer layouts, and required metadata columns (`source_id`, `ingested_at`, `payload`).
* **Gate 2 (Silver Parquet Cleanliness Gate):** Asserts zero null identifiers, minimum token counts, valid UTF-8 encoding, and zero exact duplicate documents.
* **Gate 3 (Split Leakage Gate):** Evaluates 13-gram Jaccard index and embedding cosine similarity between train and evaluation partitions to ensure zero data contamination.
* **Gate 4 (Pre-Flight Tensor Health Gate):**
* Verifies absence of NaNs/Infs and dead zero matrices across all parameters.
* Asserts Step-0 cross-entropy loss satisfies $\vert{}\mathcal{L}_0 - \ln(V)\vert{} \le 0.60\text{ nats}$.
* Validates tied embedding memory pointers (`tok_emb.weight.data_ptr() == lm_head.weight.data_ptr()`).
* Asserts 100% autograd gradient flow coverage across all `requires_grad=True` tensors.


* **Gate 5 (Automated Production Gatekeeper):**
* **Tier 1:** 100% AST syntax validity, $\ge 99.0\%$ EOS delimiter compliance, and zero PII leaks.
* **Tier 2:** Paired McNemar Chi-Square tests ($p \ge 0.05$) and Bootstrap $95\%$ non-inferiority margins ($\Delta \ge -0.5\%$).
* **Tier 3:** LLM Judge tournament with Wilson $95\%$ confidence interval lower bound $p_{\text{lower}} \ge 0.50$.
* **Tier 4:** Expected Calibration Error $\text{ECE} \le 0.06$ and Inter-Token Latency SLA compliance.



---

## 4. Repository Workspace Layout

```text
actf-core/
├── compose.yaml                  # Multi-container orchestration (CPU & GPU profiles)
├── Dockerfile.ray-cpu            # Ray Cluster image (Python 3.11, CPU execution)
├── Dockerfile.ray-gpu            # Ray Cluster image (Python 3.11, CUDA execution)
├── pyproject.toml                # Monorepo root configuration & linter standards
├── run_tests.sh                  # 5-Layer platform test runner
├── 1-raw-data-ingest/            # Module 1: Spark ingestion jobs & tests
├── 2-data-prep/                  # Module 2: Ray Data normalization & LSH deduplication
├── 3-model-training/             # Module 3: Distributed parameter optimization
│   ├── pyproject.toml
│   ├── README.md
│   ├── scripts/
│   │   ├── hardware_engine.py
│   │   ├── step_11_pre_tokenization_audit_and_schema_alignment.py
│   │   ├── step_12_tokenization_and_sequence_packing.py
│   │   ├── step_13_parameter_optimization_loop.py
│   │   └── step_14_ephemeral_staging_export.py
│   └── tests/
├── 4-model-eval/                 # Module 4: Gold benchmarks, LLM judge & Gatekeeper
│   ├── pyproject.toml
│   ├── README.md
│   ├── scripts/
│   │   ├── gate_05_automated_gatekeeper.py
│   │   ├── hardware_engine.py
│   │   ├── step_15_gold_benchmark_evaluation.py
│   │   ├── step_16_llm_judge_scoring.py
│   │   └── step_17_mlflow_registry_promotion.py
│   └── tests/
├── data/                         # Local storage mount (Bronze/Silver/Gold/Checkpoints)
└── orchestrator/                 # Airflow workflow orchestration
    ├── requirements.txt
    ├── dags/
    │   ├── dag_00_simulation_seed_postgres.py
    │   ├── dag_01_simulation_seed_files.py
    │   ├── dag_02_ingest_source_to_bronze.py
    │   ├── dag_03_prep_bronze_to_silver.py
    │   ├── dag_04_train_and_eval_model.py
    │   ├── dag_05_model_eval.py
    │   └── scripts/
    │       ├── quality_gate_1.py
    │       ├── quality_gate_2.py
    │       ├── quality_gate_3.py
    │       └── quality_gate_4.py
    └── tests/

```

---

## 5. Continuous Training Workflow Lifecycle (DAGs 00 to 05)

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               AIRFLOW CONTINUOUS TRAINING PIPELINE                               │
├────────────────────────┬─────────────────────────────────────────────────────────────────────────┤
│ DAG Identifier         │ Pipeline Phase and Operational Scope                                    │
├────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 0_simulation_seed_db   │ Generates synthetic relational transaction data into PostgreSQL.        │
├────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 1_simulation_seed_file │ Generates synthetic regulatory filings into raw storage landing zones.  │
├────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 2_ingest_to_bronze     │ Dispatches Spark & Ray jobs to land Parquet data; runs Quality Gate 1.  │
├────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 3_prep_to_silver       │ Dispatches Ray Data cleaning, LSH deduplication; runs Quality Gate 2.   │
├────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 4_train_and_eval_model │ Runs Gate 3 -> Steps 11-14 -> Gate 4 -> Steps 15-16 -> Gate 5 -> Step 17│
├────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 5_model_eval           │ Dedicated standalone evaluation, LLM-as-a-Judge, and Gate 5 promotion.  │
└────────────────────────┴─────────────────────────────────────────────────────────────────────────┘

```

---

## 6. Hardware Sizing and Profiles

The platform supports local development and GPU cluster deployments via Docker Compose profiles:

| Service Component | Baseline CPU | Memory Allocation | Operational Purpose |
| --- | --- | --- | --- |
| **Airflow (`webserver`, `scheduler`)** | 2.0 Cores | ~3.0 GB | DAG scheduling and execution tracking |
| **PostgreSQL 16** | 0.5 Cores | ~500 MB | Relational metadata store |
| **MinIO Object Store** | 0.5 Cores | ~1.0 GB | S3 Medallion storage layer |
| **Apache Spark (Master + Worker)** | 1.0 Core | ~2.5 GB | Relational transformation cluster |
| **Ray Cluster (Head + CPU Worker)** | 2.0 Cores | ~8.0 GB (`/dev/shm`) | Distributed tokenization & data prep |
| **Total Baseline Platform** | **~6.0 Cores** | **~15.0 GB RAM** | **Standard Local Development** |

### Execution Profiles

* **Standard CPU Mode (Local Workstation):**
```bash
docker compose up -d

```


* **CUDA GPU Accelerated Mode (Cloud Cluster):**
```bash
docker compose --profile gpu up -d

```



---

## 7. Quickstart Guide

### 1. Environment Configuration

Create a `.env` file in the project root:

```bash
# Database Credentials
POSTGRES_USER=admin
POSTGRES_PASSWORD=secure_password_123
POSTGRES_DB=enterprise_db

# Airflow Administrative User
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=admin_password

# Telemetry and Tracking
MLFLOW_TRACKING_URI=http://mlflow-server:5000
WANDB_API_KEY=your_wandb_key_optional

# Model Architecture Configuration
LOCAL_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
BASELINE_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
MAX_VRAM_USAGE_RATIO=0.85

```

### 2. File and Log Permissions

Airflow runs under unprivileged container UID `50000`:

```bash
mkdir -p orchestrator/logs data/checkpoints data/evaluation
sudo chown -R 50000:0 orchestrator/logs data

```

### 3. Build and Launch Services

```bash
# Build base infrastructure and start services
docker compose up -d --build

```

---

## 8. Web Endpoints and Service Console Reference

| Service Interface | URL Endpoint | Credentials / Role |
| --- | --- | --- |
| **Airflow Webserver** | `http://localhost:8081` | Defined in `.env` |
| **MinIO Console** | `http://localhost:9001` | `minioadmin` / `minioadmin` |
| **MinIO S3 API** | `http://localhost:9000` | S3 SDK Endpoint (`data` bucket) |
| **MLflow UI** | `http://localhost:5000` | Tracking & Model Registry (`@champion`) |
| **Spark Master UI** | `http://localhost:8080` | Spark Cluster Status |
| **Ray Dashboard** | `http://localhost:8265` | Ray Distributed Resource Telemetry |
| **PostgreSQL DB** | `localhost:5432` | Relational Storage Backend |

---

## 9. Testing and Quality Assurance

ACTF Core uses a 5-layer testing strategy spanning every module:

```bash
# Execute the complete containerized test suite
chmod +x run_tests.sh
./run_tests.sh

```

### Targeted Module Testing

```bash
# Layer 1: Airflow DAG Integrity Tests
docker exec -it actf-core-airflow-webserver pytest /opt/airflow/tests/ -v

# Layer 4: Model Training Module Tests
docker exec -it actf-core-ray-head pytest /home/ray/workspace/3-model-training/tests/ -v

# Layer 5: Model Evaluation and Gate 5 Tests
docker exec -it actf-core-ray-head pytest /home/ray/workspace/4-model-eval/tests/ -v

```

---

## 10. Developer Troubleshooting Playbook

### 1. Direct CLI Task Testing

Bypass the Airflow scheduler loop to run task logic directly in your shell:

```bash
docker exec -it actf-core-airflow-webserver airflow tasks test 4_train_and_eval_model tensor_quality_gate_4_preflight_check 2026-01-01

```

### 2. Interactive Breakpoint Debugging with `debugpy`

To attach VS Code to a running Airflow task:

1. Expose port `5678` under `airflow-webserver` in `compose.yaml`.
2. Configure `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to Airflow Container",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}/orchestrator/dags",
          "remoteRoot": "/opt/airflow/dags"
        }
      ]
    }
  ]
}

```


3. Launch the task listener inside the container:
```bash
docker exec -it actf-core-airflow-webserver python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m airflow tasks test 4_train_and_eval_model step_13_14_training_and_staging 2026-01-01

```


4. Press **F5** in VS Code to attach and step through breakpoints.

