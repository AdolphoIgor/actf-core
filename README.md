
---

# ACTF: Automated Continuous Training Framework AI Platform

> **An Enterprise-Grade Automated Data Platform & Continuous Training (CT) Pipeline**

---

## 1. Executive Summary & Capabilities

This repository is an **architecture showcase** demonstrating how to design, containerize, and orchestrate an end-to-end **Continuous Training (CT)** pipeline for regulatory compliance and AI applications (RegTech).

It bridges the gap between raw data ingestion and distributed ML model retraining by solving real-world data platform and LLMOps engineering challenges:

* **Production-Grade Decoupling:** Separates Airflow runtime components (`webserver`, `scheduler`, and an idempotent `init` container) to guarantee state persistence across container recreations.
* **Hybrid Distributed Compute:** Combines **Apache Spark** for high-throughput relational ETL with a **Ray Cluster** (configured with Plasma shared memory) for unstructured data preparation, fuzzy LSH deduplication, and CPU/GPU ML processing.
* **Dual-Target Execution (CPU Fallback & GPU Profiles):** Native support for ultra-lightweight models (`qwen2.5-0.5b-instruct`) on standard CPU hardware with optional CUDA acceleration via Docker Compose profiles.
* **Full-Lifecycle LLMOps Stack:** Integrates **DVC** for data versioning, **Axolotl** for fine-tuning, **Weights & Biases** for experiment tracking, **MLflow** for model registry management, and **Promptfoo** for automated quality evaluation gates.
* **Isolated Quality Gates:** Enforces storage contracts (`quality_gate_1.py`, `quality_gate_2.py`) directly inside the orchestration layer (`orchestrator/dags/scripts/`), preventing corrupt data assets from polluting downstream stages.

---

## 2. System Architecture & Tech Stack

### High-Level Architecture Flow

```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  PostgreSQL 16  │ ───>  │  Apache Spark   │ ───>  │  MinIO (S3 API) │
│   (Source DB)   │       │   (Raw ETL)     │       │ (Bronze/Silver) │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Dataset Control │ <───> │   Ray Cluster   │ <───> │   DVC Tracking  │
│ (Ray Data/DVC)  │       │(Prep, LSH & ML) │       │ (MinIO / S3)    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Fine-Tuning Exec│ ───>  │ Experiment Trk  │ ───>  │ Auto Gatekeeper │ ───> ┌──────────────────┐
│(Axolotl/PyTorch)│       │ (W&B / MLflow)  │       │(Promptfoo/Eval) │      │ MLflow Registry  │
└─────────────────┘       └─────────────────┘       └─────────────────┘      └──────────────────┘
         ▲                         ▲                         ▲
         │                         │                         │
┌────────────────────────────────────────────────────────────────────────┐
│                      Apache Airflow 2.9.2 (DAGs)                       │
│              (Orchestrator, Scheduler & Webserver)                     │
└────────────────────────────────────────────────────────────────────────┘

```

### Technology Matrix

| Layer | Technology | Operational Role |
| --- | --- | --- |
| **Orchestration** | **Apache Airflow 2.9.2** | Runs in `LocalExecutor` mode. Decoupled into `airflow-init`, `airflow-webserver`, and `airflow-scheduler`. |
| **Metadata DB** | **PostgreSQL 16** | Shared relational store for Airflow state persistence and source transaction simulation. |
| **Data Lakehouse** | **MinIO** | Local S3-compatible object storage configured with Medallion buckets (`bronze`, `silver`, `gold`). |
| **Dataset Control** | **DVC (Data Version Control)** | Tracks raw and processed dataset partitions alongside Git commits for reproducible training runs. |
| **Structured Compute** | **Apache Spark 3.5** | Master-Worker cluster for high-volume relational data extraction and Bronze layer loading. |
| **Unstructured Compute** | **Ray Cluster 2.40** | Ray Head + CPU/GPU Workers (Python 3.11) with 8GB shared memory (`/dev/shm`) for distributed data prep and ML. |
| **Fine-Tuning Engine** | **Axolotl / PyTorch** | Orchestrates instruction tuning for `qwen2.5-0.5b-instruct` (CPU LoRA locally or multi-GPU FSDP in production). |
| **Experiment Tracking** | **Weights & Biases / MLflow** | Logs training loss curves, hyperparameter sweeps, system telemetry, and model registry artifacts. |
| **Quality Gatekeeper** | **PyArrow / Promptfoo** | Enforces storage contracts at Bronze/Silver layers and automated LLM assertion checks before model promotion. |
| **Package Engine** | **Astral `uv**` | Fast Python dependency installer used across custom container image builds (`pyproject.toml` based). |

---

### Repository Structure

```text
actf-core/
├── .devcontainer/           # VS Code DevContainer workspace setup
├── .vscode/                 # Debugger launch configurations (debugpy attach)
├── 1-raw-data-ingest/       # Spark ETL scripts & PySpark extraction jobs
├── 2-data-prep/             # Ray Data scripts (normalization, LSH deduplication)
├── 3-model-training/        # Axolotl training configs, LoRA setup & W&B integration
├── 4-model-eval/            # Promptfoo eval suites & MLflow Model Registry promotion
├── data/                    # Local storage volume emulation (Bronze/Silver/Gold)
├── docs/                    # Architecture Decision Records (ADRs) & guides
├── orchestrator/            # Airflow platform setup
│   ├── dags/                # Workflow DAG definitions (dag_00_... to dag_03_...)
│   │   └── scripts/         # Standalone quality gates (quality_gate_1.py, quality_gate_2.py)
│   └── logs/                # Mounted execution logs (UID 50000 permissions)
├── tests/                   # Suite for DAG integrity and platform integration
├── .env                     # Local environment secrets (git-ignored)
├── compose.yaml             # Master multi-container Docker Compose setup
├── Dockerfile.ray-cpu       # Custom Ray cluster image definition (Python 3.11 CPU)
├── Dockerfile.ray-gpu       # Custom Ray cluster image definition (Python 3.11 CUDA)
├── pyproject.toml           # Root dependencies & workspace manifests
└── uv.lock                  # Deterministic dependency lockfile

```

---

## 3. Hardware Requirements & Deployment Modes

The platform supports both CPU-bound local simulation and CUDA-accelerated cloud deployment using **Docker Compose Profiles**.

### Running Local CPU vs. Cloud GPU

* **Local Environment (e.g., CPU-Only Laptop / Workstation):**
Spins up core infrastructure, Spark, Ray CPU workers, Airflow, and Postgres without requesting GPU drivers.
```bash
docker compose up -d

```


* **Cloud GPU Server (NVIDIA CUDA Hardware Available):**
Uses the `gpu` profile to instantiate the `ray-worker-gpu` worker alongside the core stack.
```bash
docker compose --profile gpu up -d

```



---

### Resource & Sizing Guidelines

| Service Component | Minimum CPU Allocation | Baseline RAM Footprint | Operational Role |
| --- | --- | --- | --- |
| **Airflow (`webserver`, `scheduler`, `init`)** | 1.5 Cores | ~2.5 GB | DAG scheduling, queuing, web interface |
| **PostgreSQL 16** | 0.5 Cores | ~500 MB | Airflow state & source DB simulation |
| **MinIO (S3 API & Console)** | 0.5 Cores | ~1.0 GB | S3 Medallion storage buckets |
| **Apache Spark (Master + Worker)** | 1.0 Core | ~2.5 GB | Relational Spark cluster standby |
| **Ray Cluster (Head + CPU Worker)** | 1.5 Cores | ~8.0 GB (`/dev/shm`) | Shared memory plasma store standby |
| **Total Core Stack Baseline** | **~5.0 Cores** | **~14.5 GB RAM** | **Always-On Infrastructure** |

* **Development Minimum:** 4–8 Cores, 24–32 GB System RAM (Executes DAGs sequentially).
* **Production Recommended:** 8+ Cores, 32+ GB System RAM + Dedicated NVIDIA GPU (Supports concurrent DAG runs).

---

## 4. Quickstart Guide

### 1. Environment Configuration

Create a `.env` file in the project root (**do not commit credentials to source control**):

```bash
# Database Credentials
POSTGRES_USER=admin
POSTGRES_PASSWORD=secure_password_123
POSTGRES_DB=enterprise_db

# Airflow Admin Setup
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=admin_password

# External Telemetry & APIs
GEMINI_API_KEY=your_gemini_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
MLFLOW_TRACKING_URI=http://localhost:5000

# Local Model Target
LOCAL_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct

```

### 2. Log Permissions Setup

Airflow runs under unprivileged container user `airflow` (UID `50000`). Grant write permissions to the host log directory:

```bash
mkdir -p orchestrator/logs
sudo chown -R 50000:0 orchestrator/logs

```

### 3. Launch the Stack

```bash
# Build and launch CPU core services
docker compose up -d --build

# Optional: Launch with GPU worker enabled
docker compose --profile gpu up -d --build

```

---

## 5. Web Console & Port Mapping Reference

Access dashboards once containers report healthy:

| Service | Web UI Endpoint | Credentials / Details |
| --- | --- | --- |
| **Airflow Webserver** | `http://localhost:8081` | Configured via `.env` |
| **MinIO Console** | `http://localhost:9001` | `minioadmin` / `minioadmin` |
| **MinIO S3 API** | `http://localhost:9000` | AWS S3 SDK / Boto3 Endpoint |
| **Spark Master UI** | `http://localhost:8080` | Spark Cluster Status & Jobs |
| **Ray Dashboard** | `http://localhost:8265` | Ray Resources & Active Actors |
| **MLflow UI** | `http://localhost:5000` | Experiment Runs & Model Registry |
| **PostgreSQL DB** | `localhost:5432` | Configured via `.env` |

---

## 6. Pipeline DAG Lifecycle

The Continuous Training pipeline consists of six sequential DAGs:

1. **`dag_00_simulation_seed_postgres.py`** (`0_simulation_seed_postgres`): Generates synthetic relational data using Gemini API fallback pools and inserts it into PostgreSQL.
2. **`dag_01_simulation_seed_files.py`** (`1_simulation_seed_files`): Generates unstructured text documents with boundary delimiters for file landing zones.
3. **`dag_02_ingest_source_to_bronze.py`** (`2_ingest_source_to_bronze`): Triggers PySpark relational extraction and Ray Data document ingestion into MinIO Bronze Parquet storage, ending with **Quality Gate 1**.
4. **`dag_03_prep_bronze_to_silver.py`** (`3_prep_bronze_to_silver`): Dispatches Ray Data jobs for Unicode normalization and text cleaning, ending with **Quality Gate 2**.
5. **`dag_04_train_model.py`** (`4_train_model`): Executes fine-tuning via Ray/Axolotl, streaming loss metrics directly to Weights & Biases.
6. **`dag_05_model_eval.py`** (`5_model_eval`): Runs Promptfoo evaluation benchmark suites and promotes passing candidates to the MLflow Model Registry.

---

## 7. Quality Control, Testing & Debugging

### The 4-Layer Testing Strategy

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. DAG Integrity & Topology Tests (Fast CI/CD Signal)                   │
│    - Verifies DAG imports, zero syntax errors, and valid DAG structures.│
├─────────────────────────────────────────────────────────────────────────┤
│ 2. Transformation Unit Tests (Logic Validation)                         │
│    - Validates regexes, PySpark functions, and PyArrow logic via pytest.│
├─────────────────────────────────────────────────────────────────────────┤
│ 3. In-Pipeline Quality Gates (Runtime Storage Contracts)                │
│    - Gates 1 and 2 check non-zero file sizes and schemas in Airflow.   │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. Automated Model Gatekeeper (Evaluation Suite)                        │
│    - Promptfoo + MLflow validation before registering model checkpoints.│
└─────────────────────────────────────────────────────────────────────────┘

```

#### Executing Unit Tests

Run tests inside the orchestrator environment:

```bash
# Run DAG integrity and Quality Gate unit tests
docker exec -it actf-core-airflow-webserver pytest /opt/airflow/tests/

```

---

### Developer Debugging Playbook

#### 1. Instant CLI Task Execution (`airflow tasks test`)

Bypass scheduler loops to test task logic directly in your terminal:

```bash
docker exec -it actf-core-airflow-webserver airflow tasks test 2_ingest_source_to_bronze spark_parallel_bronze_extraction 2026-01-01

```

#### 2. Visual Remote Debugging via VS Code (`debugpy`)

You can use visual breakpoints directly from VS Code on your host machine:

1. **Expose debug port `5678**` in `compose.yaml` under `airflow-webserver`.
2. **Add configuration** to `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Attach to Airflow Container",
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


3. **Execute the task with the `debugpy` listener**:
```bash
docker exec -it actf-core-airflow-webserver python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m airflow tasks test 2_ingest_source_to_bronze data_quality_gate_1_bronze_check 2026-01-01

```


4. Press **F5** in VS Code to attach the debugger to the live container process.