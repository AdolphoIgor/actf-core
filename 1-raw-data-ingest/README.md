# 1-raw-data-ingest
                            ┌────────────────────────┐
                            │   Airflow Scheduler    │
                            └───────────┬────────────┘
                                        │ (Triggers)
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
     ┌───────────────────────────┐             ┌───────────────────────────┐
     │  Apache Spark Executor    │             │   Ray Cluster Workers     │
     │  (Structured JDBC Engine) │             │ (Unstructured Doc Parser) │
     └─────────────┬─────────────┘             └─────────────┬─────────────┘
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼ (Writes)
                      ┌───────────────────────────────────┐
                      │    MinIO S3 Compatible Storage    │
                      │  (Hive-Style Partitioned Bronze)  │
                      └─────────────────┬─────────────────┘
                                        │ (Triggers Gate 1)
                                        ▼
                      ┌───────────────────────────────────┐
                      │      Data Quality Gate 1          │
                      │   (Integrity & Volumetrics)       │
                      └───────────────────────────────────┘ 


---

# `1-raw-data-ingest` — Multi-Modal Raw Ingestion Subsystem

The **Raw Ingestion Engine** forms the foundational landing layer (Bronze Lakehouse) of the platform. It orchestrates high-throughput parallel extraction from structured relational databases (Enterprise PostgreSQL) and unstructured document storage (compliance files) directly into S3-compatible object storage (MinIO).

```
+-----------------------------------------------------------------------------------+
|                            SOURCE LAYER & ORCHESTRATION                           |
|                                                                                   |
|  [ PostgreSQL DB ] -------> (Apache Spark Operator)                               |
|                                     |                                             |
|  [ File System ] ---------> (Ray Rest Job Submitter)                              |
+-------------------------------------|---------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        BRONZE LAKEHOUSE STORAGE (MinIO / S3)                      |
|                                                                                   |
|  s3://company-ai-datalake/bronze/postgres_enterprise/compliance_audit/           |
|  s3://company-ai-datalake/bronze/local_filesystem/compliance_documents/          |
+-------------------------------------|---------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                         STREAM-AWARE QUALITY GATE 1                               |
|  - Filters metadata markers (_SUCCESS, ._*)                                       |
|  - Validates non-zero byte data payloads                                         |
|  - Handles partial stream execution (TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)    |
+-----------------------------------------------------------------------------------+

```

---

## 1. Architectural Highlights

* **Multi-Engine Parallelism:** Combines **Apache Spark** (optimized for structured tabular extracts) and **Ray Cluster Data API** (optimized for distributed file I/O and document payloads).
* **High-Watermark Incremental State:** Tracks ingestion deltas per execution cycle to ensure zero duplicate data transfers during hourly DAG runs.
* **Stream-Aware Quality Gate:** Integrates dynamic state validation that evaluates data contracts without failing on skipped stream execution paths.
* **Defensive Edge Protection:** Wraps network sockets and file handles in explicit retry/quarantine mechanisms to isolate transient S3 timeouts and filesystem corruption.

---

## 2. Component Layout

```text
1-raw-data-ingest/
├── pyproject.toml              # CPU ingestion dependencies (Boto3, PyArrow, S3FS)
├── spark/
│   └── scripts/
│       └── spark_ingest.py     # Distributed Spark extraction for Postgres tables
├── ray/
│   └── scripts/
│       └── ray_ingest.py       # Distributed Ray document scanner & MinIO writer
└── tests/
    ├── test_ray_ingest.py      # Layer 2 unit tests for CLI parsing & Zero-Write rules
    └── test_quality_gate_1.py  # Layer 2 unit tests for Gate 1 metadata & 0-byte checks

```

---

## 3. High-Watermark Incremental Sync Strategy

To prevent redundant batch transfers, `ray_ingest.py` implements a **High-Watermark State Pattern** synchronized with the Airflow Metadata Database:

```
[DAG Start] 
    │
    ▼
(Fetch 'ray_unstructured_high_watermark' from Airflow State Store)
    │
    ├─► Found Timestamp: Pass '--last_sync 2026-08-04T12:00:00'
    └─► Missing / Default: Pass '--last_sync None'
    │
    ▼
(Ray Ingestion Execution)
    │
    ├─► Scan file modification time (mtime_iso)
    └─► Filter out files where mtime_iso <= last_sync
    │
    ▼
[Parse Standard Output Log]
    │
    └─► Capture stdout marker: "RAY_MARKER_MAX_TS:<ISO_TIMESTAMP>"
    │
    ▼
(Update Airflow State Store Variable with New Watermark)

```

### String Watermark Parity

Airflow CLI invocations pass `None` values as string literals (`"None"`, `"null"`). `ray_ingest.py` standardizes inputs to prevent state drift:

```python
if args.last_sync in (None, "None", "", "null"):
    args.last_sync = None

```

---

## 4. Reconciled Zero-Write Guard Rules

Because Bronze ingestion supports both **Initial Historical Loads** and **Hourly Delta Runs**, a simple "0 files written = error" check breaks incremental pipelines. The system enforces a **Reconciled Guard Strategy**:

| Run Type | `last_sync` Value | Processed Count | Status / Outcome | Architectural Reason |
| --- | --- | --- | --- | --- |
| **Initial Load** | `None` | `0` | **`RuntimeError` (FAIL)** | Initial run cannot complete with an empty Bronze landing store. Indicates bad mount path or missing source data. |
| **Initial Load** | `None` | `> 0` | **`SUCCEEDED` (PASS)** | Initial Bronze historical baseline established. Watermark output generated. |
| **Delta Run** | `<ISO_TIMESTAMP>` | `0` | **`SUCCEEDED` (PASS)** | Valid operational state. No new files arrived since the previous watermark cycle. Watermark preserved. |
| **Delta Run** | `<ISO_TIMESTAMP>` | `> 0` | **`SUCCEEDED` (PASS)** | Delta records transferred. Watermark advanced to latest `mtime_iso`. |

---

## 5. Ingestion Data Contract (Quality Gate 1)

Quality Gate 1 (`orchestrator/dags/scripts/quality_gate_1.py`) runs downstream of ingestion tasks. It enforces data sanity before Bronze datasets can be consumed by Silver data prep steps.

### Rules Enforced

1. **Dynamic Task Inspection:** Inspects `context['dag_run']` to identify which extraction streams actually reached `TaskInstanceState.SUCCESS`. Skips validation on intentionally skipped streams.
2. **Metadata Artifact Filtering:** Filters out Spark/Hadoop committer markers (`_SUCCESS`) and hidden OS files (`._*`, `.DS_Store`) during key traversal.
3. **Corrupted File Detection:** Raises an explicit `ValueError` if any valid data file contains **0 bytes**, halting downstream processing immediately.

---

## 6. Verification & Automated Test Suite

This module is covered by a multi-tier testing framework:

### Layer 1: DAG Topology & Dependency Integrity

Validates that DAG `dag_02_ingest_source_to_bronze` parses cleanly without syntax errors and configures `TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS`.

```bash
docker exec -it actf-core-airflow-webserver pytest /opt/airflow/tests/test_dag_integrity.py -v

```

### Layer 2: Ingestion Logic & Quality Gate Unit Tests

Validates watermark parameter normalization, Zero-Write Guard conditional branches, S3 exception handling, and metadata filtering in microsecond isolation.

```bash
docker exec -it actf-core-ray-head pytest /home/ray/workspace/1-raw-data-ingest/tests/ -v

```

### Universal Monorepo Test Execution

Run all platform test suites via the root runner:

```bash
./run_tests.sh

```

---