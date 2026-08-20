# ADR 0025: Decoupled Just-In-Time Tokenization and ChatML Compilation

* **Status:** Accepted
* **Date:** 2026-08-19
* **Deciders:** Continuous Training & Data Platform Architecture Team

## Context

In earlier iterations, Step 11 (Pre-Tokenization Audit & Schema Alignment) and Step 12 (Tokenization & Sequence Packing) were executed inside `2-data-prep` as part of the data preparation DAG (`dag_03_prep_bronze_to_silver.py`).

This coupled the data pipeline directly to model-specific dependencies (such as fixed tokenizer vocabularies, context length boundaries, and ChatML/Jinja template schemas). Generating static Gold tensor shards inside `2-data-prep` forced the upstream data pipeline to re-run expensive CPU curation tasks whenever the downstream training job switched base model architectures (e.g., transitioning from `Qwen2.5-0.5B` to `Llama-3.2-1B`). Furthermore, storing static binary tensor shards in object storage created unnecessary data duplication and storage bloat across experiments.

## Decision

Relocate Step 11 (`step_11_pre_tokenization_audit_and_schema_alignment.py`) and Step 12 (`step_12_tokenization_and_sequence_packing.py`) from `2-data-prep` to `3-model-training`:

1. **Model-Agnostic Silver Boundary:** The data curation pipeline (`2-data-prep`) strictly terminates at the **Silver Layer** (following Step 10 Cross-Dataset Decontamination and Gate 2 Pre-Tokenization Validation). It outputs universal, human-readable, schema-standardized JSONL/Parquet datasets (`{"messages": [{"role": "...", "content": "..."}]}`).


2. **Just-In-Time (JIT) In-Memory Compilation:** Steps 11 and 12 execute dynamically at the beginning of `dag_04_model_train.py` inside the training execution container. The worker loads the model's native Jinja chat template, encodes sub-word tokens via multi-threaded Rust BPE kernels, applies assistant-only target loss masking ($-100$ on prompts), and packs sequences directly into shared memory (`/dev/shm`).


3. **Declarative Runtime Dispatch:** Model swaps and tokenizer overrides are defined purely in declarative YAML manifests (`training_config.yaml`) passed via orchestration trigger payloads (`dag_run.conf`), leaving the upstream data pipeline untouched.



## Consequences

* **Positive:** Complete architectural decoupling between data curation and parameter optimization; the Silver dataset becomes a universal, permanent System of Record reusable across any model family; enables on-the-fly model switching without re-running data extraction; eliminates object storage overhead for static tensor shards.
* **Positive:** Centralizes pre-flight tensor verification (Gate 3 Data Leakage and Gate 4 Pre-Flight Tensor validation) directly within the training container boundary before provisioning heavy compute.


* **Negative:** Tokenization and sequence packing overhead are executed at the start of each training run, requiring sufficient CPU/RAM allocation (`/dev/shm`) during training container boot.