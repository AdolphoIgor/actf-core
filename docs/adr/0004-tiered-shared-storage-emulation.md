# ADR 0004: Tiered Shared Storage Emulation

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Local development environments require testing multi-tier cloud lakehouse storage (Bronze/Silver/Gold) without cloud API costs or network latency.

## Decision
Mount the local `./data` directory across services (e.g., `/opt/airflow/data:ro` for Airflow) to emulate tiered object storage.

## Consequences
* **Positive:** Zero-cost, high-speed local emulation of S3/MinIO bucket hierarchies.
* **Negative:** Requires disciplined path mapping across `compose.yaml` services.