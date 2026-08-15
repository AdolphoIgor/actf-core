# ADR 0006: Lazy S3 Object Pagination

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Evaluating `list()` arrays over tens of thousands of Parquet partitions during quality gate checks causes memory spikes and Airflow Worker OOM crashes.

## Decision
Implement `bucket.objects.filter(Prefix=...)` generator iteration inside `quality_gate_2.py` rather than materializing list arrays.

## Consequences
* **Positive:** Guarantees constant memory consumption regardless of dataset partition scale.
* **Negative:** Requires generator-based control flow instead of simple array length operations.