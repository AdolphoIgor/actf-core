# ADR 0008: CLI Argument & Watermark Standard

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Ingestion tasks require uniform command-line interface execution and timestamp metadata tracking across diverse data sources.

## Decision
Standardize CLI argument passing across ingestion workers and inject ISO-8601 UTC ingestion timestamps into all Bronze output schemas.

## Consequences
* **Positive:** Provides consistent interface invocations and transparent temporal lineage tracking.
* **Negative:** Mandates standard metadata schema compliance across all ingest worker implementations.