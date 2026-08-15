# ADR 0015: Auditability via Transformation Timestamps

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Silver lakehouse data requires complete execution tracking to verify when specific transformation passes were executed on each batch.

## Decision
Every curation script appends an ISO-8601 UTC timestamp column (`normalized_at`, `boilerplate_stripped_at`) to output tables.

## Consequences
* **Positive:** Provides transparent dataset lineage tracking and process auditability.
* **Negative:** Marginally increases dataset storage footprint by adding metadata columns.