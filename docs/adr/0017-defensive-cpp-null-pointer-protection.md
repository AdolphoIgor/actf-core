# ADR 0017: Defensive C++ Null-Pointer Protection

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Null values in dataset text columns cause PyArrow C++ compute kernels to fail or propagate null values silently down the pipeline.

## Decision
Sanitize input columns via `pc.fill_null(text_column, "")` before invoking any PyArrow compute operations.

## Consequences
* **Positive:** Prevents kernel crashes and guarantees deterministic string handling across batch jobs.
* **Negative:** Replaces explicit SQL `NULL` states with empty strings `""`.