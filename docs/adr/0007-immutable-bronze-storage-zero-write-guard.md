# ADR 0007: Immutable Bronze Storage (Zero-Write Guard)

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Raw data in the Bronze lakehouse tier must remain untampered and immutable for auditability and pipeline reproducibility.

## Decision
Enforce strict read-only execution constraints during raw data extraction and validation passes.

## Consequences
* **Positive:** Guarantees Bronze data reproducibility and audit integrity across re-runs.
* **Negative:** Requires downstream steps to write transformed outputs to new locations rather than mutating in place.