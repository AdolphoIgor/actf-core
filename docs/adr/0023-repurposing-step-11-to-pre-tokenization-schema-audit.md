# ADR 0023: Repurposing Step 11 to Pre-Tokenization Schema Audit

* **Status:** Accepted
* **Date:** 2026-08-07
* **Deciders:** Data Platform Architecture Team

## Context
Originally, Step 11 was conceived as a post-facto domain retention whitelist. With Step 4 now handling early metadata-driven routing, Step 11's original scope became redundant.

## Decision
Repurpose Step 11 on Node 4 (`tokenization_and_packing`) as a **Pre-Tokenization Metadata & Schema Audit**. It executes pre-flight validation on surviving schema attributes, alignment flags, and domain-specific tokenization parameters (e.g., whitespace/indentation flags) immediately before Rust tokenization.

## Consequences
* **Positive:** Prevents tokenization failures caused by missing metadata or malformed indentation schemas.
* **Negative:** Adds a validation pass prior to sequence bin packing.