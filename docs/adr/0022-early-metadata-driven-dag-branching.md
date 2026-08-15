# ADR 0022: Early Metadata-Driven DAG Branching

* **Status:** Accepted (Supersedes linear flow in ADR-0018)
* **Date:** 2026-08-07
* **Deciders:** Data Platform Architecture Team

## Context
A linear pipeline processing all documents through identical heuristic, fuzzy deduplication, and quality classification steps causes compute inflation and accidentally drops technical corpora (e.g., code, JSON, math) via prose-trained classifiers.

## Decision
Split the pipeline at Step 4 (`metadata_router.py`) into two parallel processing streams:
1. **Natural Language Branch (Steps 5a–8a):** Optimized for web prose, standard MinHash, and neural quality classifiers.
2. **Specialized Technical Branch (Steps 5b–8b):** Optimized for code/math, symbol-density checks, AST shingling, and syntax validation.
Streams rejoin at Step 9 for mandatory PII masking and decontamination.

## Consequences
* **Positive:** Eliminates silent deletion of technical corpora and cuts compute costs by bypassing heavy NLP models for non-prose streams.
* **Negative:** Increases pipeline DAG complexity and requires maintaining dual step handlers for Steps 5–8.