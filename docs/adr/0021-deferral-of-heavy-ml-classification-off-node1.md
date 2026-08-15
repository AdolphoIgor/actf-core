# ADR 0021: Deferral of Heavy ML Classification off Node 1

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Node 1 (`exact_dedup_and_heuristics`) is engineered as a high-throughput CPU/IO streaming node. Loading heavy classification binaries (`fastText`) or NLP models (`spaCy`/`NLTK`) on Node 1 degrades memory bandwidth and slows early pipeline steps.

## Decision
Explicitly defer `fastText` language identification, stop-word density scoring, and quality classification to downstream stages (Step 5, Step 7, and Step 8) running on specialized, GPU-dense worker nodes (**Node 3: `classifier_and_safety`**).

## Consequences
* **Positive:** Keeps Node 1 lightweight, maximizing streaming throughput for normalization, boilerplate stripping, and exact hashing.
* **Negative:** Requires downstream nodes to handle filtered rows that could theoretically have been dropped earlier if fastText were run on Node 1.