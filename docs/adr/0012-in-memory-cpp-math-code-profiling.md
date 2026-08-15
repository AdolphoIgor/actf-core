# ADR 0012: In-Memory C++ Math Code Profiling

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Classifying code-heavy text without incurring the overhead of heavy ML models or slow Python lexers on Node 1 (`exact_dedup_and_heuristics`).

## Decision
Compute structural syntax density (`{`, `}`, `;`) using PyArrow vectorized arithmetic kernels (`count_substring`, `divide`, `greater`) to populate the `is_code_heavy` flag.

## Consequences
* **Positive:** Operates entirely in C++ shared memory at maximum hardware throughput.
* **Negative:** Heuristic character counting is an approximation and may misclassify edge-case structured JSON.