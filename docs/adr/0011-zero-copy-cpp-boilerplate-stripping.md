# ADR 0011: Zero-Copy C++ Boilerplate Stripping

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Cleaning DOM tags, script contents, page numbers, and confidentiality stamps across millions of documents causes heavy GIL bottlenecks if executed in Python loops.

## Decision
Execute `pyarrow.compute.replace_substring_regex` with non-capturing RE2 groups (`(?i:...)`) directly inside contiguous Arrow C++ memory.

## Consequences
* **Positive:** Enables microsecond-level vectorized text sanitization without GIL overhead or string allocation.
* **Negative:** Regex logic must adhere strictly to Google RE2 syntax limits (no lookarounds).