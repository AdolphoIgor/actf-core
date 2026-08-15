# ADR 0020: DOM Parsing Offload to Bronze Ingestion

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Re-parsing raw HTML DOM trees (`lxml`, `trafilatura`) inside the downstream Silver text curation pipeline creates severe CPU and memory serialization overhead on streaming processing nodes.

## Decision
Enforce a strict architectural boundary where full DOM tree parsing and primary text extraction are executed during the Bronze extraction phase. Step 2 (`boilerplate_stripping.py`) receives clean, structured string records and applies zero-copy C++ RE2 regex kernels strictly for residual markup sanitization.

## Consequences
* **Positive:** Eliminates heavy tree-parsing memory allocations and GIL locks on Node 1 streaming tasks.
* **Negative:** Assumes input data arriving from Bronze has already undergone primary structural HTML extraction.