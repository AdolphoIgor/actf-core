# ADR 0016: Dual Data-Structure API Contract

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Transformation functions must seamlessly support both high-performance C++ batch processing (`pyarrow.Table`) and Ray Data dict-batch execution (`Dict[str, Any]`).

## Decision
Standardize batch transformation signatures to dynamically accept both `pyarrow.Table` and `Dict[str, Any]`, returning matching structures.

## Consequences
* **Positive:** Provides flexible execution across streaming Ray Data tasks and batch PyArrow pipelines.
* **Negative:** Requires conditional input/output branch handling inside transformation functions.