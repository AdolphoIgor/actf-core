# ADR 0003: Hardware Extra Isolation (PEP 508)

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
CPU-only processing nodes do not require heavy CUDA, PyTorch, or vLLM runtime libraries, which bloat image sizes and slow deployment.

## Decision
Leverage PEP 508 optional dependencies (`--extra cpu` / `[project.optional-dependencies]`) inside `pyproject.toml` to isolate hardware dependencies.

## Consequences
* **Positive:** Keeps CPU worker node image footprints lightweight and minimizes container startup overhead.
* **Negative:** Requires explicitly specifying `--extra` flags during dependency installation.