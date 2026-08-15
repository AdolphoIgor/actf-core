# ADR 0002: Deterministic Builds via UV Workspaces

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Multi-subsystem Python projects require deterministic, fast dependency resolution across separate service images without duplicating dependency definitions.

## Decision
Utilize `uv` inside `Dockerfile.ray-cpu` mapped to root `pyproject.toml` workspace members (`members = ["1-raw-data-ingest", "2-data-prep"]`) for build resolution.

## Consequences
* **Positive:** Fast, lockfile-enforced dependency resolution from a single workspace root.
* **Negative:** Requires developers to maintain workspace metadata in the root `pyproject.toml`.