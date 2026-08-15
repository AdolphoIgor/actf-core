# ADR 0014: Ephemeral Cache Directory Standard

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Executing `pytest` inside containers creates `.pytest_cache` directories on host-mounted volumes, causing file permission lockouts and cache pollution.

## Decision
Force `-o cache_dir=/tmp/.pytest_cache` across all container test invocations.

## Consequences
* **Positive:** Keeps host-mounted workspaces clean and eliminates file permission conflicts.
* **Negative:** Test cache does not persist between container restarts.