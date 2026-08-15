# ADR 0013: Multilayer Target Container Testing

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Running tests across different environments requires executing code inside its native target container to reflect production behavior accurately.

## Decision
Orchestrate system testing via `run_tests.sh` using targeted `docker exec` calls (`actf-core-airflow-webserver` for DAGs/Gates; `actf-core-ray-head` for Ray transformations).

## Consequences
* **Positive:** Ensures tests run against exact container dependencies and isolated volume mounts.
* **Negative:** Requires maintaining container target orchestration inside `run_tests.sh`.