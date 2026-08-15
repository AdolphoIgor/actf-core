# ADR 0005: Lightweight Task Gate Execution

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Data contract schemas and record counts must be verified at subsystem boundaries to prevent downstream processing of corrupted data.

## Decision
Execute data quality contracts (`quality_gate_1.py`, `quality_gate_2.py`) as lightweight Airflow `PythonOperator` tasks prior to triggering downstream stages.

## Consequences
* **Positive:** Halts pipeline execution early upon contract failure, preserving cluster compute resources.
* **Negative:** Adds explicit validation steps to Airflow DAG topology definitions.