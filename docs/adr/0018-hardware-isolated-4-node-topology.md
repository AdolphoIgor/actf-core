# ADR 0018: Hardware-Isolated 4-Node Topology

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Pipeline steps have fundamentally different resource requirements (CPU-bound regex vs RAM-dense graph algorithms vs GPU-dense classifiers).

## Decision
Isolate processing across 4 node profiles: Node 1 (CPU/IO), Node 2 (High RAM), Node 3 (GPU Dense), Node 4 (High-Core CPU).

## Consequences
* **Positive:** Optimizes cloud infrastructure costs and prevents hardware resource contention.
* **Negative:** Requires routing data batches across specialized Ray cluster worker groups.