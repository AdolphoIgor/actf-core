# ADR 0024: In-Memory Symbol Density for Snippet Disambiguation

* **Status:** Accepted
* **Date:** 2026-08-07
* **Deciders:** Data Platform Architecture Team

## Context
Web-scraped documents without explicit metadata provenance often contain embedded code blocks (`<pre><code>`). These snippets must be identified during Step 5b without unpacking string arrays into Python.

## Decision
Utilize PyArrow C++ vectorized character counting (`pc.count_substring` for `{`, `}`, `;`, `=`) in shared memory during Step 5b. If symbol density exceeds 5%, fastText microsecond code classification determines whether to route the document to the technical branch or prune it as log garbage.

## Consequences
* **Positive:** Microsecond snippet disambiguation in C++ memory without GIL or object allocation bottlenecks.
* **Negative:** Requires secondary classification passes for documents sitting near the density threshold.