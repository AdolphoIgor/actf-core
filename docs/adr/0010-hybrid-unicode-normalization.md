# ADR 0010: Hybrid Unicode Normalization

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
PyArrow's C++ kernel `pc.utf8_normalize(form="NFKC")` (driven by `utf8proc`) fails to perform canonical composition on combining accent marks (e.g., leaving `e\u0301` uncomposed).

## Decision
Adopt a hybrid approach: use PyArrow C++ RE2 kernels for zero-copy control byte stripping (`\x00-\x1F`), combined with Python's C-backed `unicodedata.normalize("NFKC")` for canonical composition.

## Consequences
* **Positive:** Guarantees 100% spec-compliant Unicode composition without C++ binding errors.
* **Negative:** Requires converting Arrow memory blocks to Python lists for the normalization pass.