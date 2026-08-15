# ADR 0009: Standard PEP 8 Package Layout

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
Numeric file prefixes (`1-normalization.py`) violate Python identifier syntax, breaking static imports and forcing dynamic `importlib` workarounds in test suites.

## Decision
Rename files to standard snake_case module names (`normalization.py`) and include `__init__.py` files to make scripts a proper Python package.

## Consequences
* **Positive:** Restores standard Python `import` semantics, IDE autocomplete, and static type checking.
* **Negative:** Replaces visual numerical file sorting in IDE file trees with alphabetical module ordering.