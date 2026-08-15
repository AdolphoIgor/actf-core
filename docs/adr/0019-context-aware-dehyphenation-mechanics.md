# ADR 0019: Context-Aware Dehyphenation Mechanics

* **Status:** Accepted
* **Date:** 2026-08-06
* **Deciders:** Data Platform Architecture Team

## Context
PDF and OCR extraction breaks words across line splits, but naive hyphen deletion destroys legitimate compound terms like `high-quality`.

## Decision
Apply word boundary and length constraints (`(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)`) during dehyphenation.

## Consequences
* **Positive:** Correctly reassembles margin splits (`en-\nterprise` $\rightarrow$ `enterprise`) while preserving compound terms.
* **Negative:** Does not reassemble single-letter prefix splits or non-alphabetic hyphen splits.