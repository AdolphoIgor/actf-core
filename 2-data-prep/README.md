# `2-data-prep` — Silver Data Curation & Pre-Tokenization Pipeline

The **Data Preparation Subsystem** handles the distributed, multi-branch transformation of raw Bronze lakehouse data into sanitized, deduplicated, quality-filtered, and tokenized Silver/Gold datasets.

```text
===================================================================================
                              SHARED TRUNK: INGESTION
===================================================================================

                            [ Raw Data Intake ]
                                     │
                                     ▼
        NODE 1: intake_and_provenance_router (Memory/IO CPU Pod)
        ├── Step 1: Normalization & Unicode Reassembly
        ├── Step 2: Boilerplate Stripping & Code Profiling
        ├── Step 3: Exact Deduplication (Bloom + RocksDB)
        └── Step 4: Metadata Inspector & Router
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
===================================     ===================================
BRANCH A: NATURAL LANGUAGE PROSE        BRANCH B: CODE & TECHNICAL DOMAINS
===================================     ===================================

NODE 2A: nl_heuristics_and_dedup        NODE 2B: code_syntax_and_dedup
(Memory-Optimized CPU Pod)              (Compute-Dense CPU Pod)
├── Step 5a: Macro-Linguistic Filters   ├── Step 5b: Syntax & Snippet Disambig.
└── Step 6a: Document MinHash LSH       └── Step 6b: Code MinHash / AST Dedup
                 │                                       │
                 ▼                                       ▼
NODE 3A: nl_quality_and_lang            NODE 3B: code_validation_and_ast
(GPU-Dense Inference Pod)               (CPU/GPU Hybrid Pod)
├── Step 7a: Classifier Quality (CQF)   ├── Step 7b: Domain Quality & Lexer
└── Step 8a: Language ID (fastText)     └── Step 8b: AST Syntax Validation
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
===================================================================================
                             SHARED TRUNK: CONVERGENCE
===================================================================================

        NODE 4: safety_and_decontamination (GPU/Compute Pod)
        ├── Step 9:  Safety Guardrails & PII Redaction
        └── Step 10: Decontamination (N-Gram Overlap Check)
                                     │
                                     ▼
        NODE 5: tokenization_and_packing (Compute-Optimized CPU Pod)
        ├── Step 11: Pre-Tokenization Audit & Schema Alignment
        └── Step 12: Tokenization & Sequence Packing (Rust BPE)

```

---

## 1. Subsystem Architecture & Node Topology

To maximize throughput and isolate hardware constraints, processing steps are mapped across specialized node profiles:

| Specialized Node | Architectural Responsibilities | Exec Engine | Hardware Profile |
| --- | --- | --- | --- |
| **Node 1: `intake_and_provenance_router**` | Unicode normalization, boilerplate stripping, exact hashing, metadata routing | Ray Data (CPU) | Memory / IO Heavy |
| **Node 2A: `nl_heuristics_and_dedup**` | Macro-linguistic prose heuristics, document-level MinHash LSH | Ray Cluster | High Memory (RAM) |
| **Node 2B: `code_syntax_and_dedup**` | Code/syntax snippet disambiguation, AST-based code deduplication | Ray Cluster | Compute Dense |
| **Node 3A: `nl_quality_and_lang**` | Classifier-based quality scoring (CQF), fastText language ID | Ray + vLLM / CPU | GPU-Dense / CPU |
| **Node 3B: `code_validation_and_ast**` | Pygments lexer check, Tree-Sitter AST syntax verification | Ray + C++ Parsers | CPU/GPU Hybrid |
| **Node 4: `safety_and_decontamination**` | Presidio PII redaction, toxicity filtering, benchmark decontamination | Ray + Transformers | GPU / Compute |
| **Node 5: `tokenization_and_packing**` | Pre-tokenization schema audit, Rust BPE tokenization, sequence packing | Ray + Rust | High Core Count |

---

## 2. Directory Structure & Dependency Isolation

This module uses **PEP 508 Optional Dependencies** inside `2-data-prep/pyproject.toml` to prevent GPU package bloat (`torch`, `vllm`) on CPU worker nodes. Source files and unit tests mirror a 1:1 phased directory structure:

```text
2-data-prep/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── __init__.py
│   ├── phase_01_shared_ingestion/
│   │   ├── __init__.py
│   │   ├── step_01_normalization.py
│   │   ├── step_02_boilerplate_stripping.py
│   │   ├── step_03_exact_deduplication.py
│   │   └── step_04_metadata_inspection_and_routing.py
│   ├── phase_02_domain_specific_processing/
│   │   ├── track_a_natural_language/
│   │   │   ├── __init__.py
│   │   │   ├── step_05a_standard_heuristics.py
│   │   │   ├── step_06a_minhash_fuzzy_deduplication.py
│   │   │   ├── step_07a_natural_language_cqf.py
│   │   │   └── step_08a_fasttext_language_id.py
│   │   └── track_b_specialized_domain/
│   │       ├── __init__.py
│   │       ├── step_05b_code_and_syntax_disambiguation.py
│   │       ├── step_06b_code_specific_minhash_ast_deduplication.py
│   │       ├── step_07b_domain_quality_check.py
│   │       └── step_08b_syntax_verification.py
│   └── phase_03_reconvergence_and_tokenization/
│       ├── __init__.py
│       ├── step_09_safety_and_pii_redaction.py
│       ├── step_10_cross_dataset_decontamination.py
│       ├── step_11_pre_tokenization_audit_and_schema_alignment.py
│       └── step_12_tokenization_and_sequence_packing.py
└── tests/
    ├── __init__.py
    ├── phase_01_shared_ingestion/
    │   ├── __init__.py
    │   ├── test_step_01_normalization.py
    │   ├── test_step_02_boilerplate_stripping.py
    │   ├── test_step_03_exact_deduplication.py
    │   └── test_step_04_metadata_inspection_and_routing.py
    ├── phase_02_domain_specific_processing/
    │   ├── track_a_natural_language/
    │   │   ├── __init__.py
    │   │   ├── test_step_05a_standard_heuristics.py
    │   │   ├── test_step_06a_minhash_fuzzy_deduplication.py
    │   │   ├── test_step_07a_natural_language_cqf.py
    │   │   └── test_step_08a_fasttext_language_id.py
    │   └── track_b_specialized_domain/
    │       ├── __init__.py
    │       ├── test_step_05b_code_and_syntax_disambiguation.py
    │       ├── test_step_06b_code_specific_minhash_ast_deduplication.py
    │       ├── test_step_07b_domain_quality_check.py
    │       └── test_step_08b_syntax_verification.py
    └── phase_03_reconvergence_and_tokenization/
        ├── __init__.py
        ├── test_step_09_safety_and_pii_redaction.py
        ├── test_step_10_cross_dataset_decontamination.py
        ├── test_step_11_pre_tokenization_audit_and_schema_alignment.py
        └── test_step_12_tokenization_and_sequence_packing.py

```

---

## 3. Pipeline Step Execution Ledger

| Step | Name | Phase & Branch | Node Assignment | Target Script |
| --- | --- | --- | --- | --- |
| **1** | **Unicode Normalization** | Phase 1 (Shared) | Node 1 (`intake_and_provenance_router`) | `scripts/phase_01_shared_ingestion/step_01_normalization.py` |
| **2** | **Boilerplate Stripping** | Phase 1 (Shared) | Node 1 (`intake_and_provenance_router`) | `scripts/phase_01_shared_ingestion/step_02_boilerplate_stripping.py` |
| **3** | **Exact Deduplication** | Phase 1 (Shared) | Node 1 (`intake_and_provenance_router`) | `scripts/phase_01_shared_ingestion/step_03_exact_deduplication.py` |
| **4** | **Metadata Inspection & Router** | Phase 1 (Shared) | Node 1 (`intake_and_provenance_router`) | `scripts/phase_01_shared_ingestion/step_04_metadata_inspection_and_routing.py` |
| **5a** | **Standard Prose Heuristics** | Phase 2 (Track A) | Node 2A (`nl_heuristics_and_dedup`) | `scripts/phase_02_domain_specific_processing/track_a_natural_language/step_05a_standard_heuristics.py` |
| **5b** | **Syntax & Snippet Disambig.** | Phase 2 (Track B) | Node 2B (`code_syntax_and_dedup`) | `scripts/phase_02_domain_specific_processing/track_b_specialized_domain/step_05b_code_and_syntax_disambiguation.py` |
| **6a** | **Document MinHash LSH** | Phase 2 (Track A) | Node 2A (`nl_heuristics_and_dedup`) | `scripts/phase_02_domain_specific_processing/track_a_natural_language/step_06a_minhash_fuzzy_deduplication.py` |
| **6b** | **Code MinHash / AST Dedup** | Phase 2 (Track B) | Node 2B (`code_syntax_and_dedup`) | `scripts/phase_02_domain_specific_processing/track_b_specialized_domain/step_06b_code_specific_minhash_ast_deduplication.py` |
| **7a** | **Prose Quality Classifier** | Phase 2 (Track A) | Node 3A (`nl_quality_and_lang`) | `scripts/phase_02_domain_specific_processing/track_a_natural_language/step_07a_natural_language_cqf.py` |
| **7b** | **Domain Quality & Lexer Check** | Phase 2 (Track B) | Node 3B (`code_validation_and_ast`) | `scripts/phase_02_domain_specific_processing/track_b_specialized_domain/step_07b_domain_quality_check.py` |
| **8a** | **FastText Language ID** | Phase 2 (Track A) | Node 3A (`nl_quality_and_lang`) | `scripts/phase_02_domain_specific_processing/track_a_natural_language/step_08a_fasttext_language_id.py` |
| **8b** | **AST Syntax Verification** | Phase 2 (Track B) | Node 3B (`code_validation_and_ast`) | `scripts/phase_02_domain_specific_processing/track_b_specialized_domain/step_08b_syntax_verification.py` |
| **9** | **Safety & PII Redaction** | Phase 3 (Shared) | Node 4 (`safety_and_decontamination`) | `scripts/phase_03_reconvergence_and_tokenization/step_09_safety_and_pii_redaction.py` |
| **10** | **Benchmark Decontamination** | Phase 3 (Shared) | Node 4 (`safety_and_decontamination`) | `scripts/phase_03_reconvergence_and_tokenization/step_10_cross_dataset_decontamination.py` |
| **11** | **Pre-Tokenization Audit** | Phase 3 (Shared) | Node 5 (`tokenization_and_packing`) | `scripts/phase_03_reconvergence_and_tokenization/step_11_pre_tokenization_audit_and_schema_alignment.py` |
| **12** | **Tokenization & Packing** | Phase 3 (Shared) | Node 5 (`tokenization_and_packing`) | `scripts/phase_03_reconvergence_and_tokenization/step_12_tokenization_and_sequence_packing.py` |

---

## 4. Deep-Dive: Implemented Pipeline Steps

### Step 01: Unicode Normalization & Layout-Hyphen Reassembly

**File:** `scripts/phase_01_shared_ingestion/step_01_normalization.py`

**Node:** Node 1 (`intake_and_provenance_router`)

Restores canonical structure across multi-source text extracts before hashing or tokenization:

* **NFKC Composition:** Applies standard compatibility composition (`NFD` $\rightarrow$ `NFC`), converting decomposed accents (`e` + `\u0301`) into precomposed characters (`\u00E9`) and resolving compatibility ligatures (`\uFB01` $\rightarrow$ `fi`).
* **Control Byte & Phantom Token Stripping:** Purges non-printable ASCII bytes (`\x00`–`\x1F` excluding `\n`, `\t`) and zero-width spaces (`\u200B`), preventing tokenizer vocabulary corruption.
* **Layout Hyphen Reassembly:** Resolves OCR/PDF margin line splits (e.g., `en-\nterprise` $\rightarrow$ `enterprise`) via context-aware regex boundary rules while preserving legitimate compound terms (`high-quality`).

---

### Step 02: Boilerplate Stripping & Code Syntactic Profiling

**File:** `scripts/phase_01_shared_ingestion/step_02_boilerplate_stripping.py`

**Node:** Node 1 (`intake_and_provenance_router`)

Applies a 100% vectorized C++ sanitization pass directly in shared memory:

* **RE2 Structural Stripping:** Employs `pyarrow.compute.replace_substring_regex` with non-capturing groups (`(?i:...)`) to purge DOM tags, `<script>`/`<style>` blocks and their internal contents, pagination artifacts (`Page X of Y`), copyright blocks, and confidentiality watermarks (`CONFIDENTIAL - INTERNAL USE ONLY`).
* **Memory Margin Trimming:** Collapses duplicate newline runs and trims margin whitespace directly in C++ via `pyarrow.compute.utf8_trim_whitespace`.
* **Code Disambiguation Profiling:** Calculates structural character density in PyArrow shared memory ($\text{CodeDensity} = \frac{\text{Count}(\{, \}, ;)}{\max(1, \text{Length}(\text{text}))}$) and flags documents exceeding 5% syntax characters (`is_code_heavy = True`) for downstream routing.

---

### Step 03: Exact Deduplication (Bloom Filter & Embedded RocksDB)

**File:** `scripts/phase_01_shared_ingestion/step_03_exact_deduplication.py`

**Node:** Node 1 (`intake_and_provenance_router`)

Enforces global character-level uniqueness across distributed streaming partitions and historical runs:

* **Zero-Copy Hash Generation:** Generates 64-bit integer signatures ($H(S)$) using PyArrow C++ native kernels (`pc.hash_64`). Strings never leave Arrow C++ memory registers during signature generation.
* **Vectorized Intra-Batch Selection:** Executes a fast `np.unique()` pass over batch arrays to purge localized intra-batch character duplicates simultaneously.
* **Bloom Filter & Direct-Mapped RocksDB Lookup:** Passes 64-bit hashes through a lightweight in-memory Bloom filter pre-filter ($O(1)$ non-membership test). Potential collisions undergo exact verification against an embedded C++ RocksDB Key-Value index stored at `/dev/shm/dedup_index` (virtual shared memory).
* **Silver Manifest Persistence:** Non-duplicated 64-bit hash signatures (`document_hash_signature`, `pa.int64()`) and execution timestamps (`exact_dedup_at`) are written out alongside Silver Parquet files to guarantee historical state lineage across execution cycles.

---

### Step 04: Metadata Inspection & Routing

**File:** `scripts/phase_01_shared_ingestion/step_04_metadata_inspection_and_routing.py`

**Node:** Node 1 (`intake_and_provenance_router`)

Splits the data stream into domain-specific processing tracks to prevent natural language filters from destroying technical corpora:

* **Tier 1 Provenance Whitelist:** Inspects schema metadata (`pyarrow.KeyValueMetadata`). Documents tagged with explicit technical origins (`github_repo`, `financial_pdf`, `latex_research`) bypass prose heuristics and route directly to Track B (`branch_id = 1`).
* **Tier 2 Vectorized Snippet Disambiguation:** Un-tagged web pages undergo C++ structural symbol density profiling ($\text{Symbol Density} = \frac{\text{Count}(\{, \}, ;, [, ], =, \rightarrow, <, >)}{\text{Total Length}}$). If density $\ge 0.15$, a fastText micro-classifier pass separates valid embedded code snippets (routed to Track B) from garbled OCR/log noise (routed to Track A for pruning).

---

### Step 05a: Standard Macro-Linguistic Heuristics

**File:** `scripts/phase_02_domain_specific_processing/track_a_natural_language/step_05a_standard_heuristics.py`

**Node:** Node 2A (`nl_heuristics_and_dedup`)

Prunes structural web noise and unformatted telemetry dumps using statistical language boundaries:

* **Punctuation & Symbol Boundaries:** Drops documents with punctuation-to-word ratios $> 0.30$ or equal to $0.0$, and discards text where operational symbols (`#`, `$`, `%`, `@`) exceed 10% of total word count.
* **Stop-Word Density Threshold:** Verifies that functional structural words make up at least 5% to 10% of total document word count ($\text{Stop-Word Density} < 0.05$), pruning automated error dumps and inventory lists.
* **Repetition Pruning:** Evaluates frequency distributions of repeating 2-gram, 3-gram, and 4-gram phrase loops resulting from bad web scraping, emitting a zero-copy Boolean mask array.

---

### Step 05b: Code & Syntax Disambiguation

**File:** `scripts/phase_02_domain_specific_processing/track_b_specialized_domain/step_05b_code_and_syntax_disambiguation.py`

**Node:** Node 2B (`code_syntax_and_dedup`)

Serves as the primary quality gate for technical datasets, bypassing prose filters while scrubbing non-code system garbage:

* **Structural Symbol Profiling:** Enforces expected code symbol boundaries ($0.05 \le \text{Symbol Density} \le 0.25$), pruning minified assets ($> 0.50$) or raw license headers ($< 0.01$).
* **Fence Isolation & Classification:** Uses C++/Rust regex engines to isolate code fences (`<pre><code>`, ```) and classifies isolated syntax blocks using a fastText code model (`fasttext_code_id.bin`).
* **Indentation & Layout Profiling:** Analyzes leading indentation ratios (`\t`, spaces) and line-length variance, dropping minified single-line bundles ($> 2000$ characters) and corrupted OCR outputs.

---

### Step 06a: Document-Level MinHash LSH Fuzzy Deduplication

**File:** `scripts/phase_02_domain_specific_processing/track_a_natural_language/step_06a_minhash_fuzzy_deduplication.py`

**Node:** Node 2A (`nl_heuristics_and_dedup`)

Purges textual near-duplicates sharing 80%–95% structural overlap across prose corpora:

* **Shingling & Signature Vectors:** Converts documents into overlapping word $N$-gram shingles and computes 128-permutation MinHash signature vectors per document.
* **LSH Banding & Graph Clustering:** Bands MinHash signatures into local RocksDB bucket shards. Colliding candidate pairs meeting Jaccard similarity ($\ge 0.85$) are resolved via a Ray distributed Connected Components (Union-Find) graph, retaining only the canonical longest document.

---

### Step 06b: Code-Specific MinHash & AST Deduplication

**File:** `scripts/phase_02_domain_specific_processing/track_b_specialized_domain/step_06b_code_specific_minhash_ast_deduplication.py`

**Node:** Node 2B (`code_syntax_and_dedup`)

Eliminates code clones, refactored utility functions, and license header false positives:

* **Boilerplate & Comment Stripping:** Removes open-source license headers (Apache, MIT) and docstrings prior to hashing to prevent unrelated files with shared headers from being false-flagged.
* **Line-Level MinHash LSH:** Computes 128 MinHash permutations over normalized, non-empty code lines, clustering pairs exceeding a $0.80$ line-level Jaccard similarity.
* **Canonical AST Hashing:** Parses unique files using Tree-Sitter C++ parsers, normalizes user identifiers (`VAR_1`, `FUNC_1`), and computes 64-bit MurmurHash3 signatures over canonical AST nodes to eliminate variable-renamed logic clones.

---

### Step 07a: Classifier-Based Quality Filtering (CQF)

**File:** `scripts/phase_02_domain_specific_processing/track_a_natural_language/step_07a_natural_language_cqf.py`

**Node:** Node 3A (`nl_quality_and_lang`)

Shifts from heuristic pattern matching to probabilistic semantic grading:

* **Target Calibration:** Trains a lightweight binary classifier comparing a curated High-Quality (HQ) reference dataset against an uncurated Low-Quality (LQ) web stream.
* **Inference & Threshold Ranking:** Scores incoming documents with a scalar quality probability ($\text{Score} \in [0.0, 1.0]$), dropping records below a retention threshold ($\text{Score} < 0.65$). The float64 score column is permanently appended to Silver Parquet metadata to monitor threshold drift over time.

---

### Step 07b: Domain Quality & Lexer Check

**File:** `scripts/phase_02_domain_specific_processing/track_b_specialized_domain/step_07b_domain_quality_check.py`

**Node:** Node 3B (`code_validation_and_ast`)

Evaluates technical health and maintainability without misclassifying valid code as "low-quality prose":

* **Lexer Token Validation:** Passes code blocks through C-implemented Pygments lexers, dropping files exceeding a 5% lexical token error ratio ($\text{Error Ratio} > 0.05$).
* **Structural Maintainability Metrics:** Evaluates docstring-to-code ratios and Cyclomatic Complexity ($M = E - N + 2P$). Prunes trivial stubs ($M < 2$) and machine-generated decision trees ($M > 50$).
* **Neural Technical Quality Model:** Runs fine-tuned INT8-quantized code quality models via ONNX Runtime, enforcing a hard score threshold ($\ge 0.60$).

---

### Step 08a: FastText Language Identification

**File:** `scripts/phase_02_domain_specific_processing/track_a_natural_language/step_08a_fasttext_language_id.py`

**Node:** Node 3A (`nl_quality_and_lang`)

Prevents tokenizer fragmentation and vocabulary dilution caused by code-switching:

* **Segmented Consensus Architecture:** Breaks documents into structured paragraph blocks using boundary regexes and executes vectorized inference per chunk using Meta's fastText `lid.176.bin` model.
* **Code-Switching Threshold:** Calculates the linguistic distribution across paragraph blocks. If secondary languages exceed 15% of total document character length, alien paragraphs are purged or the document is routed into language-segmented lake partitions (`quarantine_de/`, `quarantine_es/`).

---

### Step 08b: AST Syntax Verification

**File:** `scripts/phase_02_domain_specific_processing/track_b_specialized_domain/step_08b_syntax_verification.py`

**Node:** Node 3B (`code_validation_and_ast`)

Enforces compiler-grade syntax correctness across multi-language source code, SQL queries, and LaTeX proofs:

* **Tree-Sitter Concrete Syntax Tree:** Builds in-memory ASTs using compiled C++ parsers. Calculates structural error density ($\frac{\text{ERROR Nodes} + \text{MISSING Nodes}}{\text{Total Nodes}}$) and enforces zero-tolerance pruning on any document with an error density $> 0.0$.
* **Scope Resolution & EOF Truncation:** Verifies bracket/scope matching (`{`, `}`, `begin{equation}`, `end{equation}`) and drops truncated files ending in dangling binary operators or unclosed string literals.
* **Dialect Fallback Escalation:** Escalates syntax failures to dialect fallback parsers (`tree-sitter-python2`, `sqlglot`), retaining valid legacy code while purging un-parseable files.

---

### Step 09: Safety Guardrails & PII Redaction

**File:** `scripts/phase_03_reconvergence_and_tokenization/step_09_safety_and_pii_redaction.py`

**Node:** Node 4 (`safety_and_decontamination`)

Executes unified safety sanitization across re-converged Track A and Track B streams:

* **Hybrid PII Masking:** Combines compiled regex patterns for alphanumeric identifiers (credit cards, IP addresses) with Microsoft Presidio (spaCy/Transformer NER token classifiers) for complex entities (names, corporate identifiers), replacing private attributes with immutable tags (`[REDACTED_EMAIL]`, `[REDACTED_NAME]`).
* **Localized Toxicity Filtering:** Runs sequence classifiers over text blocks to predict toxicity vector distributions (hate speech, profanity, harassment). Documents exceeding an enterprise threshold ($> 0.40$) are isolated into a secure compliance audit directory (`toxic_quarantine/`).

---

### Step 10: Benchmark Decontamination

**File:** `scripts/phase_03_reconvergence_and_tokenization/step_10_cross_dataset_decontamination.py`

**Node:** Node 4 (`safety_and_decontamination`)

Eliminates test-set leakage against industry benchmarks (FinQA, GSM8K, HumanEval, internal golden sets):

* **Fast N-Gram Hash Pre-Filter:** Extracts sliding 13-gram to 5-gram phrases from the evaluation catalog registry (`gold_standards/`), converts them to 64-bit integer hashes, and clears training documents with zero hash matches in $O(1)$ time.
* **Longest Common Subsequence (LCS) Validation:** Candidate matches undergo explicit string alignment checks. If the LCS ratio exceeds $0.20$ or matches $> 20$ consecutive tokens verbatim, the overlapping section is surgically redacted or the document is discarded.

---

### Step 11: Pre-Tokenization Audit & Schema Alignment

**File:** `scripts/phase_03_reconvergence_and_tokenization/step_11_pre_tokenization_audit_and_schema_alignment.py`

**Node:** Node 5 (`tokenization_and_packing`)

Validates structural integrity and injects tokenizer policies prior to tensor generation:

* **Schema Harmonization & Null Pruning:** Aligns field schemas across disparate data shards and purges null/empty records (`pyarrow.compute.is_null`) resulting from upstream PII redaction or decontamination scrubbing.
* **UTF-8 Byte Validation & Context Boundary:** Validates byte structures via `pyarrow.compute.utf8_is_valid`, replacing corrupted bytes with `\uFFFD`. Splits documents exceeding upper context length bounds ($> 128,000$ characters) at logical paragraph boundaries.
* **Tokenizer Policy Injection:** Inspects `branch_id` metadata tags and attaches explicit execution payloads for Step 12: injecting standard whitespace collapsing policies for Track A prose shards, and strict layout-preservation policies (`preserve_whitespace = True`, mapping `\t` and indentation spaces to dedicated token IDs) for Track B technical shards.

---

### Step 12: Tokenization & Sequence Packing

**File:** `scripts/phase_03_reconvergence_and_tokenization/step_12_tokenization_and_sequence_packing.py`

**Node:** Node 5 (`tokenization_and_packing`)

Transforms clean text strings into numeric machine tensors for pre-training:

* **Persistent Tokenizer Actor Pools:** Initializes a stateful pool of Ray Actors holding Hugging Face Rust-backed tokenizers locked in CPU memory, bypassing Python GIL bottlenecks during parallel encoding into `input_ids` and `attention_mask` arrays.
* **Sequence Context Packing:** Groups variable-length encoded samples into unified, fixed context windows (e.g., packing four 1024-token samples into a single 4096-token sequence matrix) to eliminate empty padding tokens during model training.
* **Gold Tensor Output:** Serializes packed integer arrays as flat Apache Arrow/Parquet tensor files directly out to the feature store registry (`s3a://final-features/tokenized_tensors/`).

---

## 5. Ingestion Gate 2 Data Contract

Data Gate 2 (`orchestrator/dags/scripts/quality_gate_2.py`) validates Silver outputs downstream of Ray tasks before subsequent DAG stages can consume the data.

* **Lazy S3 Pagination:** Uses `bucket.objects.filter(Prefix=prefix)` as an iterator rather than evaluating `list()`, avoiding Airflow Worker OOM crashes when processing tens of thousands of Parquet partitions.
* **Corrupted File Detection:** Asserts that all data assets contain size $> 0$ bytes, ignoring Spark/Ray metadata flags (`_SUCCESS`, `._*`).

---

## 6. Verification & Automated Test Suite

### Layer 1: DAG Topology & Dependency Tests

Verifies that `2_prep_bronze_to_silver` parses without import errors and configures correct task dependency graphs.

```bash
docker exec -it actf-core-airflow-webserver pytest /opt/airflow/tests/test_dag_integrity.py -v

```

### Layer 2: Transformation Logic Unit Tests

Executes unit tests across phased transformation modules in microsecond isolation.

```bash
docker exec -it actf-core-ray-head pytest /home/ray/workspace/2-data-prep/tests/ -v

```

### Universal Platform Execution

Run all system tests via the root orchestration script:

```bash
./run_tests.sh

```