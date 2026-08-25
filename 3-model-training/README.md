# Model Training Engine (`3-model-training`)

The `3-model-training` module encapsulates the parameter optimization, distributed execution, and checkpoint lifecycle stages of the ACTF continuous training pipeline. It accepts curated Silver-tier datasets from Phase 1-3, verifies in-memory computational graph stability via **Gate 4**, and executes mixed-precision parameter updates before staging artifacts for downstream evaluation.

---

## Architecture and Pipeline Scope

The module coordinates execution from pre-tokenization checks through asynchronous artifact offloading:

```text
[ Curated Silver Dataset ]
          │
          ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3-MODEL-TRAINING PIPELINE LIFECYCLE                                    │
├────────────────────────────────────────────────────────────────────────┤
│ Step 11: Pre-Tokenization Audit and Schema Alignment                   │
│   • Validates structural column layouts and character encodings.      │
│                                                                        │
│ Step 12: Tokenization and Sequence Packing                             │
│   • Applies tokenizer vocabs and packs sequences to max context length.│
│                                                                        │
│ Gate 4: Pre-Flight Tensor and Gradient Health Gate                     │
│   • Verifies zero NaNs/Infs, Step-0 loss ln(V), tied pointers, and VRAM.│
│                                                                        │
│ Step 13: Distributed Parameter Optimization Loop                       │
│   • Executes AdamW optimization, cosine warmup decay, and grad clipping│
│   • Supports BFloat16/FP16 mixed precision and gradient accumulation.  │
│                                                                        │
│ Step 14: Ephemeral Staging Export and Asynchronous Offloading          │
│   • Synchronously stages recovery and stripped inference bundles.      │
│   • Dispatches background uploads with checksum validation.            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                     [ 4-model-eval / Gate 5 ]