import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "auto")
BRONZE_URI = os.environ.get("BRONZE_STORAGE_PATH", "data/bronze")
SILVER_URI = os.environ.get("SILVER_STORAGE_PATH", "data/silver")


def execute_ray_bronze_to_silver_curation(**context) -> dict[str, Any]:
    """
    Executes Steps 01 to 10 within Ray Data shared Plasma memory.

    Pipeline Structure:
    1. Phase 1 (Shared Ingestion): Steps 01 -> 02 -> 03 -> 04 (Routing)
    2. Phase 2 (In-Memory Branching):
       - Track A (Prose): Steps 05a -> 06a -> 07a -> 08a
       - Track B (Technical): Steps 05b -> 06b -> 07b -> 08b
    3. Phase 3 (Reconvergence): Steps 09 (Safety/PII) -> 10 (Decontamination)
    """
    import ray
    from scripts.phase_01_shared_ingestion.step_01_normalization import UnicodeNormalizer
    from scripts.phase_01_shared_ingestion.step_02_boilerplate_stripping import BoilerplateStripper
    from scripts.phase_01_shared_ingestion.step_03_exact_deduplication import ExactDeduplicator
    from scripts.phase_01_shared_ingestion.step_04_metadata_inspection_and_routing import (
        MetadataRouter,
    )
    from scripts.phase_02_domain_specific_processing.track_a_natural_language.step_05a_standard_heuristics import (
        StandardHeuristicsFilter,
    )
    from scripts.phase_02_domain_specific_processing.track_a_natural_language.step_06a_minhash_fuzzy_deduplication import (
        MinHashFuzzyDeduplicator,
    )
    from scripts.phase_02_domain_specific_processing.track_a_natural_language.step_07a_natural_language_cqf import (
        NaturalLanguageCQF,
    )
    from scripts.phase_02_domain_specific_processing.track_a_natural_language.step_08a_fasttext_language_id import (
        FastTextLanguageID,
    )
    from scripts.phase_02_domain_specific_processing.track_b_specialized_domain.step_05b_code_and_syntax_disambiguation import (
        CodeSyntaxDisambiguator,
    )
    from scripts.phase_02_domain_specific_processing.track_b_specialized_domain.step_06b_code_specific_minhash_ast_deduplication import (
        CodeASTDeduplicator,
    )
    from scripts.phase_02_domain_specific_processing.track_b_specialized_domain.step_07b_domain_quality_check import (
        DomainQualityChecker,
    )
    from scripts.phase_02_domain_specific_processing.track_b_specialized_domain.step_08b_syntax_verification import (
        SyntaxVerifier,
    )
    from scripts.phase_03_reconvergence_and_tokenization.step_09_safety_and_pii_redaction import (
        SafetyAndPIIRedactor,
    )
    from scripts.phase_03_reconvergence_and_tokenization.step_10_cross_dataset_decontamination import (
        CrossDatasetDecontaminator,
    )

    try:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    except Exception:
        ray.init(ignore_reinit_error=True)

    # -------------------------------------------------------------------------
    # Phase 1: Shared Ingestion Trunk (Steps 01 - 04)
    # -------------------------------------------------------------------------
    ds = ray.data.read_parquet(BRONZE_URI)
    ds = ds.map_batches(UnicodeNormalizer, batch_format="pyarrow")
    ds = ds.map_batches(BoilerplateStripper, batch_format="pyarrow")
    ds = ds.map_batches(ExactDeduplicator, batch_format="pyarrow")
    ds = ds.map_batches(MetadataRouter, batch_format="pyarrow")

    # Pin in-memory representation before lazy branching to prevent duplicate evaluation
    ds = ds.materialize()

    # -------------------------------------------------------------------------
    # Phase 2: In-Memory Domain-Specific Branching (Steps 05 - 08)
    # -------------------------------------------------------------------------
    # Track A: Natural Language Prose (branch_id == 0)
    ds_prose = ds.filter(lambda row: row.get("branch_id", 0) == 0)
    ds_prose = ds_prose.map_batches(StandardHeuristicsFilter, batch_format="pyarrow")
    ds_prose = ds_prose.map_batches(MinHashFuzzyDeduplicator, batch_format="pyarrow")
    ds_prose = ds_prose.map_batches(NaturalLanguageCQF, batch_format="pyarrow")
    ds_prose = ds_prose.map_batches(FastTextLanguageID, batch_format="pyarrow")

    # Track B: Code & Technical Domains (branch_id == 1)
    ds_code = ds.filter(lambda row: row.get("branch_id", 0) == 1)
    ds_code = ds_code.map_batches(CodeSyntaxDisambiguator, batch_format="pyarrow")
    ds_code = ds_code.map_batches(CodeASTDeduplicator, batch_format="pyarrow")
    ds_code = ds_code.map_batches(DomainQualityChecker, batch_format="pyarrow")
    ds_code = ds_code.map_batches(SyntaxVerifier, batch_format="pyarrow")

    # -------------------------------------------------------------------------
    # Phase 3: Shared Reconvergence & Global Safety (Steps 09 - 10)
    # -------------------------------------------------------------------------
    ds_reconverged = ds_prose.union(ds_code)
    ds_reconverged = ds_reconverged.map_batches(SafetyAndPIIRedactor, batch_format="pyarrow")
    ds_curated = ds_reconverged.map_batches(CrossDatasetDecontaminator, batch_format="pyarrow")

    os.makedirs(SILVER_URI, exist_ok=True)
    ds_curated.write_parquet(SILVER_URI)

    return {"status": "SUCCESS", "output_path": SILVER_URI}


DAG_DOC_MD = """
# Medallion Silver Preparation & Normalization Engine (`dag_03_prep_bronze_to_silver`)

Executes distributed Ray Data transformations across Bronze Parquet stores:
* **Phase 1 (Shared Ingestion):** Unicode normalization, zero-copy boilerplate stripping, and exact SHA-256 deduplication.
* **Phase 2 (Dual-Track Domain Processing):** 
  * Track A (NLP): Heuristic filtering, MinHash LSH fuzzy deduplication, and FastText language ID.
  * Track B (Code/SQL): AST disambiguation, syntax tree verification via Tree-Sitter, and code-specific deduplication.
* **Phase 3 (Reconvergence):** Presidio PII redaction, cross-dataset decontamination, and **Quality Gate 2** storage validation.
"""

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="dag_03_prep_bronze_to_silver",
    default_args=default_args,
    description="Distributed Ray Data In-Memory Curation: Steps 01 to 10",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["curation", "ray", "silver", "actf"],
    doc_md=DAG_DOC_MD,
) as dag:
    ray_curation_task = PythonOperator(
        task_id="ray_distributed_bronze_to_silver_curation",
        python_callable=execute_ray_bronze_to_silver_curation,
    )

    quality_gate_2_task = BashOperator(
        task_id="data_quality_gate_2_silver_check",
        bash_command="python /opt/airflow/dags/scripts/quality_gate_2.py",
    )

    ray_curation_task >> quality_gate_2_task
