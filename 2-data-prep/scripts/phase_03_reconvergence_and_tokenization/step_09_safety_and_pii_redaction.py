import datetime
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

# PII Redaction (Microsoft Presidio)
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

# Toxicity Classification (Hugging Face / GPU)
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SafetyAndPIIRedactor:
    """
    Ray Data Stateful Actor for Phase 3: Safety Guardrails & PII Redaction.
    
    ===========================================================================
    ARCHITECTURAL DESIGN:
    ===========================================================================
    Part A: Localized Toxicity Filtering
            Runs a high-throughput sequence classifier (`unitary/toxic-bert`) 
            over text blocks. If any toxicity score crosses the >0.4 threshold, 
            the document is immediately dropped.
            
    Part B: PII Redaction via Hybrid Extraction
            Uses Microsoft Presidio (Regex + NER token classifier) to detect 
            entities (e.g., Emails, Phone Numbers, Persons). Replaces private 
            identifiers with immutable generic tags (e.g., [EMAIL_ADDRESS]).
    ===========================================================================
    """
    def __init__(self, toxicity_threshold: float = 0.4, toxicity_model: str = "unitary/toxic-bert"):
        self.toxicity_threshold = toxicity_threshold
        
        # BUG FIX: Initialize instance variables to None to support 
        # dependency injection via MagicMock during local testing.
        self.analyzer = None
        self.anonymizer = None
        
        # Initialize Presidio Engines (CPU bound, fast NER)
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            
        # Initialize Toxicity Classifier (GPU accelerated if available)
        self.toxicity_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            device = 0 if torch.cuda.is_available() else -1
            self.toxicity_pipeline = pipeline(
                "text-classification", 
                model=toxicity_model, 
                device=device,
                truncation=True,
                max_length=512
            )

    def _redact_pii(self, text: str) -> str:
        """Runs Microsoft Presidio to detect and anonymize PII."""
        
        # BUG FIX: Evaluate the instance attributes instead of the global import flag. 
        # This respects injected test mocks even if Presidio is missing locally.
        if not self.analyzer or not self.anonymizer or not text.strip():
            return text
            
        # Analyze text for PII entities
        results = self.analyzer.analyze(text=text, language='en')
        
        # Anonymize (replaces with [ENTITY_TYPE] by default)
        anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized_result.text

    def _is_toxic(self, text: str) -> bool:
        """Evaluates toxicity probability. Returns True if score > threshold."""
        if not self.toxicity_pipeline or not text.strip():
            return False
            
        # toxic-bert outputs labels like 'toxic', 'severe_toxic', 'obscene', etc.
        try:
            results = self.toxicity_pipeline(text[:2000]) # Truncate for inference speed
            for result in results:
                # unitary/toxic-bert returns score for the highest probability label
                if result['score'] > self.toxicity_threshold and result['label'] != 'non-toxic':
                    return True
        except Exception:
            return False
            
        return False

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        texts = text_column.to_pylist()
        
        keep_mask = []
        redacted_texts = []
        timestamps = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for text in texts:
            # 1. Toxicity Filtering
            if self._is_toxic(text):
                keep_mask.append(False)
                redacted_texts.append("")
                continue
                
            # 2. PII Redaction (Only process surviving clean texts)
            clean_text = self._redact_pii(text)
            
            keep_mask.append(True)
            redacted_texts.append(clean_text)
            timestamps.append(current_utc)

        mask_array = pa.array(keep_mask, type=pa.bool_())
        
        # Filter texts directly using standard Python lists before converting to Arrow
        surviving_texts = [t for i, t in enumerate(redacted_texts) if keep_mask[i]]
        text_array = pa.array(surviving_texts, type=pa.string())
        time_array = pa.array([current_utc] * len(surviving_texts))

        if is_arrow_table:
            # Drop toxic rows
            filtered_table = batch.filter(mask_array)
            
            # Replace old text column with PII-redacted text column
            text_idx = filtered_table.column_names.index("text")
            filtered_table = filtered_table.set_column(text_idx, "text", text_array)
            
            if "safety_audited_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["safety_audited_at"])
                
            return filtered_table.append_column("safety_audited_at", time_array)
        else:
            filtered_dict = {
                key: [v for i, v in enumerate(value) if keep_mask[i]]
                for key, value in batch.items() if key != "text"
            }
            filtered_dict["text"] = surviving_texts
            filtered_dict["safety_audited_at"] = [current_utc] * len(surviving_texts)
            return filtered_dict