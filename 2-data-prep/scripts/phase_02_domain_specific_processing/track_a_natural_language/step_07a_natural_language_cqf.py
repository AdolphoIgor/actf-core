import os
import datetime
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False


class ProseQualityClassifier:
    """
    Ray Data Stateful Actor for Track A: Classifier-Based Quality Filtering (CQF).
    
    ===========================================================================
    ARCHITECTURAL DESIGN:
    ===========================================================================
    1. Inference: Uses a high-speed lexical classifier (fastText) to evaluate 
       text blocks against a curated "Gold Standard" baseline.
    2. Scoring: Generates a float64 quality score between 0.0 and 1.0.
    3. Metadata Preservation: Appends the `cqf_quality_score` to the schema 
       for downstream observability and dynamic thresholding.
    4. Hard Gate: Drops documents falling below the 0.65 probability threshold.
    ===========================================================================
    """
    def __init__(self, model_path: str = "/opt/models/cqf_prose_model.bin", threshold: float = 0.65):
        self.threshold = threshold
        self.fasttext_model = None
        
        if FASTTEXT_AVAILABLE and os.path.exists(model_path):
            self.fasttext_model = fasttext.load_model(model_path)

    def _score_text(self, text: str) -> float:
        """
        Runs fastText inference on a single string.
        Assumes the model is trained with __label__hq (High Quality) 
        and __label__lq (Low Quality).
        """
        if not self.fasttext_model or not text.strip():
            return 1.0  # Pass-through fallback if model is missing or text is empty

        # fastText requires newlines to be stripped for single-document inference
        safe_text = text.replace("\n", " ")[:2000] 
        labels, probs = self.fasttext_model.predict(safe_text)
        
        if not labels:
            return 0.0
            
        label = labels[0]
        prob = float(probs[0])
        
        if label == "__label__hq":
            return prob
        elif label == "__label__lq":
            return 1.0 - prob
        return 0.0

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        texts = text_column.to_pylist()
        
        keep_mask = []
        scores = []
        timestamps = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for text in texts:
            score = self._score_text(text)
            scores.append(score)
            
            if score >= self.threshold:
                keep_mask.append(True)
            else:
                keep_mask.append(False)
                
            timestamps.append(current_utc)

        mask_array = pa.array(keep_mask, type=pa.bool_())
        score_array = pa.array(scores, type=pa.float64())
        time_array = pa.array(timestamps)

        if is_arrow_table:
            # Append metadata before filtering so dropped records can theoretically 
            # be logged to a quarantine bucket in a more advanced Airflow split step
            enriched_table = batch
            for col in ("cqf_quality_score", "cqf_scored_at"):
                if col in enriched_table.column_names:
                    enriched_table = enriched_table.drop_columns([col])
                    
            enriched_table = enriched_table.append_column("cqf_quality_score", score_array)
            enriched_table = enriched_table.append_column("cqf_scored_at", time_array)
            
            # Zero-copy Boolean mask filtering
            return enriched_table.filter(mask_array)
        else:
            filtered_dict = {
                key: [v for i, v in enumerate(value) if keep_mask[i]]
                for key, value in batch.items()
            }
            # Append metadata to remaining dictionary arrays
            filtered_dict["cqf_quality_score"] = [s for i, s in enumerate(scores) if keep_mask[i]]
            filtered_dict["cqf_scored_at"] = [t for i, t in enumerate(timestamps) if keep_mask[i]]
            return filtered_dict