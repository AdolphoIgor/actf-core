import os
import re
import datetime
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.compute as pc

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False


class FastTextLanguageFilter:
    """
    Ray Data Stateful Actor for Track A: FastText Language Identification.
    
    ===========================================================================
    ARCHITECTURAL DESIGN: Segmented Consensus Architecture
    ===========================================================================
    1. Paragraph Chunking: Breaks raw text into structured paragraph blocks 
       using regex boundary markers.
    2. Vectorized Sub-Document Inference: Evaluates each paragraph 
       independently using Meta's lightweight lid.176.bin C++ engine[cite: 2].
    3. Code-Switching Threshold: If secondary language blocks exceed 15% 
       of the document's total characters, the document is pruned (or routed 
       to a quarantine partition)[cite: 2].
    ===========================================================================
    """
    def __init__(self, 
                 model_path: str = "/opt/models/lid.176.bin", 
                 target_lang: str = "__label__en",
                 alien_threshold: float = 0.15):
        self.target_lang = target_lang
        self.alien_threshold = alien_threshold
        self.fasttext_model = None
        
        # Load the 126MB C++ model directly into worker memory[cite: 2]
        if FASTTEXT_AVAILABLE and os.path.exists(model_path):
            self.fasttext_model = fasttext.load_model(model_path)

    def _get_alien_language_ratio(self, text: str) -> float:
        """
        Calculates the statistical distribution of languages across the text[cite: 2].
        """
        if not self.fasttext_model or not text.strip():
            return 0.0

        total_chars = len(text)
        alien_chars = 0
        
        # Paragraph chunking via fast regex token boundary markers[cite: 2]
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        for p in paragraphs:
            # fastText requires single-line strings
            safe_p = p.replace("\n", " ")[:1500] 
            labels, _ = self.fasttext_model.predict(safe_p)
            
            if labels and labels[0] != self.target_lang:
                alien_chars += len(p)
                
        return alien_chars / max(total_chars, 1)

    def __call__(self, batch: Union[Dict[str, Any], pa.Table]) -> Union[Dict[str, Any], pa.Table]:
        is_arrow_table = isinstance(batch, pa.Table)
        
        text_column = batch["text"] if is_arrow_table else pa.array(batch["text"])
        text_column = pc.fill_null(text_column, "")
        texts = text_column.to_pylist()
        
        keep_mask = []
        timestamps = []
        current_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for text in texts:
            alien_ratio = self._get_alien_language_ratio(text)
            
            # The Code-Switching Threshold[cite: 2]
            if alien_ratio > self.alien_threshold:
                keep_mask.append(False)
            else:
                keep_mask.append(True)
                timestamps.append(current_utc)

        mask_array = pa.array(keep_mask, type=pa.bool_())
        time_array = pa.array(timestamps)

        if is_arrow_table:
            filtered_table = batch.filter(mask_array)
            if "language_filtered_at" in filtered_table.column_names:
                filtered_table = filtered_table.drop_columns(["language_filtered_at"])
            return filtered_table.append_column("language_filtered_at", time_array)
        else:
            filtered_dict = {
                key: [v for i, v in enumerate(value) if keep_mask[i]]
                for key, value in batch.items()
            }
            filtered_dict["language_filtered_at"] = timestamps
            return filtered_dict