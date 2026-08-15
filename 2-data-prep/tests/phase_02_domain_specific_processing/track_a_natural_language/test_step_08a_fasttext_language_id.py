import pytest
import pyarrow as pa
from unittest.mock import MagicMock
from scripts.phase_02_domain_specific_processing.track_a_natural_language.step_08a_fasttext_language_id import FastTextLanguageFilter


def test_language_id_pure_english():
    """Validates that a purely English document passes the threshold."""
    actor = FastTextLanguageFilter(model_path="mock_path", target_lang="__label__en")
    
    actor.fasttext_model = MagicMock()
    actor.fasttext_model.predict.return_value = (["__label__en"], [0.99])
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_en"],
        "text": ["This is an English paragraph.\n\nThis is another English paragraph."]
    })
    
    result = actor(input_table)
    assert result.num_rows == 1


def test_language_id_erratic_code_switching():
    """Validates that erratic code-switching >15% of character length causes the file to be pruned[cite: 2]."""
    actor = FastTextLanguageFilter(model_path="mock_path", alien_threshold=0.15)
    
    actor.fasttext_model = MagicMock()
    
    # Simulate: Paragraph 1 is English, Paragraph 2 is German logs (alien)
    actor.fasttext_model.predict.side_effect = [
        (["__label__en"], [0.99]), 
        (["__label__de"], [0.95])
    ]
    
    # German paragraph is large enough to exceed 15% of the total string
    text = (
        "Short English introduction.\n\n"
        "Achtung! Kritischer Systemfehler aufgetreten. Der Server reagiert nicht mehr und muss sofort neu gestartet werden, um Datenverlust zu vermeiden."
    )
    
    input_table = pa.Table.from_pydict({
        "doc_id": ["doc_code_switch"],
        "text": [text]
    })
    
    result = actor(input_table)
    assert result.num_rows == 0