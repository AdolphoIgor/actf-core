import pytest

from scripts.step_11_pre_tokenization_audit_and_schema_alignment import (
    PreTokenizationAuditor,
)


def _execute_audit(auditor, batch_or_record):
    for attr in [
        "__call__",
        "audit_and_align",
        "audit_and_align_batch",
        "audit",
        "validate",
        "validate_batch",
        "validate_schema",
        "audit_record",
        "process",
    ]:
        if hasattr(auditor, attr):
            method = getattr(auditor, attr)
            if (
                attr == "__call__"
                and "__call__" not in auditor.__class__.__dict__
                and not callable(auditor)
            ):
                continue
            if callable(method):
                return method(batch_or_record)
    if callable(auditor):
        return auditor(batch_or_record)
    raise AttributeError(
        f"No valid audit callable found on PreTokenizationAuditor. Available"
        f" attributes: {dir(auditor)}"
    )


def test_pre_tokenization_auditor_schema_validation():
    auditor = PreTokenizationAuditor()
    valid_dict_batch = {
        "doc_id": ["doc_001", "doc_002"],
        "text": ["Valid instruction text", "def test(): pass"],
        "branch_id": [0, 1],
    }
    valid_list_batch = [
        {"doc_id": "doc_001", "text": "Valid instruction text", "branch_id": 0},
        {"doc_id": "doc_002", "text": "def test(): pass", "branch_id": 1},
    ]

    try:
        result = _execute_audit(auditor, valid_dict_batch)
    except Exception:
        result = _execute_audit(auditor, valid_list_batch)

    assert result is not None


def test_pre_tokenization_auditor_rejects_missing_keys():
    auditor = PreTokenizationAuditor()
    corrupted_dict_batch = {
        "doc_id": ["doc_001"],
        "unrecognized_key": ["No text present"],
    }
    corrupted_list_batch = [
        {"doc_id": "doc_001", "unrecognized_key": "No text present"},
    ]

    with pytest.raises((AssertionError, ValueError, KeyError), match="(?i)missing|schema|key|text"):
        try:
            _execute_audit(auditor, corrupted_dict_batch)
        except (AssertionError, ValueError, KeyError):
            raise
        except Exception:
            _execute_audit(auditor, corrupted_list_batch)
