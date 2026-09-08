"""Unit test verifying that zero extracted facts never result in a silent DONE status."""
from unittest.mock import patch
import pytest
from src.db.models import Document, JobStatus
from src.db.session import SessionLocal
from src.pipeline.orchestrator import orchestrator
from src.pipeline.pdf_parser import ParsedChunk


def test_zero_facts_sets_done_empty_status():
    """TASK: Simulate all chunks returning extraction_status=success with empty facts.

    Assert the document ends up in DONE_EMPTY state with an explicit explanation,
    and never in a plain DONE state.
    """
    db = SessionLocal()
    test_doc = Document(
        filename="test_empty_report.pdf",
        file_path="dummy_path.pdf",
        status=JobStatus.QUEUED
    )
    db.add(test_doc)
    db.commit()
    db.refresh(test_doc)
    doc_id = test_doc.id
    db.close()

    fake_chunks = [
        ParsedChunk(page_number=1, char_start=0, char_end=50, raw_text="Safe Harbor statement only.", is_table=False, needs_vision=False),
        ParsedChunk(page_number=2, char_start=0, char_end=50, raw_text="Forward-looking disclaimer only.", is_table=False, needs_vision=False)
    ]

    with patch("src.pipeline.pdf_parser.pdf_parser.parse", return_value=(fake_chunks, 2, "macro_report")), \
         patch("src.pipeline.extractor.fact_extractor.extract_from_chunk_text", return_value=[]), \
         patch("src.pipeline.llm_client.llm_client.get_last_call_status", return_value=("success", None)):

        result = orchestrator.process_document(doc_id)

    db = SessionLocal()
    updated_doc = db.query(Document).filter(Document.id == doc_id).first()

    assert updated_doc.status == JobStatus.DONE_EMPTY
    assert updated_doc.status != JobStatus.DONE
    assert "0 facts extracted from 2 processed chunks" in updated_doc.error_message
    assert "success: 2" in updated_doc.error_message

    assert result["status"] == "done_empty"
    assert result["facts_extracted"] == 0

    # Cleanup test row
    db.delete(updated_doc)
    db.commit()
    db.close()
