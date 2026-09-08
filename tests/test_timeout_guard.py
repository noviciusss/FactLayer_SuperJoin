"""Unit tests for bounded retry policy, call timeouts, and document processing timeout backstop."""
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy.orm import Session

from src.db.models import Chunk, Document, Fact, JobStatus
from src.db.session import SessionLocal, init_db
from src.pipeline.orchestrator import orchestrator


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def db():
    session: Session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_groq_persistent_429_surfaces_clean_failure_without_hanging(db):
    """
    Test 7a: Simulating a Groq client that consistently raises 429 rate limits.
    Asserts pipeline respects bounded retry ceiling, marks chunks rate_limited,
    and transitions the Document to FAILED rather than hanging indefinitely.
    """
    # Create test document
    doc = Document(
        filename="test_rate_limit_doc.pdf",
        file_path="delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        status=JobStatus.QUEUED
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Mock client.chat.completions.create to always raise 429 RateLimit error
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception(
        "Error code: 429 - Rate limit reached for model in organization on OTPM. Please try again in 0.05s."
    )

    with patch("src.pipeline.llm_client.LLMClient._get_client", return_value=mock_client):
        # Run process_document on 1 page with a short document timeout
        res = orchestrator.process_document(doc.id, max_pages=1, timeout_seconds=15.0)

        # Reload document from DB
        db.refresh(doc)
        assert doc.status == JobStatus.FAILED
        assert "rate limits" in doc.error_message.lower() or "exhausted" in doc.error_message.lower()
        assert res["status"] == "failed"

        # Verify chunks in DB are explicitly marked as rate_limited
        chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) > 0
        for c in chunks:
            assert c.extraction_status == "rate_limited"

        # Assert no runaway retries: create call count per chunk should be bounded by max_retries
        assert mock_client.chat.completions.create.call_count <= 8


def test_groq_persistent_timeout_surfaces_clean_failure_without_hanging(db):
    """
    Test 7b: Simulating a Groq client that consistently raises timeout errors.
    Asserts pipeline terminates cleanly and marks chunks with extraction_failed.
    """
    doc = Document(
        filename="test_timeout_doc.pdf",
        file_path="delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        status=JobStatus.QUEUED
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = TimeoutError("Groq request timed out after 30.0s")

    with patch("src.pipeline.llm_client.LLMClient._get_client", return_value=mock_client):
        res = orchestrator.process_document(doc.id, max_pages=1, timeout_seconds=15.0)

        db.refresh(doc)
        assert doc.status == JobStatus.FAILED
        assert res["status"] == "failed"

        chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) > 0
        for c in chunks:
            assert c.extraction_status == "extraction_failed"


def test_document_processing_timeout_backstop_triggers_failure(db):
    """
    Test 7c: Simulating a document that exceeds the overall processing deadline.
    Asserts pipeline catches TimeoutError backstop and transitions document to FAILED.
    """
    doc = Document(
        filename="test_deadline_doc.pdf",
        file_path="delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        status=JobStatus.QUEUED
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Calling with timeout_seconds=0.001 forces deadline expiration
    res = orchestrator.process_document(doc.id, max_pages=1, timeout_seconds=0.001)

    db.refresh(doc)
    assert doc.status == JobStatus.FAILED
    assert "timeout exceeded" in doc.error_message.lower()
    assert res["status"] == "failed"


def test_executor_cancellation_on_timeout_does_not_block(db):
    """
    Test Task B: Simulate a deadline exceeded mid-extraction with multiple slow futures running.
    Asserts:
      1. executor.shutdown(wait=False, cancel_futures=True) fires without blocking.
      2. Total wall-clock time stays close to the deadline (< 4.0s), NOT waiting for 10s calls.
      3. Unfinished chunks get extraction_status='timed_out'.
      4. Document transitions to FAILED with timeout error.
    """
    import time

    doc = Document(
        filename="test_cancel_futures_doc.pdf",
        file_path="delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        status=JobStatus.QUEUED
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Mock fact extraction to sleep for 10.0 seconds (simulating long in-flight HTTP calls)
    def slow_mock_extraction(*args, **kwargs):
        time.sleep(10.0)
        return []

    from src.pipeline.pdf_parser import ParsedChunk

    fake_chunks = [
        ParsedChunk(page_number=1, char_start=0, char_end=100, raw_text="Chunk 1", is_table=False, needs_vision=False),
        ParsedChunk(page_number=2, char_start=0, char_end=100, raw_text="Chunk 2", is_table=False, needs_vision=False),
        ParsedChunk(page_number=3, char_start=0, char_end=100, raw_text="Chunk 3", is_table=False, needs_vision=False),
    ]

    start_wall_clock = time.time()

    # Pass timeout_seconds=1.0 with fake chunks so timeout occurs mid-Stage B
    with patch("src.pipeline.orchestrator.pdf_parser.parse", return_value=(fake_chunks, 3, "earnings_presentation")), \
         patch("src.pipeline.orchestrator.fact_extractor.extract_from_chunk_text", side_effect=slow_mock_extraction):
        res = orchestrator.process_document(doc.id, max_pages=3, timeout_seconds=1.0)

    elapsed_wall_clock = time.time() - start_wall_clock

    db.refresh(doc)
    assert doc.status == JobStatus.FAILED
    assert "timeout" in doc.error_message.lower()
    assert res["status"] == "failed"

    # Wall-clock time should be close to 1.0s, NOT 10s (proves wait=False cancelled futures without waiting)
    assert elapsed_wall_clock < 4.0, (
        f"Executor waited for slow futures! Elapsed time was {elapsed_wall_clock:.2f}s instead of ~1.0s"
    )

    # Check chunks in DB: at least one chunk must be marked as timed_out
    chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()
    assert len(chunks) > 0
    timed_out_chunks = [c for c in chunks if c.extraction_status == "timed_out"]
    assert len(timed_out_chunks) > 0, "No chunks were marked as timed_out!"

