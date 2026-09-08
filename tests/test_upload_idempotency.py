"""Unit tests for content-hash upload idempotency guard."""
import hashlib
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.main import app
from src.db.models import Chunk, Document, Fact, JobStatus, Relationship, RelationType
from src.db.session import SessionLocal, init_db


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


@pytest.fixture
def client():
    return TestClient(app)


def test_upload_same_file_twice_returns_same_id_and_runs_pipeline_once(client, db):
    """
    Test 1: Uploading the same file twice returns the same document_id both times,
    and the second call does not trigger a second pipeline run (orchestrator called exactly once).
    """
    pdf_bytes = b"%PDF-1.4 test same file content twice \x00\x01\x02"
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Clean up prior test runs if any
    old_docs = db.query(Document).filter(Document.content_hash == expected_hash).all()
    for d in old_docs:
        db.query(Fact).filter(Fact.document_id == d.id).delete()
        db.query(Chunk).filter(Chunk.document_id == d.id).delete()
        db.delete(d)
    db.commit()

    with patch("src.api.routes.orchestrator.process_document") as mock_process:
        # First upload
        res1 = client.post(
            "/documents",
            files={"file": ("report_v1.pdf", pdf_bytes, "application/pdf")}
        )
        assert res1.status_code in (200, 202)
        data1 = res1.json()
        doc_id_1 = data1["document_id"]
        assert data1["content_hash"] == expected_hash
        assert data1["deduplicated"] is False
        assert mock_process.call_count == 1

        # Simulate first document finishing pipeline (DONE) and producing 3 facts
        doc1 = db.query(Document).filter(Document.id == doc_id_1).first()
        doc1.status = JobStatus.DONE
        dummy_chunk = Chunk(document_id=doc_id_1, page_number=1, raw_text="Sample text")
        db.add(dummy_chunk)
        db.flush()
        for i in range(3):
            f = Fact(
                document_id=doc_id_1,
                chunk_id=dummy_chunk.id,
                entity="TestCo",
                attribute=f"Metric_{i}",
                value=str(100 * i),
                evidence_quote=f"Quote {i}",
                canonical_statement=f"Statement {i}"
            )
            db.add(f)
        db.commit()

        # Second upload with identical bytes (simulating retry / double click)
        res2 = client.post(
            "/documents",
            files={"file": ("report_v1_renamed.pdf", pdf_bytes, "application/pdf")}
        )
        assert res2.status_code in (200, 202)
        data2 = res2.json()

        # Idempotency checks:
        assert data2["document_id"] == doc_id_1
        assert data2["deduplicated"] is True
        assert data2["status"] == "done"
        assert data2["facts_count"] == 3
        # Assert orchestrator's process function was NOT called a second time
        assert mock_process.call_count == 1


def test_upload_same_file_while_inflight_does_not_enqueue_duplicate_job(client, db):
    """
    Test 2: Uploading the same file while the first is still mid-processing returns
    the in-flight document_id without enqueuing a duplicate job.
    """
    pdf_bytes = b"%PDF-1.4 in-flight processing guard bytes \x10\x20\x30"
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Clean up prior test runs
    old_docs = db.query(Document).filter(Document.content_hash == expected_hash).all()
    for d in old_docs:
        db.query(Fact).filter(Fact.document_id == d.id).delete()
        db.query(Chunk).filter(Chunk.document_id == d.id).delete()
        db.delete(d)
    db.commit()

    with patch("src.api.routes.orchestrator.process_document") as mock_process:
        # First upload starts processing
        res1 = client.post(
            "/documents",
            files={"file": ("inflight.pdf", pdf_bytes, "application/pdf")}
        )
        assert res1.status_code in (200, 202)
        doc_id_1 = res1.json()["document_id"]
        assert mock_process.call_count == 1

        # Simulate document transitioning to EXTRACTING stage
        doc = db.query(Document).filter(Document.id == doc_id_1).first()
        doc.status = JobStatus.EXTRACTING
        db.commit()

        # Second upload while first is in-flight
        res2 = client.post(
            "/documents",
            files={"file": ("inflight_duplicate.pdf", pdf_bytes, "application/pdf")}
        )
        assert res2.status_code in (200, 202)
        data2 = res2.json()

        assert data2["document_id"] == doc_id_1
        assert data2["status"] == "extracting"
        assert data2["deduplicated"] is True
        # Assert no second job enqueued
        assert mock_process.call_count == 1


def test_upload_different_files_creates_separate_document_rows(client, db):
    """
    Test 3: Uploading a genuinely different file (different bytes) creates a new, separate Document row.
    """
    bytes_a = b"%PDF-1.4 file A unique bytes \xAA\xBB"
    bytes_b = b"%PDF-1.4 file B unique bytes \xCC\xDD"
    hash_a = hashlib.sha256(bytes_a).hexdigest()
    hash_b = hashlib.sha256(bytes_b).hexdigest()

    # Clean up
    for h in (hash_a, hash_b):
        for d in db.query(Document).filter(Document.content_hash == h).all():
            db.query(Fact).filter(Fact.document_id == d.id).delete()
            db.query(Chunk).filter(Chunk.document_id == d.id).delete()
            db.delete(d)
    db.commit()

    with patch("src.api.routes.orchestrator.process_document") as mock_process:
        res_a = client.post(
            "/documents",
            files={"file": ("file_a.pdf", bytes_a, "application/pdf")}
        )
        res_b = client.post(
            "/documents",
            files={"file": ("file_b.pdf", bytes_b, "application/pdf")}
        )

        assert res_a.status_code in (200, 202)
        assert res_b.status_code in (200, 202)

        data_a = res_a.json()
        data_b = res_b.json()

        assert data_a["document_id"] != data_b["document_id"]
        assert data_a["content_hash"] == hash_a
        assert data_b["content_hash"] == hash_b
        assert data_a["deduplicated"] is False
        assert data_b["deduplicated"] is False

        # Verify two distinct rows exist in the database
        db_doc_a = db.query(Document).filter(Document.content_hash == hash_a).first()
        db_doc_b = db.query(Document).filter(Document.content_hash == hash_b).first()
        assert db_doc_a is not None
        assert db_doc_b is not None
        assert db_doc_a.id != db_doc_b.id


def test_reupload_failed_document_resets_and_reprocesses_without_duplicate_row(client, db):
    """
    Test 4: Re-uploading a file whose prior Document is in FAILED status successfully
    resets and reprocesses it rather than creating a duplicate row.
    """
    pdf_bytes = b"%PDF-1.4 failed recovery test bytes \xFF\xEE\xDD"
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Clean up prior test runs
    old_docs = db.query(Document).filter(Document.content_hash == expected_hash).all()
    for d in old_docs:
        db.query(Fact).filter(Fact.document_id == d.id).delete()
        db.query(Chunk).filter(Chunk.document_id == d.id).delete()
        db.delete(d)
    db.commit()

    with patch("src.api.routes.orchestrator.process_document") as mock_process:
        # First upload
        res1 = client.post(
            "/documents",
            files={"file": ("failed_doc.pdf", pdf_bytes, "application/pdf")}
        )
        assert res1.status_code in (200, 202)
        doc_id = res1.json()["document_id"]

        # Simulate job failing after leaving partial Chunk/Fact rows
        doc = db.query(Document).filter(Document.id == doc_id).first()
        doc.status = JobStatus.FAILED
        doc.error_message = "Corrupt PDF stream on page 3"
        failed_chunk = Chunk(document_id=doc_id, page_number=1, raw_text="Orphan chunk")
        db.add(failed_chunk)
        db.flush()
        failed_fact = Fact(
            document_id=doc_id,
            chunk_id=failed_chunk.id,
            entity="FailedEntity",
            attribute="FailedAttr",
            value="0",
            evidence_quote="Failed quote",
            canonical_statement="Failed statement"
        )
        db.add(failed_fact)
        db.commit()

        # Verify old chunks/facts exist before re-upload
        assert db.query(Chunk).filter(Chunk.document_id == doc_id).count() == 1
        assert db.query(Fact).filter(Fact.document_id == doc_id).count() == 1

        mock_process.reset_mock()

        # Re-upload the exact same file
        res2 = client.post(
            "/documents",
            files={"file": ("failed_doc_retry.pdf", pdf_bytes, "application/pdf")}
        )
        assert res2.status_code in (200, 202)
        data2 = res2.json()

        # Assert same document_id is reused
        assert data2["document_id"] == doc_id
        assert data2["status"] == "queued"

        # Assert orchestrator is re-enqueued
        assert mock_process.call_count == 1

        # Assert no duplicate Document row was created
        matching_docs = db.query(Document).filter(Document.content_hash == expected_hash).all()
        assert len(matching_docs) == 1

        # Assert old chunk and fact rows were purged
        assert db.query(Chunk).filter(Chunk.document_id == doc_id).count() == 0
        assert db.query(Fact).filter(Fact.document_id == doc_id).count() == 0

        # Assert error_message was reset
        reloaded_doc = db.query(Document).filter(Document.id == doc_id).first()
        assert reloaded_doc.status == JobStatus.QUEUED
        assert reloaded_doc.error_message is None
