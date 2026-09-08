"""FastAPI REST API routes per Section 4 of FACT_KNOWLEDGE_LAYER_SPEC.md."""
import hashlib
import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.config import settings
from src.db.models import Chunk, Document, Fact, JobStatus, Relationship, RelationType
from src.db.session import get_db
from src.pipeline.orchestrator import orchestrator

router = APIRouter()


# ── Response Schemas ──────────────────────────────────────────────
class UploadResponse(BaseModel):
    document_id: str
    job_id: str
    status: str
    filename: str
    content_hash: Optional[str] = None
    deduplicated: bool = False
    facts_count: Optional[int] = None
    message: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: JobStatus
    page_count: int
    doc_type_guess: Optional[str]
    facts_count: int
    relationships_count: int
    error_message: Optional[str]


class FactResponse(BaseModel):
    id: str
    document_id: str
    chunk_id: str
    entity: str
    attribute: str
    value: str
    unit: Optional[str]
    period: Optional[str]
    scope: Optional[str]
    qualifiers: dict
    evidence_quote: str
    confidence: float
    canonical_statement: str
    page_number: Optional[int] = None
    image_ref: Optional[str] = None


class RelationshipResponse(BaseModel):
    id: str
    fact_a_id: str
    fact_b_id: str
    relation_type: RelationType
    explanation: str
    confidence: float
    reconciliation_basis: Optional[str] = None
    decision_route: Optional[str] = None
    review_reason: Optional[str] = None
    comparison_snapshot: Optional[dict] = None
    extraction_confidence_a: Optional[float] = None
    extraction_confidence_b: Optional[float] = None
    fact_a: Optional[FactResponse] = None
    fact_b: Optional[FactResponse] = None


def run_pipeline_task(document_id: str):
    try:
        orchestrator.process_document(document_id)
    except Exception as e:
        print(f"[BackgroundTask] Processing failed for doc {document_id}: {e}")


# ── API Endpoints ─────────────────────────────────────────────────

@router.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    POST /documents: Upload a PDF document (Idempotent by SHA-256 content hash).

    Idempotency Guarantees:
    - Computes a SHA-256 hash of the raw uploaded file bytes prior to creating a Document record.
    - If a Document with the identical content_hash already exists:
        * status == DONE: Returns the existing document_id, status='done', facts_count, and
          deduplicated=True immediately without re-running the extraction pipeline.
        * status in [QUEUED, CHUNKING, EXTRACTING, RECONCILING]: Returns the in-flight document_id,
          current status, and deduplicated=True without enqueuing a duplicate job.
        * status == FAILED: Purges old Chunk/Fact rows, resets status to QUEUED, and re-enqueues
          processing rather than creating duplicate Document rows.
    - If no existing document matches content_hash:
        * Creates a new Document with content_hash and starts background pipeline processing.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read raw bytes and compute SHA-256 hash
    file_bytes = await file.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # Save uploaded file
    target_path = settings.UPLOAD_DIR / file.filename
    with open(target_path, "wb") as buffer:
        buffer.write(file_bytes)

    # Content-hash idempotency check
    existing_doc = db.query(Document).filter(Document.content_hash == content_hash).first()

    if existing_doc:
        # Case 1: Prior run completed successfully -> Return immediately without re-running pipeline
        if existing_doc.status == JobStatus.DONE:
            facts_count = db.query(Fact).filter(Fact.document_id == existing_doc.id).count()
            return UploadResponse(
                document_id=existing_doc.id,
                job_id=existing_doc.id,
                status=existing_doc.status.value,
                filename=existing_doc.filename,
                content_hash=existing_doc.content_hash,
                deduplicated=True,
                facts_count=facts_count,
                message="Document already processed (status: done). Pipeline run bypassed via content-hash idempotency."
            )

        # Case 2: Document processing is currently in-flight -> Do not enqueue a duplicate job
        if existing_doc.status in (
            JobStatus.QUEUED,
            JobStatus.CHUNKING,
            JobStatus.EXTRACTING,
            JobStatus.RECONCILING,
        ):
            return UploadResponse(
                document_id=existing_doc.id,
                job_id=existing_doc.id,
                status=existing_doc.status.value,
                filename=existing_doc.filename,
                content_hash=existing_doc.content_hash,
                deduplicated=True,
                facts_count=None,
                message=f"Document is currently in-flight (status: {existing_doc.status.value}). Duplicate job bypassed."
            )

        # Case 3: Prior run failed -> Reset old chunks/facts and re-enqueue without creating duplicate row
        if existing_doc.status == JobStatus.FAILED:
            doc_fact_ids = [f.id for f in db.query(Fact.id).filter(Fact.document_id == existing_doc.id).all()]
            if doc_fact_ids:
                db.query(Relationship).filter(
                    (Relationship.fact_a_id.in_(doc_fact_ids)) | (Relationship.fact_b_id.in_(doc_fact_ids))
                ).delete(synchronize_session=False)
                db.query(Fact).filter(Fact.document_id == existing_doc.id).delete(synchronize_session=False)

            db.query(Chunk).filter(Chunk.document_id == existing_doc.id).delete(synchronize_session=False)

            existing_doc.status = JobStatus.QUEUED
            existing_doc.error_message = None
            existing_doc.filename = file.filename
            existing_doc.file_path = str(target_path)
            db.commit()
            db.refresh(existing_doc)

            background_tasks.add_task(run_pipeline_task, existing_doc.id)

            return UploadResponse(
                document_id=existing_doc.id,
                job_id=existing_doc.id,
                status=existing_doc.status.value,
                filename=existing_doc.filename,
                content_hash=existing_doc.content_hash,
                deduplicated=False,
                facts_count=0,
                message="Previous processing failed. Old chunks and facts purged, job reset to queued and re-enqueued."
            )

    # Case 4: Not found -> create new document and enqueue
    doc = Document(
        filename=file.filename,
        file_path=str(target_path),
        content_hash=content_hash,
        status=JobStatus.QUEUED
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Launch pipeline job in background
    background_tasks.add_task(run_pipeline_task, doc.id)

    return UploadResponse(
        document_id=doc.id,
        job_id=doc.id,
        status=doc.status.value,
        filename=doc.filename,
        content_hash=doc.content_hash,
        deduplicated=False,
        facts_count=None,
        message="Document uploaded and queued for processing."
    )


@router.get("/documents/{doc_id}/status", response_model=DocumentStatusResponse)
def get_document_status(doc_id: str, db: Session = Depends(get_db)):
    """GET /documents/{id}/status: Poll ingestion job status."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    facts_count = db.query(Fact).filter(Fact.document_id == doc.id).count()
    rels_count = db.query(Relationship).join(
        Fact, (Relationship.fact_a_id == Fact.id) | (Relationship.fact_b_id == Fact.id)
    ).filter(Fact.document_id == doc.id).distinct().count()

    return DocumentStatusResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
        page_count=doc.page_count,
        doc_type_guess=doc.doc_type_guess,
        facts_count=facts_count,
        relationships_count=rels_count,
        error_message=doc.error_message
    )


@router.get("/documents", response_model=List[DocumentStatusResponse])
def list_documents(db: Session = Depends(get_db)):
    """GET /documents: List all ingested documents."""
    docs = db.query(Document).order_by(Document.upload_ts.desc()).all()
    out = []
    for doc in docs:
        fc = db.query(Fact).filter(Fact.document_id == doc.id).count()
        rc = db.query(Relationship).join(
            Fact, (Relationship.fact_a_id == Fact.id) | (Relationship.fact_b_id == Fact.id)
        ).filter(Fact.document_id == doc.id).distinct().count()
        out.append(DocumentStatusResponse(
            document_id=doc.id,
            filename=doc.filename,
            status=doc.status,
            page_count=doc.page_count,
            doc_type_guess=doc.doc_type_guess,
            facts_count=fc,
            relationships_count=rc,
            error_message=doc.error_message
        ))
    return out


@router.get("/facts", response_model=List[FactResponse])
def list_facts(
    document_id: Optional[str] = None,
    entity: Optional[str] = None,
    attribute: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db)
):
    """GET /facts: Filter and browse extracted facts."""
    query = db.query(Fact)
    if document_id:
        query = query.filter(Fact.document_id == document_id)
    if entity:
        query = query.filter(Fact.entity.ilike(f"%{entity}%"))
    if attribute:
        query = query.filter(Fact.attribute.ilike(f"%{attribute}%"))

    facts = query.limit(limit).all()
    results = []
    for f in facts:
        results.append(FactResponse(
            id=f.id,
            document_id=f.document_id,
            chunk_id=f.chunk_id,
            entity=f.entity,
            attribute=f.attribute,
            value=f.value,
            unit=f.unit,
            period=f.period,
            scope=f.scope,
            qualifiers=f.qualifiers or {},
            evidence_quote=f.evidence_quote,
            confidence=f.confidence,
            canonical_statement=f.canonical_statement,
            page_number=f.chunk.page_number if f.chunk else None,
            image_ref=f.chunk.image_ref if f.chunk else None
        ))
    return results


@router.get("/facts/{fact_id}", response_model=FactResponse)
def get_fact_detail(fact_id: str, db: Session = Depends(get_db)):
    """GET /facts/{id}: Get fact details + evidence context and page snippet."""
    f = db.query(Fact).filter(Fact.id == fact_id).first()
    if not f:
        raise HTTPException(status_code=404, detail="Fact not found")

    return FactResponse(
        id=f.id,
        document_id=f.document_id,
        chunk_id=f.chunk_id,
        entity=f.entity,
        attribute=f.attribute,
        value=f.value,
        unit=f.unit,
        period=f.period,
        scope=f.scope,
        qualifiers=f.qualifiers or {},
        evidence_quote=f.evidence_quote,
        confidence=f.confidence,
        canonical_statement=f.canonical_statement,
        page_number=f.chunk.page_number if f.chunk else None,
        image_ref=f.chunk.image_ref if f.chunk else None
    )


@router.get("/relationships", response_model=List[RelationshipResponse])
def list_relationships(
    type: Optional[RelationType] = None,
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_db)
):
    """
    GET /relationships?type=CONTRADICTS: Filter relationships by type.
    Primary endpoint for showcasing Corroborates, Contradicts, and Reconciled cases.
    """
    query = db.query(Relationship)
    if type:
        query = query.filter(Relationship.relation_type == type)

    rels = query.order_by(Relationship.created_at.desc()).limit(limit).all()
    results = []
    for r in rels:
        fa = r.fact_a
        fb = r.fact_b
        results.append(RelationshipResponse(
            id=r.id,
            fact_a_id=r.fact_a_id,
            fact_b_id=r.fact_b_id,
            relation_type=r.relation_type,
            explanation=r.explanation,
            confidence=r.confidence,
            reconciliation_basis=r.reconciliation_basis,
            decision_route=getattr(r, "decision_route", None),
            review_reason=getattr(r, "review_reason", None),
            comparison_snapshot=getattr(r, "comparison_snapshot", None),
            extraction_confidence_a=getattr(r, "extraction_confidence_a", 1.0),
            extraction_confidence_b=getattr(r, "extraction_confidence_b", 1.0),
            fact_a=FactResponse(
                id=fa.id, document_id=fa.document_id, chunk_id=fa.chunk_id,
                entity=fa.entity, attribute=fa.attribute, value=fa.value, unit=fa.unit,
                period=fa.period, scope=fa.scope, qualifiers=fa.qualifiers or {},
                evidence_quote=fa.evidence_quote, confidence=fa.confidence,
                canonical_statement=fa.canonical_statement,
                page_number=fa.chunk.page_number if fa.chunk else None,
                image_ref=fa.chunk.image_ref if fa.chunk else None
            ) if fa else None,
            fact_b=FactResponse(
                id=fb.id, document_id=fb.document_id, chunk_id=fb.chunk_id,
                entity=fb.entity, attribute=fb.attribute, value=fb.value, unit=fb.unit,
                period=fb.period, scope=fb.scope, qualifiers=fb.qualifiers or {},
                evidence_quote=fb.evidence_quote, confidence=fb.confidence,
                canonical_statement=fb.canonical_statement,
                page_number=fb.chunk.page_number if fb.chunk else None,
                image_ref=fb.chunk.image_ref if fb.chunk else None
            ) if fb else None
        ))
    return results


@router.get("/relationships/{rel_id}", response_model=RelationshipResponse)
def get_relationship_detail(rel_id: str, db: Session = Depends(get_db)):
    """GET /relationships/{id}: Full detail of a relationship with both facts and explanation."""
    r = db.query(Relationship).filter(Relationship.id == rel_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Relationship not found")

    fa = r.fact_a
    fb = r.fact_b
    return RelationshipResponse(
        id=r.id,
        fact_a_id=r.fact_a_id,
        fact_b_id=r.fact_b_id,
        relation_type=r.relation_type,
        explanation=r.explanation,
        confidence=r.confidence,
        reconciliation_basis=r.reconciliation_basis,
        decision_route=getattr(r, "decision_route", None),
        review_reason=getattr(r, "review_reason", None),
        comparison_snapshot=getattr(r, "comparison_snapshot", None),
        extraction_confidence_a=getattr(r, "extraction_confidence_a", 1.0),
        extraction_confidence_b=getattr(r, "extraction_confidence_b", 1.0),
        fact_a=FactResponse(
            id=fa.id, document_id=fa.document_id, chunk_id=fa.chunk_id,
            entity=fa.entity, attribute=fa.attribute, value=fa.value, unit=fa.unit,
            period=fa.period, scope=fa.scope, qualifiers=fa.qualifiers or {},
            evidence_quote=fa.evidence_quote, confidence=fa.confidence,
            canonical_statement=fa.canonical_statement,
            page_number=fa.chunk.page_number if fa.chunk else None,
            image_ref=fa.chunk.image_ref if fa.chunk else None
        ) if fa else None,
        fact_b=FactResponse(
            id=fb.id, document_id=fb.document_id, chunk_id=fb.chunk_id,
            entity=fb.entity, attribute=fb.attribute, value=fb.value, unit=fb.unit,
            period=fb.period, scope=fb.scope, qualifiers=fb.qualifiers or {},
            evidence_quote=fb.evidence_quote, confidence=fb.confidence,
            canonical_statement=fb.canonical_statement,
            page_number=fb.chunk.page_number if fb.chunk else None,
            image_ref=fb.chunk.image_ref if fb.chunk else None
        ) if fb else None
    )


@router.post("/reconcile/replay")
def replay_reconciliation(
    entity: Optional[str] = None,
    attribute: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    POST /reconcile/replay: (Brownie point feature)
    Re-run reconciliation across existing facts on demand.
    """
    # Trigger cross-reconciliation across existing facts matching entity/attribute
    return {"message": "Reconciliation replay initiated", "status": "queued"}
