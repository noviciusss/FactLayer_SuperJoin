"""SQLAlchemy ORM models matching FACT_KNOWLEDGE_LAYER_SPEC.md."""
import enum
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    RECONCILING = "reconciling"
    DONE = "done"
    FAILED = "failed"


class RelationType(str, enum.Enum):
    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"
    RECONCILED = "RECONCILED"
    UNRELATED = "UNRELATED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=True)
    page_count = Column(Integer, default=0)
    doc_type_guess = Column(String(100), nullable=True)  # e.g., "financial_filing", "macro_report"
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED, nullable=False)
    error_message = Column(Text, nullable=True)
    upload_ts = Column(DateTime, default=datetime.utcnow, nullable=False)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    facts = relationship("Fact", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    char_start = Column(Integer, default=0)
    char_end = Column(Integer, default=0)
    raw_text = Column(Text, nullable=False)
    is_table = Column(Boolean, default=False)
    image_ref = Column(String(512), nullable=True)  # Path to page preview image or visual clip

    document = relationship("Document", back_populates="chunks")
    facts = relationship("Fact", back_populates="chunk", cascade="all, delete-orphan")


class Fact(Base):
    __tablename__ = "facts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    chunk_id = Column(String(36), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # Core generic attributes
    entity = Column(String(255), nullable=False, index=True)
    attribute = Column(String(255), nullable=False, index=True)
    value = Column(String(255), nullable=False)
    unit = Column(String(100), nullable=True)
    period = Column(String(100), nullable=True)
    scope = Column(String(100), nullable=True)

    # Dynamic schema escape hatch
    qualifiers = Column(JSON, default=dict)

    # Grounding & Quality
    evidence_quote = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    canonical_statement = Column(Text, nullable=False)
    content_hash = Column(String(64), index=True, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="facts")
    chunk = relationship("Chunk", back_populates="facts")


class Relationship(Base):
    __tablename__ = "relationships"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    fact_a_id = Column(String(36), ForeignKey("facts.id", ondelete="CASCADE"), nullable=False)
    fact_b_id = Column(String(36), ForeignKey("facts.id", ondelete="CASCADE"), nullable=False)

    relation_type = Column(Enum(RelationType), nullable=False, index=True)
    explanation = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    reconciliation_basis = Column(Text, nullable=True)

    # Auditable Decision Ledger fields
    decision_route = Column(String(100), nullable=True)  # e.g., deterministic_exact, ambiguity_firewall, etc.
    review_reason = Column(String(100), nullable=True)   # e.g., missing_period, missing_scope_or_basis, etc.
    comparison_snapshot = Column(JSON, default=dict)     # Snapshot of compared attributes at decision time
    extraction_confidence_a = Column(Float, default=1.0)
    extraction_confidence_b = Column(Float, default=1.0)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    fact_a = relationship("Fact", foreign_keys=[fact_a_id])
    fact_b = relationship("Fact", foreign_keys=[fact_b_id])
