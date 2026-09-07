"""Pydantic schemas for LLM structured output and validation."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.db.models import RelationType


class ExtractedFact(BaseModel):
    """Generic, unconstrained fact assertion extracted from a chunk."""
    entity: str = Field(
        description="The subject entity of the fact, e.g. 'Delhivery Limited', 'India', 'Reserve Bank of India'"
    )
    attribute: str = Field(
        description="The specific metric, event, or attribute, e.g. 'Revenue from operations', 'EBITDA', 'Real GDP growth rate', 'Executive Director appointment'"
    )
    value: str = Field(
        description="The asserted value, numeric or textual, e.g. '8142', '76', '8.2', 'resigned'"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement if applicable, e.g. '₹ Cr', '%', 'million tons', 'USD'"
    )
    period: Optional[str] = Field(
        default=None,
        description="Time period or point in time, e.g. 'FY24', 'Q4 FY24', 'FY 2023-24', 'as of March 31, 2024'. Null if not stated."
    )
    scope: Optional[str] = Field(
        default=None,
        description="Accounting or operational scope, e.g. 'consolidated', 'standalone', 'adjusted', 'reported', 'pro forma'. Null if not stated."
    )
    qualifiers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic escape hatch: any additional qualifiers, methodologies, footnotes, or context"
    )
    evidence_quote: str = Field(
        description="Exact verbatim span of text from the source chunk (<= 30 words) that directly supports this fact."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0. Lower if period or scope was ambiguous."
    )


class FactExtractionResponse(BaseModel):
    """Container for batch facts extracted from a single chunk."""
    facts: List[ExtractedFact] = Field(default_factory=list)


class ReconciliationDecision(BaseModel):
    """Reconciliation decision between two candidate facts, deterministic or LLM-adjudicated."""
    relation_type: RelationType = Field(
        description="CORROBORATES, CONTRADICTS, RECONCILED, UNRELATED, or NEEDS_REVIEW"
    )
    explanation: str = Field(
        description="Thorough explanation grounded in both facts' values, scopes, periods, units, and evidence quotes"
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score in this adjudication"
    )
    reconciliation_basis: Optional[str] = Field(
        default=None,
        description="If RECONCILED, state the root cause: e.g. 'accounting definition bridge', 'pro forma restatement', etc."
    )
    decision_route: Optional[str] = Field(
        default=None,
        description="Decision source: deterministic_exact, deterministic_rounding, ambiguity_firewall, llm_adjudication, irrelevant_pre_filter"
    )
    review_reason: Optional[str] = Field(
        default=None,
        description="Reason for NEEDS_REVIEW: missing_period, missing_scope_or_basis, unit_ambiguity, header_context_uncertain, adjudication_unavailable"
    )
    comparison_snapshot: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Frozen snapshot of compared attributes at decision time"
    )
