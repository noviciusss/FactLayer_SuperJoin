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
    """LLM adjudication decision between two candidate facts."""
    relation_type: RelationType = Field(
        description="CORROBORATES (same fact / agreeing numbers), CONTRADICTS (incompatible assertions), RECONCILED (apparent conflict explained by scope/timing/accounting/methodology), or UNRELATED (different facts)"
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
        description="If RECONCILED, state the root cause: e.g. 'accounting definition bridge (ESOP and lease add-backs)', 'pro forma restatement vs historical actuals', 'different publication vintage'"
    )
