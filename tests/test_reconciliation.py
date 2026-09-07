"""Unit tests for Stage E: Reconciliation rules and tolerance check (T-05)."""
import pytest
from src.db.models import Fact, RelationType
from src.pipeline.reconciliation import reconciliation_engine, values_are_close


def test_t05_numeric_tolerance_corroborates():
    """T-05: Numeric tolerance rule doesn't flag ₹8,142 Cr vs ₹8,142.3 Cr as contradiction."""
    assert values_are_close("8,142", "8,142.3") is True
    assert values_are_close("₹8142 Cr", "8142.3") is True
    assert values_are_close("₹8,142 Cr", "₹8,142 Cr") is True
    # Values that differ by more than tolerance should not be close
    assert values_are_close("8,142", "7,054") is False


def test_deterministic_corroboration_rule():
    """Verifies that matching metrics with slight rounding differences corroborate without LLM."""
    fact_a = Fact(
        id="fact-1",
        chunk_id="chunk-1",
        document_id="doc-1",
        entity="Delhivery Limited",
        attribute="Revenue from operations",
        value="8142",
        unit="₹ Cr",
        period="FY24",
        scope="consolidated",
        evidence_quote="revenue from operations for FY24 stood at ₹8,142 Cr",
        confidence=0.95
    )

    fact_b = Fact(
        id="fact-2",
        chunk_id="chunk-2",
        document_id="doc-2",
        entity="Delhivery",
        attribute="Revenue from operations",
        value="8142.3",
        unit="₹ Cr",
        period="FY24",
        scope="consolidated",
        evidence_quote="Revenue from operations was ₹8,142.3 crore in FY24",
        confidence=0.95
    )

    decision = reconciliation_engine.check_deterministic_rules(fact_a, fact_b)
    assert decision is not None
    assert decision.relation_type == RelationType.CORROBORATES
    assert decision.confidence >= 0.95
    assert "Rule-based corroboration" in decision.explanation
