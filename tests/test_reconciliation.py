"""Unit tests for Stage E: Reconciliation engine, Ambiguity Firewall, and tolerance (T-05 to T-12)."""
from decimal import Decimal
from unittest.mock import patch
import pytest

from src.db.models import Fact, RelationType
from src.pipeline.reconciliation import (
    extract_numeric,
    reconciliation_engine,
    values_are_close,
)


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
    assert decision.decision_route in ["deterministic_exact", "deterministic_rounding"]


def test_t06_missing_period_routes_to_needs_review():
    """T-06: Same numerical metric but missing period on either fact must trigger Ambiguity Firewall."""
    fact_a = Fact(
        id="f-period-1",
        chunk_id="c-1",
        document_id="doc-1",
        entity="Acme Corp",
        attribute="Revenue",
        value="500",
        unit="USD",
        period="FY24",
        scope="consolidated",
        evidence_quote="Acme Corp FY24 revenue was 500 USD",
        confidence=0.9
    )
    fact_b = Fact(
        id="f-period-2",
        chunk_id="c-2",
        document_id="doc-2",
        entity="Acme Corp",
        attribute="Revenue",
        value="500",
        unit="USD",
        period=None,  # Missing period!
        scope="consolidated",
        evidence_quote="Revenue stood at 500 USD",
        confidence=0.9
    )

    decision = reconciliation_engine.reconcile_pair(fact_a, fact_b)
    assert decision.relation_type == RelationType.NEEDS_REVIEW
    assert decision.decision_route == "ambiguity_firewall"
    assert decision.review_reason == "missing_period"
    assert "Ambiguity Firewall" in decision.explanation


def test_t07_material_difference_missing_scope_needs_review():
    """T-07: Materially different values with missing scope must route to NEEDS_REVIEW, never silent CONTRADICTS."""
    fact_a = Fact(
        id="f-ebitda-1",
        chunk_id="c-1",
        document_id="doc-1",
        entity="Delhivery",
        attribute="EBITDA",
        value="127",
        unit="₹ Cr",
        period="FY24",
        scope="reported",
        evidence_quote="Reported EBITDA of ₹127 Cr in FY24",
        confidence=0.95
    )
    fact_b = Fact(
        id="f-ebitda-2",
        chunk_id="c-2",
        document_id="doc-2",
        entity="Delhivery",
        attribute="EBITDA",
        value="76",
        unit="₹ Cr",
        period="FY24",
        scope=None,  # Missing scope!
        evidence_quote="EBITDA was ₹76 Cr in FY24",
        confidence=0.90
    )

    decision = reconciliation_engine.reconcile_pair(fact_a, fact_b)
    assert decision.relation_type == RelationType.NEEDS_REVIEW
    assert decision.relation_type != RelationType.CONTRADICTS
    assert decision.decision_route == "ambiguity_firewall"
    assert decision.review_reason == "missing_scope_or_basis"
    assert "Ambiguity Firewall" in decision.explanation


def test_t08_unit_mismatch_routes_to_needs_review():
    """T-08: Explicit incompatible units route to NEEDS_REVIEW / unit_ambiguity."""
    fact_a = Fact(
        id="f-unit-1",
        chunk_id="c-1",
        document_id="doc-1",
        entity="TechCorp",
        attribute="Net Profit",
        value="100",
        unit="₹ Cr",
        period="FY24",
        scope="standalone",
        evidence_quote="Net profit was ₹100 Cr",
        confidence=0.95
    )
    fact_b = Fact(
        id="f-unit-2",
        chunk_id="c-2",
        document_id="doc-2",
        entity="TechCorp",
        attribute="Net Profit",
        value="100",
        unit="USD",  # Currency mismatch without exchange rate
        period="FY24",
        scope="standalone",
        evidence_quote="Net profit reached 100 USD",
        confidence=0.95
    )

    decision = reconciliation_engine.reconcile_pair(fact_a, fact_b)
    assert decision.relation_type == RelationType.NEEDS_REVIEW
    assert decision.decision_route == "ambiguity_firewall"
    assert decision.review_reason == "unit_ambiguity"


def test_t09_reported_vs_adjusted_scopes_reconciliation_path():
    """T-09: Reported vs adjusted scopes explicitly supplied must not deterministically contradict."""
    fact_a = Fact(
        id="f-scope-1",
        chunk_id="c-1",
        document_id="doc-1",
        entity="Delhivery",
        attribute="EBITDA",
        value="127",
        unit="₹ Cr",
        period="FY24",
        scope="reported",
        evidence_quote="Reported EBITDA ₹127 Cr",
        confidence=0.95
    )
    fact_b = Fact(
        id="f-scope-2",
        chunk_id="c-2",
        document_id="doc-2",
        entity="Delhivery",
        attribute="EBITDA",
        value="76",
        unit="₹ Cr",
        period="FY24",
        scope="adjusted",
        evidence_quote="Adjusted EBITDA ₹76 Cr after share-based payment adjustments",
        confidence=0.95
    )

    alignment = reconciliation_engine.evaluate_alignment(fact_a, fact_b)
    # Both scopes exist and differ, so alignment['scope_match'] is False
    assert alignment["has_scope_a"] is True
    assert alignment["has_scope_b"] is True
    assert alignment["scope_match"] is False
    # Deterministic rule must NOT declare them corroborates or contradiction
    assert reconciliation_engine.check_deterministic_rules(fact_a, fact_b) is None


def test_t10_malformed_llm_adjudication_fallback():
    """T-10: LLM timeout or malformed output safely yields NEEDS_REVIEW without crashing."""
    fact_a = Fact(
        id="f-llm-1",
        chunk_id="c-1",
        document_id="doc-1",
        entity="MacroCorp",
        attribute="Growth Rate",
        value="7.2",
        unit="%",
        period="FY24",
        scope="reported",
        evidence_quote="GDP growth reported at 7.2%",
        confidence=0.95
    )
    fact_b = Fact(
        id="f-llm-2",
        chunk_id="c-2",
        document_id="doc-2",
        entity="MacroCorp",
        attribute="Growth Rate",
        value="7.8",
        unit="%",
        period="FY24",
        scope="revised",
        evidence_quote="Revised GDP growth estimated at 7.8%",
        confidence=0.95
    )

    with patch("src.pipeline.reconciliation.llm_client.extract_structured", side_effect=RuntimeError("Groq 429 Rate Limit")):
        decision = reconciliation_engine.reconcile_pair(fact_a, fact_b)
        assert decision.relation_type == RelationType.NEEDS_REVIEW
        assert decision.decision_route == "ambiguity_firewall"
        assert decision.review_reason == "adjudication_unavailable"
        assert "Ambiguity Firewall" in decision.explanation


def test_t11_relationship_decision_metadata_snapshot():
    """T-11: Relationship decision metadata snapshot contains frozen comparison fields."""
    fact_a = Fact(
        id="fa-snap",
        chunk_id="ca-snap",
        document_id="doc-1",
        entity="Global Logistics",
        attribute="Active Hubs",
        value="45",
        unit="facilities",
        period="FY24",
        scope="consolidated",
        evidence_quote="Operated 45 automated facilities in FY24",
        confidence=0.95
    )
    fact_b = Fact(
        id="fb-snap",
        chunk_id="cb-snap",
        document_id="doc-2",
        entity="Global Logistics",
        attribute="Active Hubs",
        value="45",
        unit="facilities",
        period="FY24",
        scope="consolidated",
        evidence_quote="45 automated distribution hubs in FY24",
        confidence=0.95
    )

    decision = reconciliation_engine.reconcile_pair(fact_a, fact_b)
    assert decision.comparison_snapshot is not None
    snap = decision.comparison_snapshot
    assert "fact_a" in snap and "fact_b" in snap
    assert "alignment" in snap and "tolerance" in snap
    assert snap["fact_a"]["value"] == "45"
    assert snap["fact_b"]["value"] == "45"
    assert snap["alignment"]["period_match"] is True


def test_t12_numeric_decimals_deterministic():
    """T-12: Decimal parsing handles currency, commas, percentages, and scales with exact precision."""
    assert extract_numeric("₹8,142.35 Cr") == Decimal("8142.35")
    assert extract_numeric("$1,500,000") == Decimal("1500000")
    assert extract_numeric("8.2%") == Decimal("8.2")
    assert extract_numeric("-12.45") == Decimal("-12.45")
    assert extract_numeric(None) is None
    assert extract_numeric("Text with no numbers") is None


def test_adjudication_uses_adjudication_model():
    """TASK 2: Verify LLM adjudication explicitly passes ADJUDICATION_MODEL and validates schema."""
    from unittest.mock import patch
    from src.config import settings
    from src.pipeline.schemas import ReconciliationDecision

    fact_a = Fact(
        id="f-adj-1",
        chunk_id="c-1",
        document_id="doc-1",
        entity="Delhivery",
        attribute="EBITDA",
        value="127",
        unit="₹ Cr",
        period="FY24",
        scope="reported",
        evidence_quote="Reported EBITDA was ₹127 Cr",
        confidence=0.95
    )
    fact_b = Fact(
        id="f-adj-2",
        chunk_id="c-2",
        document_id="doc-2",
        entity="Delhivery",
        attribute="EBITDA",
        value="76",
        unit="₹ Cr",
        period="FY24",
        scope="adjusted",
        evidence_quote="Adjusted EBITDA was ₹76 Cr",
        confidence=0.95
    )

    mock_resp = ReconciliationDecision(
        relation_type=RelationType.RECONCILED,
        explanation="Reconciled via ESOP add-backs",
        confidence=0.92,
        reconciliation_basis="Accounting scope adjustment"
    )

    with patch("src.pipeline.reconciliation.llm_client.extract_structured", return_value=mock_resp) as mock_extract:
        decision = reconciliation_engine.reconcile_pair(fact_a, fact_b)

        assert mock_extract.called
        call_kwargs = mock_extract.call_args.kwargs
        assert call_kwargs.get("model") == settings.ADJUDICATION_MODEL
        assert decision.relation_type == RelationType.RECONCILED
        assert decision.decision_route == "llm_adjudication"
        assert decision.reconciliation_basis == "Accounting scope adjustment"

