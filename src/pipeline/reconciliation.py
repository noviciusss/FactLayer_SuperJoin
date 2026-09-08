"""Stage E: Reconciliation Engine implementing rules-first, Ambiguity Firewall, and LLM cascade."""
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.db.models import Fact, RelationType
from src.pipeline.llm_client import llm_client
from src.pipeline.schemas import ReconciliationDecision


def extract_numeric(val: Any) -> Optional[Decimal]:
    """Clean string and parse numeric value as high-precision Decimal if present."""
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str:
        return None

    # Remove currency symbols, commas, and common unit keywords
    cleaned = re.sub(r"[₹$,€£]", "", val_str)
    # Remove unit words that might be attached
    cleaned = re.sub(r"\b(cr|crore|crores|lakh|lakhs|mn|million|billion|%)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if match:
        try:
            return Decimal(match.group())
        except (InvalidOperation, ValueError):
            return None
    return None


def get_numeric_diff(val_a: Any, val_b: Any) -> Optional[Decimal]:
    """Calculate absolute difference between two numeric values if both can be parsed."""
    num_a = extract_numeric(val_a)
    num_b = extract_numeric(val_b)
    if num_a is not None and num_b is not None:
        return abs(num_a - num_b)
    return None


def values_are_close(
    val_a: Any,
    val_b: Any,
    max_pct_diff: Decimal = Decimal("0.005"),
    abs_tolerance: Decimal = Decimal("1.0")
) -> bool:
    """
    Check if two values match within numeric rounding tolerance (T-05, T-12).
    Uses Decimal arithmetic to prevent floating-point inaccuracy.
    Returns True if values match within tolerance, False otherwise.
    """
    str_a = str(val_a).strip().lower() if val_a is not None else ""
    str_b = str(val_b).strip().lower() if val_b is not None else ""

    if str_a and str_a == str_b:
        return True

    num_a = extract_numeric(val_a)
    num_b = extract_numeric(val_b)

    if num_a is not None and num_b is not None:
        if num_a == num_b:
            return True
        diff = abs(num_a - num_b)
        if diff <= abs_tolerance:
            return True
        avg = (abs(num_a) + abs(num_b)) / Decimal("2")
        if avg > Decimal("0") and (diff / avg) <= max_pct_diff:
            return True
        return False

    return False


def normalize_metric(text: Optional[str]) -> str:
    """Standardize entity/attribute names for fuzzy comparison without hardcoded domain bias."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def normalize_unit(unit: Optional[str]) -> str:
    """Standardize unit strings for equivalence checks."""
    if not unit:
        return ""
    u = unit.lower().strip()
    u = re.sub(r"[^\w%]", "", u)
    # Common standardizations
    if u in ["cr", "crore", "crores", "inrcr", "rs cr"]:
        return "inr_cr"
    if u in ["lakh", "lakhs", "inrlakh"]:
        return "inr_lakh"
    if u in ["percent", "pct", "%"]:
        return "%"
    if u in ["usd", "dollar", "dollars"]:
        return "usd"
    return u


def make_comparison_snapshot(fact_a: Fact, fact_b: Fact, alignment: Dict[str, Any]) -> Dict[str, Any]:
    """Create a persistent, auditable snapshot of facts compared at decision time."""
    return {
        "fact_a": {
            "id": getattr(fact_a, "id", None),
            "document_id": getattr(fact_a, "document_id", None),
            "entity": getattr(fact_a, "entity", ""),
            "attribute": getattr(fact_a, "attribute", ""),
            "value": str(getattr(fact_a, "value", "")),
            "unit": getattr(fact_a, "unit", None),
            "period": getattr(fact_a, "period", None),
            "scope": getattr(fact_a, "scope", None),
            "qualifiers": getattr(fact_a, "qualifiers", {}) or {},
            "confidence": getattr(fact_a, "confidence", 1.0),
        },
        "fact_b": {
            "id": getattr(fact_b, "id", None),
            "document_id": getattr(fact_b, "document_id", None),
            "entity": getattr(fact_b, "entity", ""),
            "attribute": getattr(fact_b, "attribute", ""),
            "value": str(getattr(fact_b, "value", "")),
            "unit": getattr(fact_b, "unit", None),
            "period": getattr(fact_b, "period", None),
            "scope": getattr(fact_b, "scope", None),
            "qualifiers": getattr(fact_b, "qualifiers", {}) or {},
            "confidence": getattr(fact_b, "confidence", 1.0),
        },
        "alignment": alignment,
        "tolerance": {"relative": 0.005, "absolute": 1.0}
    }


RECONCILIATION_SYSTEM_PROMPT = """You are DealGuard's contextual financial and corporate fact reconciliation engine.
You are evaluating two factual claims extracted from corporate filings, prospectuses, annual reports, or macroeconomic publications.

DECISION PRINCIPLES:
1. "A numerical difference is not automatically a contradiction; it first has to be the same claim."
2. If two facts assert the same metric for the same entity and period, and values match within rounding: CORROBORATES.
3. If two facts differ in scope or basis (e.g. Reported EBITDA vs Adjusted EBITDA, Pro Forma restatement vs Historical Actuals, Consolidated vs Standalone), DO NOT assert a contradiction. Classify as RECONCILED and state the accounting/scope bridge.
4. Only classify as CONTRADICTS when entity, attribute, period, scope, and unit are identical and there is a genuine, irreconcilable discrepancy with no reconciling footnote or restatement basis.
5. If the comparison lacks vital contextual reporting basis or is fundamentally ambiguous, return NEEDS_REVIEW.
6. If the claims describe distinct topics, return UNRELATED.

ALLOWED RELATION TYPES:
- CORROBORATES
- CONTRADICTS
- RECONCILED
- UNRELATED
- NEEDS_REVIEW

Provide a detailed explanation grounded in both facts' values, scopes, periods, units, and evidence quotes.
If RECONCILED, identify the exact reconciliation_basis.
"""


class ReconciliationEngine:
    """
    Evidence-First Fact Reconciliation Engine with Ambiguity Firewall.
    Cascade:
    1. Deterministic Exact / Rounding Match -> CORROBORATES
    2. Irrelevant Pre-Filter -> UNRELATED
    3. Ambiguity Firewall -> NEEDS_REVIEW (missing_period, missing_scope_or_basis, unit_ambiguity, header_context_uncertain)
    4. Contextual LLM Adjudication -> RECONCILED / CONTRADICTS
    5. Fallback on Error -> NEEDS_REVIEW (adjudication_unavailable)
    """

    def _check_entity_overlap(self, ent_a: str, ent_b: str) -> bool:
        """Domain-agnostic entity equivalence check using token intersection."""
        if not ent_a or not ent_b:
            return False
        if ent_a == ent_b or ent_a in ent_b or ent_b in ent_a:
            return True
        stop_words = {"ltd", "limited", "inc", "corp", "corporation", "co", "company", "the", "of", "and"}
        words_a = set(ent_a.split()) - stop_words
        words_b = set(ent_b.split()) - stop_words
        return bool(words_a.intersection(words_b))

    def _check_attribute_overlap(self, attr_a: str, attr_b: str) -> bool:
        """Domain-agnostic attribute overlap check with standard financial synonym clusters."""
        if not attr_a or not attr_b:
            return False
        if attr_a == attr_b or attr_a in attr_b or attr_b in attr_a:
            return True
        stop_words = {"of", "the", "in", "for", "and", "on", "at", "to", "a", "an", "from", "by", "is", "as", "total"}
        words_a = set(attr_a.split()) - stop_words
        words_b = set(attr_b.split()) - stop_words
        if bool(words_a.intersection(words_b)):
            return True

        # Common financial and macroeconomic synonym clusters
        synonyms = [
            {"revenue", "turnover", "sales", "income", "topline"},
            {"ebitda", "operating profit", "operating margin", "operating result"},
            {"pat", "profit", "net profit", "bottomline", "earnings"},
            {"gdp", "growth", "growth rate", "output", "gva"},
            {"headcount", "employees", "workforce", "personnel"},
            {"debt", "borrowings", "liabilities", "leverage"}
        ]
        return any(bool(words_a.intersection(s)) and bool(words_b.intersection(s)) for s in synonyms)

    def evaluate_alignment(self, fact_a: Fact, fact_b: Fact) -> Dict[str, Any]:
        """Perform field-by-field alignment check between two facts."""
        ent_a = normalize_metric(fact_a.entity)
        ent_b = normalize_metric(fact_b.entity)
        attr_a = normalize_metric(fact_a.attribute)
        attr_b = normalize_metric(fact_b.attribute)
        period_a = normalize_metric(fact_a.period)
        period_b = normalize_metric(fact_b.period)
        scope_a = normalize_metric(fact_a.scope)
        scope_b = normalize_metric(fact_b.scope)
        unit_a = normalize_unit(fact_a.unit)
        unit_b = normalize_unit(fact_b.unit)

        is_entity_match = self._check_entity_overlap(ent_a, ent_b)
        is_attr_match = self._check_attribute_overlap(attr_a, attr_b)

        # Period match: both non-empty and matching
        has_period_a = bool(period_a)
        has_period_b = bool(period_b)
        is_period_match = has_period_a and has_period_b and (
            period_a == period_b or period_a in period_b or period_b in period_a
        )

        # Scope match: both non-empty and matching, or both empty
        has_scope_a = bool(scope_a)
        has_scope_b = bool(scope_b)
        is_scope_match = (scope_a == scope_b)

        # Unit match
        has_unit_a = bool(unit_a)
        has_unit_b = bool(unit_b)
        is_unit_match = (unit_a == unit_b) if (has_unit_a and has_unit_b) else True

        # Value comparison
        is_close = values_are_close(fact_a.value, fact_b.value)
        diff = get_numeric_diff(fact_a.value, fact_b.value)

        return {
            "entity_match": is_entity_match,
            "attribute_match": is_attr_match,
            "has_period_a": has_period_a,
            "has_period_b": has_period_b,
            "period_match": is_period_match,
            "has_scope_a": has_scope_a,
            "has_scope_b": has_scope_b,
            "scope_match": is_scope_match,
            "has_unit_a": has_unit_a,
            "has_unit_b": has_unit_b,
            "unit_match": is_unit_match,
            "value_close": is_close,
            "value_diff": str(diff) if diff is not None else None,
        }

    def check_deterministic_rules(self, fact_a: Fact, fact_b: Fact) -> Optional[ReconciliationDecision]:
        """Backward-compatible helper for deterministic corroboration rule checks."""
        alignment = self.evaluate_alignment(fact_a, fact_b)
        snapshot = make_comparison_snapshot(fact_a, fact_b, alignment)

        if alignment["entity_match"] and alignment["attribute_match"] and alignment["period_match"] and alignment["scope_match"]:
            if alignment["value_close"]:
                diff = alignment.get("value_diff")
                route = "deterministic_exact" if diff in ("0", "0.0", None) else "deterministic_rounding"
                return ReconciliationDecision(
                    relation_type=RelationType.CORROBORATES,
                    explanation=(
                        f"Rule-based corroboration ({route}): Both sources report consistent figures for "
                        f"{fact_a.entity} — {fact_a.attribute} ({fact_a.value} {fact_a.unit or ''} vs "
                        f"{fact_b.value} {fact_b.unit or ''}) for period '{fact_a.period}' "
                        f"under matching scope '{fact_a.scope or 'standard'}'."
                    ),
                    confidence=0.98,
                    reconciliation_basis=None,
                    decision_route=route,
                    review_reason=None,
                    comparison_snapshot=snapshot
                )
        return None

    def reconcile_pair(self, fact_a: Fact, fact_b: Fact) -> ReconciliationDecision:
        """Execute full rules-first, Ambiguity Firewall, and LLM cascade."""
        alignment = self.evaluate_alignment(fact_a, fact_b)
        snapshot = make_comparison_snapshot(fact_a, fact_b, alignment)

        num_a = extract_numeric(fact_a.value)
        num_b = extract_numeric(fact_b.value)
        is_numeric_pair = (num_a is not None and num_b is not None)

        # ── Priority 1: Irrelevant Pre-Filter ─────────────────────────
        if not alignment["entity_match"] or not alignment["attribute_match"]:
            return ReconciliationDecision(
                relation_type=RelationType.UNRELATED,
                explanation="Rule pre-filter: Distinct entity or non-overlapping attribute domain.",
                confidence=1.0,
                reconciliation_basis=None,
                decision_route="irrelevant_pre_filter",
                review_reason=None,
                comparison_snapshot=snapshot
            )

        # ── Priority 2: Deterministic Exact / Rounding Corroboration ──
        if alignment["period_match"] and alignment["scope_match"] and alignment["unit_match"]:
            if alignment["value_close"]:
                diff = alignment.get("value_diff")
                route = "deterministic_exact" if diff in ("0", "0.0", None) else "deterministic_rounding"
                return ReconciliationDecision(
                    relation_type=RelationType.CORROBORATES,
                    explanation=(
                        f"Deterministic {route.replace('_', ' ')}: Both sources confirm {fact_a.entity} — "
                        f"{fact_a.attribute} ({fact_a.value} {fact_a.unit or ''} vs {fact_b.value} {fact_b.unit or ''}) "
                        f"for period '{fact_a.period}' under aligned scope '{fact_a.scope or 'standard'}'."
                    ),
                    confidence=0.98,
                    reconciliation_basis=None,
                    decision_route=route,
                    review_reason=None,
                    comparison_snapshot=snapshot
                )

        # ── Ambiguity Firewall Rules ───────────────────────────────────

        # Rule 4 (T-06): Numeric pair with missing period on either fact
        if is_numeric_pair and (not alignment["has_period_a"] or not alignment["has_period_b"]):
            missing_which = []
            if not alignment["has_period_a"]:
                missing_which.append("Fact A")
            if not alignment["has_period_b"]:
                missing_which.append("Fact B")
            missing_desc = " and ".join(missing_which)

            return ReconciliationDecision(
                relation_type=RelationType.NEEDS_REVIEW,
                explanation=(
                    f"Ambiguity Firewall: Both facts assert numerical figures for '{fact_a.attribute}', "
                    f"but reporting period is missing on {missing_desc}. "
                    f"Refusing to make an automated comparison without aligned timeframes."
                ),
                confidence=0.95,
                reconciliation_basis=None,
                decision_route="ambiguity_firewall",
                review_reason="missing_period",
                comparison_snapshot=snapshot
            )

        # Rule 5 (T-07): Material numerical difference where scope/basis is missing on either fact
        if is_numeric_pair and not alignment["value_close"]:
            if not alignment["has_scope_a"] or not alignment["has_scope_b"]:
                missing_which = []
                if not alignment["has_scope_a"]:
                    missing_which.append("Fact A")
                if not alignment["has_scope_b"]:
                    missing_which.append("Fact B")
                missing_desc = " and ".join(missing_which)

                return ReconciliationDecision(
                    relation_type=RelationType.NEEDS_REVIEW,
                    explanation=(
                        f"Ambiguity Firewall: Material numeric difference ({fact_a.value} vs {fact_b.value}), "
                        f"but accounting/reporting scope is unstated on {missing_desc}. "
                        f"A numerical difference is not automatically a contradiction; "
                        f"an analyst must verify if one figure is reported, adjusted, pro forma, or standalone."
                    ),
                    confidence=0.95,
                    reconciliation_basis=None,
                    decision_route="ambiguity_firewall",
                    review_reason="missing_scope_or_basis",
                    comparison_snapshot=snapshot
                )

        # Rule 6 (T-08): Explicit incompatible units without conversion
        if alignment["has_unit_a"] and alignment["has_unit_b"] and not alignment["unit_match"]:
            return ReconciliationDecision(
                relation_type=RelationType.NEEDS_REVIEW,
                explanation=(
                    f"Ambiguity Firewall: Units '{fact_a.unit}' and '{fact_b.unit}' are incompatible "
                    f"without a confirmed exchange rate or conversion formula."
                ),
                confidence=0.95,
                reconciliation_basis=None,
                decision_route="ambiguity_firewall",
                review_reason="unit_ambiguity",
                comparison_snapshot=snapshot
            )

        # Rule 7: Uncertain table or header context
        qual_a = fact_a.qualifiers or {}
        qual_b = fact_b.qualifiers or {}
        if qual_a.get("header_uncertain") or qual_b.get("header_uncertain") or (fact_a.confidence < 0.70 or fact_b.confidence < 0.70):
            return ReconciliationDecision(
                relation_type=RelationType.NEEDS_REVIEW,
                explanation=(
                    "Ambiguity Firewall: Fact extracted from table/annex with uncertain column/row header linkage "
                    "or low extraction confidence. Routed to review for human verification."
                ),
                confidence=0.90,
                reconciliation_basis=None,
                decision_route="ambiguity_firewall",
                review_reason="header_context_uncertain",
                comparison_snapshot=snapshot
            )

        # ── Stage 4: LLM Contextual Adjudication ───────────────────────
        user_prompt = f"""Compare and reconcile the following two factual assertions:

FACT A:
- Entity: {fact_a.entity}
- Attribute: {fact_a.attribute}
- Value: {fact_a.value} {fact_a.unit or ''}
- Period: {fact_a.period or 'Not specified'}
- Scope: {fact_a.scope or 'Not specified'}
- Qualifiers: {fact_a.qualifiers}
- Evidence Quote: "{fact_a.evidence_quote}"

FACT B:
- Entity: {fact_b.entity}
- Attribute: {fact_b.attribute}
- Value: {fact_b.value} {fact_b.unit or ''}
- Period: {fact_b.period or 'Not specified'}
- Scope: {fact_b.scope or 'Not specified'}
- Qualifiers: {fact_b.qualifiers}
- Evidence Quote: "{fact_b.evidence_quote}"

ALIGNMENT SUMMARY:
- Entity Match: {alignment['entity_match']}
- Attribute Match: {alignment['attribute_match']}
- Period Match: {alignment['period_match']}
- Scope Match: {alignment['scope_match']}
- Value Close: {alignment['value_close']}

Classify into CORROBORATES, CONTRADICTS, RECONCILED, UNRELATED, or NEEDS_REVIEW.
Provide a detailed explanation grounded in both facts and their evidence quotes.
If RECONCILED, identify the exact reconciliation_basis.
"""

        try:
            llm_decision: ReconciliationDecision = llm_client.extract_structured(
                prompt=user_prompt,
                system_prompt=RECONCILIATION_SYSTEM_PROMPT,
                response_model=ReconciliationDecision,
                model=settings.ADJUDICATION_MODEL
            )
            llm_decision.decision_route = "llm_adjudication"
            llm_decision.comparison_snapshot = snapshot
            return llm_decision
        except Exception as e:
            # Rule 8 (T-10): Safe fallback on LLM failure or malformed output
            return ReconciliationDecision(
                relation_type=RelationType.NEEDS_REVIEW,
                explanation=f"Ambiguity Firewall: LLM adjudication unavailable or returned malformed output ({e}). Refusing to guess.",
                confidence=0.5,
                reconciliation_basis=None,
                decision_route="ambiguity_firewall",
                review_reason="adjudication_unavailable",
                comparison_snapshot=snapshot
            )


reconciliation_engine = ReconciliationEngine()
