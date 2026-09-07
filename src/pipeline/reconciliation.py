"""Stage E: Reconciliation Engine implementing rules-first, LLM-second cascade."""
import re
from typing import Optional, Tuple
from src.db.models import Fact, RelationType
from src.pipeline.llm_client import llm_client
from src.pipeline.schemas import ReconciliationDecision


def extract_numeric(val: str) -> Optional[float]:
    """Clean string and parse numeric value if present."""
    if not val:
        return None
    # Remove currency symbols, commas, and common unit terms
    cleaned = re.sub(r"[₹$,]", "", str(val))
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def values_are_close(val_a: str, val_b: str, max_pct_diff: float = 0.005, abs_tolerance: float = 1.0) -> bool:
    """
    Check if two values match within numeric rounding tolerance (T-05).
    Handles strings like '8142' vs '8142.3' or identical strings.
    """
    if str(val_a).strip().lower() == str(val_b).strip().lower():
        return True

    num_a = extract_numeric(val_a)
    num_b = extract_numeric(val_b)

    if num_a is not None and num_b is not None:
        if num_a == num_b:
            return True
        diff = abs(num_a - num_b)
        if diff <= abs_tolerance:
            return True
        avg = (abs(num_a) + abs(num_b)) / 2.0
        if avg > 0 and (diff / avg) <= max_pct_diff:
            return True

    return False


def normalize_metric(text: Optional[str]) -> str:
    """Standardize entity/attribute names for fuzzy comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


RECONCILIATION_SYSTEM_PROMPT = """You are an expert financial and macroeconomic fact reconciliation engine.
You are evaluating two factual claims extracted from corporate filings, annual reports, prospectuses, or macroeconomic institutional reports.

Your task is to classify their relationship and provide a clear, evidence-grounded explanation.

POSSIBLE RELATIONSHIPS:
1. CORROBORATES: Both facts state the same finding, claim, or metric for the same entity and time period (within rounding/precision tolerance), confirming each other across different sources.
2. CONTRADICTS: The two facts present genuinely incompatible, conflicting data for the same metric, entity, period, and scope without any legitimate reconciling explanation.
3. RECONCILED: An apparent contradiction or discrepancy is fully explained by contextual differences such as:
   - Scope / Accounting definition: e.g. Reported EBITDA vs Adjusted EBITDA (share-based payment, IPO expenses, lease accounting add-backs).
   - Pro Forma vs Historical Actuals: e.g. numbers restated on a pro forma basis following an acquisition (like SpotOn).
   - Publication Vintage / Methodology: e.g. Economic Survey (January/February forecast) vs RBI Annual Report vs IMF Article IV consultations for India GDP.
   - Units / Currency / Timing basis.
4. UNRELATED: The facts describe different entities, metrics, or contexts that do not compare directly.

CRITICAL INSTRUCTIONS:
- Explain WHY, not just the classification.
- Quote or cite the exact evidence quotes provided.
- If RECONCILED, explicitly state the 'reconciliation_basis'.
"""


class ReconciliationEngine:
    def check_deterministic_rules(self, fact_a: Fact, fact_b: Fact) -> Optional[ReconciliationDecision]:
        """
        Stage E Step 1: Zero-cost rule check.
        If entity, attribute, period, scope, and unit match, and values are within rounding tolerance,
        classify as CORROBORATES immediately without calling LLM.
        """
        ent_a = normalize_metric(fact_a.entity)
        ent_b = normalize_metric(fact_b.entity)
        attr_a = normalize_metric(fact_a.attribute)
        attr_b = normalize_metric(fact_b.attribute)
        period_a = normalize_metric(fact_a.period)
        period_b = normalize_metric(fact_b.period)
        scope_a = normalize_metric(fact_a.scope)
        scope_b = normalize_metric(fact_b.scope)

        # Entity match (e.g. 'delhivery' in both or exact match)
        same_entity = (ent_a == ent_b) or (ent_a in ent_b) or (ent_b in ent_a)
        # Attribute match
        same_attr = (attr_a == attr_b) or (attr_a in attr_b) or (attr_b in attr_a)

        if same_entity and same_attr:
            same_period = (period_a == period_b) and (bool(period_a))
            same_scope = (scope_a == scope_b)

            if same_period and same_scope:
                if values_are_close(fact_a.value, fact_b.value):
                    return ReconciliationDecision(
                        relation_type=RelationType.CORROBORATES,
                        explanation=(
                            f"Rule-based corroboration: Both documents report consistent figures for "
                            f"{fact_a.entity} - {fact_a.attribute} ({fact_a.value} {fact_a.unit or ''} vs "
                            f"{fact_b.value} {fact_b.unit or ''}) for period '{fact_a.period}' "
                            f"under matching scope '{fact_a.scope or 'standard'}'."
                        ),
                        confidence=0.98,
                        reconciliation_basis=None
                    )

        return None

    def reconcile_pair(self, fact_a: Fact, fact_b: Fact) -> ReconciliationDecision:
        """Run rules-first, LLM-second cascade."""
        # 1. Deterministic rules check (free and fast)
        rule_decision = self.check_deterministic_rules(fact_a, fact_b)
        if rule_decision:
            return rule_decision

        # 2. Fast heuristic rule: filter out clearly unrelated entities or disjoint attributes
        ent_a = normalize_metric(fact_a.entity)
        ent_b = normalize_metric(fact_b.entity)
        attr_a = normalize_metric(fact_a.attribute)
        attr_b = normalize_metric(fact_b.attribute)

        same_entity = (
            (ent_a == ent_b) or (ent_a in ent_b) or (ent_b in ent_a) or
            ("delhivery" in ent_a and "delhivery" in ent_b) or
            (ent_a in ["company", "delhivery", "delhivery limited"] and ent_b in ["company", "delhivery", "delhivery limited"]) or
            ("india" in ent_a and "india" in ent_b)
        )

        words_a = set(attr_a.split()) - {"of", "the", "in", "for", "and", "on", "at", "to", "a", "an", "from", "by", "is", "as"}
        words_b = set(attr_b.split()) - {"of", "the", "in", "for", "and", "on", "at", "to", "a", "an", "from", "by", "is", "as"}
        has_metric_overlap = bool(words_a.intersection(words_b))

        # Check for well-known metric synonyms (e.g. revenue and turnover, profit and pat)
        synonyms = [
            {"revenue", "turnover", "sales", "income"},
            {"ebitda", "operating profit", "operating margin"},
            {"pat", "profit", "net profit"},
            {"gdp", "growth", "growth rate", "output"}
        ]
        is_synonym = any(bool(words_a.intersection(s)) and bool(words_b.intersection(s)) for s in synonyms)

        if not same_entity or (not has_metric_overlap and not is_synonym):
            return ReconciliationDecision(
                relation_type=RelationType.UNRELATED,
                explanation="Rule pre-filter: Distinct entity or non-overlapping attribute domain.",
                confidence=1.0,
                reconciliation_basis=None
            )

        # 3. LLM Adjudication for genuinely related/competing pairs
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

Classify into CORROBORATES, CONTRADICTS, RECONCILED, or UNRELATED.
Provide a detailed explanation grounded in the facts and evidence quotes.
If RECONCILED, identify the exact reconciliation_basis.
"""

        try:
            decision: ReconciliationDecision = llm_client.extract_structured(
                prompt=user_prompt,
                system_prompt=RECONCILIATION_SYSTEM_PROMPT,
                response_model=ReconciliationDecision
            )
            return decision
        except Exception as e:
            # Graceful fallback if LLM times out or is unreachable
            return ReconciliationDecision(
                relation_type=RelationType.UNRELATED,
                explanation=f"Automated comparison could not complete: {e}",
                confidence=0.5,
                reconciliation_basis=None
            )


reconciliation_engine = ReconciliationEngine()
