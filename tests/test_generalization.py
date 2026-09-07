"""Integration tests for domain-agnostic schema generalization (I-05)."""
from pathlib import Path
import pytest
from src.pipeline.canonicalizer import canonicalizer
from src.pipeline.schemas import ExtractedFact


def test_i05_domain_agnostic_fact_representation():
    """
    I-05: Verifies that the Fact representation generalizes across entirely different domains
    (e.g., aerospace engineering, biomedical, macroeconomics) without hardcoding fields.
    """
    # 1. Aerospace engineering fact
    fact_aerospace = ExtractedFact(
        entity="SpaceX Starship",
        attribute="Raptor engine sea-level thrust",
        value="230",
        unit="tf",
        period="Block 2",
        scope="sea-level",
        qualifiers={"propellant": "liquid CH4 / LOX", "chamber_pressure_bar": 300},
        evidence_quote="Raptor 2 produces 230 tf of thrust at sea level with 300 bar chamber pressure",
        confidence=0.98
    )

    canonical_aero = canonicalizer.build_canonical_statement(fact_aerospace)
    assert "SpaceX Starship" in canonical_aero
    assert "230 tf" in canonical_aero

    # 2. Macroeconomic policy fact
    fact_macro = ExtractedFact(
        entity="Reserve Bank of India",
        attribute="Policy Repo Rate",
        value="6.50",
        unit="%",
        period="August 2024",
        scope="Monetary Policy Committee",
        qualifiers={"stance": "withdrawal of accommodation", "vote": "4 to 2"},
        evidence_quote="The MPC decided to keep the policy repo rate unchanged at 6.50 per cent",
        confidence=0.99
    )

    canonical_macro = canonicalizer.build_canonical_statement(fact_macro)
    assert "Reserve Bank of India" in canonical_macro
    assert "6.50 %" in canonical_macro

    # Both schemas embed cleanly without any domain restriction
    embeddings = canonicalizer.embed_statements([canonical_aero, canonical_macro])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
