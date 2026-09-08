"""Unit tests for Stage B.2: Self-check verification and deduplication (T-03, T-04)."""
import pytest
from src.pipeline.schemas import ExtractedFact
from src.pipeline.self_check import self_checker


def test_t03_self_check_verifies_exact_quote():
    """T-03: Verifies fact with exact verbatim quote passes self-check."""
    chunk_text = (
        "Delhivery Limited reported consolidated revenue from operations "
        "of ₹8,142 Cr for the fiscal year ended March 31, 2024."
    )

    fact = ExtractedFact(
        entity="Delhivery Limited",
        attribute="revenue from operations",
        value="8142",
        unit="₹ Cr",
        period="FY24",
        scope="consolidated",
        evidence_quote="revenue from operations of ₹8,142 Cr for the fiscal year",
        confidence=0.95
    )

    is_valid, adjusted_conf, reason = self_checker.verify_grounding(fact, chunk_text)
    assert is_valid is True
    assert adjusted_conf == 0.95


def test_t03_self_check_rejects_hallucinated_quote():
    """T-03: Verifies fact with ungrounded/hallucinated quote is rejected or heavily penalized."""
    chunk_text = "The Board of Directors met on May 17, 2024 to approve the audited financial results."

    hallucinated_fact = ExtractedFact(
        entity="Delhivery Limited",
        attribute="EBITDA",
        value="500",
        unit="₹ Cr",
        period="FY24",
        scope="consolidated",
        evidence_quote="adjusted EBITDA reached an all-time high of ₹500 Cr",
        confidence=0.9
    )

    is_valid, adjusted_conf, reason = self_checker.verify_grounding(hallucinated_fact, chunk_text)
    assert is_valid is False
    assert adjusted_conf < 0.3
    assert "Unsubstantiated" in reason


def test_t04_idempotent_duplicate_deduplication():
    """T-04: Same fact ingested twice does not create duplicate rows."""
    chunk_text = "Revenue from operations was ₹8,142 Cr."

    fact1 = ExtractedFact(
        entity="Delhivery Limited",
        attribute="Revenue",
        value="8142",
        unit="₹ Cr",
        period="FY24",
        scope="consolidated",
        evidence_quote="Revenue from operations was ₹8,142 Cr",
        confidence=0.95
    )

    fact2 = ExtractedFact(
        entity="delhivery limited",
        attribute="revenue",
        value="8142",
        unit="₹ Cr",
        period="FY24",
        scope="consolidated",
        evidence_quote="Revenue from operations was ₹8,142 Cr",
        confidence=0.95
    )

    seen_hashes = set()
    verified = self_checker.process_facts([fact1, fact2], chunk_text, "doc-1", seen_hashes)

    # Only 1 unique fact should survive deduplication
    assert len(verified) == 1


def test_vision_extraction_path_grounding_pass():
    """TASK 1: Facts marked with extraction_path='vision' pass self-check and have confidence capped at 0.7."""
    vision_fact = ExtractedFact(
        entity="Delhivery Limited",
        attribute="Hub Count",
        value="24",
        unit="hubs",
        period="FY24",
        scope="operational",
        qualifiers={"extraction_path": "vision"},
        evidence_quote="Infographic shows 24 automated hubs",
        confidence=0.92
    )

    is_valid, adjusted_conf, reason = self_checker.verify_grounding(vision_fact, "completely different chunk text")
    assert is_valid is True
    assert adjusted_conf == 0.7
    assert "Vision extraction path" in reason


def test_self_check_multiline_and_footnoted_quote_passes():
    """Verify that multi-line breaks and footnote markers (e.g., superscripts, brackets) pass grounding check."""
    # Chunk containing newline break and unicode superscript footnote marker
    chunk_text = (
        "In Fiscal Year 2025, India had a nominal\n"
        "GDP of ₹332 trillion¹ (US$3.91 trillion).\n"
        "Note(s): 1. Second revised estimates."
    )

    # LLM quote without the newline or superscript
    fact = ExtractedFact(
        entity="India",
        attribute="nominal GDP",
        value="332",
        unit="₹ trillion",
        period="FY 2025",
        evidence_quote="India had a nominal GDP of ₹332 trillion",
        confidence=0.95
    )

    is_valid, adjusted_conf, reason = self_checker.verify_grounding(fact, chunk_text)
    assert is_valid is True
    assert adjusted_conf >= 0.85
    assert "Verified" in reason


