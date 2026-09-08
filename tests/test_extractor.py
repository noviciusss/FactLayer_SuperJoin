"""Unit tests for Stage B: Fact extractor boilerplate suppression (T-02)."""
import pytest
from src.pipeline.extractor import fact_extractor


def test_t02_boilerplate_safe_harbor_returns_empty():
    """T-02: Fact extractor on a chunk with no numeric facts returns empty list, not hallucinations."""
    safe_harbor_chunk = (
        "SAFE HARBOR STATEMENT: Certain statements contained in this presentation may be statements of future "
        "expectations and other forward-looking statements that are based on management's current views and assumptions "
        "and involve known and unknown risks and uncertainties. Actual results, performance, or events may differ materially."
    )

    facts = fact_extractor.extract_from_chunk_text(safe_harbor_chunk, page_num=2, doc_type="earnings_presentation")
    assert isinstance(facts, list)
    assert len(facts) == 0


def test_two_hop_vision_extraction_flow(tmp_path):
    """TASK 1: Verify two-hop vision path calls transcribe_image, pipes to text extractor, flags vision, and caps confidence at 0.7."""
    from unittest.mock import patch, MagicMock
    from src.pipeline.schemas import ExtractedFact

    # Create dummy image file
    dummy_img = tmp_path / "page_1.png"
    dummy_img.write_bytes(b"dummy_png_bytes")

    mock_fact = ExtractedFact(
        entity="Delhivery Limited",
        attribute="Revenue",
        value="8142",
        unit="₹ Cr",
        period="FY24",
        scope="consolidated",
        qualifiers={},
        evidence_quote="Revenue ₹8,142 Cr",
        confidence=0.98
    )

    with patch("src.pipeline.extractor.llm_client.transcribe_image", return_value="Slide shows Delhivery Revenue: 8142 Cr FY24") as mock_transcribe:
        with patch.object(fact_extractor, "extract_from_chunk_text", return_value=[mock_fact]) as mock_text_extract:
            facts = fact_extractor.extract_from_chunk_vision(str(dummy_img), page_num=1, doc_type="presentation")

            mock_transcribe.assert_called_once()
            mock_text_extract.assert_called_once_with(
                chunk_text="Slide shows Delhivery Revenue: 8142 Cr FY24",
                page_num=1,
                doc_type="presentation"
            )
            assert len(facts) == 1
            f = facts[0]
            assert f.qualifiers.get("extraction_path") == "vision"
            assert f.confidence == 0.7  # Capped at 0.7

