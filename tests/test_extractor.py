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
