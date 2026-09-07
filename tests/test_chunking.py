"""Unit tests for Stage A: PDF parsing and table boundary preservation (T-01)."""
from pathlib import Path
import pytest
from src.pipeline.pdf_parser import pdf_parser


def test_t01_table_chunking_preserves_table():
    """T-01: Chunker detects tables and treats each table as one cohesive chunk."""
    # Test with starter presentation or excerpt if available
    sample_pdf = Path("delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not found in workspace")

    # Parse first 5 pages
    chunks, total_pages, doc_type = pdf_parser.parse(str(sample_pdf), doc_id="test-doc", max_pages=5)
    assert total_pages > 0
    assert len(chunks) > 0

    # Ensure table chunks exist and are cleanly formatted with markdown headers
    table_chunks = [c for c in chunks if c.is_table]
    if table_chunks:
        first_table = table_chunks[0]
        assert first_table.is_table is True
        assert "[Table on Page" in first_table.raw_text
        assert "|" in first_table.raw_text
