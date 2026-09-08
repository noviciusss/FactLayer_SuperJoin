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


def test_coalesce_synthetic_fragments():
    """
    Test page-level coalescing against a synthetic page with 5 small fragments,
    including one sub-100-char fragment containing numbers.
    Asserts:
      (a) Fragment count drops from 5 to expected level (1 merged chunk).
      (b) Sub-100-char fragment content is present verbatim in merged output.
    """
    from src.pipeline.pdf_parser import ParsedChunk

    sub_100_char_frag = "| 11 (5%) |"
    fragments = [
        ParsedChunk(page_number=1, char_start=0, char_end=58, raw_text="Digital payments ecosystem expanded significantly during FY24.", is_table=False, needs_vision=False),
        ParsedChunk(page_number=1, char_start=0, char_end=82, raw_text="[Table on Page 1]\n| Category | Growth |\n| --- | --- |\n| Merchant POS | 28% |", is_table=True, needs_vision=False),
        ParsedChunk(page_number=1, char_start=0, char_end=len(sub_100_char_frag), raw_text=sub_100_char_frag, is_table=True, needs_vision=False),
        ParsedChunk(page_number=1, char_start=0, char_end=72, raw_text="UPI transaction volume reached over 131 billion transactions annually.", is_table=False, needs_vision=False),
        ParsedChunk(page_number=1, char_start=0, char_end=64, raw_text="Consumer adoption across Tier-2 and Tier-3 cities reached 65%.", is_table=False, needs_vision=False),
    ]

    # Run coalescer on synthetic page
    merged = pdf_parser.coalesce_page_chunks(fragments, max_chunk_size=2400)

    # (a) Fragment count drops to expected level (all 5 fit comfortably in 2400 chars -> 1 chunk)
    assert len(merged) == 1, f"Expected 1 coalesced chunk, got {len(merged)}"

    # (b) The sub-100-char fragment's content is present verbatim somewhere in merged output
    assert sub_100_char_frag in merged[0].raw_text, "Sub-100-char fragment content was dropped!"

    # Invariants: Table status is preserved if any merged fragment was a table
    assert merged[0].is_table is True
    assert "[Table on Page 1]" in merged[0].raw_text

    # CRITICAL SAFETY RULE: Every single fragment must be present verbatim in the merged chunk
    for frag in fragments:
        assert frag.raw_text in merged[0].raw_text, f"Fragment '{frag.raw_text[:30]}' was dropped!"


def test_industry_report_fixture_coalescing():
    """
    Parse 41-page industry-report.pdf fixture (chunking only, no LLM calls).
    Asserts:
      1. Chunk count drops meaningfully from the 80 un-coalesced baseline.
      2. Total character content across all new chunks >= baseline 166,063 characters
         (proving no fragments were dropped, only merged).
      3. Sub-100-char table fragment on Page 15 is present verbatim in merged output.
    """
    pdf_path = Path("industry-report.pdf")
    if not pdf_path.exists():
        pytest.skip("industry-report.pdf not found in workspace")

    chunks, total_pages, doc_type = pdf_parser.parse(str(pdf_path), doc_id="test-coalesce-eval")
    assert total_pages == 41

    # 1. Chunk count drops meaningfully (baseline was 80 chunks due to over-segmentation)
    assert len(chunks) < 80, f"Chunk count did not drop: got {len(chunks)}"
    assert len(chunks) <= 55, f"Expected <= 55 chunks after coalescing 41 pages, got {len(chunks)}"

    # 2. Total character content >= baseline 166,063 chars (proves nothing dropped)
    total_chars = sum(len(c.raw_text) for c in chunks)
    assert total_chars >= 166063, f"Character loss detected: {total_chars} < 166063"

    # 3. Verify sub-100-char fragment from Page 15 is present verbatim
    p15_chunks = [c for c in chunks if c.page_number == 15]
    assert len(p15_chunks) > 0
    assert any("| 11 ( 5 %) |" in c.raw_text for c in p15_chunks), "Page 15 micro-table fragment was dropped!"

