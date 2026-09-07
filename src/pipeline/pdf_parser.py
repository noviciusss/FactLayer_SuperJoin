"""Stage A: PDF parsing, table preservation, image density detection, and page image extraction."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import pymupdf as fitz
import pdfplumber
from src.config import settings


@dataclass
class ParsedChunk:
    page_number: int
    char_start: int
    char_end: int
    raw_text: str
    is_table: bool
    needs_vision: bool
    image_ref: Optional[str] = None


class PDFParser:
    def __init__(self):
        self.preview_dir = settings.PREVIEW_DIR
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def extract_document_type(self, sample_text: str) -> str:
        """Heuristic and keyword-based document type guess."""
        lower_text = sample_text.lower()
        if "prospectus" in lower_text or "initial public offering" in lower_text:
            return "prospectus"
        elif "annual report" in lower_text or "board of directors" in lower_text:
            return "annual_report"
        elif "earnings presentation" in lower_text or "investor presentation" in lower_text or "q4 fy" in lower_text:
            return "earnings_presentation"
        elif "economic survey" in lower_text or "gdp growth" in lower_text or "macroeconomic" in lower_text:
            return "macro_report"
        elif "article iv" in lower_text or "international monetary fund" in lower_text:
            return "imf_report"
        return "general_document"

    def render_page_preview(self, doc_id: str, doc_fitz: fitz.Document, page_num: int) -> str:
        """Render page to a PNG preview image and return relative/absolute path."""
        try:
            page = doc_fitz.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=150)
            img_filename = f"{doc_id}_page_{page_num}.png"
            img_path = self.preview_dir / img_filename
            pix.save(str(img_path))
            return str(img_path)
        except Exception:
            return None

    def parse(self, file_path: str, doc_id: str, max_pages: Optional[int] = None) -> Tuple[List[ParsedChunk], int, str]:
        """
        Parse PDF file into structured chunks.
        Preserves tables as single cohesive chunks.
        Detects image/chart-heavy slides for vision fallback.
        """
        chunks: List[ParsedChunk] = []
        path = Path(file_path)
        first_pages_text = []

        doc_fitz = fitz.open(str(path))
        total_pages = len(doc_fitz)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        with pdfplumber.open(str(path)) as pdf:
            for page_idx in range(pages_to_process):
                page_num = page_idx + 1
                plumber_page = pdf.pages[page_idx]

                # Render page preview for evidence UI
                preview_image_path = self.render_page_preview(doc_id, doc_fitz, page_num)

                # 1. Detect tables using pdfplumber
                # Table finder gives precise bounding boxes
                tables = plumber_page.find_tables()
                table_bboxes = [t.bbox for t in tables]

                # 2. Extract tables as structured markdown chunks
                extracted_tables = plumber_page.extract_tables()
                for t_idx, table_data in enumerate(extracted_tables):
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # Format as clean markdown table
                    clean_rows = []
                    for row in table_data:
                        clean_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
                        clean_rows.append(" | ".join(clean_row))

                    if len(clean_rows) >= 2:
                        header = clean_rows[0]
                        separator = " | ".join(["---"] * len(clean_rows[0].split(" | ")))
                        body = "\n".join(clean_rows[1:])
                        table_md = f"| {header} |\n| {separator} |\n" + "\n".join([f"| {r} |" for r in clean_rows[1:] if r.strip()])
                        
                        chunks.append(ParsedChunk(
                            page_number=page_num,
                            char_start=0,
                            char_end=len(table_md),
                            raw_text=f"[Table on Page {page_num}]\n{table_md}",
                            is_table=True,
                            needs_vision=False,
                            image_ref=preview_image_path
                        ))

                # 3. Extract non-table text or entire page text outside table bboxes
                # To prevent table text from being shredded across text chunks, filter out table bounding boxes if possible
                try:
                    non_table_page = plumber_page
                    for bbox in table_bboxes:
                        # bbox: (x0, top, x1, bottom)
                        non_table_page = non_table_page.filter(
                            lambda obj: not (
                                obj["x0"] >= bbox[0] and obj["x1"] <= bbox[2] and
                                obj["top"] >= bbox[1] and obj["bottom"] <= bbox[3]
                            )
                        )
                    page_text = non_table_page.extract_text() or ""
                except Exception:
                    page_text = plumber_page.extract_text() or ""

                page_text = page_text.strip()
                if page_idx < 3 and page_text:
                    first_pages_text.append(page_text)

                # Check if page is image-heavy or infographic-style slide
                # (e.g. presentation slide with large visuals, very low text density)
                char_count = len(page_text)
                is_low_text = char_count < 80 and len(plumber_page.images) > 0

                if page_text:
                    # Split into reasonable paragraph chunks if text is long
                    paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                    running_pos = 0
                    for p in paragraphs:
                        if len(p) < 20:  # Skip tiny headers/footers
                            continue
                        chunks.append(ParsedChunk(
                            page_number=page_num,
                            char_start=running_pos,
                            char_end=running_pos + len(p),
                            raw_text=p,
                            is_table=False,
                            needs_vision=is_low_text,
                            image_ref=preview_image_path
                        ))
                        running_pos += len(p) + 2
                elif is_low_text:
                    # Page has almost no text but has images/charts (e.g. Infographic slide)
                    chunks.append(ParsedChunk(
                        page_number=page_num,
                        char_start=0,
                        char_end=0,
                        raw_text=f"[Visual / Chart Page {page_num}]",
                        is_table=False,
                        needs_vision=True,
                        image_ref=preview_image_path
                    ))

        doc_fitz.close()
        doc_type = self.extract_document_type("\n".join(first_pages_text))
        return chunks, total_pages, doc_type


pdf_parser = PDFParser()
