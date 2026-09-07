"""CLI benchmark script to demonstrate the 4 assignment cases and incremental ingestion timing."""
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from src.config import settings
from src.db.models import Document, Fact, JobStatus, Relationship, RelationType
from src.db.session import SessionLocal, init_db
from src.pipeline.orchestrator import orchestrator


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def ingest_file(pdf_path: Path, max_pages: int = 15) -> str:
    db = SessionLocal()
    target_path = settings.UPLOAD_DIR / pdf_path.name
    if not target_path.exists():
        import shutil
        shutil.copy(pdf_path, target_path)

    doc = Document(
        filename=pdf_path.name,
        file_path=str(target_path),
        status=JobStatus.QUEUED
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc_id = doc.id
    db.close()

    print(f"\n[Ingesting] {pdf_path.name} (pages 1-{max_pages})...")
    start_ts = time.time()
    res = orchestrator.process_document(doc_id, max_pages=max_pages)
    elapsed = time.time() - start_ts

    print(f" -> Completed in {elapsed:.2f}s | Facts Extracted: {res['facts_extracted']} | Relationships: {res['relationships_found']}")
    return doc_id, elapsed


def evaluate_cases():
    print_banner("EVALUATING THE 4 REQUIRED CASES")
    db = SessionLocal()

    # Case 1: Corroboration
    print("\n[CASE 1: Corroborated Fact]")
    corrob_rels = db.query(Relationship).filter(Relationship.relation_type == RelationType.CORROBORATES).all()
    if corrob_rels:
        for r in corrob_rels[:3]:
            fa = r.fact_a
            fb = r.fact_b
            print(f" - Fact A: [{fa.document.filename if fa.document else 'Doc A'}] {fa.entity} {fa.attribute} = {fa.value} {fa.unit or ''} ({fa.period or 'N/A'})")
            print(f" - Fact B: [{fb.document.filename if fb.document else 'Doc B'}] {fb.entity} {fb.attribute} = {fb.value} {fb.unit or ''} ({fb.period or 'N/A'})")
            print(f" - Explanation: {r.explanation}")
            print(f" - Confidence: {r.confidence}\n")
    else:
        print(" - No corroboration found yet. Ensure multi-document ingestion has completed.")

    # Case 2: Genuine Contradiction
    print("\n[CASE 2: Genuine / Likely Contradiction]")
    contra_rels = db.query(Relationship).filter(Relationship.relation_type == RelationType.CONTRADICTS).all()
    if contra_rels:
        for r in contra_rels[:3]:
            fa = r.fact_a
            fb = r.fact_b
            print(f" - Fact A: [{fa.document.filename if fa.document else 'Doc A'}] {fa.entity} {fa.attribute} = {fa.value} {fa.unit or ''}")
            print(f" - Fact B: [{fb.document.filename if fb.document else 'Doc B'}] {fb.entity} {fb.attribute} = {fb.value} {fb.unit or ''}")
            print(f" - Conflict Explanation: {r.explanation}\n")
    else:
        print(" - Ingest both 2022 Prospectus and FY24 Earnings Presentation to detect pro forma vs historical actual contradiction.")

    # Case 3: Reconciled Ambiguity (EBITDA Bridge)
    print("\n[CASE 3: Apparent Contradiction Reconciled by Context (EBITDA Bridge)]")
    reconciled_rels = db.query(Relationship).filter(Relationship.relation_type == RelationType.RECONCILED).all()
    if reconciled_rels:
        for r in reconciled_rels[:3]:
            fa = r.fact_a
            fb = r.fact_b
            print(f" - Fact A: {fa.entity} {fa.attribute} = {fa.value} {fa.unit or ''} [{fa.scope or 'reported'}]")
            print(f" - Fact B: {fb.entity} {fb.attribute} = {fb.value} {fb.unit or ''} [{fb.scope or 'adjusted'}]")
            print(f" - Reconciliation Basis: {r.reconciliation_basis}")
            print(f" - Grounded Explanation: {r.explanation}\n")
    else:
        print(" - Ingest Earnings Presentation to evaluate EBITDA bridge.")

    # Case 4: Extraction/Reasoning Failure
    print("\n[CASE 4: Documented Extraction / Reasoning Failure Mode]")
    print(" - Failure Scenario: In complex presentation tables with rotated headers or multi-level column groupings,")
    print("   an LLM can extract metric numbers while failing to bind the 'scope' (reported vs adjusted) or period qualifier.")
    print(" - Consequence: Unscoped metrics risk spurious direct comparisons.")
    print(" - Pipeline Defense: Verbatim self-check node + rule-based scope requirement before allowing auto-corroboration.\n")

    db.close()


def main():
    parser = argparse.ArgumentParser(description="Run Fact Knowledge Layer Benchmark")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing database")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages per document to process")
    args = parser.parse_args()

    init_db()

    if args.eval_only:
        evaluate_cases()
        return

    print_banner("BENCHMARK INGESTION & INCREMENTAL CLUSTERING")

    deck_path = Path("delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf")
    annual_rep_path = Path("delhivery/02-delhivery-annual-report-fy24-excerpt.pdf")
    prospectus_path = Path("delhivery/01-delhivery-prospectus-2022-excerpt.pdf")

    if deck_path.exists():
        doc1_id, t1 = ingest_file(deck_path, max_pages=min(args.max_pages, 27))
    if annual_rep_path.exists():
        doc2_id, t2 = ingest_file(annual_rep_path, max_pages=args.max_pages)
        print(f"\n[Incremental Clustering Metric] Doc 2 processed incrementally against existing Doc 1 vector space.")

    evaluate_cases()


if __name__ == "__main__":
    main()
