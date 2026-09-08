"""Master Ingestion Orchestrator running Stages A through E."""
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from src.config import settings
from src.db.models import Chunk, Document, Fact, JobStatus, Relationship, RelationType
from src.db.session import SessionLocal
from src.db.vector_store import vector_store
from src.pipeline.canonicalizer import canonicalizer
from src.pipeline.extractor import fact_extractor
from src.pipeline.llm_client import llm_client
from src.pipeline.pdf_parser import pdf_parser
from src.pipeline.reconciliation import reconciliation_engine
from src.pipeline.schemas import ExtractedFact
from src.pipeline.self_check import self_checker


class PipelineOrchestrator:
    def __init__(self):
        # Initialize vector store collection
        try:
            vector_store.init_collection(vector_size=384)
        except Exception as e:
            print(f"[Orchestrator] Note on vector store init: {e}")

    def process_document(
        self,
        document_id: str,
        max_pages: Optional[int] = None,
        timeout_seconds: Optional[float] = None
    ) -> Dict:
        """
        End-to-end execution of Stages A -> B -> C -> D -> E for a given document
        with document-level timeout backstop and chunk failure tracking.
        """
        doc_timeout = timeout_seconds or getattr(settings, "DOCUMENT_PROCESSING_TIMEOUT", 300.0)
        start_time = time.time()
        deadline = start_time + doc_timeout

        db: Session = SessionLocal()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            db.close()
            raise ValueError(f"Document {document_id} not found")

        try:
            # ───────────────────────────────────────────────────────────
            # Stage A: Ingest & Chunk
            # ───────────────────────────────────────────────────────────
            doc.status = JobStatus.CHUNKING
            db.commit()

            parsed_chunks, total_pages, doc_type = pdf_parser.parse(
                doc.file_path,
                doc_id=doc.id,
                max_pages=max_pages
            )
            doc.page_count = total_pages
            doc.doc_type_guess = doc_type

            db_chunks = []
            for pc in parsed_chunks:
                db_chunk = Chunk(
                    document_id=doc.id,
                    page_number=pc.page_number,
                    char_start=pc.char_start,
                    char_end=pc.char_end,
                    raw_text=pc.raw_text,
                    is_table=pc.is_table,
                    image_ref=pc.image_ref,
                    extraction_status="pending"
                )
                db.add(db_chunk)
                db_chunks.append((db_chunk, pc))
            db.commit()

            if time.time() > deadline:
                raise TimeoutError(f"Document chunking exceeded total processing deadline of {doc_timeout}s.")

            # ───────────────────────────────────────────────────────────
            # Stage B: Fact Extraction (Parallelized) & Self-Check
            # ───────────────────────────────────────────────────────────
            doc.status = JobStatus.EXTRACTING
            db.commit()

            def process_single_chunk(pair):
                db_chunk_obj, p_chunk = pair
                if p_chunk.needs_vision and p_chunk.image_ref:
                    raw_facts = fact_extractor.extract_from_chunk_vision(
                        p_chunk.image_ref,
                        p_chunk.page_number,
                        doc_type=doc_type
                    )
                else:
                    raw_facts = fact_extractor.extract_from_chunk_text(
                        p_chunk.raw_text,
                        p_chunk.page_number,
                        doc_type=doc_type
                    )
                c_status, c_err = llm_client.get_last_call_status()
                if not raw_facts and c_status in ("rate_limited", "extraction_failed"):
                    final_chunk_status = c_status
                else:
                    final_chunk_status = "success"
                return db_chunk_obj, raw_facts, final_chunk_status

            extracted_facts_by_chunk = []
            remaining_b = max(5.0, deadline - time.time())
            timed_out_stage_b = False

            with ThreadPoolExecutor(max_workers=settings.MAX_EXTRACTION_WORKERS) as executor:
                future_to_chunk = {
                    executor.submit(process_single_chunk, pair): pair for pair in db_chunks
                }
                try:
                    for future in as_completed(future_to_chunk, timeout=remaining_b):
                        try:
                            db_chunk_obj, raw_facts, c_status = future.result()
                            db_chunk_obj.extraction_status = c_status
                            if raw_facts:
                                extracted_facts_by_chunk.append((db_chunk_obj, raw_facts))
                        except Exception as exc:
                            print(f"[Orchestrator] Error processing chunk: {exc}")
                except TimeoutError:
                    timed_out_stage_b = True
                    print(f"[Orchestrator Timeout] Document processing deadline ({doc_timeout}s) reached during Stage B.")
                    for fut, (c_obj, _) in future_to_chunk.items():
                        if not fut.done():
                            fut.cancel()
                            c_obj.extraction_status = "rate_limited"

            # Self-check, hallucination filtering, and deduplication
            seen_hashes = set()
            new_facts: List[Fact] = []
            canonical_statements: List[str] = []

            for db_chunk_obj, raw_facts in extracted_facts_by_chunk:
                verified = self_checker.process_facts(
                    facts=raw_facts,
                    chunk_text=db_chunk_obj.raw_text,
                    doc_id=doc.id,
                    seen_hashes=seen_hashes
                )

                for fact_obj, chash in verified:
                    canonical_stmt = canonicalizer.build_canonical_statement(fact_obj)
                    db_fact = Fact(
                        chunk_id=db_chunk_obj.id,
                        document_id=doc.id,
                        entity=fact_obj.entity,
                        attribute=fact_obj.attribute,
                        value=fact_obj.value,
                        unit=fact_obj.unit,
                        period=fact_obj.period,
                        scope=fact_obj.scope,
                        qualifiers=fact_obj.qualifiers,
                        evidence_quote=fact_obj.evidence_quote,
                        confidence=fact_obj.confidence,
                        canonical_statement=canonical_stmt,
                        content_hash=chash
                    )
                    db.add(db_fact)
                    new_facts.append(db_fact)
                    canonical_statements.append(canonical_stmt)

            db.commit()

            # Count chunk statuses
            failed_chunks_count = sum(
                1 for c, _ in db_chunks
                if getattr(c, "extraction_status", "success") in ("rate_limited", "extraction_failed")
            )

            if timed_out_stage_b and not new_facts:
                raise TimeoutError(f"Document processing deadline of {doc_timeout}s reached during Stage B with 0 facts extracted.")

            if not new_facts and failed_chunks_count > 0:
                doc.status = JobStatus.FAILED
                doc.error_message = f"Extraction failed: {failed_chunks_count} chunk(s) exhausted retries (rate_limited/extraction_failed) without recovering facts."
                db.commit()
                return {
                    "document_id": doc.id,
                    "status": "failed",
                    "error": doc.error_message,
                    "facts_extracted": 0,
                    "failed_chunks": failed_chunks_count
                }

            if time.time() > deadline:
                raise TimeoutError(f"Document processing deadline of {doc_timeout}s exceeded before Stage C.")

            # ───────────────────────────────────────────────────────────
            # Stage C: Canonicalization & Vector Embeddings
            # ───────────────────────────────────────────────────────────
            if new_facts:
                vectors = canonicalizer.embed_statements(canonical_statements)
                fact_ids = [f.id for f in new_facts]
                payloads = [
                    {
                        "fact_id": f.id,
                        "document_id": doc.id,
                        "entity": f.entity,
                        "attribute": f.attribute,
                        "canonical_statement": f.canonical_statement
                    }
                    for f in new_facts
                ]
                vector_store.upsert_facts(fact_ids, vectors, payloads)

            if time.time() > deadline:
                raise TimeoutError(f"Document processing deadline of {doc_timeout}s exceeded before Stage D/E.")

            # ───────────────────────────────────────────────────────────
            # Stage D & E: Candidate Clustering & Reconciliation
            # ───────────────────────────────────────────────────────────
            doc.status = JobStatus.RECONCILING
            db.commit()

            reconciled_pairs = set()

            if new_facts:
                for idx, fact in enumerate(new_facts):
                    if time.time() > deadline:
                        print(f"[Orchestrator Timeout] Reconciliation exceeded deadline ({doc_timeout}s). Halting pair search.")
                        break

                    query_vector = vectors[idx]
                    candidates = vector_store.search_candidates(
                        query_vector=query_vector,
                        limit=4,
                        score_threshold=0.76,
                        exclude_fact_id=fact.id
                    )

                    for cand in candidates:
                        if time.time() > deadline:
                            break

                        cand_fact_id = cand["fact_id"]
                        pair_key = tuple(sorted([fact.id, cand_fact_id]))
                        if pair_key in reconciled_pairs:
                            continue
                        reconciled_pairs.add(pair_key)

                        cand_fact = db.query(Fact).filter(Fact.id == cand_fact_id).first()
                        if not cand_fact or cand_fact.document_id == fact.document_id:
                            continue

                        # Check existing relationship in DB
                        existing_rel = db.query(Relationship).filter(
                            ((Relationship.fact_a_id == fact.id) & (Relationship.fact_b_id == cand_fact.id)) |
                            ((Relationship.fact_a_id == cand_fact.id) & (Relationship.fact_b_id == fact.id))
                        ).first()
                        if existing_rel:
                            continue

                        decision = reconciliation_engine.reconcile_pair(fact, cand_fact)
                        if decision.relation_type != RelationType.UNRELATED:
                            rel = Relationship(
                                fact_a_id=fact.id,
                                fact_b_id=cand_fact.id,
                                relation_type=decision.relation_type,
                                explanation=decision.explanation,
                                confidence=decision.confidence,
                                reconciliation_basis=decision.reconciliation_basis,
                                decision_route=decision.decision_route,
                                review_reason=decision.review_reason,
                                comparison_snapshot=decision.comparison_snapshot or {},
                                extraction_confidence_a=fact.confidence,
                                extraction_confidence_b=cand_fact.confidence
                            )
                            db.add(rel)

                db.commit()

            doc.status = JobStatus.DONE
            if failed_chunks_count > 0:
                doc.error_message = f"Completed with warnings: {len(new_facts)} facts extracted; {failed_chunks_count} chunk(s) rate-limited or failed extraction."
            else:
                doc.error_message = None
            db.commit()

            return {
                "document_id": doc.id,
                "status": "done",
                "facts_extracted": len(new_facts),
                "relationships_found": len(reconciled_pairs),
                "failed_chunks": failed_chunks_count
            }

        except TimeoutError as te:
            db.rollback()
            doc.status = JobStatus.FAILED
            doc.error_message = f"Document processing timeout exceeded ({doc_timeout}s): {str(te)}"
            db.commit()
            print(f"[Orchestrator Timeout] {doc.error_message}")
            return {
                "document_id": doc.id,
                "status": "failed",
                "error": doc.error_message
            }
        except Exception as e:
            db.rollback()
            doc.status = JobStatus.FAILED
            doc.error_message = f"{str(e)}\n{traceback.format_exc()}"
            db.commit()
            print(f"[Orchestrator Failure] {doc.error_message}")
            raise e
        finally:
            db.close()


orchestrator = PipelineOrchestrator()
