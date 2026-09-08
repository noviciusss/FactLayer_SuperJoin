"""Stage B.2: Self-check node for grounding validation, hallucination suppression, and idempotency."""
import hashlib
import re
from typing import List, Optional, Tuple
from src.pipeline.schemas import ExtractedFact


def normalize_text(text: str) -> str:
    """Normalize text by lowering case, stripping footnote markers/superscripts, collapsing whitespace, and stripping punctuation."""
    if not text:
        return ""
    text = text.lower()
    # Strip unicode superscripts and footnote symbols
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡]", " ", text)
    # Strip bracketed footnote references like [1], (1)
    text = re.sub(r"\[\d+\]|\(\d+\)", " ", text)
    # Normalize zero-width spaces, non-breaking spaces, soft hyphens
    text = re.sub(r"[\u200b\xa0\xad]", " ", text)
    # Strip punctuation and symbols
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SelfChecker:
    def verify_grounding(self, fact: ExtractedFact, chunk_text: str) -> Tuple[bool, float, str]:
        """
        Verify that evidence_quote is genuinely grounded in the source chunk.
        Returns: (is_valid, adjusted_confidence, reason)
        """
        # Check if extracted via two-hop vision path (has no independent text grounding check)
        if fact.qualifiers and fact.qualifiers.get("extraction_path") == "vision":
            capped_conf = round(min(fact.confidence, 0.7), 2)
            return True, capped_conf, "Vision extraction path (confidence capped at 0.7)"

        if not fact.evidence_quote or not fact.evidence_quote.strip():
            return False, 0.0, "Missing evidence quote"

        norm_quote = normalize_text(fact.evidence_quote)
        norm_chunk = normalize_text(chunk_text)

        # 1. Exact normalized substring match
        if norm_quote in norm_chunk:
            return True, fact.confidence, "Verified exact match in source chunk"

        quote_words = norm_quote.split()
        if not quote_words:
            return False, 0.0, "Empty evidence quote words"

        # 2. Token overlap and short quote support
        matched_words = sum(1 for w in quote_words if w in norm_chunk)
        overlap_ratio = matched_words / len(quote_words)

        # For short quotes (1-2 words), if all words appear in chunk, accept
        if len(quote_words) <= 2:
            if matched_words == len(quote_words) or norm_quote in norm_chunk.replace(" ", ""):
                return True, fact.confidence, "Verified short quote match in source chunk"

        # For multi-line, footnoted, or slightly varied quotes (>= 70% overlap)
        if overlap_ratio >= 0.70:
            adj = 0.95 if overlap_ratio >= 0.85 else 0.90
            adjusted_conf = round(max(0.3, fact.confidence * adj), 2)
            return True, adjusted_conf, f"Verified fuzzy match ({round(overlap_ratio*100)}% token overlap)"

        # 3. Grounding failed: evidence quote does NOT appear in source text
        # Per T-03: fact is dropped or confidence heavily penalized (< 0.3)
        return False, 0.2, f"Unsubstantiated evidence quote (only {round(overlap_ratio*100)}% match)"

    def compute_content_hash(self, doc_id: str, fact: ExtractedFact) -> str:
        """
        Deterministic hash for idempotency and duplicate deduplication (T-04).
        Combines doc_id, normalized entity, attribute, period, scope, and value.
        """
        canonical_key = (
            f"{doc_id}|"
            f"{normalize_text(fact.entity)}|"
            f"{normalize_text(fact.attribute)}|"
            f"{normalize_text(fact.period or '')}|"
            f"{normalize_text(fact.scope or '')}|"
            f"{normalize_text(str(fact.value))}"
        )
        return hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()

    def process_facts(
        self,
        facts: List[ExtractedFact],
        chunk_text: str,
        doc_id: str,
        seen_hashes: set
    ) -> List[Tuple[ExtractedFact, str]]:
        """
        Filter hallucinations and deduplicate facts.
        Returns list of (verified_fact, content_hash).
        """
        verified_results = []
        for fact in facts:
            is_valid, adjusted_conf, reason = self.verify_grounding(fact, chunk_text)
            
            # If grounding failed or confidence dropped below threshold, reject hallucination
            if not is_valid and adjusted_conf < 0.3:
                # Dropped as hallucination
                continue

            fact.confidence = adjusted_conf
            chash = self.compute_content_hash(doc_id, fact)

            # Idempotency check: drop exact duplicate facts within or across uploads
            if chash in seen_hashes:
                continue

            seen_hashes.add(chash)
            verified_results.append((fact, chash))

        return verified_results


self_checker = SelfChecker()
