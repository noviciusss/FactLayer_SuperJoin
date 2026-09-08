"""Stage B: Unconstrained, open-vocabulary Fact Extraction via LLM structured outputs."""
from pathlib import Path
from typing import List, Optional
from src.config import settings
from src.pipeline.llm_client import llm_client
from src.pipeline.schemas import ExtractedFact, FactExtractionResponse

SYSTEM_EXTRACTION_PROMPT = """You are an expert fact extraction engine.
Your task is to extract verifiable, factual assertions from the provided document chunk into structured Fact objects.

CRITICAL EXTRACTION RULES:
1. OPEN VOCABULARY: Do NOT restrict yourself to any predefined set of metrics or fields. 'entity' and 'attribute' must be open free text that faithfully describes what is in the document (e.g. 'Revenue from operations', 'EBITDA', 'Real GDP growth rate', 'PTL freight tonnage', 'Resignation of Director').
2. STRICT GROUNDING: Only extract facts that are explicitly stated or directly computable from this text. Do NOT infer facts that are not directly present.
3. AMBIGUITY HANDLING: If a number or fact lacks a clear period or scope, still extract it, but leave 'period' or 'scope' as null and lower the 'confidence' score (e.g. 0.6 - 0.7).
4. BOILERPLATE & SAFE HARBOR: If the chunk contains only legal disclaimers, forward-looking statements, safe-harbor boilerplate, or table of contents with no factual data, return an EMPTY list of facts. Do NOT manufacture or hallucinate facts.
5. EVIDENCE QUOTE: Every single fact MUST include an 'evidence_quote' that is a short, VERBATIM snippet (<= 30 words) appearing directly in the chunk text.
6. VALUES: Numbers should be extracted as clean strings (e.g., '8142', '76', '8.2%'). If a unit is present (e.g. '₹ Cr', 'million tons', '%'), place the unit in the 'unit' field. Qualitative facts (e.g. 'Director resigned', 'Acquisition completed') are valid facts too.
7. SCOPE & QUALIFIERS: Pay close attention to qualifiers like 'consolidated', 'standalone', 'adjusted', 'reported', 'pro forma', 'preliminary', or footnotes. Place extra contextual notes into the 'qualifiers' JSON object.
"""


class FactExtractor:
    def extract_from_chunk_text(self, chunk_text: str, page_num: int, doc_type: Optional[str] = None) -> List[ExtractedFact]:
        """Extract facts from chunk text."""
        # Pre-filter: if chunk is just boilerplate notice
        lower = chunk_text.lower()
        if "forward-looking statements" in lower and "safe harbor" in lower and len(chunk_text) < 600:
            return []

        user_prompt = f"Page Number: {page_num}\nDocument Type Hint: {doc_type or 'General'}\n\nChunk Text:\n\"\"\"\n{chunk_text}\n\"\"\"\n\nExtract all factual assertions according to the instructions."

        try:
            res: FactExtractionResponse = llm_client.extract_structured(
                prompt=user_prompt,
                system_prompt=SYSTEM_EXTRACTION_PROMPT,
                response_model=FactExtractionResponse
            )
            return res.facts
        except Exception as e:
            print(f"[FactExtractor] Error extracting from chunk (page {page_num}): {e}")
            return []

    def extract_from_chunk_vision(self, image_path: str, page_num: int, doc_type: Optional[str] = None) -> List[ExtractedFact]:
        """
        Two-hop vision extraction:
        1. Plain-text visual description/transcription via vision model (qwen/qwen3.6-27b)
        2. Structured Fact extraction via existing text extractor (openai/gpt-oss-20b)
        3. Flag with extraction_path="vision" and cap confidence at 0.7
        """
        p = Path(image_path)
        if not p.exists():
            return []
        try:
            image_bytes = p.read_bytes()

            # Hop 1: Send image to vision model for plain-text description of numbers and labels
            vision_prompt = (
                f"Page {page_num} of document (slide / infographic).\n"
                "Transcribe and describe in detail all visible numbers, labels, metrics, tables, chart points, "
                "and their visual association (which label belongs to which number)."
            )
            transcription = llm_client.transcribe_image(
                image_bytes=image_bytes,
                prompt=vision_prompt,
                model=settings.VISION_MODEL
            )

            if not transcription or not transcription.strip():
                return []

            # Hop 2: Pipe transcription through existing extractor in strict json_schema mode
            extracted_facts = self.extract_from_chunk_text(
                chunk_text=transcription,
                page_num=page_num,
                doc_type=doc_type
            )

            # Hop 3: Flag with extraction_path="vision" and cap confidence at 0.7
            for fact in extracted_facts:
                if fact.qualifiers is None:
                    fact.qualifiers = {}
                fact.qualifiers["extraction_path"] = "vision"
                fact.confidence = min(fact.confidence, 0.7)

            return extracted_facts
        except Exception as e:
            print(f"[FactExtractor Vision] Error extracting from page {page_num}: {e}")
            return []


fact_extractor = FactExtractor()
