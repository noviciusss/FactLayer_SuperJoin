"""Stage C: Fact canonicalization and dense local vector embeddings using FastEmbed."""
from typing import List
from fastembed import TextEmbedding
from src.config import settings
from src.pipeline.schemas import ExtractedFact


class FactCanonicalizer:
    def __init__(self):
        # Local, lightweight, ONNX-accelerated embedding model (384 dimensions)
        self.embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)

    def build_canonical_statement(self, fact: ExtractedFact) -> str:
        """
        Generate a standardized one-line canonical assertion for vector search.
        Format: Entity Attribute Period Scope: Value Unit
        Example: 'Delhivery Limited EBITDA FY24 consolidated: 127 ₹ Cr'
        """
        parts = [fact.entity.strip()]
        if fact.attribute:
            parts.append(fact.attribute.strip())
        if fact.period:
            parts.append(f"({fact.period.strip()})")
        if fact.scope:
            parts.append(f"[{fact.scope.strip()}]")

        val_str = str(fact.value).strip()
        if fact.unit:
            val_str += f" {fact.unit.strip()}"

        canonical = f"{' '.join(parts)}: {val_str}"
        return canonical

    def embed_statements(self, statements: List[str]) -> List[List[float]]:
        """Batch embed canonical statements into 384-dimensional dense vectors."""
        if not statements:
            return []
        embeddings = list(self.embedding_model.embed(statements))
        return [emb.tolist() for emb in embeddings]


canonicalizer = FactCanonicalizer()
