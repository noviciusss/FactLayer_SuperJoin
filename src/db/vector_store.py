"""Qdrant vector store interface supporting Docker and local embedded storage."""
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.config import settings


class VectorStore:
    def __init__(self):
        self._client = None
        self.collection_name = settings.QDRANT_COLLECTION

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            try:
                if settings.QDRANT_URL:
                    self._client = QdrantClient(
                        url=settings.QDRANT_URL,
                        api_key=settings.QDRANT_API_KEY
                    )
                else:
                    self._client = QdrantClient(path=str(settings.QDRANT_STORAGE_PATH))
            except Exception as e:
                print(f"[VectorStore Warning] Could not acquire disk lock on {settings.QDRANT_STORAGE_PATH}: {e}")
                # In-memory fallback if another process holds the disk lock
                self._client = QdrantClient(location=":memory:")
        return self._client

    def init_collection(self, vector_size: int = 384):
        """Ensure the collection exists with the required vector dimension and cosine metric."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE
                )
            )

    def upsert_facts(
        self,
        fact_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ):
        """Batch upsert fact vectors with payload metadata."""
        if not fact_ids:
            return
        points = [
            qmodels.PointStruct(
                id=f_id,
                vector=vector,
                payload=payload
            )
            for f_id, vector, payload in zip(fact_ids, vectors, payloads)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_candidates(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.72,
        exclude_fact_id: Optional[str] = None,
        exclude_document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find candidate facts matching the query vector above score threshold."""
        must_not_conditions = []
        if exclude_fact_id:
            must_not_conditions.append(
                qmodels.HasIdCondition(has_id=[exclude_fact_id])
            )
        if exclude_document_id:
            must_not_conditions.append(
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=exclude_document_id)
                )
            )

        query_filter = None
        if must_not_conditions:
            query_filter = qmodels.Filter(must_not=must_not_conditions)

        try:
            # Modern qdrant-client >= 1.10 uses query_points
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            ).points
        except AttributeError:
            # Fallback for older client versions if search() exists
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )

        return [
            {
                "fact_id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]


# Singleton instance
vector_store = VectorStore()
