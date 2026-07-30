"""
Elasticsearch search engine.

Supports:
- BM25 text search
- Vector similarity search
- Hybrid search
"""

from typing import Any, Dict, List

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from eval.evaluate_utils import rrf


class SearchEngine:
    def __init__(self, host: str, index_name: str, embedding_field: str = "embedding"):
        self.client = Elasticsearch(host)

        self.index_name = index_name
        self.embedding_field = embedding_field
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def text_search(
        self,
        query: str,
        size: int = 5,
        field_boosts: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        """

        BM25 keyword search.

        """

        if field_boosts is None:
            field_boosts = {
                "page_title": 1.0,
                "section": 1.0,
                "text": 1.0,
            }

        fields = [
            f"{field}^{boost}" if boost != 1 else field
            for field, boost in field_boosts.items()
        ]
        

        response = self.client.search(
            index=self.index_name,
            size=size,
            query={
                "multi_match": {
                    "query": query,
                    "fields": fields,
                }
            },
        )

        return self._format_results(response)

    def vector_search(
        self, query: str, k: int = 5, num_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search using kNN.
        """

        query_vector = self.model.encode(query).tolist()

        response = self.client.search(
            index=self.index_name,
            knn={
                "field": self.embedding_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
            },
        )

        return self._format_results(response)

    def hybrid_search(
        self, query: str, num_results=5, rrf_k=60
    ) -> List[Dict[str, Any]]:
        """
        Hybrid BM25 + Vector search.

        Uses Reciprocal Rank Fusion style merging.
        """
        text_results = self.text_search(query)
        vector_results = self.vector_search(query)
        return rrf([text_results, vector_results], k=rrf_k, num_results=num_results)

    def _merge_results(
        self, text_results, vector_results, text_weight, vector_weight, size
    ):

        scores = {}

        for rank, doc in enumerate(text_results):
            doc_id = doc["id"]
            score = text_weight / (rank + 1)

            scores[doc_id] = scores.get(doc_id, 0) + score

        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            score = vector_weight / (rank + 1)

            scores[doc_id] = scores.get(doc_id, 0) + score

        ranked_ids = sorted(scores, key=scores.get, reverse=True)

        result_map = {}
        for doc in text_results + vector_results:
            result_map[doc["id"]] = doc

        results = []
        for doc_id in ranked_ids[:size]:
            result = result_map[doc_id]
            result["hybrid_score"] = scores[doc_id]
            results.append(result)
        return results

    def _format_results(self, response):

        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                {
                    "id": source.get("id", hit["_id"]),
                    "score": hit.get("_score"),
                    "text": source.get("text"),
                    "city": source.get("city"),
                    "section": source.get("section"),
                }
            )

        return results
