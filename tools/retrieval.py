"""Hybrid search implementation combining BM25 and vector similarity."""

from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi
from strands import tool

from config.settings import get_settings
from vector_store.chromadb_client import get_chromadb_client


class HybridSearchEngine:
    """Combines BM25 keyword search with vector similarity using RRF fusion."""

    def __init__(self):
        self.settings = get_settings()
        self.chromadb = get_chromadb_client()
        self._bm25_index = None
        self._doc_ids: List[str] = []
        self._documents: List[str] = []
        self._metadatas: List[dict] = []

    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection."""
        # Get all documents from collection
        all_docs = self.chromadb.collection.get(include=["documents", "metadatas"])

        if not all_docs["documents"]:
            self._bm25_index = None
            self._doc_ids = []
            self._documents = []
            self._metadatas = []
            return

        self._documents = all_docs["documents"]
        self._doc_ids = all_docs["ids"]
        self._metadatas = all_docs["metadatas"]

        # Tokenize documents for BM25 (simple whitespace tokenization)
        tokenized_docs = [doc.lower().split() for doc in self._documents]
        self._bm25_index = BM25Okapi(tokenized_docs)

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """Combine vector and BM25 results using Reciprocal Rank Fusion.

        Args:
            vector_results: List of (doc_id, distance) from vector search.
            bm25_results: List of (doc_id, score) from BM25 search.
            k: RRF constant (default 60).

        Returns:
            List of (doc_id, rrf_score) sorted by score descending.
        """
        rrf_scores: Dict[str, float] = {}

        # Add vector search ranks (weighted)
        for rank, (doc_id, _) in enumerate(vector_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                self.settings.vector_weight / (rank + k)
            )

        # Add BM25 ranks (weighted)
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                self.settings.bm25_weight / (rank + k)
            )

        # Sort by RRF score (descending)
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and BM25.

        Args:
            query: Search query string.
            top_k: Number of results to return (default from settings).

        Returns:
            List of result dictionaries with id, document, metadata, and score.
        """
        top_k = top_k or self.settings.top_k_results

        # Rebuild BM25 index if needed (lazy initialization)
        if self._bm25_index is None:
            self._build_bm25_index()

        if not self._documents:
            return []

        # Determine how many results to fetch for ranking
        fetch_k = min(top_k * 2, len(self._documents))

        # Vector search via ChromaDB
        vector_results = self.chromadb.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        # Extract vector results as (id, distance) pairs
        vector_ranked = list(
            zip(vector_results["ids"][0], vector_results["distances"][0])
        )

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25_index.get_scores(tokenized_query)

        # Get top BM25 results as (id, score) pairs
        bm25_ranked = sorted(
            zip(self._doc_ids, bm25_scores), key=lambda x: x[1], reverse=True
        )[:fetch_k]

        # RRF fusion
        fused_results = self._rrf_fusion(vector_ranked, bm25_ranked)[:top_k]

        # Build final results with documents and metadata
        results = []
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self._doc_ids)}

        for doc_id, rrf_score in fused_results:
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                results.append(
                    {
                        "id": doc_id,
                        "document": self._documents[idx],
                        "metadata": self._metadatas[idx],
                        "score": rrf_score,
                    }
                )

        return results

    def refresh_index(self):
        """Force rebuild of BM25 index."""
        self._bm25_index = None
        self._build_bm25_index()


# Global search engine instance (singleton)
_search_engine = None


def get_search_engine() -> HybridSearchEngine:
    """Get the singleton search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearchEngine()
    return _search_engine


@tool
def retrieve_qa_context(question: str, top_k: int = 5) -> str:
    """
    Retrieve relevant document context for answering a question using hybrid search.

    This tool performs hybrid search combining BM25 keyword matching with
    vector similarity to find the most relevant documents for the given question.
    Always use this tool before answering questions about the documents.

    Args:
        question: The user's question to find relevant context for.
        top_k: Number of document chunks to retrieve (default: 5).

    Returns:
        Formatted string containing numbered documents with their sources,
        ready for the LLM to use as context for answering.
    """
    try:
        search_engine = get_search_engine()
        results = search_engine.search(question, top_k=top_k)

        if not results:
            return "No relevant documents found in the knowledge base."

        # Format results for LLM consumption
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            chunk_idx = result["metadata"].get("chunk_index", 0)
            total_chunks = result["metadata"].get("total_chunks", 1)

            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source} (chunk {chunk_idx + 1}/{total_chunks})\n"
                f"Content: {result['document']}\n"
            )

        return "\n---\n".join(context_parts)

    except Exception as e:
        return (
            f"Error retrieving context: {str(e)}. "
            "Please try rephrasing your question."
        )
