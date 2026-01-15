"""Tests for hybrid search and retrieval."""

import pytest

from tools.retrieval import HybridSearchEngine


class TestHybridSearchEngine:
    """Tests for HybridSearchEngine."""

    def test_rrf_fusion_basic(self):
        """Test basic RRF fusion logic."""
        engine = HybridSearchEngine()

        # Vector results: doc1 first, doc2 second
        vector_results = [("doc1", 0.1), ("doc2", 0.2), ("doc3", 0.3)]
        # BM25 results: doc2 first, doc1 second
        bm25_results = [("doc2", 10.0), ("doc1", 8.0), ("doc4", 5.0)]

        fused = engine._rrf_fusion(vector_results, bm25_results)

        # doc1 and doc2 appear in both, should have highest scores
        fused_ids = [doc_id for doc_id, _ in fused]
        assert "doc1" in fused_ids[:2]
        assert "doc2" in fused_ids[:2]

    def test_rrf_fusion_empty_lists(self):
        """Test RRF fusion with empty lists."""
        engine = HybridSearchEngine()

        fused = engine._rrf_fusion([], [])
        assert fused == []

    def test_rrf_fusion_one_empty(self):
        """Test RRF fusion when one list is empty."""
        engine = HybridSearchEngine()

        vector_results = [("doc1", 0.1), ("doc2", 0.2)]
        fused = engine._rrf_fusion(vector_results, [])

        assert len(fused) == 2
        fused_ids = [doc_id for doc_id, _ in fused]
        assert "doc1" in fused_ids
        assert "doc2" in fused_ids

    def test_rrf_fusion_no_overlap(self):
        """Test RRF fusion when results don't overlap."""
        engine = HybridSearchEngine()

        vector_results = [("doc1", 0.1), ("doc2", 0.2)]
        bm25_results = [("doc3", 10.0), ("doc4", 8.0)]

        fused = engine._rrf_fusion(vector_results, bm25_results)

        assert len(fused) == 4
        fused_ids = [doc_id for doc_id, _ in fused]
        assert set(fused_ids) == {"doc1", "doc2", "doc3", "doc4"}

    def test_rrf_fusion_scores_are_positive(self):
        """Test all RRF scores are positive."""
        engine = HybridSearchEngine()

        vector_results = [("doc1", 0.1), ("doc2", 0.2)]
        bm25_results = [("doc2", 10.0), ("doc3", 8.0)]

        fused = engine._rrf_fusion(vector_results, bm25_results)

        for _, score in fused:
            assert score > 0

    def test_rrf_fusion_sorted_descending(self):
        """Test RRF results are sorted by score descending."""
        engine = HybridSearchEngine()

        vector_results = [("doc1", 0.1), ("doc2", 0.2), ("doc3", 0.3)]
        bm25_results = [("doc2", 10.0), ("doc1", 8.0), ("doc3", 5.0)]

        fused = engine._rrf_fusion(vector_results, bm25_results)
        scores = [score for _, score in fused]

        assert scores == sorted(scores, reverse=True)

    def test_search_empty_collection(self):
        """Test search on empty collection returns empty list."""
        engine = HybridSearchEngine()
        # Force empty state
        engine._documents = []
        engine._doc_ids = []
        engine._bm25_index = None

        results = engine.search("test query")
        assert results == []


class TestRetrievalTool:
    """Tests for retrieve_qa_context tool."""

    def test_retrieve_qa_context_returns_string(self):
        """Test retrieve_qa_context returns a string."""
        from tools.retrieval import retrieve_qa_context

        result = retrieve_qa_context("What is machine learning?")
        assert isinstance(result, str)

    def test_retrieve_qa_context_handles_empty(self):
        """Test retrieve_qa_context handles empty results gracefully."""
        from tools.retrieval import retrieve_qa_context

        result = retrieve_qa_context("completely random query xyz123")
        assert isinstance(result, str)
        # Should return either results or "No relevant documents" message
        assert len(result) > 0
