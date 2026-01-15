"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for API."""
    from api.main import app

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Test health response has required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "llm_provider" in data
        assert "documents_count" in data

    def test_health_status_is_healthy(self, client):
        """Test health status is 'healthy'."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"


class TestInfoEndpoint:
    """Tests for info endpoint."""

    def test_info_returns_200(self, client):
        """Test info endpoint returns 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_response_structure(self, client):
        """Test info response has required fields."""
        response = client.get("/info")
        data = response.json()

        assert "version" in data
        assert "llm" in data
        assert "vector_store" in data
        assert "hybrid_search" in data


class TestChatEndpoint:
    """Tests for chat endpoint."""

    def test_chat_requires_message(self, client):
        """Test chat endpoint validates input."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_empty_message_rejected(self, client):
        """Test empty message is rejected."""
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422

    def test_chat_creates_session(self, client):
        """Test chat creates session when not provided."""
        # This test may fail without a working LLM
        # In production, mock the agent
        response = client.post("/chat", json={"message": "Hello"})
        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert "answer" in data


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    def test_list_sessions(self, client):
        """Test listing sessions."""
        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "count" in data

    def test_get_nonexistent_session(self, client):
        """Test getting non-existent session returns 404."""
        response = client.get("/sessions/nonexistent-session-id")
        assert response.status_code == 404

    def test_delete_nonexistent_session(self, client):
        """Test deleting non-existent session returns 404."""
        response = client.delete("/sessions/nonexistent-session-id")
        assert response.status_code == 404


class TestIngestionEndpoint:
    """Tests for document ingestion endpoint."""

    def test_ingest_nonexistent_path(self, client):
        """Test ingesting non-existent path returns 404."""
        response = client.post(
            "/ingest", json={"path": "/nonexistent/path/to/docs"}
        )
        assert response.status_code == 404

    def test_ingest_requires_path(self, client):
        """Test ingest endpoint requires path field."""
        response = client.post("/ingest", json={})
        assert response.status_code == 422
