"""Pytest fixtures and configuration for RAG system tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Set test environment before importing modules
os.environ["VECTOR_STORE_PATH"] = tempfile.mkdtemp()
os.environ["SESSION_DIR"] = tempfile.mkdtemp()
os.environ["LLM_PROVIDER"] = "ollama"  # Use ollama for tests (can mock)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text(
        "This is a sample document about machine learning. "
        "Machine learning is a subset of artificial intelligence. "
        "It allows computers to learn from data without being explicitly programmed."
    )
    return file_path


@pytest.fixture
def sample_markdown_file(temp_dir: Path) -> Path:
    """Create a sample markdown file for testing."""
    file_path = temp_dir / "readme.md"
    file_path.write_text(
        "# Sample Document\n\n"
        "This is a **markdown** document.\n\n"
        "## Section 1\n\n"
        "Some content here about testing.\n\n"
        "## Section 2\n\n"
        "More content about RAG systems."
    )
    return file_path


@pytest.fixture
def sample_html_file(temp_dir: Path) -> Path:
    """Create a sample HTML file for testing."""
    file_path = temp_dir / "page.html"
    file_path.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<header>Header content to ignore</header>
<main>
<h1>Main Content</h1>
<p>This is the main content of the page about document retrieval.</p>
<p>RAG systems combine retrieval with generation.</p>
</main>
<footer>Footer to ignore</footer>
</body>
</html>"""
    )
    return file_path


@pytest.fixture
def sample_documents_dir(temp_dir: Path) -> Path:
    """Create a directory with sample documents for testing."""
    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()

    # Create text file
    (docs_dir / "doc1.txt").write_text(
        "Python is a popular programming language. "
        "It is widely used for data science and machine learning."
    )

    # Create another text file
    (docs_dir / "doc2.txt").write_text(
        "Vector databases store embeddings for semantic search. "
        "ChromaDB is an open-source vector database."
    )

    # Create markdown file
    (docs_dir / "notes.md").write_text(
        "# Notes\n\n"
        "These are notes about the RAG system.\n"
        "It uses hybrid search for better results."
    )

    return docs_dir


@pytest.fixture
def mock_settings(temp_dir: Path):
    """Create mock settings with temporary paths."""
    from config.settings import RagSettings

    return RagSettings(
        vector_store_path=str(temp_dir / "vector_store"),
        session_dir=str(temp_dir / "sessions"),
        llm_provider="ollama",
        ollama_model="llama3",
    )
