"""Tests for document ingestion pipeline."""

from pathlib import Path

import pytest

from ingestion.chunker import Chunk, TextChunker
from ingestion.loaders import (
    DocumentLoaderRegistry,
    HTMLLoader,
    MarkdownLoader,
    TextLoader,
)


class TestLoaders:
    """Tests for document loaders."""

    def test_text_loader_supports_txt(self):
        """Test TextLoader supports .txt files."""
        loader = TextLoader()
        assert loader.supports(Path("test.txt"))
        assert not loader.supports(Path("test.pdf"))

    def test_text_loader_load(self, sample_text_file: Path):
        """Test TextLoader loads text files correctly."""
        loader = TextLoader()
        content = loader.load(sample_text_file)
        assert "machine learning" in content.lower()
        assert len(content) > 0

    def test_markdown_loader_supports_md(self):
        """Test MarkdownLoader supports .md files."""
        loader = MarkdownLoader()
        assert loader.supports(Path("readme.md"))
        assert not loader.supports(Path("readme.txt"))

    def test_markdown_loader_load(self, sample_markdown_file: Path):
        """Test MarkdownLoader extracts text from markdown."""
        loader = MarkdownLoader()
        content = loader.load(sample_markdown_file)
        # Should contain text without markdown syntax
        assert "Sample Document" in content
        assert "Section 1" in content
        # Should not contain raw markdown
        assert "**markdown**" not in content

    def test_html_loader_supports_html(self):
        """Test HTMLLoader supports .html files."""
        loader = HTMLLoader()
        assert loader.supports(Path("page.html"))
        assert loader.supports(Path("page.htm"))
        assert not loader.supports(Path("page.txt"))

    def test_html_loader_removes_scripts_and_nav(self, sample_html_file: Path):
        """Test HTMLLoader removes non-content elements."""
        loader = HTMLLoader()
        content = loader.load(sample_html_file)
        # Should contain main content
        assert "Main Content" in content
        assert "document retrieval" in content
        # Should not contain header/footer (removed)
        assert "Header content to ignore" not in content
        assert "Footer to ignore" not in content

    def test_registry_selects_correct_loader(self):
        """Test DocumentLoaderRegistry selects appropriate loader."""
        registry = DocumentLoaderRegistry()

        assert isinstance(registry.get_loader(Path("test.txt")), TextLoader)
        assert isinstance(registry.get_loader(Path("test.md")), MarkdownLoader)
        assert isinstance(registry.get_loader(Path("test.html")), HTMLLoader)
        assert registry.get_loader(Path("test.unknown")) is None

    def test_registry_is_supported(self):
        """Test DocumentLoaderRegistry.is_supported method."""
        registry = DocumentLoaderRegistry()

        assert registry.is_supported(Path("file.txt"))
        assert registry.is_supported(Path("file.md"))
        assert registry.is_supported(Path("file.html"))
        assert registry.is_supported(Path("file.pdf"))
        assert registry.is_supported(Path("file.docx"))
        assert not registry.is_supported(Path("file.xyz"))

    def test_registry_load(self, sample_text_file: Path):
        """Test DocumentLoaderRegistry.load method."""
        registry = DocumentLoaderRegistry()
        content = registry.load(sample_text_file)
        assert "machine learning" in content.lower()

    def test_registry_load_unsupported_raises(self, temp_dir: Path):
        """Test loading unsupported format raises ValueError."""
        registry = DocumentLoaderRegistry()
        unsupported_file = temp_dir / "file.xyz"
        unsupported_file.write_text("content")

        with pytest.raises(ValueError, match="No loader available"):
            registry.load(unsupported_file)


class TestChunker:
    """Tests for text chunking."""

    def test_chunk_basic(self):
        """Test basic chunking functionality."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a test. " * 20  # ~320 characters
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.text) <= 100

    def test_chunk_respects_size(self):
        """Test chunks respect size limits."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        text = "word " * 100  # ~500 characters

        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            assert len(chunk.text) <= 50

    def test_chunk_empty_text(self):
        """Test chunking empty text returns empty list."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text returns empty list."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_text("   \n\t  ")
        assert chunks == []

    def test_chunk_small_text(self):
        """Test text smaller than chunk size creates single chunk."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "Small text"
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == "Small text"

    def test_chunk_indices_are_sequential(self):
        """Test chunk indices are sequential starting from 0."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        text = "word " * 50

        chunks = chunker.chunk_text(text)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_with_metadata(self, sample_text_file: Path):
        """Test chunking with metadata generation."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = sample_text_file.read_text()
        source = str(sample_text_file)

        results = chunker.chunk_with_metadata(text, source)

        assert len(results) > 0
        for chunk_text, metadata in results:
            assert isinstance(chunk_text, str)
            assert metadata["source"] == source
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert "start_char" in metadata
            assert "end_char" in metadata

    def test_chunk_metadata_total_chunks_consistent(self):
        """Test all chunks have same total_chunks value."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        text = "word " * 50

        results = chunker.chunk_with_metadata(text, "test.txt")
        total_chunks_values = [m["total_chunks"] for _, m in results]

        assert len(set(total_chunks_values)) == 1  # All same value
        assert total_chunks_values[0] == len(results)
