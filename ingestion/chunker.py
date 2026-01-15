"""Text chunking utilities for document processing."""

from dataclasses import dataclass
from typing import List, Tuple

from config.settings import get_ingestion_settings


@dataclass
class Chunk:
    """Represents a text chunk with position information."""

    text: str
    index: int
    start_char: int
    end_char: int


class TextChunker:
    """Fixed-size text chunker with overlap and word boundary awareness."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize chunker with optional custom parameters.

        Args:
            chunk_size: Maximum characters per chunk (default from settings).
            chunk_overlap: Overlap between consecutive chunks (default from settings).
        """
        settings = get_ingestion_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk_text(self, text: str) -> List[Chunk]:
        """Split text into chunks with overlap.

        Attempts to break at word boundaries when possible.

        Args:
            text: Text to chunk.

        Returns:
            List of Chunk objects.
        """
        chunks = []
        # Normalize whitespace
        text = " ".join(text.split())

        if not text:
            return chunks

        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at word boundary if not at end of text
            if end < len(text):
                # Look for last space within chunk
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=index,
                        start_char=start,
                        end_char=min(end, len(text)),
                    )
                )
                index += 1

            # Move start position with overlap
            new_start = end - self.chunk_overlap
            # Prevent infinite loop
            if new_start <= start:
                new_start = end
            start = new_start

        return chunks

    def chunk_with_metadata(
        self, text: str, source: str
    ) -> List[Tuple[str, dict]]:
        """Chunk text and return with metadata for ChromaDB ingestion.

        Args:
            text: Text to chunk.
            source: Source file path or identifier.

        Returns:
            List of (chunk_text, metadata) tuples.
        """
        chunks = self.chunk_text(text)
        total_chunks = len(chunks)

        return [
            (
                chunk.text,
                {
                    "source": source,
                    "chunk_index": chunk.index,
                    "total_chunks": total_chunks,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                },
            )
            for chunk in chunks
        ]
