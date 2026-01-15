"""Document ingestion pipeline for processing and indexing documents."""

import hashlib
from pathlib import Path
from typing import List

from config.settings import get_ingestion_settings
from ingestion.chunker import TextChunker
from ingestion.loaders import DocumentLoaderRegistry
from vector_store.chromadb_client import get_chromadb_client


class IngestionPipeline:
    """Orchestrates document loading, chunking, and indexing."""

    def __init__(self):
        self.settings = get_ingestion_settings()
        self.loader_registry = DocumentLoaderRegistry()
        self.chunker = TextChunker()
        self.chromadb = get_chromadb_client()

    def _generate_doc_id(self, source: str, chunk_index: int) -> str:
        """Generate unique document ID using MD5 hash.

        Args:
            source: Source file path.
            chunk_index: Index of the chunk within the document.

        Returns:
            Unique document ID.
        """
        hash_input = f"{source}_{chunk_index}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _process_file(self, file_path: Path) -> List[tuple]:
        """Process a single file and return chunks with metadata.

        Args:
            file_path: Path to the file to process.

        Returns:
            List of (chunk_text, metadata) tuples.
        """
        try:
            # Load document
            text = self.loader_registry.load(file_path)

            # Normalize text if configured
            if self.settings.normalize_text:
                text = " ".join(text.split())

            # Chunk document
            chunks_with_metadata = self.chunker.chunk_with_metadata(
                text, source=str(file_path)
            )

            return chunks_with_metadata
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single file into the vector store.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            Number of chunks indexed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.loader_registry.is_supported(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        chunks_with_metadata = self._process_file(file_path)

        if not chunks_with_metadata:
            return 0

        documents = []
        metadatas = []
        ids = []

        for chunk_text, metadata in chunks_with_metadata:
            doc_id = self._generate_doc_id(
                metadata["source"], metadata["chunk_index"]
            )
            documents.append(chunk_text)
            metadatas.append(metadata)
            ids.append(doc_id)

        # Add to ChromaDB in batches
        batch_size = self.settings.batch_size
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            self.chromadb.collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )

        return len(documents)

    def ingest_directory(self, directory: Path, recursive: bool = True) -> dict:
        """Ingest all supported files from a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to process subdirectories.

        Returns:
            Dictionary with ingestion statistics.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        results = {
            "processed_files": 0,
            "total_chunks": 0,
            "errors": [],
        }

        # Get supported file extensions
        supported_extensions = set(self.settings.supported_formats)

        # Collect files
        pattern = "**/*" if recursive else "*"
        files = [
            f
            for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        for file_path in files:
            try:
                chunk_count = self.ingest_file(file_path)
                results["processed_files"] += 1
                results["total_chunks"] += chunk_count
                print(f"Ingested {chunk_count} chunks from {file_path}")
            except Exception as e:
                error_info = {"file": str(file_path), "error": str(e)}
                results["errors"].append(error_info)
                print(f"Error ingesting {file_path}: {e}")

        return results

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection.

        Returns:
            Dictionary with collection statistics.
        """
        return {
            "document_count": self.chromadb.collection.count(),
            "collection_name": self.chromadb.collection.name,
        }
