"""ChromaDB client initialization and collection management."""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config.settings import get_ingestion_settings, get_settings


class ChromaDBClient:
    """Singleton ChromaDB client for vector storage operations."""

    _instance = None
    _client = None
    _collection = None
    _embedding_fn = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        settings = get_settings()
        ingestion_settings = get_ingestion_settings()

        # Ensure storage directory exists
        Path(settings.vector_store_path).mkdir(parents=True, exist_ok=True)

        # Initialize persistent client
        self._client = chromadb.PersistentClient(
            path=settings.vector_store_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create embedding function using SentenceTransformer
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=ingestion_settings.embedding_model
        )

        # Get or create collection with cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def client(self):
        """Get the ChromaDB client instance."""
        return self._client

    @property
    def collection(self):
        """Get the documents collection."""
        return self._collection

    @property
    def embedding_function(self):
        """Get the embedding function."""
        return self._embedding_fn

    def reset_collection(self):
        """Delete and recreate the collection."""
        settings = get_settings()
        self._client.delete_collection(settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )


def get_chromadb_client() -> ChromaDBClient:
    """Get the singleton ChromaDB client instance."""
    return ChromaDBClient()
