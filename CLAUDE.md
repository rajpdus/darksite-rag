# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Document Retrieval Q&A Chatbot system implementing RAG (Retrieval-Augmented Generation) architecture. The system combines BM25 keyword search with vector similarity search (hybrid retrieval) for optimal document question-answering performance.

**Key Architecture**: Single-container deployment using ChromaDB, FastAPI backend, and LLM agents (Claude/GPT/Open-source).

## Core Technology Stack

- **Vector Database**: ChromaDB with hybrid search (BM25 + vector embeddings)
- **Backend API**: FastAPI with async support
- **LLM Integration**: Anthropic Claude (cloud) or Ollama (local/open-source) via strands framework
- **Agent Framework**: strands (model-agnostic, supports both cloud and local LLMs)
- **Local LLM Runtime**: Ollama for running open-source models offline
- **Deployment**: Single Docker container with persistent volumes

## Architecture Layers

### 1. Hybrid Vector Storage Layer (ChromaDB)
- ChromaDB PersistentClient stores document embeddings and enables BM25 search
- Collections organize documents by domain/topic
- Default hybrid search weights: BM25=0.3, Vector=0.7
- Storage path: `data/vector_store`

### 2. RAG Tools Layer
- Tools are decorated with `@tool` from strands framework
- Primary tool: `retrieve_qa_context(question, top_k)` performs hybrid search
- Tools format retrieved documents into LLM-consumable context strings
- Error handling returns degraded responses on retrieval failures

### 3. Agent Orchestration Layer
- Single-agent architecture: `RagAgent` or `DocumentQAAgent` classes
- Agent initialization includes:
  - Model configuration (AnthropicModel, Ollama, or other providers)
  - System prompts (domain-specific Q&A instructions)
  - Tool registration (retrieval functions)
- Memory management handles conversation history
- Supports both cloud (Claude/GPT) and local (Ollama) LLMs via Strands

### 4. API Service Layer (FastAPI)
- REST endpoints for chat interactions
- Session management for multi-turn conversations
- Streaming support for real-time responses
- Health check endpoints for monitoring

## Open Source LLM Integration

### Strands with Ollama

Strands provides model-agnostic support for both cloud and local LLMs. For offline/open-source LLMs, use Ollama with Strands via LiteLLM.

### Ollama Setup

Ollama enables running open-source LLMs locally without internet connectivity:

```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.ai

# Start Ollama service
ollama serve

# Download a model (e.g., Llama 3)
ollama pull llama3

# Verify installation
ollama list
```

### Strands with Ollama Integration Pattern

```python
from strands import Agent
from strands.models import LiteLLMModel

# Configure LiteLLM to use Ollama
llm = LiteLLMModel(
    model="ollama/llama3",
    api_base="http://localhost:11434",
    temperature=0.7
)

# Create agent with retrieval tools
agent = Agent(
    name="document_qa_agent",
    model=llm,
    system_prompt="You are a Q&A assistant. Use retrieve_qa_context before answering.",
    tools=[retrieve_qa_context]
)
```

### Model Selection Guidance

Recommended open-source models for RAG:
- **Llama 3** (8B/70B): Best balance of performance and resource usage
- **Mistral 7B**: Efficient for smaller deployments
- **Phi-3**: Lightweight option for resource-constrained environments
- **Neural Chat**: Good for conversational Q&A

Hardware requirements:
- **7B models**: 8-16GB RAM, suitable for single-container deployments
- **70B models**: 64GB+ RAM, requires larger infrastructure

## Strands Framework for Offline Agents

### Why Strands?

Strands is an excellent choice for offline RAG agents because:

- **Model-Agnostic**: Seamlessly supports both cloud (Anthropic, OpenAI) and local (Ollama) LLMs
- **Advanced Tool Orchestration**: Powerful multi-tool chaining and agent workflows
- **Offline Support**: Works with Ollama via LiteLLM for completely offline operation
- **Multi-Agent Systems**: Supports complex multi-agent collaboration
- **AWS Integration**: Native observability and scaling capabilities
- **Flexible Deployment**: From local development to production on AWS services

### Strands Features for RAG

- Tool decoration with `@tool` for easy retrieval function integration
- Model abstraction layer via LiteLLM (supports 100+ LLM providers)
- Memory management for conversation history
- Streaming support for real-time responses
- Error handling and graceful degradation

## Project Structure (To Be Implemented)

```
darksite-rag/
├── data/
│   ├── vector_store/     # ChromaDB persistent storage
│   └── sessions/         # User session data
├── api/
│   └── main.py          # FastAPI application
├── agents/
│   └── qa_agent.py      # RAG agent implementation
├── tools/
│   └── retrieval.py     # Hybrid search tools
├── vector_store/
│   └── chromadb_client.py  # Vector store initialization
├── config/
│   └── settings.py      # Pydantic configuration
├── requirements.txt
├── Dockerfile
└── spec.md
```

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For open-source LLM support, install Ollama
brew install ollama  # macOS
# or download from https://ollama.ai

# Start Ollama service
ollama serve

# Download a model (e.g., Llama 3)
ollama pull llama3

# Set required environment variables
# For cloud LLM (Anthropic)
export ANTHROPIC_API_KEY="your-key-here"

# For local LLM (Ollama)
export LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3"

# Vector store configuration
export VECTOR_STORE_PATH="data/vector_store"
```

### Running the Application
```bash
# Start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker build and run
docker build -t darksite-rag .
docker run -p 8000:8000 -v $(pwd)/data:/app/data darksite-rag
```

### Vector Store Initialization

**Note**: Initialization and data ingestion are separate concerns:

- **Initialization**: Creates ChromaDB client, collections, and infrastructure
- **Data Ingestion**: Processes documents (loading, chunking, embedding, indexing)

```bash
# Initialize ChromaDB infrastructure only
python -m vector_store.initialize

# Ingest documents into vector store (separate step)
python -m vector_store.ingest --collection documents --path ./data/docs
```

### Testing
```bash
# Run all tests
pytest

# Test hybrid retrieval only
pytest tests/test_retrieval.py

# Test with coverage
pytest --cov=. --cov-report=html
```

## Key Implementation Patterns

### ChromaDB Client Initialization
```python
import chromadb

client = chromadb.PersistentClient(path="./data/vector_store")
collection = client.get_or_create_collection(
    "documents",
    metadata={"hnsw:space": "cosine"}
)
```

### Hybrid Search Pattern
```python
# ChromaDB query automatically uses hybrid search
results = collection.query(
    query_texts=[query],
    n_results=top_k,
    include=["documents", "metadatas", "distances"]
)
```

### Tool Definition Pattern
```python
from strands import tool

@tool
def retrieve_qa_context(question: str, top_k: int = 3) -> str:
    """Retrieve relevant context using hybrid search."""
    results = search_documents(question, top_k=top_k)
    # Format and return context
```

### Agent Initialization Pattern

#### Strands with Cloud LLM (Anthropic Claude)
```python
from strands import Agent, AnthropicModel

agent = Agent(
    name="document_qa_agent",
    model=AnthropicModel(model_id="claude-3-sonnet-20240229"),
    system_prompt="You are a Q&A assistant. Use retrieve_qa_context before answering.",
    tools=[retrieve_qa_context]
)
```

#### Strands with Open-Source LLM (Ollama)
```python
from strands import Agent
from strands.models import LiteLLMModel

# Configure LiteLLM to use Ollama
llm = LiteLLMModel(
    model="ollama/llama3",
    api_base="http://localhost:11434",
    temperature=0.7
)

agent = Agent(
    name="document_qa_agent",
    model=llm,
    system_prompt="You are a Q&A assistant. Use retrieve_qa_context before answering.",
    tools=[retrieve_qa_context]
)
```

## Configuration Management

### Environment Variables

#### LLM Provider Configuration
```bash
# Model provider selection: anthropic, ollama, openai
LLM_PROVIDER=anthropic

# Anthropic Claude (cloud)
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (cloud)
OPENAI_API_KEY=sk-...

# Ollama (local/open-source)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# LLM settings
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
```

#### Vector Store Configuration
```bash
VECTOR_STORE_PATH=data/vector_store
BM25_WEIGHT=0.3
VECTOR_WEIGHT=0.7
TOP_K_RESULTS=5
COLLECTION_NAME=documents
```

#### Data Ingestion Configuration
```bash
# Chunking
INGESTION_CHUNK_SIZE=512
INGESTION_CHUNK_OVERLAP=50
INGESTION_CHUNKING_STRATEGY=fixed

# Batch processing
INGESTION_BATCH_SIZE=100
INGESTION_MAX_WORKERS=4

# Embeddings
INGESTION_EMBEDDING_MODEL=all-MiniLM-L6-v2
INGESTION_EMBEDDING_DIMENSION=384
INGESTION_EMBEDDING_BATCH_SIZE=32

# File handling
INGESTION_MAX_FILE_SIZE_MB=50
INGESTION_SUPPORTED_FORMATS=.pdf,.docx,.txt,.md,.html
INGESTION_REMOVE_DUPLICATES=true
INGESTION_NORMALIZE_TEXT=true
```

#### Service Configuration
```bash
SESSION_DIR=data/sessions
PORT=8000
HOST=0.0.0.0
```

### Configuration Classes

#### Main RAG Settings
```python
from pydantic import BaseSettings
from typing import Literal

class RagSettings(BaseSettings):
    # LLM provider selection
    llm_provider: Literal["anthropic", "ollama", "openai"] = "anthropic"
    
    # Anthropic settings
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-sonnet-20240229"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    
    # LLM parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Vector store settings
    vector_store_path: str = "data/vector_store"
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    top_k_results: int = 5
    collection_name: str = "documents"
    
    # API settings
    port: int = 8000
    host: str = "0.0.0.0"
    session_dir: str = "data/sessions"
    
    class Config:
        env_file = ".env"
```

#### Ingestion Settings (see Data Ingestion Configuration section)

### Hybrid Search Tuning
- **Factual Q&A**: BM25=0.4-0.5 (precise keyword matching)
- **Conceptual Q&A**: Vector=0.7-0.8 (semantic understanding)
- **Balanced**: BM25=0.3 / Vector=0.7 (default)

### Model Provider Selection

The system supports multiple LLM providers. Configure via `LLM_PROVIDER` environment variable:

- **anthropic**: Use Anthropic Claude (requires `ANTHROPIC_API_KEY`)
- **ollama**: Use local Ollama models (requires Ollama service running)
- **openai**: Use OpenAI models (requires `OPENAI_API_KEY`)

Example provider switching:
```python
from config.settings import RagSettings
from strands import Agent
from strands.models import AnthropicModel, LiteLLMModel

settings = RagSettings()

if settings.llm_provider == "ollama":
    # Use Ollama with Strands via LiteLLM
    llm = LiteLLMModel(
        model=f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_base_url
    )
elif settings.llm_provider == "anthropic":
    # Use Anthropic with Strands
    llm = AnthropicModel(model_id=settings.anthropic_model)
elif settings.llm_provider == "openai":
    # Use OpenAI with Strands via LiteLLM
    llm = LiteLLMModel(model=settings.openai_model)

# Create agent with selected model
agent = Agent(
    name="document_qa_agent",
    model=llm,
    system_prompt="You are a Q&A assistant. Use retrieve_qa_context before answering.",
    tools=[retrieve_qa_context]
)
```

## Vector Store Initialization vs Data Ingestion

### Vector Store Initialization

Initialization creates the ChromaDB infrastructure but does NOT handle data ingestion:

```python
import chromadb

# Initialization: Creates client and collections
client = chromadb.PersistentClient(path="./data/vector_store")
collection = client.get_or_create_collection(
    "documents",
    metadata={"hnsw:space": "cosine"}
)
# At this point, collection is empty - no documents ingested
```

### Data Ingestion Pipeline

Data ingestion is a separate process that processes documents:

1. **Document Loading**: Load documents from various formats (PDF, DOCX, TXT, etc.)
2. **Text Chunking**: Split documents into appropriately sized chunks
3. **Metadata Extraction**: Extract source, date, author, category, etc.
4. **Embedding Generation**: Generate vector embeddings for each chunk
5. **Indexing**: Add documents to ChromaDB collection with `collection.add()`
6. **BM25 Indexing**: ChromaDB automatically builds BM25 index during `add()`

The ingestion pipeline must be explicitly configured and executed separately from initialization.

## Document Ingestion Pipeline

When adding documents to the system:
1. Load documents from source (file system, API, database)
2. Chunk documents appropriately (consider document type and chunking strategy)
3. Extract metadata (source, date, author, category)
4. Generate embeddings (using configured embedding model)
5. Use `collection.add()` with documents, metadatas, and ids
6. ChromaDB handles BM25 indexing automatically during `add()`

## Data Ingestion Configuration

Data ingestion requires explicit configuration for optimal performance. Create a dedicated configuration class:

### Ingestion Configuration Class

```python
from pydantic import BaseSettings
from typing import List, Optional

class IngestionConfig(BaseSettings):
    # Chunking parameters
    chunk_size: int = 512  # Characters per chunk
    chunk_overlap: int = 50  # Overlap between chunks
    chunking_strategy: str = "fixed"  # Options: fixed, semantic, sentence
    
    # Batch processing
    batch_size: int = 100  # Documents per batch
    max_workers: int = 4  # Parallel processing threads
    
    # Metadata extraction
    extract_metadata: bool = True
    metadata_fields: List[str] = ["source", "date", "author", "category"]
    auto_extract_dates: bool = True
    
    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"  # HuggingFace model
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    
    # File format support
    supported_formats: List[str] = [".pdf", ".docx", ".txt", ".md", ".html"]
    max_file_size_mb: int = 50
    
    # Processing options
    remove_duplicates: bool = True
    normalize_text: bool = True
    language: str = "en"
    
    class Config:
        env_file = ".env"
        env_prefix = "INGESTION_"
```

### Environment Variables for Ingestion

```bash
# Chunking
INGESTION_CHUNK_SIZE=512
INGESTION_CHUNK_OVERLAP=50
INGESTION_CHUNKING_STRATEGY=fixed

# Batch processing
INGESTION_BATCH_SIZE=100
INGESTION_MAX_WORKERS=4

# Embeddings
INGESTION_EMBEDDING_MODEL=all-MiniLM-L6-v2
INGESTION_EMBEDDING_DIMENSION=384
INGESTION_EMBEDDING_BATCH_SIZE=32

# File handling
INGESTION_MAX_FILE_SIZE_MB=50
INGESTION_SUPPORTED_FORMATS=.pdf,.docx,.txt,.md,.html

# Processing
INGESTION_REMOVE_DUPLICATES=true
INGESTION_NORMALIZE_TEXT=true
INGESTION_LANGUAGE=en
```

### Chunking Strategies

**Fixed-size chunking** (default):
- Simple, predictable chunk sizes
- Good for uniform documents
- Fast processing

**Semantic chunking**:
- Splits at semantic boundaries
- Better context preservation
- Slower but higher quality

**Sentence-aware chunking**:
- Respects sentence boundaries
- Good for natural language documents
- Balanced performance/quality

### Ingestion Pipeline Implementation Pattern

```python
from config.settings import IngestionConfig
import chromadb
from pathlib import Path

config = IngestionConfig()

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./data/vector_store")
collection = client.get_or_create_collection("documents")

# Load and process documents
def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """Simple fixed-size chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Process documents in batches
documents_dir = Path("./data/docs")
all_chunks = []
all_metadatas = []
all_ids = []

for file_path in documents_dir.glob("**/*"):
    if file_path.suffix in config.supported_formats:
        # Load document (implement based on file type)
        text = load_document(file_path)  # Implement document loader
        
        # Chunk document
        chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
        
        # Create metadata for each chunk
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": str(file_path),
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            all_ids.append(f"{file_path.stem}_{i}")

# Add to ChromaDB in batches
for i in range(0, len(all_chunks), config.batch_size):
    batch_chunks = all_chunks[i:i + config.batch_size]
    batch_metadatas = all_metadatas[i:i + config.batch_size]
    batch_ids = all_ids[i:i + config.batch_size]
    
    collection.add(
        documents=batch_chunks,
        metadatas=batch_metadatas,
        ids=batch_ids
    )
```

## Critical Implementation Notes

### Tool Return Format
- Tools MUST return formatted strings for LLM consumption
- Include document source metadata in context
- Number documents for LLM reference (Document 1, Document 2, etc.)

### Error Handling in Retrieval
- Gracefully degrade when retrieval fails
- Return error messages as strings (not exceptions)
- Agent should indicate when context is unavailable

### Session Management
- File-based session storage in `data/sessions`
- Session IDs should be user-specific or generated
- Store conversation history for context continuity

### Agent System Prompts
- Instruct agent to use retrieval tools BEFORE answering
- Define appropriate uncertainty expressions
- Include guidelines for multi-document synthesis

## Single-Container Constraints

- Vector store size: ~10-50GB typical
- Concurrent users: ~10-50 active sessions
- Memory usage: 2-8GB RAM for typical workloads
- All data persists in mounted volumes

## Scaling Considerations

For future scaling beyond single container:
- Deploy ChromaDB as separate service
- Use external session store (Redis)
- Load balance multiple API instances
- Implement shared vector store access

## Security Considerations

- Validate and sanitize all user inputs
- Implement rate limiting per session
- Secure API keys in environment (never commit)
- Consider authentication for production deployments

## Reference: spec.md

The complete specification is in `spec.md` including:
- Detailed code examples for each component
- ChromaDB hybrid search implementation details
- Dockerfile structure
- Score fusion strategies
- Domain adaptation guidelines
