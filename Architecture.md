# Architecture Documentation

This document describes the architecture of the Document Q&A RAG System, including component design, data flow, and implementation details.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                    │
│                     (React + TypeScript + Vite)                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP/SSE
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API Layer (FastAPI)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  /chat   │  │ /ingest  │  │ /health  │  │/sessions │               │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └────┬─────┘               │
└───────┼─────────────┼───────────────────────────┼───────────────────────┘
        │             │                           │
        ▼             ▼                           ▼
┌───────────────┐ ┌───────────────┐     ┌───────────────┐
│  Agent Layer  │ │   Ingestion   │     │    Session    │
│  (Strands)    │ │   Pipeline    │     │   Manager     │
└───────┬───────┘ └───────┬───────┘     └───────┬───────┘
        │                 │                     │
        ▼                 │                     ▼
┌───────────────┐         │             ┌───────────────┐
│  RAG Tools    │         │             │  File-based   │
│ (Retrieval)   │         │             │   Sessions    │
└───────┬───────┘         │             └───────────────┘
        │                 │
        ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Vector Store Layer (ChromaDB)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ Vector Embeddings│  │   BM25 Index    │  │    Metadata     │         │
│  │ (all-MiniLM-L6) │  │  (rank_bm25)    │  │    Storage      │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM Provider Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │  Anthropic  │  │   OpenAI    │  │   Ollama    │                     │
│  │   Claude    │  │    GPT      │  │   (Local)   │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Layer Architecture

### 1. API Layer (`api/`)

The FastAPI application serves as the HTTP interface for all client interactions.

**Components:**
- `main.py` - Application entry point with route definitions
- `models.py` - Pydantic schemas for request/response validation

**Endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System health check |
| `/info` | GET | Detailed configuration info |
| `/chat` | POST | Synchronous Q&A |
| `/chat/stream` | POST | SSE streaming Q&A |
| `/ingest` | POST | Document ingestion |
| `/sessions/*` | CRUD | Session management |

**Key Features:**
- CORS middleware for frontend communication
- Lifespan events for service initialization
- SSE streaming via `sse-starlette`

### 2. Agent Layer (`agents/`)

Orchestrates LLM interactions using the Strands Agents framework.

**Components:**

#### `model_factory.py`
Factory pattern for creating LLM model instances:

```python
def create_model():
    if provider == "anthropic":
        return AnthropicModel(...)
    elif provider == "openai":
        return LiteLLMModel(model_id="gpt-4o", ...)
    elif provider == "ollama":
        return LiteLLMModel(model_id="ollama/llama3", ...)
```

#### `qa_agent.py`
Document Q&A agent with tool access:

```python
class DocumentQAAgent:
    def __init__(self):
        self.agent = Agent(
            model=create_model(),
            system_prompt=SYSTEM_PROMPT,
            tools=[retrieve_qa_context],  # RAG tool
        )
```

**System Prompt Strategy:**
The agent is instructed to:
1. Always use `retrieve_qa_context` tool before answering
2. Base answers only on retrieved context
3. Cite document sources
4. Indicate uncertainty when information is unavailable

#### `session.py`
File-based session persistence:

```
data/sessions/
├── {session_id_1}.json
├── {session_id_2}.json
└── ...
```

Each session file contains:
```json
{
  "session_id": "uuid",
  "created_at": "ISO timestamp",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ]
}
```

### 3. Tools Layer (`tools/`)

RAG-specific tools that agents can invoke.

#### `retrieval.py` - Hybrid Search Engine

**Architecture:**
```
Query
  │
  ├──────────────────┬──────────────────┐
  ▼                  ▼                  │
┌──────────┐   ┌──────────┐            │
│  Vector  │   │   BM25   │            │
│  Search  │   │  Search  │            │
│(ChromaDB)│   │(rank_bm25)│            │
└────┬─────┘   └────┬─────┘            │
     │              │                   │
     ▼              ▼                   │
┌─────────────────────────────┐        │
│    Reciprocal Rank Fusion   │        │
│         (RRF, k=60)         │        │
└──────────────┬──────────────┘        │
               │                        │
               ▼                        │
        Ranked Results ─────────────────┘
```

**RRF Score Calculation:**
```python
rrf_score[doc] = Σ (weight / (rank + k))
```

Where:
- `k = 60` (constant to prevent division by zero effects)
- `vector_weight = 0.7` (default)
- `bm25_weight = 0.3` (default)

**Tool Interface:**
```python
@tool
def retrieve_qa_context(question: str, top_k: int = 5) -> str:
    """Returns formatted context for LLM consumption."""
```

Output format:
```
[Document 1]
Source: /path/to/doc.pdf (chunk 1/5)
Content: ...

---

[Document 2]
Source: /path/to/doc.txt (chunk 3/10)
Content: ...
```

### 4. Ingestion Layer (`ingestion/`)

Processes documents from various formats into searchable chunks.

#### Pipeline Flow

```
Document Files
      │
      ▼
┌─────────────────┐
│    Loaders      │  PDFLoader, DOCXLoader, TextLoader,
│   (Strategy)    │  MarkdownLoader, HTMLLoader
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunker      │  Fixed-size: 512 chars
│                 │  Overlap: 50 chars
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Pipeline      │  Batch processing (100/batch)
│                 │  MD5 deduplication
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │  Vector embeddings
│                 │  BM25 indexing
└─────────────────┘
```

#### Document Loaders

| Loader | Formats | Implementation |
|--------|---------|----------------|
| `PDFLoader` | .pdf | pypdf |
| `DOCXLoader` | .docx | python-docx |
| `TextLoader` | .txt | Built-in |
| `MarkdownLoader` | .md | markdown + BeautifulSoup |
| `HTMLLoader` | .html, .htm | BeautifulSoup (strips nav/footer) |

#### Chunking Strategy

**Fixed-Size Chunking with Word Boundary Awareness:**

```python
def chunk_text(text, chunk_size=512, overlap=50):
    # 1. Normalize whitespace
    # 2. Find chunk boundary
    # 3. Prefer breaking at word boundaries
    # 4. Apply overlap for context continuity
```

**Chunk Metadata:**
```python
{
    "source": "/path/to/document.pdf",
    "chunk_index": 0,
    "total_chunks": 5,
    "start_char": 0,
    "end_char": 512
}
```

### 5. Vector Store Layer (`vector_store/`)

ChromaDB-based persistent storage with hybrid search capabilities.

#### ChromaDB Configuration

```python
client = chromadb.PersistentClient(
    path="data/vector_store",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    ),
    metadata={"hnsw:space": "cosine"}
)
```

**Storage Structure:**
```
data/vector_store/
├── chroma.sqlite3          # Metadata and mappings
└── {collection_id}/        # Vector data
    ├── data_level0.bin     # HNSW index
    ├── header.bin
    ├── index_metadata.json
    └── length.bin
```

#### Embedding Model

**all-MiniLM-L6-v2:**
- Dimensions: 384
- Size: ~80MB
- Performance: Good balance of speed and quality
- Use case: General-purpose semantic similarity

### 6. Configuration Layer (`config/`)

Pydantic-based configuration management with environment variable support.

#### Settings Classes

```python
class RagSettings(BaseSettings):
    # LLM Configuration
    llm_provider: Literal["anthropic", "ollama", "openai"]
    anthropic_api_key: str
    ollama_base_url: str

    # Vector Store
    vector_store_path: str
    bm25_weight: float
    vector_weight: float

    class Config:
        env_file = ".env"

class IngestionSettings(BaseSettings):
    chunk_size: int
    chunk_overlap: int
    embedding_model: str

    class Config:
        env_prefix = "INGESTION_"
```

#### Singleton Pattern

```python
@lru_cache()
def get_settings() -> RagSettings:
    return RagSettings()
```

## Data Flow

### Query Flow (Chat)

```
1. User sends question via POST /chat
                │
                ▼
2. API creates/retrieves session
                │
                ▼
3. Agent receives question
                │
                ▼
4. Agent invokes retrieve_qa_context tool
                │
                ▼
5. HybridSearchEngine performs:
   a. Vector search (ChromaDB)
   b. BM25 search (rank_bm25)
   c. RRF fusion
                │
                ▼
6. Formatted context returned to agent
                │
                ▼
7. Agent generates answer using LLM
                │
                ▼
8. Response saved to session
                │
                ▼
9. Answer returned to user
```

### Ingestion Flow

```
1. User calls POST /ingest with path
                │
                ▼
2. Pipeline discovers supported files
                │
                ▼
3. For each file:
   a. Loader extracts text
   b. Chunker splits into chunks
   c. Metadata generated (source, index)
   d. Document ID computed (MD5)
                │
                ▼
4. Batch insert into ChromaDB:
   a. Embeddings generated
   b. BM25 index updated
   c. Metadata stored
                │
                ▼
5. Search engine index refreshed
                │
                ▼
6. Statistics returned to user
```

## Hybrid Search Algorithm

### Why Hybrid Search?

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| BM25 Only | Exact keyword matches, acronyms | Misses synonyms, context |
| Vector Only | Semantic similarity | Can miss exact terms |
| **Hybrid** | **Best of both** | Slightly more complex |

### RRF Fusion Algorithm

Reciprocal Rank Fusion combines rankings from multiple search methods:

```python
def rrf_fusion(vector_results, bm25_results, k=60):
    scores = {}

    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (rank + k)

    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (rank + k)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Why k=60?**
- Standard value from research
- Prevents over-weighting top results
- Balances contribution across rank positions

### Weight Tuning Guidelines

| Use Case | BM25 Weight | Vector Weight |
|----------|-------------|---------------|
| Factual Q&A (dates, names) | 0.4-0.5 | 0.5-0.6 |
| Conceptual Q&A | 0.2-0.3 | 0.7-0.8 |
| **General (default)** | **0.3** | **0.7** |

## Scalability Considerations

### Current Limits (Single Container)

| Resource | Limit |
|----------|-------|
| Vector Store Size | 10-50GB |
| Concurrent Users | 10-50 |
| Memory Usage | 2-8GB RAM |

### Scaling Strategies

1. **Horizontal API Scaling**
   - Multiple API containers
   - Shared ChromaDB volume or external ChromaDB server

2. **Vector Store Scaling**
   - ChromaDB server mode
   - Migration to Qdrant or Pinecone

3. **Session Store Scaling**
   - Redis for session storage
   - Database-backed sessions

## Security Considerations

### Input Validation
- Pydantic models validate all API inputs
- Path traversal prevention in session IDs
- File type validation in ingestion

### Secrets Management
- API keys via environment variables
- Never logged or exposed in responses
- `.env` file excluded from git

### Production Recommendations
- Enable authentication middleware
- Configure CORS for specific origins
- Rate limiting per session/IP
- HTTPS termination at load balancer

## Testing Strategy

### Unit Tests
- Loaders: Format-specific text extraction
- Chunker: Size limits, overlap, boundaries
- RRF: Score calculation, ranking

### Integration Tests
- API endpoints with TestClient
- Ingestion pipeline end-to-end
- Search with indexed documents

### Test Fixtures
```python
@pytest.fixture
def sample_documents_dir(temp_dir):
    """Creates temporary documents for testing."""
    # Creates .txt, .md files
    return docs_dir
```

## Deployment Architecture

### Docker Single Container

```dockerfile
FROM python:3.11-slim

# Pre-download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Volume mount for persistence
VOLUME /app/data

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Services

```yaml
services:
  rag-api:
    build: .
    volumes:
      - ./data:/app/data
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER}

  ollama:  # Optional local LLM
    image: ollama/ollama
    profiles: [local-llm]
```

## Future Enhancements

1. **Multi-Collection Support**: Separate collections per document domain
2. **Reranking**: Add cross-encoder reranking for improved relevance
3. **Caching**: Query cache for frequent questions
4. **Analytics**: Query logging and relevance feedback
5. **Multi-Modal**: Support for image-containing documents
