# Document Q&A RAG System

A Retrieval-Augmented Generation (RAG) chatbot for document question-answering with hybrid search capabilities. Built with FastAPI, ChromaDB, Strands Agents, and React.

## Features

- **Hybrid Search**: Combines BM25 keyword matching with vector similarity using Reciprocal Rank Fusion (RRF)
- **Multi-LLM Support**: Works with Anthropic Claude, OpenAI GPT, or local Ollama models
- **Document Ingestion**: Supports PDF, DOCX, TXT, Markdown, and HTML formats
- **Streaming Responses**: Real-time response streaming via Server-Sent Events
- **Session Management**: Persistent conversation history
- **Single-Container Deployment**: Self-contained Docker deployment

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- Docker (optional, for containerized deployment)
- Ollama (optional, for local LLM)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd darksite-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Configuration

Edit `.env` with your settings:

```bash
# Choose LLM provider: anthropic, openai, or ollama
LLM_PROVIDER=anthropic

# For Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your-key-here

# For OpenAI
OPENAI_API_KEY=sk-your-key-here

# For Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### Running the Application

#### 1. Ingest Documents

Place your documents in a directory and run:

```bash
python -m ingestion.cli --path ./path/to/documents --stats
```

#### 2. Start the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3. Start the Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

Access the application:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:5173

## Docker Deployment

### Using Docker Compose

```bash
# With cloud LLM (Anthropic/OpenAI)
docker-compose up -d

# With local Ollama
docker-compose --profile local-llm up -d
```

### Using Docker directly

```bash
docker build -t darksite-rag .
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e ANTHROPIC_API_KEY=your-key \
  darksite-rag
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with system status |
| GET | `/info` | Detailed system information |
| POST | `/chat` | Send a question, receive an answer |
| POST | `/chat/stream` | Stream response via SSE |
| POST | `/ingest` | Ingest documents into vector store |
| GET | `/sessions` | List all sessions |
| GET | `/sessions/{id}` | Get session info |
| GET | `/sessions/{id}/history` | Get conversation history |
| DELETE | `/sessions/{id}` | Delete a session |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What topics are covered in the documents?"}'

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/docs", "recursive": true}'
```

## Using Local LLMs with Ollama

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### Download a Model

```bash
ollama pull llama3      # Recommended for RAG
ollama pull mistral     # Lightweight alternative
ollama pull phi3        # Smallest option
```

### Configure for Ollama

```bash
# In .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## Project Structure

```
darksite-rag/
├── api/                 # FastAPI application
│   ├── main.py         # API endpoints
│   └── models.py       # Request/response schemas
├── agents/             # LLM agent orchestration
│   ├── model_factory.py # Multi-provider LLM factory
│   ├── qa_agent.py     # Document Q&A agent
│   └── session.py      # Session management
├── config/             # Configuration
│   └── settings.py     # Pydantic settings
├── ingestion/          # Document processing
│   ├── loaders.py      # Format-specific loaders
│   ├── chunker.py      # Text chunking
│   ├── pipeline.py     # Ingestion orchestration
│   └── cli.py          # CLI interface
├── tools/              # RAG tools
│   └── retrieval.py    # Hybrid search implementation
├── vector_store/       # Vector database
│   └── chromadb_client.py
├── frontend/           # React frontend
│   └── src/
│       ├── App.tsx     # Chat interface
│       └── App.css     # Styling
├── tests/              # Test suite
├── data/               # Runtime data
│   ├── vector_store/   # ChromaDB storage
│   └── sessions/       # Session files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── CLAUDE.md           # AI assistant guidance
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_ingestion.py -v
```

## Configuration Reference

### LLM Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider: anthropic, openai, ollama | anthropic |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |
| `OLLAMA_MODEL` | Ollama model name | llama3 |
| `TEMPERATURE` | LLM temperature | 0.7 |
| `MAX_TOKENS` | Max response tokens | 2048 |

### Vector Store Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_STORE_PATH` | ChromaDB storage path | data/vector_store |
| `COLLECTION_NAME` | Collection name | documents |
| `BM25_WEIGHT` | BM25 search weight | 0.3 |
| `VECTOR_WEIGHT` | Vector search weight | 0.7 |
| `TOP_K_RESULTS` | Results per query | 5 |

### Ingestion Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `INGESTION_CHUNK_SIZE` | Characters per chunk | 512 |
| `INGESTION_CHUNK_OVERLAP` | Overlap between chunks | 50 |
| `INGESTION_BATCH_SIZE` | Batch size for indexing | 100 |
| `INGESTION_EMBEDDING_MODEL` | Embedding model | all-MiniLM-L6-v2 |

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
