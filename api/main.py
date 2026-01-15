"""FastAPI application for Document Q&A RAG API."""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from agents.model_factory import get_provider_info
from agents.qa_agent import get_qa_agent
from agents.session import get_session_manager
from api.models import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    SessionInfo,
)
from config.settings import get_settings
from ingestion.pipeline import IngestionPipeline
from tools.retrieval import get_search_engine
from vector_store.chromadb_client import get_chromadb_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup: Pre-initialize services
    print("Starting RAG API...")
    get_chromadb_client()
    print("ChromaDB initialized")
    get_qa_agent()
    print("Q&A Agent initialized")
    print("RAG API started successfully")

    yield

    # Shutdown
    print("Shutting down RAG API...")


app = FastAPI(
    title="Document Q&A RAG API",
    description="RAG-powered document question-answering API with hybrid search",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint returning system status."""
    settings = get_settings()
    chromadb = get_chromadb_client()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider=settings.llm_provider,
        documents_count=chromadb.collection.count(),
    )


@app.get("/info")
async def get_info():
    """Get detailed system information."""
    settings = get_settings()
    chromadb = get_chromadb_client()
    provider_info = get_provider_info()

    return {
        "version": "1.0.0",
        "llm": provider_info,
        "vector_store": {
            "path": settings.vector_store_path,
            "collection": settings.collection_name,
            "documents_count": chromadb.collection.count(),
        },
        "hybrid_search": {
            "bm25_weight": settings.bm25_weight,
            "vector_weight": settings.vector_weight,
            "top_k": settings.top_k_results,
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for question-answering.

    Sends user message to RAG agent and returns the response.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Get session manager and save user message
        session_mgr = get_session_manager()
        session_mgr.add_message(session_id, "user", request.message)

        # Get agent response
        agent = get_qa_agent()
        answer = agent.ask(request.message)

        # Save assistant response
        session_mgr.add_message(session_id, "assistant", answer)

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            sources=[],  # TODO: Extract sources from retrieval results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events.

    Streams the agent's response as it's generated.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def generate() -> AsyncGenerator[dict, None]:
        try:
            # Save user message
            session_mgr = get_session_manager()
            session_mgr.add_message(session_id, "user", request.message)

            # Stream agent response
            agent = get_qa_agent()
            full_response = ""

            async for chunk in agent.ask_stream(request.message):
                full_response += chunk
                yield {"event": "message", "data": chunk}

            # Save complete response
            session_mgr.add_message(session_id, "assistant", full_response)

            # Send completion event
            yield {"event": "done", "data": session_id}

        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(generate())


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest):
    """Ingest documents into the vector store.

    Supports files and directories with PDF, DOCX, TXT, MD, HTML formats.
    """
    path = Path(request.path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    pipeline = IngestionPipeline()

    try:
        if path.is_file():
            chunk_count = pipeline.ingest_file(path)
            # Refresh search engine index after ingestion
            get_search_engine().refresh_index()
            return IngestionResponse(
                processed_files=1,
                total_chunks=chunk_count,
                errors=[],
            )
        elif path.is_dir():
            results = pipeline.ingest_directory(path, recursive=request.recursive)
            # Refresh search engine index after ingestion
            get_search_engine().refresh_index()
            return IngestionResponse(**results)
        else:
            raise HTTPException(status_code=400, detail="Invalid path type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session information."""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at,
        message_count=len(session.messages),
    )


@app.get("/sessions/{session_id}/history", response_model=ConversationHistory)
async def get_session_history(session_id: str, limit: int = 20):
    """Get conversation history for a session."""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    history = session_mgr.get_conversation_history(session_id, limit=limit)
    return ConversationHistory(session_id=session_id, messages=history)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session_mgr = get_session_manager()
    deleted = session_mgr.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    """List all sessions."""
    session_mgr = get_session_manager()
    sessions = session_mgr.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}
