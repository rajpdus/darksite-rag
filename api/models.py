"""Pydantic request/response models for the API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User's question or message", min_length=1)
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID")
    sources: List[str] = Field(default=[], description="Document sources used")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    llm_provider: str = Field(..., description="Configured LLM provider")
    documents_count: int = Field(..., description="Number of documents in vector store")


class IngestionRequest(BaseModel):
    """Request model for document ingestion."""

    path: str = Field(..., description="Path to file or directory")
    recursive: bool = Field(True, description="Recursively process directories")


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint."""

    processed_files: int = Field(..., description="Number of files processed")
    total_chunks: int = Field(..., description="Total chunks indexed")
    errors: List[dict] = Field(default=[], description="Errors encountered")


class SessionInfo(BaseModel):
    """Session information response."""

    session_id: str = Field(..., description="Session identifier")
    created_at: str = Field(..., description="Session creation timestamp")
    message_count: int = Field(..., description="Number of messages in session")


class MessageItem(BaseModel):
    """Individual message in conversation."""

    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")


class ConversationHistory(BaseModel):
    """Conversation history response."""

    session_id: str = Field(..., description="Session identifier")
    messages: List[dict] = Field(..., description="List of messages")
