"""File-based session management for conversation history."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import get_settings


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str


@dataclass
class Session:
    """Represents a conversation session."""

    session_id: str
    created_at: str
    messages: List[Message]


class SessionManager:
    """File-based session storage for conversation history."""

    def __init__(self):
        """Initialize the session manager."""
        self.settings = get_settings()
        self.session_dir = Path(self.settings.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session.

        Args:
            session_id: The session identifier.

        Returns:
            Path to the session file.
        """
        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.session_dir / f"{safe_id}.json"

    def create_session(self, session_id: str) -> Session:
        """Create a new session.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            The created Session object.
        """
        session = Session(
            session_id=session_id,
            created_at=datetime.utcnow().isoformat(),
            messages=[],
        )
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session.

        Args:
            session_id: The session identifier.

        Returns:
            Session object or None if not found.
        """
        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return Session(
                session_id=data["session_id"],
                created_at=data["created_at"],
                messages=[Message(**m) for m in data["messages"]],
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def get_or_create_session(self, session_id: str) -> Session:
        """Get an existing session or create a new one.

        Args:
            session_id: The session identifier.

        Returns:
            Session object.
        """
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id)
        return session

    def add_message(self, session_id: str, role: str, content: str) -> Session:
        """Add a message to a session.

        Args:
            session_id: The session identifier.
            role: Message role ("user" or "assistant").
            content: Message content.

        Returns:
            Updated Session object.
        """
        session = self.get_or_create_session(session_id)
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
        )
        session.messages.append(message)
        self._save_session(session)
        return session

    def _save_session(self, session: Session):
        """Save a session to file.

        Args:
            session: The Session object to save.
        """
        path = self._get_session_path(session.session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "messages": [asdict(m) for m in session.messages],
                },
                f,
                indent=2,
            )

    def get_conversation_history(
        self, session_id: str, limit: int = 10
    ) -> List[Dict]:
        """Get recent conversation history formatted for context.

        Args:
            session_id: The session identifier.
            limit: Maximum number of messages to return.

        Returns:
            List of message dictionaries.
        """
        session = self.get_session(session_id)
        if session is None:
            return []

        recent_messages = session.messages[-limit:]
        return [{"role": m.role, "content": m.content} for m in recent_messages]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session identifier.

        Returns:
            True if deleted, False if not found.
        """
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all session IDs.

        Returns:
            List of session identifiers.
        """
        return [p.stem for p in self.session_dir.glob("*.json")]


def get_session_manager() -> SessionManager:
    """Get a SessionManager instance.

    Returns:
        SessionManager instance.
    """
    return SessionManager()
