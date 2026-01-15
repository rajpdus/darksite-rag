"""Document Q&A Agent implementation using Strands framework."""

from typing import AsyncGenerator

from strands import Agent

from agents.model_factory import create_model
from tools.retrieval import retrieve_qa_context

SYSTEM_PROMPT = """You are a helpful document Q&A assistant. Your role is to answer questions based on the documents in the knowledge base.

IMPORTANT INSTRUCTIONS:
1. ALWAYS use the retrieve_qa_context tool BEFORE answering any question about the documents.
2. Base your answers ONLY on the retrieved context. Do not make up information.
3. If the retrieved context doesn't contain relevant information, say so clearly.
4. Cite the document sources when providing information (e.g., "According to Document 1...").
5. If asked about topics outside the knowledge base, politely explain you can only answer questions about the indexed documents.

When answering:
- Be concise but thorough
- Use bullet points for lists when appropriate
- Quote relevant passages when helpful
- Indicate confidence level when uncertain
- If multiple documents provide different information, synthesize or note the differences

Remember: Always retrieve context first, then formulate your answer based on what you found."""


class DocumentQAAgent:
    """RAG Agent for document question-answering."""

    def __init__(self):
        """Initialize the Q&A agent with model and tools."""
        self.model = create_model()
        self.agent = Agent(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            tools=[retrieve_qa_context],
        )

    def ask(self, question: str) -> str:
        """Ask a question and get an answer based on retrieved documents.

        Args:
            question: The user's question.

        Returns:
            The agent's answer based on retrieved context.
        """
        response = self.agent(question)
        return response.message

    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        """Ask a question and stream the response.

        Args:
            question: The user's question.

        Yields:
            Response chunks as they are generated.
        """
        async for event in self.agent.stream_async(question):
            if hasattr(event, "data"):
                yield event.data


# Singleton instance
_agent_instance = None


def get_qa_agent() -> DocumentQAAgent:
    """Get or create the Q&A agent singleton.

    Returns:
        The singleton DocumentQAAgent instance.
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DocumentQAAgent()
    return _agent_instance


def reset_agent():
    """Reset the agent singleton (useful for testing or reconfiguration)."""
    global _agent_instance
    _agent_instance = None
