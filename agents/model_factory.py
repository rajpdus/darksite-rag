"""Factory for creating LLM model instances based on provider configuration."""

from typing import Union

from strands.models import AnthropicModel, BedrockModel
from strands.models.litellm import LiteLLMModel

from config.settings import get_settings


def create_model() -> Union[AnthropicModel, LiteLLMModel]:
    """Create appropriate LLM model based on configuration.

    Supports:
    - anthropic: Anthropic Claude models (direct integration)
    - openai: OpenAI GPT models (via LiteLLM)
    - ollama: Local Ollama models (via LiteLLM)

    Returns:
        Configured model instance.

    Raises:
        ValueError: If provider is not supported.
    """
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        return AnthropicModel(
            model_id=settings.anthropic_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    elif settings.llm_provider == "openai":
        return LiteLLMModel(
            model_id=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    elif settings.llm_provider == "ollama":
        return LiteLLMModel(
            model_id=f"ollama/{settings.ollama_model}",
            api_base=settings.ollama_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def get_provider_info() -> dict:
    """Get information about the current LLM provider configuration.

    Returns:
        Dictionary with provider information.
    """
    settings = get_settings()

    info = {
        "provider": settings.llm_provider,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }

    if settings.llm_provider == "anthropic":
        info["model"] = settings.anthropic_model
    elif settings.llm_provider == "openai":
        info["model"] = settings.openai_model
    elif settings.llm_provider == "ollama":
        info["model"] = settings.ollama_model
        info["base_url"] = settings.ollama_base_url

    return info
