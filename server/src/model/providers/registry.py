"""LLM provider registry — single source of truth for model selection.

This is the only file that reads provider configuration from settings.
All other code should import `primary` or `cheap` from here rather than
instantiating providers directly.

Configuration (.env):
    # Which provider backend to use:
    PRIMARY_PROVIDER=claude            # or openai_compatible
    PRIMARY_MODEL=claude-sonnet-4-6

    CHEAP_PROVIDER=claude              # or openai_compatible
    CHEAP_MODEL=claude-haiku-4-5-20251001

    # Anthropic (required when provider = "claude"):
    ANTHROPIC_API_KEY=sk-ant-...

    # OpenAI-compatible (required when provider = "openai_compatible"):
    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://api.deepseek.com/v1   # omit for standard OpenAI

Example configurations:

    DeepSeek for both roles:
        PRIMARY_PROVIDER=openai_compatible
        PRIMARY_MODEL=deepseek-chat
        CHEAP_PROVIDER=openai_compatible
        CHEAP_MODEL=deepseek-chat
        OPENAI_API_KEY=sk-...
        OPENAI_BASE_URL=https://api.deepseek.com/v1

    OpenRouter (200+ models):
        PRIMARY_PROVIDER=openai_compatible
        PRIMARY_MODEL=anthropic/claude-sonnet-4-6
        CHEAP_PROVIDER=openai_compatible
        CHEAP_MODEL=google/gemini-flash-1.5
        OPENAI_API_KEY=sk-or-...
        OPENAI_BASE_URL=https://openrouter.ai/api/v1

    Mixed (Claude primary, cheap via OpenRouter):
        PRIMARY_PROVIDER=claude
        PRIMARY_MODEL=claude-sonnet-4-6
        CHEAP_PROVIDER=openai_compatible
        CHEAP_MODEL=google/gemini-flash-1.5
        ANTHROPIC_API_KEY=sk-ant-...
        OPENAI_API_KEY=sk-or-...
        OPENAI_BASE_URL=https://openrouter.ai/api/v1
"""

from src.config import settings
from src.model.providers.base import LLMProvider

_SUPPORTED_PROVIDERS = ("claude", "openai_compatible")


def _make_provider(provider_name: str, model: str) -> LLMProvider:
    """Instantiate a provider from a name string and model ID."""
    if provider_name == "claude":
        from src.model.providers.claude import ClaudeProvider
        return ClaudeProvider(model)

    if provider_name == "openai_compatible":
        from src.model.providers.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            model=model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    raise ValueError(
        f"Unknown provider {provider_name!r}. "
        f"Supported values: {', '.join(_SUPPORTED_PROVIDERS)}"
    )


# ---------------------------------------------------------------------------
# Module-level singletons — import these throughout the codebase.
# They are constructed once at import time from settings.
# ---------------------------------------------------------------------------

primary: LLMProvider = _make_provider(settings.primary_provider, settings.primary_model)
"""High-capability provider — used for extraction, formalization, reasoning."""

cheap: LLMProvider = _make_provider(settings.cheap_provider, settings.cheap_model)
"""Cost-optimised provider — used for deduplication, classification, descriptions."""
