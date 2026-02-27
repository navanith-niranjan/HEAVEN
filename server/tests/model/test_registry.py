"""Tests for src/model/providers/registry.py."""

from unittest.mock import patch

import pytest

from src.model.providers.base import LLMProvider


def _make_registry(primary_provider: str, primary_model: str,
                   cheap_provider: str, cheap_model: str,
                   openai_api_key: str = "", openai_base_url: str = "https://api.openai.com/v1"):
    """Re-invoke _make_provider with custom settings without re-importing the module."""
    from src.model.providers import registry
    return (
        registry._make_provider(primary_provider, primary_model),
        registry._make_provider(cheap_provider, cheap_model),
    )


def test_claude_provider_name_returns_claude_provider():
    from src.model.providers.claude import ClaudeProvider
    from src.model.providers.registry import _make_provider
    provider = _make_provider("claude", "claude-sonnet-4-6")
    assert isinstance(provider, ClaudeProvider)


def test_openai_compatible_provider_name_returns_openai_provider():
    from src.model.providers.openai_compatible import OpenAICompatibleProvider
    from src.model.providers.registry import _make_provider
    provider = _make_provider("openai_compatible", "gpt-4o")
    assert isinstance(provider, OpenAICompatibleProvider)


def test_unknown_provider_raises_value_error():
    from src.model.providers.registry import _make_provider
    with pytest.raises(ValueError, match="Unknown provider"):
        _make_provider("gemini_native", "gemini-pro")


def test_primary_and_cheap_are_llm_providers():
    from src.model.providers.registry import cheap, primary
    assert isinstance(primary, LLMProvider)
    assert isinstance(cheap, LLMProvider)


@patch("src.model.providers.registry.settings")
def test_registry_reads_primary_model_from_settings(mock_settings):
    mock_settings.primary_provider = "claude"
    mock_settings.primary_model = "claude-opus-4-6"
    mock_settings.cheap_provider = "claude"
    mock_settings.cheap_model = "claude-haiku-4-5-20251001"
    mock_settings.openai_api_key = ""
    mock_settings.openai_base_url = "https://api.openai.com/v1"

    from src.model.providers.claude import ClaudeProvider
    from src.model.providers.registry import _make_provider
    p = _make_provider(mock_settings.primary_provider, mock_settings.primary_model)
    assert isinstance(p, ClaudeProvider)
    assert p._model == "claude-opus-4-6"


@patch("src.model.providers.registry.settings")
def test_registry_builds_openai_provider_with_correct_params(mock_settings):
    mock_settings.openai_api_key = "sk-deepseek-key"
    mock_settings.openai_base_url = "https://api.deepseek.com/v1"

    from src.model.providers.openai_compatible import OpenAICompatibleProvider
    from src.model.providers.registry import _make_provider
    p = _make_provider("openai_compatible", "deepseek-chat")
    assert isinstance(p, OpenAICompatibleProvider)
    assert p._model == "deepseek-chat"
    assert p._api_key == "sk-deepseek-key"
    assert "deepseek" in p._base_url
