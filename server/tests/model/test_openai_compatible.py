"""Tests for src/model/providers/openai_compatible.py."""

from unittest.mock import MagicMock, patch

import httpx

from src.model.providers.openai_compatible import OpenAICompatibleProvider


def _make_response(content: str, model: str = "gpt-4o") -> MagicMock:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _provider(base_url: str = "https://api.openai.com/v1") -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        model="gpt-4o",
        api_key="test-key",
        base_url=base_url,
    )


@patch("src.model.providers.openai_compatible.httpx.Client")
def test_complete_returns_llm_response(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _make_response("Hello world")

    provider = _provider()
    result = provider.complete(system="Be helpful.", messages=[{"role": "user", "content": "Hi"}])

    assert result.content == "Hello world"
    assert result.model == "gpt-4o"
    assert result.input_tokens == 10
    assert result.output_tokens == 5


@patch("src.model.providers.openai_compatible.httpx.Client")
def test_system_prepended_as_first_message(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _make_response("ok")

    provider = _provider()
    provider.complete(
        system="You are a math expert.",
        messages=[{"role": "user", "content": "Prove it."}],
    )

    _, kwargs = mock_client.post.call_args
    sent_messages = kwargs["json"]["messages"]
    assert sent_messages[0] == {"role": "system", "content": "You are a math expert."}
    assert sent_messages[1] == {"role": "user", "content": "Prove it."}


@patch("src.model.providers.openai_compatible.httpx.Client")
def test_correct_endpoint_called(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _make_response("ok")

    provider = _provider(base_url="https://api.deepseek.com/v1")
    provider.complete(system="s", messages=[{"role": "user", "content": "u"}])

    url_called = mock_client.post.call_args[0][0]
    assert url_called == "https://api.deepseek.com/v1/chat/completions"


@patch("src.model.providers.openai_compatible.httpx.Client")
def test_trailing_slash_in_base_url_normalised(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _make_response("ok")

    provider = _provider(base_url="https://openrouter.ai/api/v1/")
    provider.complete(system="s", messages=[{"role": "user", "content": "u"}])

    url_called = mock_client.post.call_args[0][0]
    assert url_called == "https://openrouter.ai/api/v1/chat/completions"
    assert "//" not in url_called.replace("https://", "")


@patch("src.model.providers.openai_compatible.httpx.Client")
def test_max_tokens_and_temperature_forwarded(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _make_response("ok")

    provider = _provider()
    provider.complete(
        system="s",
        messages=[{"role": "user", "content": "u"}],
        max_tokens=512,
        temperature=0.7,
    )

    payload = mock_client.post.call_args[1]["json"]
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0.7


@patch("src.model.providers.openai_compatible.httpx.Client")
def test_empty_choices_raises_value_error(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    bad_resp = MagicMock(spec=httpx.Response)
    bad_resp.raise_for_status = MagicMock()
    bad_resp.json.return_value = {"choices": [], "model": "gpt-4o", "usage": {}}
    mock_client.post.return_value = bad_resp

    import pytest
    provider = _provider()
    with pytest.raises(ValueError, match="no choices"):
        provider.complete(system="s", messages=[{"role": "user", "content": "u"}])


@patch("src.model.providers.openai_compatible.httpx.Client")
def test_http_error_propagates(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.side_effect = httpx.HTTPError("connection refused")

    import pytest
    provider = _provider()
    with pytest.raises(httpx.HTTPError):
        provider.complete(system="s", messages=[{"role": "user", "content": "u"}])


@patch("src.model.providers.openai_compatible.httpx.Client")
def test_client_created_lazily_and_reused(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _make_response("ok")

    provider = _provider()
    provider.complete(system="s", messages=[{"role": "user", "content": "1"}])
    provider.complete(system="s", messages=[{"role": "user", "content": "2"}])

    # httpx.Client constructed only once despite two calls
    assert mock_client_cls.call_count == 1
    assert mock_client.post.call_count == 2
