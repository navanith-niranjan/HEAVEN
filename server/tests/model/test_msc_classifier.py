"""Tests for src/model/reasoning/msc_classifier.py."""

import json
from unittest.mock import MagicMock

from src.model.providers.base import LLMResponse
from src.model.reasoning.msc_classifier import classify_msc


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = LLMResponse(
        content=content,
        model="claude-haiku-4-5-20251001",
        input_tokens=30,
        output_tokens=10,
    )
    return provider


def test_valid_codes_returned():
    provider = _make_provider(json.dumps(["11", "14", "13A15"]))
    result = classify_msc("Number theory paper about primes", provider)
    assert result == ["11", "14", "13A15"]


def test_empty_array_returned_on_vague_input():
    provider = _make_provider("[]")
    result = classify_msc("general stuff", provider)
    assert result == []


def test_invalid_json_returns_empty():
    provider = _make_provider("not json")
    result = classify_msc("text", provider)
    assert result == []


def test_non_array_json_returns_empty():
    provider = _make_provider('{"code": "11"}')
    result = classify_msc("text", provider)
    assert result == []


def test_top_k_respected():
    provider = _make_provider(json.dumps(["11", "14", "13", "57", "55"]))
    result = classify_msc("broad paper", provider, top_k=3)
    assert len(result) == 3
    assert result == ["11", "14", "13"]


def test_provider_exception_returns_empty():
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("LLM down")
    result = classify_msc("text", provider)
    assert result == []


def test_markdown_fences_stripped():
    codes = ["11", "14"]
    content = "```json\n" + json.dumps(codes) + "\n```"
    provider = _make_provider(content)
    result = classify_msc("text", provider)
    assert result == ["11", "14"]


def test_integer_codes_converted_to_string():
    provider = _make_provider(json.dumps([11, 14]))
    result = classify_msc("text", provider)
    assert result == ["11", "14"]


def test_default_top_k_is_three():
    provider = _make_provider(json.dumps(["11", "14", "13", "57"]))
    result = classify_msc("text", provider)
    assert len(result) == 3


def test_query_includes_input_text():
    provider = _make_provider("[]")
    classify_msc("algebraic topology and cohomology", provider)
    call_args = provider.complete.call_args
    messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][1]
    assert "algebraic topology and cohomology" in messages[0]["content"]
