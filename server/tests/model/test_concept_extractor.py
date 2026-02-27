"""Tests for src/model/extraction/concept_extractor.py."""

import json
from unittest.mock import MagicMock

from src.model.extraction.concept_extractor import extract_concepts
from src.model.providers.base import LLMResponse


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = LLMResponse(
        content=content,
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
    )
    return provider


def test_valid_response_returns_concepts():
    payload = [
        {
            "name": "Pythagorean Theorem",
            "concept_type": "theorem",
            "latex_statement": r"a^2 + b^2 = c^2",
            "description": "Relates sides of a right triangle.",
            "msc_codes": ["51"],
        }
    ]
    provider = _make_provider(json.dumps(payload))
    result = extract_concepts("some chunk", provider)
    assert len(result) == 1
    assert result[0].name == "Pythagorean Theorem"
    assert result[0].concept_type == "theorem"
    assert result[0].msc_codes == ["51"]


def test_empty_json_array_returns_empty():
    provider = _make_provider("[]")
    result = extract_concepts("some chunk", provider)
    assert result == []


def test_invalid_json_returns_empty():
    provider = _make_provider("not json at all")
    result = extract_concepts("some chunk", provider)
    assert result == []


def test_non_array_json_returns_empty():
    provider = _make_provider('{"name": "foo"}')
    result = extract_concepts("some chunk", provider)
    assert result == []


def test_markdown_fences_stripped():
    payload = [
        {
            "name": "Fermat Last Theorem",
            "concept_type": "theorem",
            "latex_statement": r"a^n + b^n \neq c^n",
            "description": "No solutions for n > 2.",
            "msc_codes": [],
        }
    ]
    content = "```json\n" + json.dumps(payload) + "\n```"
    provider = _make_provider(content)
    result = extract_concepts("chunk", provider)
    assert len(result) == 1
    assert result[0].name == "Fermat Last Theorem"


def test_markdown_fences_no_language_tag_stripped():
    payload = [{"name": "Euclid", "concept_type": "theorem",
                "latex_statement": r"p \to \infty", "description": "d", "msc_codes": []}]
    content = "```\n" + json.dumps(payload) + "\n```"
    provider = _make_provider(content)
    result = extract_concepts("chunk", provider)
    assert len(result) == 1


def test_missing_required_field_skipped():
    # Missing latex_statement
    payload = [{"name": "Bad", "concept_type": "theorem", "description": "d", "msc_codes": []}]
    provider = _make_provider(json.dumps(payload))
    result = extract_concepts("chunk", provider)
    assert result == []


def test_multiple_concepts_all_returned():
    payload = [
        {"name": "A", "concept_type": "theorem",
         "latex_statement": "x", "description": "d", "msc_codes": []},
        {"name": "B", "concept_type": "definition",
         "latex_statement": "y", "description": "e", "msc_codes": ["11"]},
    ]
    provider = _make_provider(json.dumps(payload))
    result = extract_concepts("chunk", provider)
    assert len(result) == 2
    assert result[0].name == "A"
    assert result[1].name == "B"


def test_provider_exception_returns_empty():
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("network error")
    result = extract_concepts("chunk", provider)
    assert result == []


def test_source_hint_passed_to_provider():
    payload = []
    provider = _make_provider(json.dumps(payload))
    extract_concepts("chunk", provider, source_hint="My Paper Title")
    call_args = provider.complete.call_args
    messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][1]
    assert "My Paper Title" in messages[0]["content"]


def test_context_snippet_set_from_chunk():
    payload = [
        {"name": "X", "concept_type": "lemma",
         "latex_statement": "z", "description": "d", "msc_codes": []}
    ]
    provider = _make_provider(json.dumps(payload))
    chunk = "A" * 300
    result = extract_concepts(chunk, provider)
    assert result[0].context_snippet == chunk[:200]
