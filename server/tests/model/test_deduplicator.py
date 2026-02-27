"""Tests for src/model/extraction/deduplicator.py."""

import json
from unittest.mock import MagicMock, patch

from src.ingestion.extractor import ExtractedConcept
from src.model.providers.base import LLMResponse


def _candidate() -> ExtractedConcept:
    return ExtractedConcept(
        name="Pythagorean Theorem",
        concept_type="theorem",
        latex_statement=r"a^2 + b^2 = c^2",
        description="Relates sides of a right triangle.",
    )


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = LLMResponse(
        content=content,
        model="claude-haiku-4-5-20251001",
        input_tokens=20,
        output_tokens=10,
    )
    return provider


@patch("src.model.extraction.deduplicator.collections")
def test_no_chroma_results_returns_none(mock_collections):
    mock_collections.query_concepts.return_value = {
        "ids": [[]], "distances": [[]], "documents": [[]]
    }
    provider = _make_provider("{}")
    from src.model.extraction.deduplicator import find_duplicate
    result = find_duplicate(_candidate(), provider)
    assert result is None
    provider.complete.assert_not_called()


@patch("src.model.extraction.deduplicator.collections")
def test_high_distance_returns_none_without_llm(mock_collections):
    # Distance 0.5 > (1 - 0.92) = 0.08 — not similar enough
    mock_collections.query_concepts.return_value = {
        "ids": [["existing-id"]],
        "distances": [[0.5]],
        "documents": [["some text"]],
    }
    provider = _make_provider("{}")
    from src.model.extraction.deduplicator import find_duplicate
    result = find_duplicate(_candidate(), provider)
    assert result is None
    provider.complete.assert_not_called()


@patch("src.model.extraction.deduplicator.collections")
def test_low_distance_same_verdict_returns_existing_id(mock_collections):
    mock_collections.query_concepts.return_value = {
        "ids": [["existing-uuid"]],
        "distances": [[0.03]],  # below threshold
        "documents": [["Name: Pythagorean Theorem\nStatement: a^2+b^2=c^2"]],
    }
    provider = _make_provider(json.dumps({"same": True, "reason": "Identical concept."}))
    from src.model.extraction.deduplicator import find_duplicate
    result = find_duplicate(_candidate(), provider)
    assert result == "existing-uuid"


@patch("src.model.extraction.deduplicator.collections")
def test_low_distance_different_verdict_returns_none(mock_collections):
    mock_collections.query_concepts.return_value = {
        "ids": [["existing-uuid"]],
        "distances": [[0.05]],
        "documents": [["Different concept text"]],
    }
    provider = _make_provider(json.dumps({"same": False, "reason": "Different objects."}))
    from src.model.extraction.deduplicator import find_duplicate
    result = find_duplicate(_candidate(), provider)
    assert result is None


@patch("src.model.extraction.deduplicator.collections")
def test_chroma_exception_returns_none(mock_collections):
    mock_collections.query_concepts.side_effect = RuntimeError("chroma down")
    provider = _make_provider("{}")
    from src.model.extraction.deduplicator import find_duplicate
    result = find_duplicate(_candidate(), provider)
    assert result is None


@patch("src.model.extraction.deduplicator.collections")
def test_llm_exception_returns_none(mock_collections):
    mock_collections.query_concepts.return_value = {
        "ids": [["existing-uuid"]],
        "distances": [[0.02]],
        "documents": [["text"]],
    }
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("llm error")
    from src.model.extraction.deduplicator import find_duplicate
    result = find_duplicate(_candidate(), provider)
    assert result is None


@patch("src.model.extraction.deduplicator.collections")
def test_invalid_json_from_llm_returns_none(mock_collections):
    mock_collections.query_concepts.return_value = {
        "ids": [["existing-uuid"]],
        "distances": [[0.02]],
        "documents": [["text"]],
    }
    provider = _make_provider("not json")
    from src.model.extraction.deduplicator import find_duplicate
    result = find_duplicate(_candidate(), provider)
    assert result is None


@patch("src.model.extraction.deduplicator.collections")
def test_custom_threshold_respected(mock_collections):
    # With threshold=0.5, distance 0.3 should trigger LLM check
    mock_collections.query_concepts.return_value = {
        "ids": [["existing-uuid"]],
        "distances": [[0.3]],
        "documents": [["text"]],
    }
    provider = _make_provider(json.dumps({"same": True, "reason": "Same."}))
    from src.model.extraction.deduplicator import find_duplicate
    result = find_duplicate(_candidate(), provider, similarity_threshold=0.5)
    assert result == "existing-uuid"
    provider.complete.assert_called_once()
