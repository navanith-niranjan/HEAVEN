"""Tests for src/model/extraction/relationship_extractor.py."""

import json
from unittest.mock import MagicMock

from src.ingestion.extractor import ExtractedConcept
from src.model.extraction.relationship_extractor import (
    extract_relationships,
)
from src.model.providers.base import LLMResponse


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = LLMResponse(
        content=content,
        model="claude-sonnet-4-6",
        input_tokens=50,
        output_tokens=30,
    )
    return provider


def _concept(name: str, ctype: str = "theorem") -> ExtractedConcept:
    return ExtractedConcept(
        name=name,
        concept_type=ctype,
        latex_statement=r"x + y = z",
        description="A concept.",
    )


def test_empty_concepts_returns_empty():
    provider = _make_provider("[]")
    result = extract_relationships([], provider)
    assert result == []
    provider.complete.assert_not_called()


def test_valid_relationship_returned():
    concepts = [_concept("Theorem A"), _concept("Lemma B", "lemma")]
    payload = [
        {
            "source_concept_name": "Theorem A",
            "target_concept_name": "Lemma B",
            "relationship_type": "proves",
            "description": "Theorem A proves Lemma B.",
        }
    ]
    provider = _make_provider(json.dumps(payload))
    result = extract_relationships(concepts, provider)
    assert len(result) == 1
    assert result[0].source_concept_name == "Theorem A"
    assert result[0].target_concept_name == "Lemma B"
    assert result[0].relationship_type == "proves"


def test_unknown_concept_name_filtered_out():
    concepts = [_concept("Theorem A")]
    payload = [
        {
            "source_concept_name": "Theorem A",
            "target_concept_name": "Unknown Concept",
            "relationship_type": "depends_on",
            "description": "...",
        }
    ]
    provider = _make_provider(json.dumps(payload))
    result = extract_relationships(concepts, provider)
    assert result == []


def test_invalid_relationship_type_filtered_out():
    concepts = [_concept("A"), _concept("B")]
    payload = [
        {
            "source_concept_name": "A",
            "target_concept_name": "B",
            "relationship_type": "invented_type",
            "description": "...",
        }
    ]
    provider = _make_provider(json.dumps(payload))
    result = extract_relationships(concepts, provider)
    assert result == []


def test_all_valid_relationship_types_accepted():
    valid_types = [
        "proves", "depends_on", "generalizes", "is_special_case_of",
        "contradicts", "cited_by", "equivalent_to", "extends",
    ]
    for rel_type in valid_types:
        concepts = [_concept("A"), _concept("B")]
        payload = [{"source_concept_name": "A", "target_concept_name": "B",
                    "relationship_type": rel_type, "description": "ok"}]
        provider = _make_provider(json.dumps(payload))
        result = extract_relationships(concepts, provider)
        assert len(result) == 1, f"Failed for type: {rel_type}"


def test_invalid_json_returns_empty():
    provider = _make_provider("not json")
    result = extract_relationships([_concept("A")], provider)
    assert result == []


def test_non_array_json_returns_empty():
    provider = _make_provider('{"source_concept_name": "A"}')
    result = extract_relationships([_concept("A")], provider)
    assert result == []


def test_provider_exception_returns_empty():
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("network error")
    result = extract_relationships([_concept("A"), _concept("B")], provider)
    assert result == []


def test_markdown_fences_stripped():
    concepts = [_concept("X"), _concept("Y")]
    payload = [{"source_concept_name": "X", "target_concept_name": "Y",
                "relationship_type": "extends", "description": "ok"}]
    content = "```json\n" + json.dumps(payload) + "\n```"
    provider = _make_provider(content)
    result = extract_relationships(concepts, provider)
    assert len(result) == 1
