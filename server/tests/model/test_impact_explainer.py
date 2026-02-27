"""Tests for src/model/reasoning/impact_explainer.py."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.model.providers.base import LLMResponse
from src.model.reasoning.impact_explainer import explain_impacts
from src.schemas.models import ConceptRead, DiscoveryCreate


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = LLMResponse(
        content=content,
        model="claude-haiku-4-5-20251001",
        input_tokens=50,
        output_tokens=30,
    )
    return provider


def _discovery() -> DiscoveryCreate:
    return DiscoveryCreate(
        name="Modified Pythagoras",
        base_concept_id="base-uuid",
        modified_latex_statement=r"a^3 + b^3 = c^3",
        modification_description="Cubes instead of squares.",
    )


def _base_concept() -> ConceptRead:
    return ConceptRead(
        id="base-uuid",
        name="Pythagorean Theorem",
        concept_type="theorem",
        latex_statement=r"a^2 + b^2 = c^2",
        description="Squares.",
        msc_codes=[],
        lean_verification_status="unverified",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def test_empty_affected_returns_empty():
    provider = _make_provider("[]")
    result = explain_impacts(_discovery(), _base_concept(), {}, provider)
    assert result == []
    provider.complete.assert_not_called()


def test_valid_response_returns_impacts():
    affected = {"proves": ["concept-a"], "depends_on": ["concept-b"]}
    payload = [
        {
            "affected_concept_id": "concept-a",
            "impact_type": "extends",
            "description": "Extends the result.",
            "confidence_score": 0.9,
        },
        {
            "affected_concept_id": "concept-b",
            "impact_type": "contradicts",
            "description": "Contradicts assumption.",
            "confidence_score": 0.7,
        },
    ]
    provider = _make_provider(json.dumps(payload))
    result = explain_impacts(_discovery(), _base_concept(), affected, provider)
    assert len(result) == 2
    assert result[0].affected_concept_id == "concept-a"
    assert result[0].impact_type == "extends"
    assert result[0].confidence_score == 0.9


def test_unknown_concept_id_filtered():
    affected = {"proves": ["concept-a"]}
    payload = [
        {"affected_concept_id": "unknown-id", "impact_type": "extends",
         "description": "...", "confidence_score": 0.5}
    ]
    provider = _make_provider(json.dumps(payload))
    result = explain_impacts(_discovery(), _base_concept(), affected, provider)
    assert result == []


def test_invalid_impact_type_filtered():
    affected = {"proves": ["concept-a"]}
    payload = [
        {"affected_concept_id": "concept-a", "impact_type": "invented_type",
         "description": "...", "confidence_score": 0.5}
    ]
    provider = _make_provider(json.dumps(payload))
    result = explain_impacts(_discovery(), _base_concept(), affected, provider)
    assert result == []


def test_all_valid_impact_types_accepted():
    valid_types = ["extends", "contradicts", "generalizes", "enables", "invalidates"]
    for itype in valid_types:
        affected = {"proves": ["concept-a"]}
        payload = [{"affected_concept_id": "concept-a", "impact_type": itype,
                    "description": "ok", "confidence_score": 0.8}]
        provider = _make_provider(json.dumps(payload))
        result = explain_impacts(_discovery(), _base_concept(), affected, provider)
        assert len(result) == 1, f"Failed for impact type: {itype}"


def test_confidence_clamped_to_zero_one():
    affected = {"proves": ["concept-a"]}
    payload = [
        {"affected_concept_id": "concept-a", "impact_type": "extends",
         "description": "...", "confidence_score": 1.5},
    ]
    provider = _make_provider(json.dumps(payload))
    result = explain_impacts(_discovery(), _base_concept(), affected, provider)
    assert result[0].confidence_score == 1.0


def test_invalid_json_returns_empty():
    affected = {"proves": ["concept-a"]}
    provider = _make_provider("not json")
    result = explain_impacts(_discovery(), _base_concept(), affected, provider)
    assert result == []


def test_provider_exception_returns_empty():
    affected = {"proves": ["concept-a"]}
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("network error")
    result = explain_impacts(_discovery(), _base_concept(), affected, provider)
    assert result == []


def test_markdown_fences_stripped():
    affected = {"proves": ["concept-a"]}
    payload = [{"affected_concept_id": "concept-a", "impact_type": "extends",
                "description": "ok", "confidence_score": 0.8}]
    content = "```json\n" + json.dumps(payload) + "\n```"
    provider = _make_provider(content)
    result = explain_impacts(_discovery(), _base_concept(), affected, provider)
    assert len(result) == 1
