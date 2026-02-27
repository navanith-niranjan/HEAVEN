"""Tests for src/model/reasoning/conflict_explainer.py."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.model.providers.base import LLMResponse
from src.model.reasoning.conflict_explainer import explain_conflicts
from src.schemas.models import ConceptRead, DiscoveryCreate


def _make_provider(content: str) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = LLMResponse(
        content=content,
        model="claude-haiku-4-5-20251001",
        input_tokens=40,
        output_tokens=20,
    )
    return provider


def _discovery() -> DiscoveryCreate:
    return DiscoveryCreate(
        name="Modified Axiom",
        base_concept_id="base-uuid",
        modified_latex_statement=r"x + y \neq y + x",
        modification_description="Commutativity removed.",
    )


def _base_concept() -> ConceptRead:
    return ConceptRead(
        id="base-uuid",
        name="Commutativity Axiom",
        concept_type="axiom",
        latex_statement=r"x + y = y + x",
        description="Commutativity of addition.",
        msc_codes=[],
        lean_verification_status="unverified",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def test_empty_conflict_ids_returns_empty_without_llm():
    provider = _make_provider("[]")
    result = explain_conflicts(_discovery(), _base_concept(), [], provider)
    assert result == []
    provider.complete.assert_not_called()


def test_valid_response_returned():
    conflict_ids = ["concept-x", "concept-y"]
    payload = [
        {
            "conflict_concept_id": "concept-x",
            "explanation": "This contradicts concept x.",
            "severity": "severe",
        },
        {
            "conflict_concept_id": "concept-y",
            "explanation": "Minor framing difference.",
            "severity": "minor",
        },
    ]
    provider = _make_provider(json.dumps(payload))
    result = explain_conflicts(_discovery(), _base_concept(), conflict_ids, provider)
    assert len(result) == 2
    assert result[0].conflict_concept_id == "concept-x"
    assert result[0].severity == "severe"
    assert result[1].severity == "minor"


def test_unknown_concept_id_filtered():
    conflict_ids = ["concept-x"]
    payload = [
        {"conflict_concept_id": "not-in-list", "explanation": "...", "severity": "severe"}
    ]
    provider = _make_provider(json.dumps(payload))
    result = explain_conflicts(_discovery(), _base_concept(), conflict_ids, provider)
    assert result == []


def test_invalid_severity_defaults_to_minor():
    conflict_ids = ["concept-x"]
    payload = [
        {"conflict_concept_id": "concept-x", "explanation": "ok", "severity": "catastrophic"}
    ]
    provider = _make_provider(json.dumps(payload))
    result = explain_conflicts(_discovery(), _base_concept(), conflict_ids, provider)
    assert len(result) == 1
    assert result[0].severity == "minor"


def test_all_valid_severities_accepted():
    for severity in ("minor", "moderate", "severe"):
        payload = [{"conflict_concept_id": "cid", "explanation": "ok", "severity": severity}]
        provider = _make_provider(json.dumps(payload))
        result = explain_conflicts(_discovery(), _base_concept(), ["cid"], provider)
        assert result[0].severity == severity


def test_invalid_json_returns_empty():
    provider = _make_provider("not json")
    result = explain_conflicts(_discovery(), _base_concept(), ["cid"], provider)
    assert result == []


def test_provider_exception_returns_empty():
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("network error")
    result = explain_conflicts(_discovery(), _base_concept(), ["cid"], provider)
    assert result == []


def test_markdown_fences_stripped():
    payload = [{"conflict_concept_id": "cid", "explanation": "ok", "severity": "moderate"}]
    content = "```json\n" + json.dumps(payload) + "\n```"
    provider = _make_provider(content)
    result = explain_conflicts(_discovery(), _base_concept(), ["cid"], provider)
    assert len(result) == 1


def test_conflict_ids_included_in_prompt():
    conflict_ids = ["abc-123", "def-456"]
    provider = _make_provider("[]")
    explain_conflicts(_discovery(), _base_concept(), conflict_ids, provider)
    call_args = provider.complete.call_args
    messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][1]
    msg_content = messages[0]["content"]
    assert "abc-123" in msg_content
    assert "def-456" in msg_content
