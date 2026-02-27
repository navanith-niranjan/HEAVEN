"""Tests for src/model/formalization/formalizer.py."""

from unittest.mock import MagicMock, patch

from src.model.formalization.formalizer import _extract_lean, formalize
from src.model.providers.base import LLMResponse
from src.verification.lean import LeanResult


def _make_provider(responses: list[str]) -> MagicMock:
    """Return a provider whose complete() returns each string in sequence."""
    provider = MagicMock()
    provider.complete.side_effect = [
        LLMResponse(content=r, model="claude-sonnet-4-6", input_tokens=10, output_tokens=5)
        for r in responses
    ]
    return provider


def _lean_ok() -> LeanResult:
    return LeanResult(success=True, output="", errors=[])


def _lean_fail(errors: list[str] | None = None) -> LeanResult:
    errs = errors or ["error: unknown identifier 'foo'"]
    return LeanResult(success=False, output="\n".join(errs), errors=errs)


@patch("src.model.formalization.formalizer.lean")
def test_success_on_first_attempt(mock_lean):
    mock_lean.verify.return_value = _lean_ok()
    provider = _make_provider(["theorem sq_nonneg (n : ℕ) : n ^ 2 ≥ 0 := by positivity"])
    result = formalize("n^2 \\geq 0", "sq_nonneg", provider)
    assert result.success is True
    assert result.attempts == 1
    assert mock_lean.verify.call_count == 1


@patch("src.model.formalization.formalizer.lean")
def test_success_on_second_attempt(mock_lean):
    mock_lean.verify.side_effect = [_lean_fail(), _lean_ok()]
    provider = _make_provider([
        "theorem bad := sorry",
        "theorem sq_nonneg (n : ℕ) : n ^ 2 ≥ 0 := by positivity",
    ])
    result = formalize("n^2 \\geq 0", "sq_nonneg", provider, max_attempts=3)
    assert result.success is True
    assert mock_lean.verify.call_count == 2


@patch("src.model.formalization.formalizer.lean")
def test_all_attempts_exhausted_returns_failure(mock_lean):
    # Each round returns a DIFFERENT error to prevent early-abort,
    # so all max_attempts rounds run.
    mock_lean.verify.side_effect = [
        _lean_fail(["error: unknown identifier 'a'"]),
        _lean_fail(["error: type mismatch at 'b'"]),
        _lean_fail(["error: elaboration failed 'c'"]),
    ]
    provider = _make_provider(["attempt 1", "attempt 2", "attempt 3"])
    result = formalize("bad latex", "bad", provider, max_attempts=3)
    assert result.success is False
    assert mock_lean.verify.call_count == 3


@patch("src.model.formalization.formalizer.lean")
def test_early_abort_on_repeated_errors(mock_lean):
    repeated_errors = ["error: unknown identifier 'x'"]
    # Both rounds return the same error — should abort after round 2
    mock_lean.verify.return_value = _lean_fail(repeated_errors)
    provider = _make_provider(["attempt 1", "attempt 2", "attempt 3"])
    result = formalize("x", "x", provider, max_attempts=3)
    assert result.success is False
    # Only 2 Lean calls: round 1 sets last_errors, round 2 matches → abort
    assert mock_lean.verify.call_count == 2


@patch("src.model.formalization.formalizer.lean")
def test_provider_exception_stops_loop(mock_lean):
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("network error")
    result = formalize("x^2", "sq", provider, max_attempts=3)
    assert result.success is False
    mock_lean.verify.assert_not_called()


def test_extract_lean_strips_markdown_fences():
    text = "```lean\ntheorem foo := sorry\n```"
    assert _extract_lean(text) == "theorem foo := sorry"


def test_extract_lean_strips_plain_fences():
    text = "```\ntheorem foo := sorry\n```"
    assert _extract_lean(text) == "theorem foo := sorry"


def test_extract_lean_no_fences_returns_as_is():
    text = "theorem foo := sorry"
    assert _extract_lean(text) == "theorem foo := sorry"


@patch("src.model.formalization.formalizer.lean")
def test_error_message_appended_to_next_turn(mock_lean):
    mock_lean.verify.side_effect = [
        _lean_fail(["error: type mismatch"]),
        _lean_ok(),
    ]
    provider = _make_provider(["attempt 1", "attempt 2"])
    formalize("latex", "name", provider, max_attempts=3)
    # Second call should include the error in messages
    second_call_messages = provider.complete.call_args_list[1][1]["messages"]
    last_user_msg = [m for m in second_call_messages if m["role"] == "user"][-1]
    assert "type mismatch" in last_user_msg["content"]
