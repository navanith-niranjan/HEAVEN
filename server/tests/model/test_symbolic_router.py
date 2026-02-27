"""Tests for src/model/symbolic/router.py."""

from unittest.mock import patch

from src.model.symbolic.router import SymbolicBackend, route_and_check


def test_axiom_is_skipped():
    result = route_and_check("x = x", "axiom")
    assert result.backend == SymbolicBackend.SKIP
    assert result.passed is None
    assert "axiom" in result.output.lower() or "definitional" in result.output.lower()


def test_definition_is_skipped():
    result = route_and_check("A group is a set...", "definition")
    assert result.backend == SymbolicBackend.SKIP
    assert result.passed is None


@patch("src.model.symbolic.router.check_latex_expression")
def test_sympy_pass_returns_sympy_result(mock_check):
    from src.verification.sympy_check import SympyCheckResult
    mock_check.return_value = SympyCheckResult(
        passed=True,
        output="Parsed OK",
        simplified_form="x**2",
    )
    result = route_and_check("x^2", "theorem")
    assert result.backend == SymbolicBackend.SYMPY
    assert result.passed is True
    assert result.simplified_form == "x**2"


@patch("src.model.symbolic.router.check_latex_expression")
def test_sympy_simplification_fail_returns_sympy_false(mock_check):
    from src.verification.sympy_check import SympyCheckResult
    mock_check.return_value = SympyCheckResult(
        passed=False,
        output="Simplification failed: division by zero",
    )
    result = route_and_check(r"\frac{1}{0}", "theorem")
    assert result.backend == SymbolicBackend.SYMPY
    assert result.passed is False


@patch("src.model.symbolic.router.settings")
@patch("src.model.symbolic.router.wolfram_query")
@patch("src.model.symbolic.router.check_latex_expression")
def test_sympy_parse_fail_falls_to_wolfram(mock_check, mock_wolfram, mock_settings):
    from src.ingestion.wolfram_client import WolframResult
    from src.verification.sympy_check import SympyCheckResult
    mock_check.return_value = SympyCheckResult(passed=False, output="LaTeX parse failed: ...")
    mock_settings.wolfram_app_id = "test-app-id"
    mock_wolfram.return_value = WolframResult(
        query="...", plaintext="True", pods=[]
    )
    result = route_and_check(r"\exists x", "theorem")
    assert result.backend == SymbolicBackend.WOLFRAM
    assert result.passed is True


@patch("src.model.symbolic.router.settings")
@patch("src.model.symbolic.router.check_latex_expression")
def test_sympy_parse_fail_no_wolfram_returns_skip(mock_check, mock_settings):
    from src.verification.sympy_check import SympyCheckResult
    mock_check.return_value = SympyCheckResult(passed=False, output="LaTeX parse failed: ...")
    mock_settings.wolfram_app_id = ""
    result = route_and_check(r"\exists x", "theorem")
    assert result.backend == SymbolicBackend.SKIP
    assert result.passed is None


@patch("src.model.symbolic.router.settings")
@patch("src.model.symbolic.router.wolfram_query")
@patch("src.model.symbolic.router.check_latex_expression")
def test_wolfram_returns_empty_plaintext_is_false(mock_check, mock_wolfram, mock_settings):
    from src.ingestion.wolfram_client import WolframResult
    from src.verification.sympy_check import SympyCheckResult
    mock_check.return_value = SympyCheckResult(passed=False, output="LaTeX parse failed: ...")
    mock_settings.wolfram_app_id = "key"
    mock_wolfram.return_value = WolframResult(query="...", plaintext="", pods=[])
    result = route_and_check(r"\exists x", "theorem")
    assert result.backend == SymbolicBackend.WOLFRAM
    assert result.passed is False


@patch("src.model.symbolic.router.settings")
@patch("src.model.symbolic.router.wolfram_query")
@patch("src.model.symbolic.router.check_latex_expression")
def test_wolfram_none_falls_to_skip(mock_check, mock_wolfram, mock_settings):
    from src.verification.sympy_check import SympyCheckResult
    mock_check.return_value = SympyCheckResult(passed=False, output="LaTeX parse failed: ...")
    mock_settings.wolfram_app_id = "key"
    mock_wolfram.return_value = None
    result = route_and_check(r"\exists x", "theorem")
    assert result.backend == SymbolicBackend.SKIP
