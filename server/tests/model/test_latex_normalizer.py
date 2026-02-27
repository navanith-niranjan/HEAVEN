"""Tests for src/model/formalization/latex_normalizer.py — pure Python, no mocking."""

from src.model.formalization.latex_normalizer import normalize


def test_strips_display_math_dollars():
    result = normalize(r"$$x^2 + 1 = 0$$")
    assert "$$" not in result
    assert "x^2" in result


def test_strips_display_math_brackets():
    result = normalize(r"\[x^2 + 1\]")
    assert r"\[" not in result
    assert r"\]" not in result
    assert "x^2" in result


def test_strips_inline_math_parens():
    result = normalize(r"\(x + y\)")
    assert r"\(" not in result
    assert r"\)" not in result
    assert "x + y" in result


def test_strips_label_command():
    result = normalize(r"f(x) \label{eq:main}")
    assert r"\label" not in result
    assert "f(x)" in result


def test_strips_ref_command():
    result = normalize(r"As in \ref{thm:1}, we have...")
    assert r"\ref" not in result
    assert "we have" in result


def test_strips_eqref_command():
    result = normalize(r"From \eqref{eq:main} it follows")
    assert r"\eqref" not in result


def test_strips_cite_command():
    result = normalize(r"This was proved in \cite{smith2020}")
    assert r"\cite" not in result
    assert "This was proved" in result


def test_strips_cref_command():
    result = normalize(r"By \cref{thm:euler} we get")
    assert r"\cref" not in result


def test_expands_such_that():
    result = normalize(r"there exists x s.t. f(x) = 0")
    assert "s.t." not in result
    assert "such that" in result


def test_expands_with_respect_to():
    result = normalize(r"derivative w.r.t. x")
    assert "w.r.t." not in result
    assert "with respect to" in result


def test_expands_iff():
    result = normalize(r"A iff B")
    assert " iff " not in result
    assert "if and only if" in result


def test_collapses_multiple_spaces():
    result = normalize("x  +   y   =   z")
    assert "  " not in result


def test_strips_trailing_whitespace():
    result = normalize("  x + y  ")
    assert result == result.strip()


def test_multiline_display_math():
    result = normalize("$$\n  \\forall x \\in \\mathbb{R}\n$$")
    assert "$$" not in result
    assert "\\forall" in result


def test_preserves_math_notation():
    latex = r"\forall n \in \mathbb{Z},\, n^2 \geq 0"
    result = normalize(latex)
    assert r"\forall" in result
    assert r"\mathbb{Z}" in result
    assert r"n^2" in result


def test_combined_cleanup():
    latex = r"\[f(x) \geq 0 \label{eq:pos} \cite{ref}\]"
    result = normalize(latex)
    assert r"\[" not in result
    assert r"\label" not in result
    assert r"\cite" not in result
    assert "f(x)" in result
