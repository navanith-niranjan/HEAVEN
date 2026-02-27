"""Symbolic verification router.

Decides which symbolic backend (SymPy, Wolfram Alpha, or neither) to use
for a given mathematical statement, based on concept type and available services.
No LLM is involved — this is a fast, deterministic routing heuristic.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from src.config import settings
from src.ingestion.wolfram_client import query as wolfram_query
from src.verification.sympy_check import check_latex_expression

logger = logging.getLogger(__name__)


class SymbolicBackend(Enum):
    SYMPY = "sympy"
    WOLFRAM = "wolfram"
    SKIP = "skip"


@dataclass
class SymbolicResult:
    backend: SymbolicBackend
    passed: bool | None       # None means "not checked" (SKIP)
    output: str
    simplified_form: str | None = None


# Concept types that are definitional — not symbolically verifiable
_SKIP_TYPES = frozenset({"axiom", "definition"})


def route_and_check(latex: str, concept_type: str) -> SymbolicResult:
    """Route a LaTeX statement to the appropriate symbolic backend and check it.

    Decision logic (no LLM):
    1. SKIP if concept_type is 'axiom' or 'definition' — definitional, not verifiable.
    2. Try SymPy: parse the expression symbolically.
       - Success → return SYMPY result.
       - Parse failure with algebraic content → fall through to Wolfram.
    3. If Wolfram is configured, try it.
    4. Otherwise return SKIP with a "no backend" message.

    Args:
        latex: LaTeX mathematical expression (should be normalized first).
        concept_type: One of the valid concept type literals from the schema.

    Returns:
        SymbolicResult with backend, pass/fail, and any output text.
    """
    if concept_type in _SKIP_TYPES:
        return SymbolicResult(
            backend=SymbolicBackend.SKIP,
            passed=None,
            output=f"Skipped: {concept_type} statements are definitional — not verifiable.",
        )

    # Attempt SymPy
    try:
        sympy_result = check_latex_expression(latex)
        if sympy_result.passed:
            return SymbolicResult(
                backend=SymbolicBackend.SYMPY,
                passed=True,
                output=sympy_result.output,
                simplified_form=sympy_result.simplified_form,
            )
        # SymPy failed — check if it's a parse error (likely too complex for SymPy)
        is_parse_error = "parse" in sympy_result.output.lower()
        if not is_parse_error:
            # SymPy parsed but simplification failed — the expression is ill-formed
            return SymbolicResult(
                backend=SymbolicBackend.SYMPY,
                passed=False,
                output=sympy_result.output,
            )
        # Parse error — fall through to Wolfram
    except Exception as exc:
        logger.warning("SymPy check raised unexpectedly: %s", exc)
        # Fall through

    # Attempt Wolfram Alpha if configured
    if settings.wolfram_app_id:
        try:
            wolfram_result = wolfram_query(latex)
            if wolfram_result is not None:
                passed = bool(wolfram_result.plaintext)
                return SymbolicResult(
                    backend=SymbolicBackend.WOLFRAM,
                    passed=passed,
                    output=wolfram_result.plaintext or "No result text returned.",
                )
        except Exception as exc:
            logger.warning("Wolfram Alpha query failed: %s", exc)

    return SymbolicResult(
        backend=SymbolicBackend.SKIP,
        passed=None,
        output="No symbolic backend could handle this expression.",
    )
