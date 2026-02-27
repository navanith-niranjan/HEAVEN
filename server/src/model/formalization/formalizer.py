"""LaTeX → Lean 4 autoformalization with iterative error correction.

Converts a normalized LaTeX mathematical statement into valid Lean 4 / Mathlib syntax,
then verifies it via the Lean 4 subprocess wrapper. On failure it feeds the error back
to the LLM for correction, up to max_attempts.

Early-abort rule: if the same Lean errors repeat on consecutive attempts, further
retries will not help — break immediately.
"""

import logging
import re
from dataclasses import dataclass, field

from src.model.providers.base import LLMProvider
from src.verification import lean

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in Lean 4 and Mathlib4. Your task is to formalize a mathematical \
statement written in LaTeX into valid Lean 4 syntax that can be verified by Lean.

Guidelines:
- Import Mathlib at the top of the file (it is already imported — do NOT include import statements).
- Write either a `#check` expression for simple type-checks, or a `theorem` with `by sorry` \
  as the body if a full proof is not required.
- Use Mathlib4 identifiers (not Lean 3 / Mathlib3 names).
- Keep the output concise — one or two lines of Lean code only.

Few-shot examples:

LaTeX: "For all integers n, n^2 \\geq 0"
Lean: theorem sq_nonneg_int (n : ℤ) : n ^ 2 ≥ 0 := by positivity

LaTeX: "The sum of the first n natural numbers equals n(n+1)/2"
Lean: theorem sum_first_n (n : ℕ) :
        Finset.sum (Finset.range (n + 1)) id = n * (n + 1) / 2 := by sorry

LaTeX: "A continuous function on a closed interval attains its maximum"
Lean: #check IsCompact.exists_isMaxOn

Respond with ONLY the Lean 4 code — no explanations, no markdown fences.
"""


@dataclass
class FormalizationResult:
    lean_source: str | None
    attempts: int
    final_lean_output: str | None
    success: bool
    errors: list[str] = field(default_factory=list)


def formalize(
    latex_statement: str,
    concept_name: str,
    provider: LLMProvider,
    max_attempts: int = 3,
) -> FormalizationResult:
    """Formalize a LaTeX statement into Lean 4 with iterative error correction.

    Args:
        latex_statement: Normalized LaTeX (output of latex_normalizer.normalize).
        concept_name: Human-readable name used as a Lean comment and for logging.
        provider: LLM backend (sonnet recommended for code generation).
        max_attempts: Maximum LLM+verify rounds. Defaults to 3.

    Returns:
        FormalizationResult indicating success/failure and the final Lean source.
    """
    messages: list[dict] = [
        {
            "role": "user",
            "content": f"Formalize this mathematical statement into Lean 4:\n\n{latex_statement}",
        }
    ]

    last_errors: list[str] = []
    lean_source: str | None = None
    lean_output: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = provider.complete(
                system=_SYSTEM_PROMPT,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("LLM call failed in formalizer (attempt %d): %s", attempt, exc)
            break

        lean_source = _extract_lean(response.content)
        messages.append({"role": "assistant", "content": response.content})

        result = lean.verify(lean_source, statement_comment=concept_name)
        lean_output = result.output

        if result.success:
            logger.info("Formalization succeeded on attempt %d for: %s", attempt, concept_name)
            return FormalizationResult(
                lean_source=lean_source,
                attempts=attempt,
                final_lean_output=lean_output,
                success=True,
                errors=[],
            )

        # Early-abort: same errors as last round — retrying won't help
        if result.errors == last_errors and last_errors:
            logger.info(
                "Early abort: repeated Lean errors after %d attempts for: %s",
                attempt, concept_name,
            )
            break

        last_errors = result.errors
        error_text = "\n".join(result.errors) if result.errors else lean_output or "Unknown error"
        messages.append({
            "role": "user",
            "content": (
                f"The above produced this Lean error:\n```\n{error_text}\n```\nPlease correct it."
            ),
        })

    return FormalizationResult(
        lean_source=lean_source,
        attempts=min(max_attempts, len(messages) // 2),
        final_lean_output=lean_output,
        success=False,
        errors=last_errors,
    )


def _extract_lean(text: str) -> str:
    """Strip markdown code fences from an LLM response if present."""
    text = text.strip()
    # Strip ```lean ... ``` or ``` ... ```
    fence_match = re.match(r"^```(?:lean)?\n?(.*?)```$", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text
