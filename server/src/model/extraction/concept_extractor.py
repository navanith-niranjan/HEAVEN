"""LLM-based mathematical concept extractor.

Given a chunk of paper text, asks Claude to identify and structure every
theorem, definition, lemma, axiom, conjecture, corollary, or proposition
it contains.
"""

import json
import logging

from src.ingestion.extractor import ExtractedConcept
from src.model.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert mathematical concept extractor. Given a passage of mathematical text \
(which may contain LaTeX), identify every formal mathematical statement: theorems, definitions, \
lemmas, axioms, conjectures, corollaries, and propositions.

For each concept found, output a JSON object with these fields:
- "name": short descriptive name (e.g. "Pythagorean Theorem", "Cauchy-Schwarz Inequality")
- "concept_type": one of: theorem, definition, lemma, axiom, conjecture, corollary, proposition
- "latex_statement": the complete LaTeX mathematical statement (keep all math notation intact)
- "description": one or two sentences explaining the concept in plain English
- "msc_codes": list of relevant MSC 2020 codes as strings (e.g. ["11A41", "26A15"]), or []

Output ONLY a JSON array of such objects. If no formal concepts are found, output [].
Do not include proofs. Do not include informal explanations as concepts.
Only extract concepts that have an explicit mathematical statement in LaTeX notation.
"""


def extract_concepts(
    chunk: str,
    provider: LLMProvider,
    source_hint: str = "",
) -> list[ExtractedConcept]:
    """Extract mathematical concepts from a single paper chunk.

    Args:
        chunk: A section of paper text (output of chunker.chunk_paper).
        provider: LLM backend to use (sonnet recommended for accuracy).
        source_hint: Paper title or ID — included in the user message for context.

    Returns:
        List of ExtractedConcept instances. Returns [] on parse failure.
    """
    hint_prefix = f"[Source: {source_hint}]\n\n" if source_hint else ""
    user_message = (
        f"{hint_prefix}Extract all mathematical concepts from the following passage:\n\n{chunk}"
    )

    try:
        response = provider.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=4096,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("LLM call failed during concept extraction: %s", exc)
        return []

    raw = response.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse concept extraction response as JSON: %s", exc)
        return []

    if not isinstance(items, list):
        logger.warning("Concept extraction response was not a JSON array")
        return []

    concepts: list[ExtractedConcept] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            concepts.append(
                ExtractedConcept(
                    name=str(item["name"]),
                    concept_type=item["concept_type"],
                    latex_statement=str(item["latex_statement"]),
                    description=str(item.get("description", "")),
                    msc_codes=list(item.get("msc_codes", [])),
                    context_snippet=chunk[:200],
                )
            )
        except (KeyError, TypeError) as exc:
            logger.debug("Skipping malformed concept item: %s", exc)
            continue

    return concepts
