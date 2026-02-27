"""Conflict explainer.

Given a discovery and the concept IDs that the knowledge graph flagged as
potential contradictions (via 'contradicts' edges), asks Claude to explain
the nature of each conflict and assess its severity.
"""

import json
import logging
from dataclasses import dataclass

from src.model.providers.base import LLMProvider
from src.schemas.models import ConceptRead, DiscoveryCreate

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in mathematical reasoning. A user has created a mathematical discovery \
(a modification to an existing concept). You are given the discovery, its base concept, \
and a list of concept IDs that are connected to the base concept via 'contradicts' edges \
in the knowledge graph — meaning they may conflict with this discovery.

For each conflicting concept, explain:
1. WHY this discovery conflicts with or tensions against that concept (one or two sentences)
2. The severity of the conflict: "minor" (notation/framing only), "moderate" (some \
   incompatibility), or "severe" (direct logical contradiction)

Output a JSON array. Each element:
{
  "conflict_concept_id": "<uuid>",
  "explanation": "<why these concepts conflict>",
  "severity": "<minor|moderate|severe>"
}

If you are not confident there is a real conflict (perhaps the graph edge was over-broad), \
still include the entry but set severity to "minor" and explain the uncertainty.
Output [] only if no concept IDs were provided.
"""


@dataclass
class ConflictExplanation:
    conflict_concept_id: str
    explanation: str
    severity: str  # "minor" | "moderate" | "severe"


def explain_conflicts(
    discovery: DiscoveryCreate,
    base_concept: ConceptRead,
    conflict_ids: list[str],
    provider: LLMProvider,
) -> list[ConflictExplanation]:
    """Generate explanations for all potential conflicts a discovery introduces.

    Args:
        discovery: The user's discovery (modification to base_concept).
        base_concept: The concept being modified.
        conflict_ids: Concept IDs connected via 'contradicts' edges from
            knowledge_graph.find_potential_conflicts(). May be empty.
        provider: LLM backend (haiku is sufficient).

    Returns:
        List of ConflictExplanation. Returns [] if conflict_ids is empty.
    """
    if not conflict_ids:
        return []

    id_lines = "\n".join(f"- {cid}" for cid in conflict_ids)
    user_message = (
        f"Discovery name: {discovery.name}\n"
        f"Modification: {discovery.modification_description}\n"
        f"Modified statement: {discovery.modified_latex_statement}\n\n"
        f"Base concept: {base_concept.name}\n"
        f"Base concept type: {base_concept.concept_type}\n"
        f"Base concept statement: {base_concept.latex_statement}\n\n"
        f"Potentially conflicting concept IDs (from knowledge graph):\n{id_lines}\n\n"
        "Explain the conflict between this discovery and each of the above concepts."
    )

    try:
        response = provider.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2048,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("LLM call failed in conflict_explainer: %s", exc)
        return []

    raw = response.content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse conflict explanation response as JSON: %s", exc)
        return []

    if not isinstance(items, list):
        return []

    valid_severities = {"minor", "moderate", "severe"}
    results: list[ConflictExplanation] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        cid = item.get("conflict_concept_id", "")
        explanation = item.get("explanation", "")
        severity = item.get("severity", "minor")

        if cid not in conflict_ids:
            logger.debug("Skipping unknown conflict concept ID: %s", cid)
            continue
        if severity not in valid_severities:
            severity = "minor"

        results.append(ConflictExplanation(
            conflict_concept_id=cid,
            explanation=str(explanation),
            severity=severity,
        ))

    return results
