"""Impact explainer.

Given a discovery and its affected concepts (from NetworkX graph traversal),
asks Claude to explain WHY each concept is affected and assign a confidence score.
All affected concepts are batched into a single LLM call to minimise cost.
"""

import json
import logging
from dataclasses import dataclass

from src.model.providers.base import LLMProvider
from src.schemas.models import ConceptRead, DiscoveryCreate, ImpactType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in mathematical reasoning. A user has created a mathematical discovery \
(a modification to an existing concept). You are given the discovery, its base concept, \
and a list of other concepts that are reachable from the base concept in the knowledge graph.

For each affected concept, explain:
1. WHY the discovery affects it (one or two sentences)
2. The nature of the impact — one of: extends, contradicts, generalizes, enables, invalidates
3. A confidence score from 0.0 to 1.0 reflecting how certain you are of the impact

Output a JSON array. Each element:
{
  "affected_concept_id": "<uuid>",
  "impact_type": "<one of: extends, contradicts, generalizes, enables, invalidates>",
  "description": "<explanation>",
  "confidence_score": <0.0-1.0>
}

Only include concepts where you are confident there is a real mathematical impact \
(confidence >= 0.3). Output [] if no confident impacts are found.
"""


@dataclass
class ExplainedImpact:
    affected_concept_id: str
    impact_type: ImpactType
    description: str
    confidence_score: float


def explain_impacts(
    discovery: DiscoveryCreate,
    base_concept: ConceptRead,
    affected: dict[str, list[str]],
    provider: LLMProvider,
) -> list[ExplainedImpact]:
    """Generate human-readable explanations for all discovery impacts.

    Args:
        discovery: The user's discovery (modification to base_concept).
        base_concept: The concept being modified.
        affected: Output of knowledge_graph.get_impact_subgraph() —
                  maps relationship_type → list of concept IDs.
        provider: LLM backend (haiku is fine; impacts are description-heavy, not code).

    Returns:
        List of ExplainedImpact ready to be persisted as DiscoveryImpact rows.
    """
    # Flatten to a unique set of affected concept IDs
    all_affected_ids = list({cid for ids in affected.values() for cid in ids})

    if not all_affected_ids:
        return []

    # Build relationship context per concept
    concept_context_lines = []
    for rel_type, ids in affected.items():
        for cid in ids:
            concept_context_lines.append(f"- Concept ID {cid} (relationship: {rel_type})")
    concept_context = "\n".join(concept_context_lines)

    user_message = (
        f"Discovery name: {discovery.name}\n"
        f"Modification: {discovery.modification_description}\n"
        f"Modified statement: {discovery.modified_latex_statement}\n\n"
        f"Base concept: {base_concept.name}\n"
        f"Base concept type: {base_concept.concept_type}\n"
        f"Base concept statement: {base_concept.latex_statement}\n\n"
        f"Affected concepts reachable in the knowledge graph:\n{concept_context}\n\n"
        "Explain the impact of this discovery on each affected concept."
    )

    try:
        response = provider.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=4096,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("LLM call failed in impact_explainer: %s", exc)
        return []

    raw = response.content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse impact explanation response as JSON: %s", exc)
        return []

    if not isinstance(items, list):
        return []

    valid_impact_types = {"extends", "contradicts", "generalizes", "enables", "invalidates"}
    results: list[ExplainedImpact] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        cid = item.get("affected_concept_id", "")
        impact_type = item.get("impact_type", "")
        description = item.get("description", "")
        confidence = item.get("confidence_score", 0.0)

        if cid not in all_affected_ids:
            logger.debug("Skipping unknown concept ID in impact explanation: %s", cid)
            continue
        if impact_type not in valid_impact_types:
            logger.debug("Skipping invalid impact_type: %s", impact_type)
            continue

        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.5

        score = max(0.0, min(1.0, score))

        results.append(ExplainedImpact(
            affected_concept_id=cid,
            impact_type=impact_type,  # type: ignore[arg-type]
            description=str(description),
            confidence_score=score,
        ))

    return results
