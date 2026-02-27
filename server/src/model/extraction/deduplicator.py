"""Concept deduplicator.

Before persisting a new concept, checks whether an equivalent concept
already exists in ChromaDB using embedding similarity + LLM confirmation.
"""

import json
import logging

from src.db.chroma import collections
from src.ingestion.extractor import ExtractedConcept, build_concept_embedding_text
from src.model.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a mathematical ontology expert. Determine whether two mathematical concept \
statements describe the same concept.

Consider them the same if they express identical mathematical content, even if worded differently \
or using different notation. Consider them different if there is any meaningful mathematical \
distinction.

Respond with a JSON object: {"same": true/false, "reason": "one sentence explanation"}
"""


def find_duplicate(
    candidate: ExtractedConcept,
    provider: LLMProvider,
    similarity_threshold: float = 0.92,
) -> str | None:
    """Check whether an equivalent concept already exists in ChromaDB.

    Steps:
    1. Query ChromaDB for the nearest existing concept by embedding distance.
    2. If the nearest neighbor's distance is above the threshold, no duplicate — return None.
    3. If below threshold, ask the LLM to confirm semantic equivalence.
    4. Return the existing concept's ID if confirmed duplicate, else None.

    Args:
        candidate: Newly extracted concept to check.
        provider: LLM backend (haiku is sufficient for binary classification).
        similarity_threshold: Cosine distance threshold. ChromaDB uses (1 - cosine_similarity),
            so 0.92 means the vectors must be at least 8% apart to be distinct.

    Returns:
        Existing concept ID if a duplicate is confirmed, else None.
    """
    embedding_text = build_concept_embedding_text(
        candidate.name,
        candidate.latex_statement,
        candidate.description,
    )

    try:
        results = collections.query_concepts(embedding_text, n_results=1)
    except Exception as exc:
        logger.warning("ChromaDB query failed in deduplicator: %s", exc)
        return None

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]

    if not ids or not distances:
        return None

    top_distance = distances[0]
    top_id = ids[0]
    top_document = documents[0] if documents else ""

    # ChromaDB cosine distance: 0 = identical, 2 = opposite
    # We use 1 - threshold as the max distance for a potential match
    if top_distance > (1.0 - similarity_threshold):
        return None  # Far enough apart — definitely not a duplicate

    # Close match — ask the LLM to confirm
    user_message = (
        f"Concept A:\n{embedding_text}\n\n"
        f"Concept B:\n{top_document}\n\n"
        "Are these the same mathematical concept?"
    )

    try:
        response = provider.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=256,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("LLM deduplication check failed: %s", exc)
        return None

    raw = response.content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Deduplicator LLM response was not valid JSON: %r", raw)
        return None

    if verdict.get("same") is True:
        logger.info(
            "Duplicate detected: %s matches existing %s (distance=%.4f)",
            candidate.name, top_id, top_distance,
        )
        return top_id

    return None
