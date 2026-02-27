"""Model layer pipeline orchestration.

Two top-level entry points:

- ingest_paper(): chunk → extract → deduplicate → classify MSC → optionally formalize
                  → extract relationships → persist to SQLite + ChromaDB + NetworkX
- run_discovery(): symbolic check → formalize → Lean verify → impact traversal
                   → explain impacts → explain conflicts → persist to SQLite
"""

import logging
import uuid
from dataclasses import dataclass, field

import networkx as nx

from src.db.chroma import collections
from src.db.sqlite.models import (
    Concept,
    ConceptRelationship,
    Discovery,
    DiscoveryImpact,
    Paper,
)
from src.db.sqlite.session import get_session
from src.graph import knowledge_graph
from src.ingestion.extractor import build_concept_embedding_text
from src.model.extraction import (
    chunker,
    concept_extractor,
    deduplicator,
    relationship_extractor,
)
from src.model.formalization import formalizer, latex_normalizer
from src.model.formalization.formalizer import FormalizationResult
from src.model.providers.base import LLMProvider
from src.model.providers.registry import cheap as _cheap_default
from src.model.providers.registry import primary as _primary_default
from src.model.reasoning import conflict_explainer, impact_explainer, msc_classifier
from src.model.reasoning.conflict_explainer import ConflictExplanation
from src.model.reasoning.impact_explainer import ExplainedImpact
from src.model.symbolic import router as symbolic_router
from src.model.symbolic.router import SymbolicResult
from src.schemas.models import ConceptRead, DiscoveryCreate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper ingestion pipeline
# ---------------------------------------------------------------------------

@dataclass
class PaperIngestionResult:
    paper_id: str
    concepts_created: int
    concepts_deduplicated: int
    relationships_created: int
    concepts_formalized: int = 0
    concept_ids: list[str] = field(default_factory=list)


def ingest_paper(
    paper_meta,
    content: str,
    provider: LLMProvider | None = None,
    cheap_provider: LLMProvider | None = None,
    graph: nx.DiGraph | None = None,
    formalize_concepts: bool = False,
) -> PaperIngestionResult:
    """Ingest a paper: extract concepts, deduplicate, classify MSC, build graph edges.

    Args:
        paper_meta: SQLAlchemy Paper ORM instance (already persisted).
        content: Transiently fetched paper text (HTML or plain text).
        provider: LLM provider for extraction + formalization tasks. Defaults to sonnet.
        cheap_provider: LLM provider for cost-sensitive tasks (dedup confirmation, MSC
            classification). Defaults to haiku. Useful when running tests or when a
            non-Claude provider is the primary but you still want a cheap model for
            classification.
        graph: Live NetworkX graph to update. If None, graph updates are skipped.
        formalize_concepts: If True, run the Lean 4 formalizer on each newly created
            concept and update its lean_verification_status. Slow — off by default.

    Returns:
        PaperIngestionResult with counts of created/deduplicated/formalized concepts
        and relationships.
    """
    if provider is None:
        provider = _primary_default
    if cheap_provider is None:
        cheap_provider = _cheap_default

    paper_id = paper_meta.id
    concepts_created = 0
    concepts_deduplicated = 0
    relationships_created = 0
    concepts_formalized = 0

    # Step 1 — Chunk the paper
    chunks = chunker.chunk_paper(content)
    logger.info("Paper %s: split into %d chunks", paper_id, len(chunks))

    # Step 2 — Extract concepts from each chunk
    all_extracted = []
    for i, chunk in enumerate(chunks):
        try:
            extracted = concept_extractor.extract_concepts(
                chunk,
                provider=provider,
                source_hint=getattr(paper_meta, "title", paper_id),
            )
            all_extracted.extend(extracted)
        except Exception as exc:
            logger.warning(
                "Concept extraction failed on chunk %d of paper %s: %s", i, paper_id, exc
            )
            continue

    logger.info("Paper %s: extracted %d raw concepts", paper_id, len(all_extracted))

    # Step 3 — Deduplicate and persist new concepts
    # name → persisted concept UUID (for relationship resolution)
    name_to_id: dict[str, str] = {}
    new_concepts = []  # ExtractedConcept instances that were persisted
    new_concept_ids: list[str] = []  # parallel list for formalization

    for extracted in all_extracted:
        try:
            existing_id = deduplicator.find_duplicate(extracted, provider=cheap_provider)
        except Exception as exc:
            logger.warning(
                "Deduplication check failed for concept %s: %s", extracted.name, exc
            )
            existing_id = None

        if existing_id is not None:
            concepts_deduplicated += 1
            name_to_id[extracted.name] = existing_id
            continue

        # New concept — persist to SQLite
        concept_id = str(uuid.uuid4())
        with get_session() as session:
            db_concept = Concept(
                id=concept_id,
                name=extracted.name,
                concept_type=extracted.concept_type,
                latex_statement=extracted.latex_statement,
                description=extracted.description,
                msc_codes=extracted.msc_codes,
                source_paper_id=paper_id,
                lean_verification_status="unverified",
                chroma_embedding_id=concept_id,
            )
            session.add(db_concept)

        # Upsert to ChromaDB
        embedding_text = build_concept_embedding_text(
            extracted.name,
            extracted.latex_statement,
            extracted.description,
        )
        collections.upsert_concept(
            concept_id=concept_id,
            text=embedding_text,
            metadata={
                "name": extracted.name,
                "concept_type": extracted.concept_type,
                "source_paper_id": paper_id,
            },
        )

        # Update live graph
        if graph is not None:
            knowledge_graph.add_concept_node(graph, concept_id, {
                "name": extracted.name,
                "concept_type": extracted.concept_type,
                "lean_status": "unverified",
                "msc_codes": extracted.msc_codes,
            })

        name_to_id[extracted.name] = concept_id
        new_concepts.append(extracted)
        new_concept_ids.append(concept_id)
        concepts_created += 1

    # Step 4 — Optionally formalize each new concept against Lean 4
    if formalize_concepts:
        for extracted, concept_id in zip(new_concepts, new_concept_ids):
            try:
                normalized = latex_normalizer.normalize(extracted.latex_statement)
                result = formalizer.formalize(
                    latex_statement=normalized,
                    concept_name=extracted.name,
                    provider=provider,
                )
                lean_status = "verified" if result.success else "failed"
                with get_session() as session:
                    db_c = session.get(Concept, concept_id)
                    if db_c is not None:
                        db_c.lean_verification_status = lean_status
                        db_c.lean_output = result.final_lean_output
                if result.success:
                    concepts_formalized += 1
            except Exception as exc:
                logger.warning(
                    "Formalization failed for concept %s: %s", extracted.name, exc
                )

    # Step 5 — Classify MSC codes for the paper (write back to Paper row)
    if paper_meta.abstract:
        try:
            msc_text = f"{paper_meta.title}\n\n{paper_meta.abstract}"
            msc_codes = msc_classifier.classify_msc(msc_text, provider=cheap_provider)
            if msc_codes:
                with get_session() as session:
                    db_paper = session.get(Paper, paper_id)
                    if db_paper is not None:
                        db_paper.msc_codes = msc_codes
        except Exception as exc:
            logger.warning("MSC classification failed for paper %s: %s", paper_id, exc)

    # Step 6 — Extract relationships among newly created concepts
    if new_concepts:
        try:
            pending_rels = relationship_extractor.extract_relationships(
                new_concepts, provider=provider
            )
        except Exception as exc:
            logger.warning(
                "Relationship extraction failed for paper %s: %s", paper_id, exc
            )
            pending_rels = []

        # Step 7 — Resolve names → IDs and persist relationships
        for pending in pending_rels:
            src_id = name_to_id.get(pending.source_concept_name)
            tgt_id = name_to_id.get(pending.target_concept_name)
            if not src_id or not tgt_id:
                logger.debug(
                    "Could not resolve concept names for relationship: %s → %s",
                    pending.source_concept_name, pending.target_concept_name,
                )
                continue

            rel_id = str(uuid.uuid4())
            try:
                with get_session() as session:
                    db_rel = ConceptRelationship(
                        id=rel_id,
                        source_concept_id=src_id,
                        target_concept_id=tgt_id,
                        relationship_type=pending.relationship_type,
                        description=pending.description,
                        source_paper_id=paper_id,
                    )
                    session.add(db_rel)

                if graph is not None:
                    knowledge_graph.add_relationship_edge(
                        graph,
                        source_id=src_id,
                        target_id=tgt_id,
                        relationship_type=pending.relationship_type,
                        description=pending.description,
                    )
                relationships_created += 1
            except Exception as exc:
                logger.warning(
                    "Failed to persist relationship %s → %s: %s",
                    pending.source_concept_name, pending.target_concept_name, exc,
                )

    logger.info(
        "Paper %s ingestion complete: %d created, %d deduplicated, "
        "%d formalized, %d relationships",
        paper_id, concepts_created, concepts_deduplicated,
        concepts_formalized, relationships_created,
    )
    return PaperIngestionResult(
        paper_id=paper_id,
        concepts_created=concepts_created,
        concepts_deduplicated=concepts_deduplicated,
        relationships_created=relationships_created,
        concepts_formalized=concepts_formalized,
        concept_ids=new_concept_ids,
    )


# ---------------------------------------------------------------------------
# Discovery pipeline
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryPipelineResult:
    discovery_id: str
    sympy_result: SymbolicResult
    formalization_result: FormalizationResult
    impacts: list[ExplainedImpact] = field(default_factory=list)
    conflict_ids: list[str] = field(default_factory=list)
    conflict_explanations: list[ConflictExplanation] = field(default_factory=list)


def run_discovery(
    discovery_create: DiscoveryCreate,
    graph: nx.DiGraph,
    provider: LLMProvider | None = None,
    cheap_provider: LLMProvider | None = None,
) -> DiscoveryPipelineResult:
    """Run the full discovery processing pipeline.

    Steps:
    1. Persist Discovery row with status unchecked/unverified.
    2. Symbolic pre-check (SymPy → Wolfram) — update sympy_check_status.
    3. Normalize LaTeX.
    4. Formalize to Lean 4 with iterative error correction.
    5. Update lean_verification_status.
    6. Graph impact traversal.
    7. Find potential conflicts.
    8. Explain impacts via LLM.
    9. Explain conflicts via LLM.
    10. Persist DiscoveryImpact rows.

    Args:
        discovery_create: Pydantic schema with the user's discovery data.
        graph: Live NetworkX knowledge graph for impact traversal.
        provider: LLM provider for formalization. Defaults to sonnet.
        cheap_provider: LLM provider for impact/conflict explanation. Defaults to haiku.

    Returns:
        DiscoveryPipelineResult with all intermediate and final results, including
        conflict_ids (raw concept IDs) and conflict_explanations (LLM-generated).
    """
    if provider is None:
        provider = _primary_default
    if cheap_provider is None:
        cheap_provider = _cheap_default

    discovery_id = str(uuid.uuid4())

    # Step 1 — Persist discovery with initial pending statuses
    with get_session() as session:
        db_discovery = Discovery(
            id=discovery_id,
            name=discovery_create.name,
            base_concept_id=discovery_create.base_concept_id,
            modified_latex_statement=discovery_create.modified_latex_statement,
            modification_description=discovery_create.modification_description,
            sympy_check_status="unchecked",
            lean_verification_status="unverified",
        )
        session.add(db_discovery)

    # Step 2 — Symbolic pre-check
    # Determine concept type for routing (default to "theorem" if no base concept)
    concept_type = "theorem"
    base_concept: ConceptRead | None = None
    if discovery_create.base_concept_id:
        with get_session() as session:
            db_concept = session.get(Concept, discovery_create.base_concept_id)
            if db_concept is not None:
                concept_type = db_concept.concept_type
                base_concept = ConceptRead.model_validate(db_concept)

    sympy_result = symbolic_router.route_and_check(
        discovery_create.modified_latex_statement,
        concept_type=concept_type,
    )

    sympy_status = "unchecked"
    if sympy_result.passed is True:
        sympy_status = "passed"
    elif sympy_result.passed is False:
        sympy_status = "failed"

    with get_session() as session:
        db_discovery = session.get(Discovery, discovery_id)
        if db_discovery is not None:
            db_discovery.sympy_check_status = sympy_status
            db_discovery.sympy_check_output = sympy_result.output

    # Step 3 — Normalize LaTeX
    normalized_latex = latex_normalizer.normalize(discovery_create.modified_latex_statement)

    # Step 4 — Formalize to Lean 4
    formalization_result = formalizer.formalize(
        latex_statement=normalized_latex,
        concept_name=discovery_create.name,
        provider=provider,
        max_attempts=3,
    )

    # Step 5 — Update lean verification status
    lean_status = "verified" if formalization_result.success else "failed"
    with get_session() as session:
        db_discovery = session.get(Discovery, discovery_id)
        if db_discovery is not None:
            db_discovery.lean_verification_status = lean_status
            db_discovery.lean_output = formalization_result.final_lean_output

    # Steps 6–10 — Graph traversal, impact + conflict explanation
    impacts: list[ExplainedImpact] = []
    raw_conflict_ids: list[str] = []
    conflict_explanations: list[ConflictExplanation] = []

    if discovery_create.base_concept_id and base_concept is not None:
        affected = knowledge_graph.get_impact_subgraph(
            graph, discovery_create.base_concept_id
        )
        raw_conflict_ids = knowledge_graph.find_potential_conflicts(
            graph, discovery_create.base_concept_id
        )

        # Step 8 — Explain impacts via LLM
        if affected:
            try:
                impacts = impact_explainer.explain_impacts(
                    discovery=discovery_create,
                    base_concept=base_concept,
                    affected=affected,
                    provider=cheap_provider,
                )
            except Exception as exc:
                logger.warning(
                    "Impact explanation failed for discovery %s: %s", discovery_id, exc
                )

        # Step 9 — Explain conflicts via LLM
        if raw_conflict_ids:
            try:
                conflict_explanations = conflict_explainer.explain_conflicts(
                    discovery=discovery_create,
                    base_concept=base_concept,
                    conflict_ids=raw_conflict_ids,
                    provider=cheap_provider,
                )
            except Exception as exc:
                logger.warning(
                    "Conflict explanation failed for discovery %s: %s", discovery_id, exc
                )

        # Step 10 — Persist impact rows
        for impact in impacts:
            impact_id = str(uuid.uuid4())
            try:
                with get_session() as session:
                    db_impact = DiscoveryImpact(
                        id=impact_id,
                        discovery_id=discovery_id,
                        affected_concept_id=impact.affected_concept_id,
                        impact_type=impact.impact_type,
                        description=impact.description,
                        confidence_score=impact.confidence_score,
                    )
                    session.add(db_impact)
            except Exception as exc:
                logger.warning("Failed to persist impact %s: %s", impact_id, exc)

    logger.info(
        "Discovery %s pipeline complete: sympy=%s lean=%s impacts=%d conflicts=%d",
        discovery_id, sympy_status, lean_status, len(impacts), len(raw_conflict_ids),
    )
    return DiscoveryPipelineResult(
        discovery_id=discovery_id,
        sympy_result=sympy_result,
        formalization_result=formalization_result,
        impacts=impacts,
        conflict_ids=raw_conflict_ids,
        conflict_explanations=conflict_explanations,
    )
