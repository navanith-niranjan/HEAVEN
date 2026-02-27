# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HEAVEN** — an AI-native research assistant for mathematicians. Users apply new properties to existing mathematical concepts and the system shows what changes, what conflicts arise, how other concepts are affected, and where corrections can be made. Theoretical discoveries can then be stress-tested against real-world scenarios.

Three planned layers (implement in order):
1. **Data Layer** ✓ complete
2. **Model Layer** ✓ complete — LLM orchestration, autoformalization, concept extraction
3. **Orchestration Layer** — session management, user-facing API

## Development Setup

All server code is Python. Use `uv` as the package manager.

```bash
cd server

# Install dependencies
uv sync

# Run database migrations
uv run alembic upgrade head

# Run tests
uv run pytest

# Lint
uv run ruff check src/
uv run ruff format src/

# Run a single test file
uv run pytest tests/path/to/test_file.py

# Run a single test by name
uv run pytest -k "test_name"
```

## Architecture

### Storage — what goes where

| Store | Contents |
|---|---|
| SQLite (`heaven.db`) | Structured relational data: papers metadata, extracted concepts, graph relationships, discoveries, impact analysis results |
| ChromaDB (`chroma_data/`) | Vector embeddings: concept statements (for semantic search), paper abstracts |
| NetworkX (in-memory) | Live knowledge graph reconstructed from SQLite `concept_relationships` at startup |

**Paper content is never stored.** Only metadata (title, authors, abstract, arXiv ID, DOI, URL) is persisted. Full paper text is fetched on demand from external APIs and discarded after concept extraction.

### Key Data Flow

```
User query
  → ChromaDB semantic search (concepts + paper abstracts)
  → Identify relevant papers by metadata
  → Fetch paper content live (arXiv ar5iv / Scholar)
  → model/pipeline.ingest_paper() — full extraction + graph build
  → Store extracted concepts in SQLite + ChromaDB
  → Build NetworkX edges from concept relationships
  → Discard raw paper content

User creates a Discovery (modification to a concept)
  → model/pipeline.run_discovery()
  → SymPy pre-check (fast, symbolic) → src/verification/sympy_check.py
  → latex_normalizer.normalize() → formalizer.formalize() (up to 3 LLM+Lean rounds)
  → Lean 4 formal verification → src/verification/lean.py
  → NetworkX impact traversal → src/graph/knowledge_graph.py
  → impact_explainer.explain_impacts() → DiscoveryImpact rows
  → Store discovery + impacts in SQLite
```

### Module Map

```
server/
├── src/
│   ├── config.py                   # Pydantic settings from .env (incl. anthropic_api_key)
│   ├── db/
│   │   ├── sqlite/
│   │   │   ├── models.py           # SQLAlchemy ORM (Paper, Concept, ConceptRelationship, Discovery, DiscoveryImpact)
│   │   │   └── session.py          # get_session() context manager
│   │   └── chroma/
│   │       └── collections.py      # ChromaDB get/upsert/query helpers
│   ├── graph/
│   │   └── knowledge_graph.py      # build_graph(), get_impact_subgraph(), get_dependencies(), find_potential_conflicts()
│   ├── ingestion/
│   │   ├── arxiv_client.py         # On-demand arXiv search + fetch
│   │   ├── wolfram_client.py       # On-demand Wolfram Alpha queries
│   │   ├── scholar_client.py       # On-demand Semantic Scholar search
│   │   └── extractor.py            # Delegates to model/extraction/concept_extractor.py
│   ├── model/                      # ← Model layer
│   │   ├── providers/
│   │   │   ├── base.py             # LLMProvider ABC + LLMResponse dataclass
│   │   │   ├── claude.py           # ClaudeProvider (Anthropic SDK)
│   │   │   ├── openai_compatible.py# OpenAICompatibleProvider (httpx; OpenRouter/DeepSeek/Gemini/vLLM)
│   │   │   └── registry.py         # primary + cheap singletons resolved from .env settings
│   │   ├── extraction/
│   │   │   ├── chunker.py          # chunk_paper(): LaTeX-env-aware text splitter
│   │   │   ├── concept_extractor.py# extract_concepts(): LLM → list[ExtractedConcept]
│   │   │   ├── relationship_extractor.py # extract_relationships(): LLM → list[PendingRelationship]
│   │   │   └── deduplicator.py     # find_duplicate(): ChromaDB + LLM duplicate check
│   │   ├── formalization/
│   │   │   ├── latex_normalizer.py # normalize(): pure-text LaTeX cleanup
│   │   │   └── formalizer.py       # formalize(): LaTeX → Lean 4 with retry loop
│   │   ├── symbolic/
│   │   │   └── router.py           # route_and_check(): SymPy → Wolfram → SKIP heuristic
│   │   ├── reasoning/
│   │   │   ├── conflict_explainer.py # explain_conflicts(): LLM severity + explanation per conflict
│   │   │   ├── impact_explainer.py # explain_impacts(): batched LLM impact descriptions
│   │   │   └── msc_classifier.py   # classify_msc(): LLM → MSC 2020 code list
│   │   └── pipeline.py             # ingest_paper(), run_discovery() — top-level orchestration
│   ├── verification/
│   │   ├── sympy_check.py          # Symbolic pre-verification (fast)
│   │   └── lean.py                 # Lean 4 subprocess wrapper (authoritative)
│   └── schemas/
│       └── models.py               # Pydantic schemas for all entities
└── alembic/                        # Database migrations
    └── versions/001_initial_schema.py
```

### Model Layer — LLM Selection

Models are configured via `.env`. The registry exports two singletons used throughout the codebase:
- `registry.primary` — high-capability model; used for extraction, formalization, relationship reasoning
- `registry.cheap` — cost-optimised model; used for dedup confirmation, MSC classification, impact/conflict descriptions

| Task | Provider role | Rationale |
|---|---|---|
| Concept extraction | `primary` | High accuracy needed; math notation is complex |
| Relationship extraction | `primary` | Semantic reasoning between statements |
| Autoformalization (LaTeX → Lean 4) | `primary` | Code generation requires strong reasoning |
| Deduplication confirmation | `cheap` | Binary yes/no; cost-sensitive at scale |
| MSC classification | `cheap` | Classification from fixed taxonomy; simple |
| Impact/conflict explanation | `cheap` | Descriptions, not mathematical proofs |

To switch providers, edit `.env` — no code changes needed. Supported values for `PRIMARY_PROVIDER` / `CHEAP_PROVIDER`:
- `"claude"` — Anthropic API (requires `ANTHROPIC_API_KEY`)
- `"openai_compatible"` — any OpenAI-format API (requires `OPENAI_API_KEY` + `OPENAI_BASE_URL`)

### Model Layer — Key Design Decisions

- **Synchronous throughout** — matches data layer; async deferred to orchestration layer.
- **Formalizer retry strategy** — up to 3 LLM+Lean rounds. Each failure feeds Lean error output back to the LLM as a correction request. Early-abort if consecutive rounds produce identical errors.
- **Deduplication threshold** — ChromaDB cosine distance ≤ 0.08 (i.e., similarity ≥ 0.92) triggers an LLM confirmation step before declaring a duplicate.
- **Partial ingestion** — chunk-level failures are logged and skipped; a paper is never fully rejected due to a single bad chunk.
- **No content stored** — `ingest_paper()` receives transiently fetched text and discards it; only structured concepts are persisted.

### SQLite Schema (key tables)

- **`papers`** — metadata only; `arxiv_id` and `doi` are unique keys
- **`concepts`** — extracted mathematical knowledge; `concept_type` is one of: theorem, definition, lemma, axiom, conjecture, corollary, proposition
- **`concept_relationships`** — persisted graph edges; `relationship_type` is one of: proves, depends_on, generalizes, is_special_case_of, contradicts, cited_by, equivalent_to, extends
- **`discoveries`** — user mutations of concepts; has both `sympy_check_status` and `lean_verification_status`
- **`discovery_impacts`** — what a discovery affects; `impact_type` is one of: extends, contradicts, generalizes, enables, invalidates

### Data Sources

| Source | Client | Notes |
|---|---|---|
| arXiv | `src/ingestion/arxiv_client.py` | Uses ar5iv.org HTML for structured content |
| Semantic Scholar | `src/ingestion/scholar_client.py` | Preferred over Google Scholar scraping |
| Wolfram Alpha | `src/ingestion/wolfram_client.py` | Requires `WOLFRAM_APP_ID` in `.env` |

### Verification Pipeline

Two-stage: SymPy first (cheap, catches most hallucinations), then Lean 4 (authoritative).

**Lean 4 prerequisites** — must be set up manually before `src/verification/lean.py` is usable:
1. Install Lean 4 via elan: `curl https://elan.lean-lang.org/elan-init.sh -sSf | sh`
2. Fetch pre-compiled Mathlib binaries (never compile from scratch — takes hours):
   `cd server/lean_project && lake exe cache get`
3. Verify setup: `lake env lean HEAVEN/Basic.lean` — should print nothing (no errors)

The lean_project is already scaffolded at `server/lean_project/`. `lake env lean <file>` is used (not bare `lean <file>`) so Mathlib imports resolve correctly.

**Autoformalization** (LaTeX → Lean 4 syntax) is handled by `src/model/formalization/formalizer.py`. `lean.py` assumes it receives valid Lean 4 source.

### Embeddings

Default embedding model: `all-MiniLM-L6-v2` (via `sentence-transformers`). Configured via `EMBEDDING_MODEL` in `.env`. This is a placeholder — swap for a math-aware model when the model layer is decided. ChromaDB uses cosine similarity.

### MSC Codes

Mathematics Subject Classification codes are stored as JSON arrays on both `papers` and `concepts`. Use standard 2-digit or 5-character MSC codes (e.g., `"57"` for Manifolds, `"11A41"` for Primes). arXiv does not expose MSC codes directly — they must be inferred or manually assigned.

### Orchestration Layer — Startup Contract

The orchestration layer (FastAPI app) must do the following at startup before handling any requests:

```python
from src.db.sqlite.session import init_db
from src.graph.knowledge_graph import build_graph

init_db()                 # creates tables if they don't exist (idempotent)
graph = build_graph()     # loads all concepts + relationships into NetworkX
```

`graph` must be kept in application state (e.g., `app.state.graph`) and passed into both pipeline calls:

```python
from src.model.pipeline import ingest_paper, run_discovery

# Paper ingestion — paper_meta must be a persisted SQLAlchemy Paper ORM instance
result = ingest_paper(paper_meta, content=fetched_text, graph=app.state.graph)
# result.concept_ids → list of UUIDs created; result.concepts_created → int count

# Discovery processing
result = run_discovery(discovery_create, graph=app.state.graph)
# result.discovery_id, result.sympy_result, result.formalization_result,
# result.impacts, result.conflict_ids, result.conflict_explanations
```

**Paper ingestion flow** (orchestration layer's responsibility before calling `ingest_paper`):
1. Call `arxiv_client.search()` or `scholar_client.search()` — get paper metadata
2. Persist `Paper` ORM row to SQLite (check for `arxiv_id` / `doi` duplicates first)
3. Call `arxiv_client.fetch_content_transiently()` — get full paper text
4. Call `ingest_paper(paper_meta, content, graph=graph)` — model layer handles the rest
5. Discard `content` — never store it
