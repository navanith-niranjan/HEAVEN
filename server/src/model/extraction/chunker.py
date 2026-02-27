"""Paper content chunker.

Splits raw paper HTML/text into chunks suitable for LLM concept extraction.
LaTeX theorem environments are kept intact as individual chunks.
Surrounding prose is grouped into paragraph-level chunks up to max_chunk_chars.
"""

import re

# LaTeX environments that contain complete mathematical statements
_ENV_NAMES = (
    "theorem", "definition", "lemma", "proof",
    "corollary", "proposition", "axiom", "conjecture",
    "remark", "example", "claim",
)
_ENV_PATTERN = re.compile(
    r"\\begin\{(" + "|".join(_ENV_NAMES) + r")\}.*?\\end\{\1\}",
    re.DOTALL | re.IGNORECASE,
)


def chunk_paper(content: str, max_chunk_chars: int = 8000) -> list[str]:
    """Split paper content into LLM-ready chunks.

    Strategy:
    1. Extract every LaTeX theorem/definition/lemma/… environment as its own chunk.
    2. Split the remaining text on paragraph breaks.
    3. Group consecutive paragraphs until max_chunk_chars is reached.

    LaTeX environments are never split mid-environment.

    Args:
        content: Raw HTML or plain text fetched transiently from arXiv.
        max_chunk_chars: Soft upper bound for prose chunks (characters).

    Returns:
        List of string chunks ready for concept_extractor.extract_concepts().
    """
    chunks: list[str] = []

    # Find all LaTeX environment positions so we can exclude them from prose
    env_spans: list[tuple[int, int]] = []
    for match in _ENV_PATTERN.finditer(content):
        chunks.append(match.group(0).strip())
        env_spans.append((match.start(), match.end()))

    # Build prose: content with all environment text blanked out
    prose = content
    for start, end in reversed(env_spans):
        prose = prose[:start] + prose[end:]

    # Split prose on blank lines (paragraph breaks)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", prose) if p.strip()]

    # Group paragraphs into chunks up to max_chunk_chars
    current_parts: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > max_chunk_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(para)
        current_len += len(para)

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if c]
