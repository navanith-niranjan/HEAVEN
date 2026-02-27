"""Tests for src/model/extraction/chunker.py — pure Python, no mocking needed."""

from src.model.extraction.chunker import chunk_paper


def test_empty_input_returns_empty():
    assert chunk_paper("") == []


def test_whitespace_only_returns_empty():
    assert chunk_paper("   \n\n   ") == []


def test_prose_only_returns_single_chunk():
    content = "This is a paper about topology. It has several results."
    chunks = chunk_paper(content)
    assert len(chunks) == 1
    assert "topology" in chunks[0]


def test_single_latex_environment_is_own_chunk():
    content = r"""
\begin{theorem}
For all n, n^2 >= 0.
\end{theorem}
"""
    chunks = chunk_paper(content)
    # The theorem environment should be its own chunk
    env_chunks = [c for c in chunks if r"\begin{theorem}" in c]
    assert len(env_chunks) == 1
    assert "n^2" in env_chunks[0]


def test_multiple_environments_each_own_chunk():
    content = r"""
Preamble text here.

\begin{theorem}
Theorem statement.
\end{theorem}

Some middle prose.

\begin{definition}
Definition statement.
\end{definition}

Conclusion.
"""
    chunks = chunk_paper(content)
    theorem_chunks = [c for c in chunks if r"\begin{theorem}" in c]
    definition_chunks = [c for c in chunks if r"\begin{definition}" in c]
    assert len(theorem_chunks) == 1
    assert len(definition_chunks) == 1


def test_prose_surrounding_environments_is_grouped():
    content = r"""
Introduction paragraph.

\begin{theorem}
T1.
\end{theorem}

More prose. Even more prose.
"""
    chunks = chunk_paper(content)
    # There should be a prose chunk containing both prose paragraphs (or separate ones)
    prose_chunks = [c for c in chunks if r"\begin" not in c]
    assert len(prose_chunks) >= 1
    combined = " ".join(prose_chunks)
    assert "Introduction" in combined
    assert "More prose" in combined


def test_prose_grouped_within_max_chunk_chars():
    # 10 paragraphs of 50 chars each = 500 chars total — all fit in one chunk at 8000 default
    para = "A" * 50
    content = "\n\n".join([para] * 10)
    chunks = chunk_paper(content, max_chunk_chars=8000)
    assert len(chunks) == 1


def test_prose_split_when_exceeds_max_chunk_chars():
    para = "A" * 100
    content = "\n\n".join([para] * 10)  # 1000 chars
    chunks = chunk_paper(content, max_chunk_chars=250)
    assert len(chunks) > 1


def test_all_supported_environments():
    env_names = [
        "theorem", "definition", "lemma", "proof",
        "corollary", "proposition", "axiom", "conjecture",
    ]
    for env in env_names:
        content = rf"\begin{{{env}}}content\end{{{env}}}"
        chunks = chunk_paper(content)
        assert any(env in c for c in chunks), f"Environment '{env}' not found in chunks"


def test_no_empty_chunks_in_output():
    content = r"""
\begin{theorem}
Something.
\end{theorem}

A paragraph.
"""
    chunks = chunk_paper(content)
    for chunk in chunks:
        assert chunk.strip() != "", "Got an empty chunk"
