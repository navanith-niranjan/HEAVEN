"""LaTeX normalizer — pure text preprocessing, no LLM.

Cleans LaTeX statements before sending them to the formalizer.
Removes markup that is irrelevant to Lean 4 formalization (labels, refs,
display delimiters) and expands common abbreviations.
"""

import re


def normalize(latex: str) -> str:
    """Normalize a LaTeX mathematical statement for autoformalization.

    Steps (in order):
    1. Strip display math delimiters ($$...$$, \\[...\\], \\(...\\))
    2. Remove \\label{...}, \\ref{...}, \\cite{...} commands
    3. Expand common abbreviations (s.t., w.r.t., iff, etc.)
    4. Collapse multiple whitespace / newlines

    Args:
        latex: Raw LaTeX string from a paper or user input.

    Returns:
        Cleaned LaTeX string with display delimiters and metadata commands removed.
    """
    text = latex

    # Strip display math delimiters — keep the inner content
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text, flags=re.DOTALL)

    # Remove cross-reference and citation commands
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\ref\{[^}]*\}", "", text)
    text = re.sub(r"\\eqref\{[^}]*\}", "", text)
    text = re.sub(r"\\cite\{[^}]*\}", "", text)
    text = re.sub(r"\\cref\{[^}]*\}", "", text)

    # Expand common abbreviations
    # Note: patterns ending in '.' cannot use a trailing \b (period is non-word),
    # so we use a lookahead (?=\s|$) instead.
    abbreviations = {
        r"\bs\.t\.(?=\s|$)": "such that",
        r"\bw\.r\.t\.(?=\s|$)": "with respect to",
        r"\bi\.e\.(?=\s|$)": "that is",
        r"\be\.g\.(?=\s|$)": "for example",
        r"\biff\b": "if and only if",
        r"\bw\.l\.o\.g\.(?=\s|$)": "without loss of generality",
        r"\bWLOG\b": "without loss of generality",
        r"\bLHS\b": "left-hand side",
        r"\bRHS\b": "right-hand side",
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text
