"""MSC 2020 classifier.

Assigns Mathematics Subject Classification codes to papers or concepts using Claude.
The LLM is given a curated list of top-level MSC areas and asked to choose the
most relevant ones.
"""

import json
import logging

from src.model.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# MSC 2020 top-level areas (two-digit codes).
# Source: https://zbmath.org/classification/
_MSC_AREAS = """
00 General and overarching topics; collections
01 History and biography
03 Mathematical logic and foundations
05 Combinatorics
06 Order, lattices, ordered algebraic structures
08 General algebraic systems
11 Number theory
12 Field theory and polynomials
13 Commutative algebra
14 Algebraic geometry
15 Linear and multilinear algebra; matrix theory
16 Associative rings and algebras
17 Nonassociative rings and algebras
18 Category theory; homological algebra
19 K-theory
20 Group theory and generalizations
22 Topological groups, Lie groups
26 Real functions
28 Measure and integration
30 Functions of a complex variable
31 Potential theory
32 Several complex variables and analytic spaces
33 Special functions
34 Ordinary differential equations
35 Partial differential equations
37 Dynamical systems and ergodic theory
39 Difference and functional equations
40 Sequences, series, summability
41 Approximations and expansions
42 Harmonic analysis on Euclidean spaces
43 Abstract harmonic analysis
44 Integral transforms, operational calculus
45 Integral equations
46 Functional analysis
47 Operator theory
49 Calculus of variations and optimal control; optimization
51 Geometry
52 Convex and discrete geometry
53 Differential geometry
54 General topology
55 Algebraic topology
57 Manifolds and cell complexes
58 Global analysis, analysis on manifolds
60 Probability theory and stochastic processes
62 Statistics
65 Numerical analysis
68 Computer science
70 Mechanics of particles and systems
74 Mechanics of deformable solids
76 Fluid mechanics
78 Optics, electromagnetic theory
80 Classical thermodynamics, heat transfer
81 Quantum theory
82 Statistical mechanics, structure of matter
83 Relativity and gravitational theory
85 Astronomy and astrophysics
86 Geophysics
90 Operations research, mathematical programming
91 Game theory, economics, finance
92 Biology and other natural sciences
93 Systems theory; control
94 Information and communication theory, circuits
97 Mathematics education
""".strip()

_SYSTEM_PROMPT = f"""\
You are a Mathematics Subject Classification (MSC 2020) expert.

Given a description of a mathematical paper or concept, assign the most relevant MSC codes.
Prefer 2-digit (top-level) codes unless the content is clearly specific enough for a \
5-character code.

MSC 2020 top-level areas:
{_MSC_AREAS}

Respond with ONLY a JSON array of MSC code strings, most relevant first.
Example: ["11", "14", "13A15"]

If the content is too vague to classify, return [].
"""


def classify_msc(
    text: str,
    provider: LLMProvider,
    top_k: int = 3,
) -> list[str]:
    """Assign MSC 2020 codes to a paper or concept.

    Args:
        text: Concatenation of title + abstract (paper) or name + description +
              latex_statement (concept).
        provider: LLM backend (haiku is sufficient for classification).
        top_k: Maximum number of MSC codes to return.

    Returns:
        List of MSC code strings (up to top_k), ordered by relevance. [] on failure.
    """
    user_message = (
        f"Classify the following mathematical content with up to {top_k} MSC codes:\n\n{text}"
    )

    try:
        response = provider.complete(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=256,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("LLM call failed in msc_classifier: %s", exc)
        return []

    raw = response.content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        codes = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("MSC classifier response was not valid JSON: %r", raw)
        return []

    if not isinstance(codes, list):
        return []

    # Validate and truncate
    validated = [str(c).strip() for c in codes if isinstance(c, (str, int)) and str(c).strip()]
    return validated[:top_k]
