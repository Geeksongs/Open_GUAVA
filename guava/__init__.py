"""GUAVA harness reproduction inside CaP-X.

A faithful re-implementation of the harness from Liu et al. (2026),
"Guava: An Effective and Universal Harness for Embodied Manipulation",
built on top of the CaP-X low-level stack.

Three GUAVA design principles, all reproduced here:
  1. iterative perception-reasoning-action (ReAct) loop  -> ``guava.agent``
  2. semantic action abstractions (9 tools, Appendix B)   -> ``guava.tools``
  3. multimodal observations (image + symbolic state)     -> ``guava.agent``

The exact system prompt and tool spec live in ``guava.prompts``.
"""

from guava.agent import EpisodeResult, GuavaAgent, StepRecord
from guava.tools import GuavaToolError, GuavaTools

__all__ = [
    "GuavaAgent",
    "GuavaTools",
    "GuavaToolError",
    "EpisodeResult",
    "StepRecord",
]
