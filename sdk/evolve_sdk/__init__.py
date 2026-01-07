"""
Agentic Evolve SDK - Agent SDK-powered evolutionary algorithm discovery.

This package provides fine-grained control over evolution using the Claude Agent SDK,
with hierarchical agents, clean context per generation, and validation hooks.

Usage:
    # As CLI
    python -m evolve_sdk "shortest Python sort" --mode=size

    # As library
    from evolve_sdk import EvolutionRunner
    runner = EvolutionRunner(problem="shortest sort", mode="size")
    await runner.run()
"""

__version__ = "0.1.0"

from .runner import EvolutionRunner
from .config import EvolutionConfig

__all__ = ["EvolutionRunner", "EvolutionConfig", "__version__"]
