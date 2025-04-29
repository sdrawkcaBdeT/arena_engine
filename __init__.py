from __future__ import annotations
import random

__all__ = ["__version__", "init_rng", "ArenaEngineError"]
__version__ = "0.1.0.dev0"


class ArenaEngineError(Exception):
    """Public umbrella exception for engine misuse."""


def init_rng(seed: int | None = None) -> random.Random:
    """Return the single RNG used by the simulation."""
    return random.Random(seed)
