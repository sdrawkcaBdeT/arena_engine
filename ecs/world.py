from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any

from .entity import EntityIDGenerator

@dataclass
class World:
    """Shared simulation state object handed to every System."""
    rng: random.Random
    tick: int = 0
    entities: EntityIDGenerator = field(default_factory=EntityIDGenerator)
    components: Dict[type, Any] = field(default_factory=dict)  # type -> ComponentStore
    events: deque = field(default_factory=deque)               # transient per-tick queues
    deferred: list[Any] = field(default_factory=list)

    # Sprint-0: just wipe transient queues; more later
    def flush(self) -> None:
        self.events.clear()
        self.deferred.clear()
