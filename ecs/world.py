from __future__ import annotations
import copy, random
from collections import deque
from dataclasses import dataclass, field

from .entity import EntityIDGenerator


@dataclass
class World:
    rng: random.Random
    entities: EntityIDGenerator = field(default_factory=EntityIDGenerator)
    components: dict[str, object] = field(default_factory=dict)
    events: deque = field(default_factory=deque)
    deferred: list[tuple[str, tuple, dict]] = field(default_factory=list)

    def copy(self):
        return copy.deepcopy(self)

    def flush(self):
        self.events.clear()
        self.deferred.clear()
