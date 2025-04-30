from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List

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
    
    # >>>>>>>>>>>>>>>  NEW stores for the health module  <<<<<<<<<<<<<<<<<
    limbs: Dict[int, list] = field(default_factory=dict)
    organs: Dict[int, list | None] = field(default_factory=dict)
    vitals: Dict[int, "health.Vitals"] = field(default_factory=dict)

    # simple ordered system list
    _systems: List[Callable[["World", int], None]] = field(default_factory=list)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Sprint-0: just wipe transient queues; more later
    def flush(self) -> None:
        self.events.clear()
        self.deferred.clear()
        
    # -------------- NEW helper methods -----------------
    def add_system(self, fn: Callable[["World", int], None]) -> None:
        self._systems.append(fn)

    def post_event(self, evt: Any) -> None:
        self.events.append(evt)

    def consume_events(self, cls: type) -> list:
        matched, rest = [], []
        for e in self.events:
            (matched if isinstance(e, cls) else rest).append(e)
        self.events = deque(rest)
        return matched

    # optional: single-step loop (if you want it)
    def step(self, dt_ns: int = 20_000_000) -> None:
        self.tick += 1
        for sysfn in self._systems:
            sysfn(self, dt_ns)
        self.flush()
    # ---------------------------------------------------
