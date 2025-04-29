from __future__ import annotations
import bisect
from typing import Protocol, List, Callable, runtime_checkable


@runtime_checkable
class System(Protocol):
    priority: int = 0

    def __call__(self, world, rng, tick: int, dt_ns: int): ...


class SystemRegistry:
    __slots__ = ("_systems",)

    def __init__(self) -> None:
        self._systems: List[Callable] = []

    def register(self, system: System) -> None:
        keys = [s.priority for s in self._systems]
        pos = bisect.bisect(keys, system.priority)
        self._systems.insert(pos, system)

    @property
    def systems(self) -> List[Callable]:
        return self._systems.copy()


registry = SystemRegistry()
