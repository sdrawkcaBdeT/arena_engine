from __future__ import annotations
import bisect
from typing import Protocol, runtime_checkable, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from .world import World

@runtime_checkable
class System(Protocol):
    """Callable chunk of game logic executed each fixed tick."""
    priority: int  # default 0 → executes earlier when lower

    def __call__(self, world: "World", dt_ns: int) -> None: ...

class SystemRegistry:
    """Keeps systems sorted by ``priority`` for deterministic iteration."""
    __slots__ = ("_systems",)

    def __init__(self) -> None:
        self._systems: List[Callable[["World", int], None]] = []

    def register(self, system: System) -> None:
        pr = getattr(system, "priority", 0)
        idx = bisect.bisect([getattr(s, "priority", 0) for s in self._systems], pr)
        self._systems.insert(idx, system)

    @property
    def systems(self) -> tuple[Callable[["World", int], None], ...]:
        """Immutable tuple prevents runtime re-ordering."""
        return tuple(self._systems)

# Global registry singleton (optional convenience)
registry = SystemRegistry()
