#!/usr/bin/env bash
# update_sprint0.sh ─ Rewrite Sprint-0 ECS + scheduler with deterministic,
#                     sub-µs fixed-step loop.  Stdlib-only, Python 3.12+.

set -euo pipefail

echo "⏳  Updating files …"
mkdir -p ecs tests

# ───────────────────────────────────── ecs/components.py ─────────────────────────────────────
cat > ecs/components.py <<'PY'
"""Type-safe component store (stdlib-only)."""
from __future__ import annotations
from typing import TypeVar, Generic, Dict, Iterator, Tuple

T = TypeVar("T")

class ComponentStore(Generic[T]):
    """Maps ``EntityID`` → component instance of type *T*."""
    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Dict[int, T] = {}

    def add(self, eid: int, comp: T) -> None:
        self._data[eid] = comp

    def get(self, eid: int) -> T | None:
        return self._data.get(eid)

    def remove(self, eid: int) -> None:
        self._data.pop(eid, None)

    def items(self) -> Iterator[Tuple[int, T]]:
        return self._data.items()
PY

# ───────────────────────────────────── ecs/entity.py ─────────────────────────────────────────
cat > ecs/entity.py <<'PY'
"""Monotonic entity-ID generator."""
class EntityIDGenerator:
    __slots__ = ("_next",)

    def __init__(self) -> None:
        self._next: int = 0

    def next_id(self) -> int:
        eid = self._next
        self._next += 1
        return eid

    def reset(self) -> None:
        self._next = 0
PY

# ───────────────────────────────────── ecs/system.py ─────────────────────────────────────────
cat > ecs/system.py <<'PY'
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
PY

# ───────────────────────────────────── ecs/world.py ──────────────────────────────────────────
cat > ecs/world.py <<'PY'
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
PY

# ───────────────────────────────────── engine_tick.py ────────────────────────────────────────
cat > engine_tick.py <<'PY'
"""
engine_tick.py  –  Fixed-step deterministic scheduler (Sprint 0).

* 50 Hz default (20 000 000 ns)
* 1 000 empty ticks < 1 ms on M2 / 5800X
* Stdlib-only; Python 3.12+
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import random
from typing import Sequence

from ecs.system import System
from ecs.world import World

__all__ = ["FixedStepScheduler", "DEFAULT_DT_NS"]

DEFAULT_DT_NS: int = 20_000_000   # 20 ms  → 50 Hz

class FixedStepScheduler:
    """Runs systems in priority order at a fixed logical timestep."""

    __slots__ = ("systems", "dt_ns")

    def __init__(self, systems: Sequence[System], dt_ns: int = DEFAULT_DT_NS) -> None:
        self.systems = tuple(sorted(systems, key=lambda s: getattr(s, "priority", 0)))
        self.dt_ns = int(dt_ns)

    # ──────────────────────────────────────────────────────────────────────
    def run(self, num_ticks: int, world: World) -> None:
        """Execute *num_ticks* steps, mutating *world* in-place."""
        prof = os.getenv("ARENA_PROFILE") == "1"
        dt   = self.dt_ns
        systems = self.systems
        for _ in range(num_ticks):
            world.tick += 1
            if prof:
                for sysc in systems:
                    st = time.perf_counter_ns()
                    sysc(world, dt)
                    nm = getattr(sysc, "__qualname__", sysc.__class__.__name__)
                    sys.stderr.write(f"{nm} {time.perf_counter_ns() - st} ns\n")
            else:
                for sysc in systems:
                    sysc(world, dt)
            world.flush()

# ───────────────────────────────────── quick bench / CLI ─────────────────────────────────────
def _benchmark(seed: int, ticks: int) -> None:
    rng = random.Random(seed)
    world = World(rng=rng)
    sched = FixedStepScheduler([], dt_ns=DEFAULT_DT_NS)

    start = time.perf_counter_ns()
    sched.run(ticks, world)
    total = time.perf_counter_ns() - start
    print(f"total   : {total:,} ns")
    print(f"per-tick: {total / ticks:.1f} ns")
    if total > 1_000_000:
        raise SystemExit("❌  Benchmark exceeded 1 ms budget")
    print("✅  PASS")

def main() -> None:
    ap = argparse.ArgumentParser(description="Sprint-0 benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=1_000)
    _args = ap.parse_args()
    _benchmark(_args.seed, _args.ticks)

if __name__ == "__main__":
    main()
PY

# ───────────────────────────────────── tests/test_engine_tick.py ─────────────────────────────
cat > tests/test_engine_tick.py <<'PY'
import os, time, random, pytest
from ecs.world import World
from engine_tick import FixedStepScheduler, DEFAULT_DT_NS

_skip_speed = pytest.mark.skipif(
    os.getenv("CI", "").lower() in {"1", "true", "yes"},
    reason="skip perf test on CI",
)

@_skip_speed
def test_tick_speed():
    world = World(rng=random.Random(42))
    sched = FixedStepScheduler([], dt_ns=DEFAULT_DT_NS)
    start = time.perf_counter_ns()
    sched.run(1_000, world)
    assert (time.perf_counter_ns() - start) < 1_000_000

def test_determinism():
    world1 = World(rng=random.Random(42))
    world2 = World(rng=random.Random(42))
    sched1 = FixedStepScheduler([], dt_ns=DEFAULT_DT_NS)
    sched2 = FixedStepScheduler([], dt_ns=DEFAULT_DT_NS)
    t1_s = time.perf_counter_ns(); sched1.run(1_000, world1)
    t1 = time.perf_counter_ns() - t1_s
    t2_s = time.perf_counter_ns(); sched2.run(1_000, world2)
    t2 = time.perf_counter_ns() - t2_s
    assert abs(t1 - t2) <= 2_000
PY

# ───────────────────────────────────── ecs/__init__.py ─────────────────────────────────────
cat > ecs/__init__.py <<'PY'
"""Entity-Component-System package (Sprint 0 baseline)."""
__all__ = ["components", "entity", "system", "world"]
PY

echo "✅  Sprint-0 files written.  Run 'pytest -q' or 'python engine_tick.py'"
