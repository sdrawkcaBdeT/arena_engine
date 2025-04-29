#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bootstrap_inplace.sh – Sprint-0 scaffold (writes into the current directory)
#
#   $ cd arena_engine          # <- you’re already here
#   $ bash bootstrap_inplace.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PKG="arena_engine"                    # import name
HERE="$(basename "$PWD")"

# Decide where to write the package files
if [[ "$HERE" == "$PKG" ]]; then
    PKG_DIR="."                       # we’re already inside arena_engine/
else
    PKG_DIR="$PKG"                    # fall back to creating the folder
    mkdir -p "$PKG_DIR"
fi

echo "▶ Package root: $PKG_DIR"

# 1) Sub-directories --------------------------------------------------------
mkdir -p "$PKG_DIR/ecs"

# 2) pyproject.toml ---------------------------------------------------------
cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=65"]
build-backend = "setuptools.build_meta"

[project]
name = "arena-engine-core"
version = "0.1.0.dev0"
description = "Sprint-0 deterministic tick loop & ECS plumbing for THE ARENA"
authors = [{ name = "Your Name" }]
requires-python = ">=3.12"
EOF

# 3) __init__.py ------------------------------------------------------------
cat > "$PKG_DIR/__init__.py" <<'EOF'
from __future__ import annotations
import random

__all__ = ["__version__", "init_rng", "ArenaEngineError"]
__version__ = "0.1.0.dev0"


class ArenaEngineError(Exception):
    """Public umbrella exception for engine misuse."""


def init_rng(seed: int | None = None) -> random.Random:
    """Return the single RNG used by the simulation."""
    return random.Random(seed)
EOF

# 4) engine_tick.py ---------------------------------------------------------
cat > "$PKG_DIR/engine_tick.py" <<'EOF'
from __future__ import annotations

import os, time, argparse, random, sys, pathlib
from typing import Sequence, Callable

from .ecs.system import System
from .ecs.world import World
from . import init_rng, ArenaEngineError


class FixedStepScheduler:
    """Runs systems at a fixed logical timestep."""

    def __init__(
        self,
        systems: Sequence[Callable],
        dt_ns: int,
        rng: random.Random,
    ) -> None:
        self.systems = tuple(sorted(systems, key=lambda s: s.priority))
        self.dt_ns = dt_ns
        self.rng = rng

    def run(self, num_ticks: int, world: World) -> None:
        profile = os.getenv("ARENA_PROFILE") == "1"
        for tick in range(num_ticks):
            for system in self.systems:
                if profile:
                    start = time.perf_counter_ns()
                    system(world, self.rng, tick, self.dt_ns)
                    sys.stderr.write(
                        f"{system.__class__.__name__} {time.perf_counter_ns()-start} ns\n"
                    )
                else:
                    system(world, self.rng, tick, self.dt_ns)
            world.flush()  # placeholder


def _benchmark(seed: int, ticks: int) -> None:
    rng = init_rng(seed)
    world = World(rng)
    scheduler = FixedStepScheduler(systems=(), dt_ns=20_000_000, rng=rng)

    start = time.perf_counter_ns()
    scheduler.run(ticks, world)
    total = time.perf_counter_ns() - start

    print(f"total   : {total:,} ns")
    print(f"per-tick: {total/ticks:.1f} ns")
    if total > 1_000_000:
        raise ArenaEngineError("Benchmark exceeded 1 ms budget")
    print("PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sprint-0 benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=1_000)
    args = ap.parse_args()
    _benchmark(args.seed, args.ticks)


if __name__ == "__main__":
    main()
EOF

# 5) ecs/__init__.py --------------------------------------------------------
echo '"""ECS sub-package – minimal for Sprint-0."""' > "$PKG_DIR/ecs/__init__.py"

# 6) ecs/entity.py ----------------------------------------------------------
cat > "$PKG_DIR/ecs/entity.py" <<'EOF'
class EntityIDGenerator:
    __slots__ = ("_next",)

    def __init__(self) -> None:
        self._next = 0

    def next_id(self) -> int:
        eid = self._next
        self._next += 1
        return eid

    def reset(self) -> None:
        self._next = 0
EOF

# 7) ecs/components.py ------------------------------------------------------
cat > "$PKG_DIR/ecs/components.py" <<'EOF'
from __future__ import annotations
from typing import TypeVar, Dict, Generic

T = TypeVar("T")


class ComponentStore(Generic[T]):
    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Dict[int, T] = {}

    def add(self, eid: int, comp: T) -> None:
        self._data[eid] = comp

    def get(self, eid: int) -> T | None:
        return self._data.get(eid)

    def remove(self, eid: int) -> None:
        self._data.pop(eid, None)

    def items(self):
        return self._data.items()
EOF

# 8) ecs/system.py ----------------------------------------------------------
cat > "$PKG_DIR/ecs/system.py" <<'EOF'
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
EOF

# 9) ecs/world.py -----------------------------------------------------------
cat > "$PKG_DIR/ecs/world.py" <<'EOF'
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
EOF

echo "✔ Files written."

# 10) Run benchmark from *parent* dir so the package is importable ----------
echo "▶ Running benchmark ..."
PYTHONPATH="$(pwd)/.." python -m arena_engine.engine_tick
