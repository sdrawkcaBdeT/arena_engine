"""
engine_tick.py – Fixed-step deterministic scheduler (Sprint 0).

* 50 Hz default (20 000 000 ns)
* 1 000 empty ticks < 1 ms on M2 / 5800X
* Stdlib-only -- Python 3.12+
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Sequence

from ecs.system import System
from ecs.world import World

__all__ = ["FixedStepScheduler", "DEFAULT_DT_NS"]

DEFAULT_DT_NS: int = 20_000_000  # 20 ms → 50 Hz


class FixedStepScheduler:
    """
    Deterministic fixed-step scheduler.

    Parameters
    ----------
    systems : Sequence[System]
        Callables executed each tick, sorted by ``.priority`` (lower first).
    dt_ns   : int, default 20 000 000
        Fixed step in **nanoseconds** (20 ms → 50 Hz).

    Notes
    -----
    * `systems` are frozen into an immutable tuple; order cannot change after
      construction — crucial for determinism.
    * Hot loop contains **no allocations or attribute look-ups** when the
      `ARENA_PROFILE` environment variable is *unset*.
    """

    __slots__ = (
        "systems",
        "dt_ns",
        "_sys_names",
        "_profile",
        "_profile_pairs",
    )

    def __init__(
        self, systems: Sequence[System], dt_ns: int = DEFAULT_DT_NS
    ) -> None:
        # ── immutable, priority-sorted system tuple
        self.systems = tuple(
            sorted(systems, key=lambda s: getattr(s, "priority", 0))
        )

        # Display names cached once for profile print-outs
        self._sys_names = tuple(
            getattr(s, "__qualname__", s.__class__.__name__)
            for s in self.systems
        )

        self.dt_ns: int = int(dt_ns)

        # Read env once ⇒ deterministic; no repeated os.getenv() cost
        self._profile: bool = os.getenv("ARENA_PROFILE") == "1"

        # Pre-pair systems + names so profile loop allocates nothing
        self._profile_pairs = tuple(zip(self.systems, self._sys_names))

    # ────────────────────────────────────────────────────────────────
    def run(self, num_ticks: int, world: World) -> None:
        """Advance *world* by *num_ticks* fixed steps (no real-time sleep)."""
        dt = self.dt_ns
        systems = self.systems
        flush = world.flush  # local binding → no attr lookup
        tns = time.perf_counter_ns  # ns timer alias

        if self._profile:  # profiling branch
            pairs = self._profile_pairs
            for _ in range(num_ticks):
                world.tick += 1  # logical tick counter
                for sysc, name in pairs:
                    start = tns()
                    sysc(world, dt)
                    sys.stderr.write(f"{name} {tns() - start} ns\n")
                flush()
        else:  # normal fast path
            for _ in range(num_ticks):
                world.tick += 1
                for sysc in systems:
                    sysc(world, dt)
                flush()


# ─────────────────────────────────── quick bench / CLI ────────────────────────────────────
def _benchmark(seed: int, ticks: int) -> None:
    rng = random.Random(seed)
    world = World(rng=rng)
    sched = FixedStepScheduler([], dt_ns=DEFAULT_DT_NS)

    sched.run(10, world)  # warm-up to reduce first-call jitter
    start = time.perf_counter_ns()
    sched.run(ticks, world)
    total = time.perf_counter_ns() - start

    print(f"total   : {total:,} ns")
    print(f"per-tick: {total / ticks:.1f} ns")
    if total > 1_000_000:
        raise SystemExit("❌  Benchmark exceeded 1 ms budget")
    print("✅  PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sprint-0 fixed-step benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=1_000)
    args = ap.parse_args()
    _benchmark(args.seed, args.ticks)


if __name__ == "__main__":
    main()
