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
