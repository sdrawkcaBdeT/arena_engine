# =============================================================
#  tests/demo_damage.py – showcase + pytest for health systems
# =============================================================
"""
Executable demo + unit tests for the *health* module.

Run directly:
    python tests/demo_damage.py

The file adds the repo root to ``sys.path`` so importing ``health`` works
no matter where the test file lives.
"""
from __future__ import annotations

import os
import random
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import List, Dict

# --- locate repo root ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import health  # noqa: E402

# ---------------------------------------------------------------------------
#  Stub ECS World good enough for this demo / test
# ---------------------------------------------------------------------------


class StubWorld:
    """Tiny ECS-ish container that matches health.Damage/Bleed expectations."""

    def __init__(self, rng: random.Random):
        self.tick: int = 0
        self._systems: List = []
        self._evq: List = []
        self.rng = rng

        # component stores – dict[entity_id] = component/structure
        self.limbs: Dict[int, List[health.Limb]] = {}
        self.organs: Dict[int, List | None] = {}
        self.vitals: Dict[int, health.Vitals] = {}

    # ---- ECS helpers ------------------------------------------------------

    def add_system(self, fn):
        self._systems.append(fn)

    def post_event(self, evt):
        self._evq.append(evt)

    def consume_events(self, cls):
        matched, rest = [], []
        for e in self._evq:
            (matched if isinstance(e, cls) else rest).append(e)
        self._evq[:] = rest
        return matched

    # ---- driver -----------------------------------------------------------

    def run(self, ticks: int, dt_ns: int = 20_000_000):
        for _ in range(ticks):
            self.tick += 1
            for sysfn in self._systems:
                sysfn(self, dt_ns)


# ---------------------------------------------------------------------------
#  ImpactEvent dataclass for demo
# ---------------------------------------------------------------------------


class _ImpactEvent:  # matches health.ImpactEvent protocol
    __slots__ = (
        "tick",
        "attacker_id",
        "defender_id",
        "relative_speed",
        "weapon_mass",
        "edge_type",
        "limb_idx",
    )

    def __init__(
        self,
        tick: int,
        attacker_id: int,
        defender_id: int,
        relative_speed: float,
        weapon_mass: float,
        edge_type: str,
        limb_idx: int,
    ):
        self.tick = tick
        self.attacker_id = attacker_id
        self.defender_id = defender_id
        self.relative_speed = relative_speed
        self.weapon_mass = weapon_mass
        self.edge_type = edge_type
        self.limb_idx = limb_idx


# ---------------------------------------------------------------------------
#  Demo harness
# ---------------------------------------------------------------------------


def _spawn_dummy(world: StubWorld, ent_id: int):
    limbs, organs = health.build_default_anatomy()
    world.limbs[ent_id] = limbs
    world.organs[ent_id] = organs
    world.vitals[ent_id] = health.Vitals()


def demo() -> None:
    rng = random.Random(42)
    world = StubWorld(rng)
    health.register_health_systems(world)

    _spawn_dummy(world, 1)
    _spawn_dummy(world, 2)

    limb_count = len(health.REGIONS)

    # 50 % chance every 20 ticks
    while True:
        if world.tick % 20 == 0:
            if rng.random() < 0.5:
                attacker, defender = (1, 2) if rng.random() < 0.5 else (2, 1)
                world.post_event(
                    _ImpactEvent(
                        world.tick,
                        attacker,
                        defender,
                        relative_speed=7.0,
                        weapon_mass=3.0,
                        edge_type=rng.choice(["blunt", "slash", "pierce"]),
                        limb_idx=rng.randrange(limb_count),
                    )
                )

        tic = time.perf_counter_ns()
        world.run(1)
        toc = time.perf_counter_ns()

        # death?
        for evt in world.consume_events(health.DeathEvent):
            print(f"[tick {evt.tick}] entity {evt.entity_id} died → {evt.cause}")
            return

        # safety stop
        if world.tick >= 10_000:
            print("no deaths in 10 000 ticks")
            return


# ---------------------------------------------------------------------------
#  Pytest section
# ---------------------------------------------------------------------------

def _death_trace(seed: int = 42):
    rng = random.Random(seed)
    world = StubWorld(rng)
    health.register_health_systems(world)
    _spawn_dummy(world, 1)
    _spawn_dummy(world, 2)
    limb_count = len(health.REGIONS)

    hashes = sha256()
    while world.tick < 10_000:
        if world.tick % 20 == 0 and rng.random() < 0.5:
            attacker, defender = (1, 2) if rng.random() < 0.5 else (2, 1)
            world.post_event(
                _ImpactEvent(
                    world.tick,
                    attacker,
                    defender,
                    7.0,
                    3.0,
                    rng.choice(["blunt", "slash", "pierce"]),
                    rng.randrange(limb_count),
                )
            )
        world.run(1)
        for evt in world.consume_events(health.DeathEvent):
            hashes.update(str(evt.tick).encode())
            hashes.update(evt.cause.encode())
            return hashes.hexdigest(), evt.tick, evt.cause
    return hashes.hexdigest(), None, None


def test_deterministic_death():
    h1, t1, c1 = _death_trace(42)
    h2, t2, c2 = _death_trace(42)
    assert h1 == h2
    assert t1 == t2
    assert c1 == c2


def test_perf_100_entities():
    import pytest

    if os.environ.get("CI") == "true":
        pytest.skip("skip perf on constrained CI runners")
    rng = random.Random(0)
    world = StubWorld(rng)
    health.register_health_systems(world)
    for i in range(100):
        _spawn_dummy(world, i)
    start = time.perf_counter_ns()
    world.run(1_000)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    assert elapsed_ms <= 4.0, f"perf {elapsed_ms:.2f} ms > 4 ms"


# ---------------------------------------------------------------------------
#  CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo()
