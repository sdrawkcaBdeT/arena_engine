"""
attack.py – melee swing & hit-detection micro-package for *The Arena*

Stdlib-only • Python 3.12  • integrates with engine_tick.FixedStepScheduler +
ecs.world.World (deterministic RNG).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Callable, List, Literal, Tuple

# ── components / helpers from movement_collision ───────────────────────────
from movement_collision import (  # type: ignore
    Position2D,
    CollisionRadius,
    _require_store,
)

Vec2 = Tuple[float, float]

ANG120 = math.radians(120.0)       # constant → saves per-tick math

##############################################################################
# ───────────────────────────────  Schema  ──────────────────────────────────
##############################################################################


class Phase(StrEnum):
    IDLE = auto()
    WINDUP = auto()
    ACTIVE = auto()
    RECOVERY = auto()


@dataclass(slots=True)
class AttackProfile:
    kind: Literal["swing", "thrust"]
    windup_ticks: int
    active_ticks: int
    recovery_ticks: int
    path_fn: Callable[[float, Vec2, Vec2], Vec2]  # kept for future use


@dataclass(slots=True)
class HitSegment:
    offset_m: float
    radius_m: float
    tag: str


@dataclass(slots=True)
class Weapon:
    profiles: List[AttackProfile]
    hit_segments: List[HitSegment]
    mass_kg: float = 1.0
    edge_type: str = "blunt"
    max_offset: float = 0.0          # populated in __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_offset",
                           max(seg.offset_m for seg in self.hit_segments))


@dataclass(slots=True)
class AttackState:
    phase: Phase = Phase.IDLE
    ticks_left: int = 0
    target_id: int | None = None
    profile_idx: int = 0
    swing_id: int = 0
    has_hit: bool = False            # true after first contact


@dataclass(slots=True)
class ImpactEvent:
    tick: int
    attacker_id: int
    defender_id: int
    contact_xy: Vec2
    relative_speed: float
    weapon_mass: float
    edge_type: str
    contact_part: str


@dataclass(slots=True)
class Opponent:
    opponent_id: int


##############################################################################
# ───────────────────────────  World-patch helpers  ─────────────────────────
##############################################################################


def _ensure_event_api(world) -> None:
    if hasattr(world, "post_event"):
        return

    world._event_q: List[object] = []

    def _post(evt):             world._event_q.append(evt)
    def _consume():             q, world._event_q = world._event_q, []; return q

    world.post_event = _post            # type: ignore[attr-defined]
    world.consume_events = _consume     # type: ignore[attr-defined]


##############################################################################
# ─────────────────────────────  Core system  ───────────────────────────────
##############################################################################


class AttackSystem:
    """Fixed-step melee FSM + circle-circle contact check."""

    DEFAULT_AUTOSWING = 60  # ticks between auto-intents (≈1 s at 50 Hz)

    def __call__(self, world, dt_ns: int) -> None:  # noqa: N802
        _ensure_event_api(world)

        pos_s = _require_store(world, Position2D)
        col_s = _require_store(world, CollisionRadius)
        wep_s = _require_store(world, Weapon)
        st_s  = _require_store(world, AttackState)
        opp_s = _require_store(world, Opponent)

        dt_s  = dt_ns * 1e-9
        tick  = world.tick

        # bootstrap states
        for eid in wep_s._data:                     # type: ignore[attr-defined]
            st_s._data.setdefault(eid, AttackState())   # type: ignore[attr-defined]

        for eid, state in st_s.items():
            weapon = wep_s.get(eid)
            if weapon is None:
                continue

            opp_id = opp_s.get(eid).opponent_id     # type: ignore[union-attr]
            phase  = state.phase                    # local ref → fewer attr hits

            # ── auto intent
            if phase is Phase.IDLE and tick % self.DEFAULT_AUTOSWING == 0:
                state.profile_idx  = 0
                prof               = weapon.profiles[0]
                state.phase        = Phase.WINDUP
                state.ticks_left   = prof.windup_ticks
                state.target_id    = opp_id
                state.has_hit      = False

            # ── countdown & phase changes
            if state.phase is not Phase.IDLE:
                state.ticks_left -= 1
                if state.ticks_left <= 0:
                    prof = weapon.profiles[state.profile_idx]
                    if state.phase is Phase.WINDUP:
                        state.phase, state.ticks_left = Phase.ACTIVE, prof.active_ticks
                    elif state.phase is Phase.ACTIVE:
                        state.phase, state.ticks_left = Phase.RECOVERY, prof.recovery_ticks
                    else:  # RECOVERY
                        state.phase, state.swing_id = Phase.IDLE, state.swing_id + 1

            # ── early-out
            if state.phase is not Phase.ACTIVE or state.has_hit:
                continue

            prof  = weapon.profiles[state.profile_idx]
            ap    = pos_s.get(eid)
            dp    = pos_s.get(state.target_id)      # type: ignore[arg-type]
            dc    = col_s.get(state.target_id)      # type: ignore[arg-type]
            if ap is None or dp is None or dc is None:
                continue

            # direction & tip progress
            dx, dy = dp.x - ap.x, dp.y - ap.y
            dist   = math.hypot(dx, dy) or 1e-9
            Hx, Hy = dx / dist, dy / dist
            t      = 1.0 - (state.ticks_left / prof.active_ticks)

            if prof.kind == "swing":
                Rx, Ry = -Hy, Hx
                theta  = -ANG120 / 2 + ANG120 * t
                ux = math.cos(theta) * Hx + math.sin(theta) * Rx
                uy = math.cos(theta) * Hy + math.sin(theta) * Ry
            else:
                ux, uy = Hx, Hy

            # linear speed (m s⁻¹)
            reach = weapon.max_offset
            v_lin = (
                reach * ANG120 / prof.active_ticks    # swing
                if prof.kind == "swing"
                else reach / (prof.active_ticks * dt_s)  # thrust
            )

            factor_const = t if prof.kind == "thrust" else 1.0
            for seg in weapon.hit_segments:
                cx = ap.x + ux * seg.offset_m * factor_const
                cy = ap.y + uy * seg.offset_m * factor_const
                ddx, ddy = dp.x - cx, dp.y - cy
                if ddx * ddx + ddy * ddy <= (seg.radius_m + dc.r) ** 2:
                    world.post_event(
                        ImpactEvent(
                            tick=tick,
                            attacker_id=eid,
                            defender_id=state.target_id,           # type: ignore[arg-type]
                            contact_xy=(cx, cy),
                            relative_speed=v_lin,
                            weapon_mass=weapon.mass_kg,
                            edge_type=weapon.edge_type,
                            contact_part=seg.tag,
                        )
                    )
                    state.has_hit = True
                    break


##############################################################################
# ─────────────────────────  Minimal demo  ─────────────────────────────────
##############################################################################


def _build_maul() -> Weapon:
    swing  = AttackProfile("swing", 10, 5, 10, lambda *_: (0, 0))   # path_fn placeholders
    thrust = AttackProfile("thrust", 6, 4, 8,  lambda *_: (0, 0))
    return Weapon(
        profiles=[swing, thrust],
        hit_segments=[HitSegment(0.6, 0.05, "shaft"),
                      HitSegment(1.4, 0.20, "head")],
        mass_kg=5.0,
    )


if __name__ == "__main__":  # pragma: no cover
    from engine_tick import FixedStepScheduler, DEFAULT_DT_NS, World  # type: ignore

    world = World(rng=random.Random(42))
    sched = FixedStepScheduler([AttackSystem()], DEFAULT_DT_NS)

    weapon = _build_maul()
    a, b = world.entities.next_id(), world.entities.next_id()
    _require_store(world, Position2D).add(a, Position2D(-0.75, 0.0))
    _require_store(world, Position2D).add(b, Position2D( 0.75, 0.0))
    _require_store(world, CollisionRadius).add(a, CollisionRadius(0.25))
    _require_store(world, CollisionRadius).add(b, CollisionRadius(0.25))
    _require_store(world, Weapon).add(a, weapon); _require_store(world, Weapon).add(b, weapon)
    _require_store(world, Opponent).add(a, Opponent(b)); _require_store(world, Opponent).add(b, Opponent(a))

    start = time.perf_counter_ns()
    sched.run(1_000, world)
    dur   = time.perf_counter_ns() - start
    print("Impact ticks:", [e.tick for e in world.consume_events()])
    print(f"avg tick time: {dur/1_000:.0f} ns")


##############################################################################
# ───────────────────────────────  Tests  ──────────────────────────────────
##############################################################################
try:
    import pytest  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pytest = None  # type: ignore


@pytest.mark.skipif(pytest is None, reason="pytest unavailable")  # type: ignore
def test_two_dummies_register_hits():  # type: ignore
    import random
    from engine_tick import World, FixedStepScheduler, DEFAULT_DT_NS  # type: ignore
    from movement_collision import create_movement_collision_systems  # type: ignore

    world = World(rng=random.Random(2))
    move, col = create_movement_collision_systems(world)
    atk      = AttackSystem()
    sched    = FixedStepScheduler([move, col, atk], DEFAULT_DT_NS)

    weapon = _build_maul()
    a, b = world.entities.next_id(), world.entities.next_id()
    _require_store(world, Position2D).add(a, Position2D(-0.75, 0.0))
    _require_store(world, Position2D).add(b, Position2D( 0.75, 0.0))
    _require_store(world, CollisionRadius).add(a, CollisionRadius(0.25))
    _require_store(world, CollisionRadius).add(b, CollisionRadius(0.25))
    _require_store(world, Weapon).add(a, weapon); _require_store(world, Weapon).add(b, weapon)
    _require_store(world, Opponent).add(a, Opponent(b)); _require_store(world, Opponent).add(b, Opponent(a))

    sched.run(1_000, world)
    assert any(isinstance(e, ImpactEvent) for e in world.consume_events())


@pytest.mark.skipif(
    __import__("os").environ.get("ARENA_FAST_MACHINE") != "1",
    reason="perf test runs only when ARENA_FAST_MACHINE=1",
)  # type: ignore
def test_perf_200_entities():  # type: ignore
    import random
    from engine_tick import World, FixedStepScheduler, DEFAULT_DT_NS  # type: ignore
    from movement_collision import create_movement_collision_systems  # type: ignore

    world = World(rng=random.Random(99))
    move, col = create_movement_collision_systems(world)
    atk      = AttackSystem()
    sched    = FixedStepScheduler([move, col, atk], DEFAULT_DT_NS)

    weapon = _build_maul()
    for i in range(0, 200, 2):
        a, b = world.entities.next_id(), world.entities.next_id()
        _require_store(world, Position2D).add(a, Position2D(float(i), 0.0))
        _require_store(world, Position2D).add(b, Position2D(float(i) + 0.75, 0.0))
        _require_store(world, CollisionRadius).add(a, CollisionRadius(0.25))
        _require_store(world, CollisionRadius).add(b, CollisionRadius(0.25))
        _require_store(world, Weapon).add(a, weapon); _require_store(world, Weapon).add(b, weapon)
        _require_store(world, Opponent).add(a, Opponent(b)); _require_store(world, Opponent).add(b, Opponent(a))

    start = time.perf_counter_ns()
    sched.run(1_000, world)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    assert elapsed_ms <= 3.0, f"took {elapsed_ms:.2f} ms (>3 ms budget)"
