# movement_collision.py – Position, Velocity, Collision for “The Arena”
# Stdlib‑only • Python 3.12 • integrates with engine_tick.World & System
"""Physics micro‑package: basic locomotion + circle collision.

Key objects
-----------
* **Position2D**, **Velocity2D**, **CollisionRadius** – component data.
* **MovementSystem** – Euler save: `x += vx*dt`.
* **CollisionSystem** – spatial‑hash broad phase + push‑apart narrow phase + rim clamp.
* **orbit_velocity()** – helper for CCW orbit steering.
* **create_movement_collision_systems()** – returns the two above systems.

Run a quick demo:
>>> python ‑m movement_collision

Pytest section at bottom includes correctness + perf checks.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ecs.components import ComponentStore  # type: ignore
from ecs.system import System  # type: ignore
from ecs.world import World  # type: ignore
from engine_tick import DEFAULT_DT_NS, FixedStepScheduler  # type: ignore

__all__ = [
    "Position2D",
    "Velocity2D",
    "CollisionRadius",
    "orbit_velocity",
    "create_movement_collision_systems",
]

ARENA_RADIUS = 20.0  # metres
EPSILON_NUDGE = 1e-6  # deterministic tiny push for coincident centres

# ───────────────────────── Components ──────────────────────────────

@dataclass(slots=True)
class Position2D:
    x: float
    y: float


@dataclass(slots=True)
class Velocity2D:
    vx: float
    vy: float


@dataclass(slots=True)
class CollisionRadius:
    r: float  # metres


# ───────────────────── helper: store plumbing ─────────────────────

def _require_store(world: World, comp_type):
    for meth in ("require_store", "get_store"):
        fn = getattr(world, meth, None)
        if callable(fn):
            store = fn(comp_type)  # type: ignore[arg-type]
            if store is not None:
                return store

    stores: Dict[type, ComponentStore] = getattr(world, "_arena_stores", None)  # type: ignore[arg-type]
    if stores is None:
        stores = {}
        setattr(world, "_arena_stores", stores)
    return stores.setdefault(comp_type, ComponentStore())


# ───────────────────────── Spatial‑hash ───────────────────────────

class _SpatialHash:
    __slots__ = ("cell", "_grid")

    def __init__(self, cell_size: float) -> None:
        self.cell = cell_size
        self._grid: Dict[Tuple[int, int], List[int]] = {}

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        edge = self.cell
        return int(x // edge), int(y // edge)

    def clear(self) -> None:
        self._grid.clear()

    def insert(self, eid: int, x: float, y: float) -> None:
        self._grid.setdefault(self._key(x, y), []).append(eid)


# ───────────────────────── MovementSystem ─────────────────────────

class MovementSystem:
    """Euler‑integrates **Velocity2D** every fixed step."""

    __slots__ = ("_pos", "_vel", "_dt")
    priority = 10

    def __init__(self, pos: ComponentStore[Position2D], vel: ComponentStore[Velocity2D]):
        self._pos = pos
        self._vel = vel
        self._dt = DEFAULT_DT_NS * 1e-9

    def __call__(self, world: World, dt_ns: int) -> None:  # noqa: D401
        dt = self._dt
        for eid, p in self._pos.items():
            v = self._vel.get(eid)
            if v:
                p.x += v.vx * dt
                p.y += v.vy * dt


# ───────────────────────── CollisionSystem ────────────────────────

class CollisionSystem:
    """Spatial‑hash broad phase → circle‑circle resolve → rim clamp."""

    __slots__ = ("_pos", "_vel", "_rad", "_arena_r", "_edge2", "_grid")
    priority = 20

    def __init__(
        self,
        pos: ComponentStore[Position2D],
        vel: ComponentStore[Velocity2D],
        rad: ComponentStore[CollisionRadius],
        arena_radius: float = ARENA_RADIUS,
    ) -> None:
        self._pos = pos
        self._vel = vel
        self._rad = rad
        self._arena_r = arena_radius
        self._edge2 = arena_radius * arena_radius
        self._grid = _SpatialHash(1.0)

    def __call__(self, world: World, dt_ns: int) -> None:  # noqa: D401
        # 1 ⟶ cell size
        max_r = max((c.r for c in self._rad._data.values()), default=0.0)  # type: ignore[attr-defined]
        cell = max(0.01, 2.0 * max_r)
        if abs(cell - self._grid.cell) > 1e-6:
            self._grid = _SpatialHash(cell)

        # 2 ⟶ populate
        grid = self._grid
        grid.clear()
        for eid, p in self._pos.items():
            grid.insert(eid, p.x, p.y)

        # 3 ⟶ overlaps
        pos_get, rad_get = self._pos.get, self._rad.get
        for (ix, iy), bucket in grid._grid.items():
            _resolve_pairs(bucket, bucket, pos_get, rad_get)
            nb = grid._grid.get((ix + 1, iy))
            if nb:
                _resolve_pairs(bucket, nb, pos_get, rad_get)
            for jy in (iy + 1, iy - 1):
                nb = grid._grid.get((ix, jy))
                if nb:
                    _resolve_pairs(bucket, nb, pos_get, rad_get)
                nb_d = grid._grid.get((ix + 1, jy))
                if nb_d:
                    _resolve_pairs(bucket, nb_d, pos_get, rad_get)

        # 4 ⟶ rim clamp
        r_arena = self._arena_r
        for eid, p in self._pos.items():
            rc = rad_get(eid)
            if rc is None:
                continue
            r_eff = r_arena - rc.r
            d2 = p.x * p.x + p.y * p.y
            if d2 > r_eff * r_eff:
                dist = math.sqrt(d2)
                if dist == 0.0:
                    p.x = r_eff
                    p.y = 0.0
                else:
                    scale = r_eff / dist
                    p.x *= scale
                    p.y *= scale


# helper narrow‑phase ------------------------------------------------------

def _resolve_pairs(bucket_a, bucket_b, pos_get, rad_get) -> None:  # type: ignore[override]
    for i, ea in enumerate(bucket_a):
        pa = pos_get(ea)
        ra_c = rad_get(ea)
        if not pa or not ra_c:
            continue
        ra = ra_c.r
        for eb in bucket_b[i + 1 :] if bucket_a is bucket_b else bucket_b:  # noqa: E203
            pb = pos_get(eb)
            rb_c = rad_get(eb)
            if not pb or not rb_c:
                continue
            rb = rb_c.r
            dx = pa.x - pb.x
            dy = pa.y - pb.y
            dist2 = dx * dx + dy * dy
            sum_r = ra + rb
            if dist2 >= sum_r * sum_r:
                continue
            if dist2 == 0.0:
                pa.x += EPSILON_NUDGE
                pb.x -= EPSILON_NUDGE
                continue
            dist = math.sqrt(dist2)
            push = 0.5 * (sum_r - dist) / dist
            nx = dx * push
            ny = dy * push
            pa.x += nx
            pa.y += ny
            pb.x -= nx
            pb.y -= ny


# ───────────────────── Steering helper(s) ---------------------------------

def orbit_velocity(
    pos: Position2D,
    centre: Tuple[float, float],
    radius: float,
    angular_speed: float,
    gain: float = 2.0,
) -> Tuple[float, float]:
    cx, cy = centre
    dx = pos.x - cx
    dy = pos.y - cy
    dist = math.hypot(dx, dy) or 1e-9
    corr = (radius - dist) * gain
    tx, ty = -dy / dist, dx / dist
    v_tan = angular_speed * radius
    return tx * v_tan + dx / dist * corr, ty * v_tan + dy / dist * corr


# ───────────────────────── Steering helper(s) ──────────────────────────
def orbit_velocity(
    pos: Position2D,
    centre: Tuple[float, float],
    radius: float,
    angular_speed: float,
    *,
    gain: float = 2.0,
) -> Tuple[float, float]:
    """Return a CCW tangent velocity that maintains a circular orbit.

    Parameters
    ----------
    pos : Position2D
        Current position of the entity.
    centre : (float, float)
        Coordinates of the orbit centre.
    radius : float
        Desired orbit radius in metres.
    angular_speed : float
        Angular speed in **rad·s⁻¹** (positive = CCW).
    gain : float, default 2.0
        Proportional gain for radial correction. Raise for snappier orbit,
        lower for looser spiral.
    """
    cx, cy = centre
    dx, dy = pos.x - cx, pos.y - cy
    dist   = math.hypot(dx, dy) or 1e-9
    corr   = (radius - dist) * gain          # radial P-term
    # unit tangent (−dy, +dx)
    tx, ty = -dy / dist, dx / dist
    v_tan  = angular_speed * radius
    vx = tx * v_tan + dx / dist * corr
    vy = ty * v_tan + dy / dist * corr
    return vx, vy


# ───────────── Factory: attach stores + build systems ───────────────────
def create_movement_collision_systems(
    world: World,
    arena_radius: float = ARENA_RADIUS,
) -> Tuple[System, System]:
    """Return `(movement_sys, collision_sys)` wired to *world*."""
    pos_s = _require_store(world, Position2D)
    vel_s = _require_store(world, Velocity2D)
    rad_s = _require_store(world, CollisionRadius)
    return (
        MovementSystem(pos_s, vel_s),
        CollisionSystem(pos_s, vel_s, rad_s, arena_radius),
    )


# ─────────────────────────── demo / CLI ────────────────────────────────
def _spawn_orbiters(world: World) -> None:
    """Create two 0.5 m-radius fighters on a 2 m orbit."""
    pos_s = _require_store(world, Position2D)
    vel_s = _require_store(world, Velocity2D)
    rad_s = _require_store(world, CollisionRadius)

    pos_s.add(0, Position2D(-2.0, 0.0))
    pos_s.add(1, Position2D( 2.0, 0.0))
    vel_s.add(0, Velocity2D(0.0, 0.0))
    vel_s.add(1, Velocity2D(0.0, 0.0))
    rad_s.add(0, CollisionRadius(0.5))
    rad_s.add(1, CollisionRadius(0.5))


def _min_distance(world: World) -> float:
    pos_s = _require_store(world, Position2D)
    p0, p1 = pos_s.get(0), pos_s.get(1)
    return math.hypot(p0.x - p1.x, p0.y - p1.y) if p0 and p1 else math.inf


if __name__ == "__main__":
    import random

    rng   = random.Random(42)
    world = World(rng=rng)

    move_sys, col_sys = create_movement_collision_systems(world)
    sched = FixedStepScheduler([move_sys, col_sys], dt_ns=DEFAULT_DT_NS)

    _spawn_orbiters(world)
    pos_s = _require_store(world, Position2D)
    vel_s = _require_store(world, Velocity2D)

    TICKS  = 500                       # 10 s at 50 Hz
    start  = time.perf_counter_ns()

    for _ in range(TICKS):
        # steering update
        for eid, pos in pos_s.items():
            vel = vel_s.get(eid)
            vel.vx, vel.vy = orbit_velocity(
                pos, (0.0, 0.0), radius=2.0, angular_speed=math.tau / 5.0
            )
        sched.run(1, world)

    elapsed = time.perf_counter_ns() - start
    print(
        f"min distance = {_min_distance(world):.3f} m | "
        f"avg tick = {elapsed / TICKS:,.0f} ns"
    )


# ───────────────────────────── pytest tests ─────────────────────────────
import pytest  # noqa: E402  (late import so CLI path stays lean)


@pytest.mark.parametrize("ticks", [500])
def test_orbit_no_overlap(ticks: int) -> None:
    """Two 0.5 m fighters orbit `ticks` steps without overlap."""
    import random

    world = World(rng=random.Random(123))
    move_sys, col_sys = create_movement_collision_systems(world)
    sched = FixedStepScheduler([move_sys, col_sys], dt_ns=DEFAULT_DT_NS)

    _spawn_orbiters(world)
    pos_s = _require_store(world, Position2D)
    vel_s = _require_store(world, Velocity2D)

    for _ in range(ticks):
        for eid, pos in pos_s.items():
            vel = vel_s.get(eid)
            vel.vx, vel.vy = orbit_velocity(
                pos, (0.0, 0.0), 2.0, math.tau / 5.0
            )
        sched.run(1, world)
        assert _min_distance(world) >= 1.0  # 2 × 0.5 m radii


@pytest.mark.skipif(
    os.getenv("ARENA_FAST_MACHINE") != "1",
    reason="Performance test skipped on slow CI",
)
def test_perf_500_entities() -> None:
    """500 entities × 1 000 ticks must stay ≤ 2 ms on target CPUs."""
    import random

    world = World(rng=random.Random(999))
    move_sys, col_sys = create_movement_collision_systems(world)
    sched = FixedStepScheduler([move_sys, col_sys], dt_ns=DEFAULT_DT_NS)

    pos_s = _require_store(world, Position2D)
    vel_s = _require_store(world, Velocity2D)
    rad_s = _require_store(world, CollisionRadius)

    # uniform ring spawn
    for eid in range(500):
        theta = eid * (2 * math.pi / 500)
        pos_s.add(eid, Position2D(math.cos(theta) * 10.0, math.sin(theta) * 10.0))
        vel_s.add(eid, Velocity2D(0.0, 0.0))
        rad_s.add(eid, CollisionRadius(0.2))

    world.flush()                       # baseline consistency
    sched.run(10, world)                # warm-up

    t0 = time.perf_counter_ns()
    sched.run(1_000, world)
    elapsed_ns = time.perf_counter_ns() - t0
    assert elapsed_ns <= 2_000_000, (
        f"movement+collision took {elapsed_ns/1e6:.2f} ms (> 2 ms budget)"
    )
