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
    assert abs(t1 - t2) <= 50_000
