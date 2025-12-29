#!/usr/bin/env python3
from deap import gp

from llmgp import EasyRunResult, quick_start


class DummyState(dict):
    pass


def feat_x(state: DummyState) -> float:
    return float(state.get("x", 0.0))


def feat_y(state: DummyState) -> float:
    return float(state.get("y", 0.0))


def cost_fn(pf):
    samples = [
        (DummyState(x=1, y=2), 0.0),
        (DummyState(x=3, y=4), 1.0),
    ]
    return sum((pf(s) - t) ** 2 for s, t in samples) / len(samples)


def test_quick_start_smoke():
    res: EasyRunResult = quick_start(
        prompt="regress a simple function",
        cost_fn=cost_fn,
        feature_fns=[feat_x, feat_y],
        state_type=DummyState,
        pop_size=4,
        ngen=1,
        n_threads=1,
        seed=0,
    )
    assert res.hof  # Hall of fame not empty
    pf = gp.compile(expr=res.hof[0], pset=res.pset)
    pred = pf(DummyState(x=2, y=3))
    assert isinstance(pred, float)


def op_scale(a: float, b: float) -> float:
    return a * b


def test_extra_primitives_and_terminals():
    res = quick_start(
        prompt="test custom primitives",
        cost_fn=cost_fn,
        feature_fns=[feat_x],
        state_type=DummyState,
        extra_primitives=[op_scale],
        extra_terminals=[2.0],
        pop_size=3,
        ngen=1,
        n_threads=1,
        seed=1,
    )
    pf = gp.compile(expr=res.hof[0], pset=res.pset)
    pred = pf(DummyState(x=1))
    assert isinstance(pred, float)
