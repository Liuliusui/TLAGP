"""
Minimal "no GP jargon" example using quick_start.
Replace DummyState/feature/cost with your own task logic.
"""
import sys
from pathlib import Path

# Allow running directly from repo without install: add src/ to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmgp import quick_start


class DummyState(dict):
    """Example state; in practice use your own object/dict."""
    pass

def feat_x(state: DummyState) -> float:
    return float(state.get("x", 0.0))


def feat_y(state: DummyState) -> float:
    return float(state.get("y", 0.0))


def cost_fn(pf):
    # Toy dataset: two states, we want pf(state) close to target.
    samples = [
        (DummyState(x=1, y=2), 0.0),
        (DummyState(x=3, y=4), 1.0),
    ]
    err = 0.0
    for s, target in samples:
        pred = pf(s)
        err += (pred - target) ** 2
    return err / len(samples)


result = quick_start(
    prompt="Regress a simple function of x and y.",
    cost_fn=cost_fn,
    feature_fns=[feat_x, feat_y],
    state_type=DummyState,
    pop_size=10,
    ngen=5,
)

print("Top individual:", result.hof[0])
