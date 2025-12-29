"""
Generic task template: plug in your data loader, feature functions, and simulator.
No GP/DEAP knowledge required—only implement the TODOs.
"""
import sys
from pathlib import Path
from typing import Any, List, Tuple

# Allow running without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmgp import quick_start


class State(dict):
    """Adapt this to your data structure; keep a single state object as input."""


def load_data() -> List[Tuple[State, float]]:
    """
    TODO: return a list of (state, target) pairs for training.
    Example below uses two toy samples; replace with your loader.
    """
    return [
        (State(x=1, y=2), 0.0),
        (State(x=3, y=4), 1.0),
    ]


# ---- Feature functions: read values from State and return float ----
def feat_x(state: State) -> float:
    return float(state.get("x", 0.0))


def feat_y(state: State) -> float:
    return float(state.get("y", 0.0))


# ---- Cost/simulator: evaluate the candidate program pf on your data ----
def cost_fn(pf):
    data = load_data()
    return sum((pf(s) - target) ** 2 for s, target in data) / len(data)


def main():
    result = quick_start(
        prompt="Describe your task here (used for LLM scoring, optional).",
        cost_fn=cost_fn,
        feature_fns=[feat_x, feat_y],  # add your features here
        state_type=State,
        pop_size=10,
        ngen=5,
        n_threads=1,
    )
    pf = result.pset.compile(expr=result.hof[0], pset=result.pset)
    print("Top individual:", result.hof[0])
    # Demo prediction on a new sample:
    print("Prediction for x=2,y=3:", pf(State(x=2, y=3)))


if __name__ == "__main__":
    main()
