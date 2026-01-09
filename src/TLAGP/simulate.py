#!/usr/bin/env python3
"""
Beginner-friendly simulator模板，支持两种用法：
  - 继承并覆写 load_data()/feature_fns()（可选 cost_fn/extra_*）
  - 构造函数直接注入 data_loader/feature_fns/cost_fn/extra_*，无需子类
调用 run() 触发 quick_start，best_pf() 会编译 HOF 个体。
"""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from deap import gp

from .quickstart import EasyRunResult, quick_start
from .llm import DEFAULT_ALPHA, DEFAULT_K_SELECT


State = Any
FeatureFn = Callable[[State], float]
CostFn = Callable[[Callable[[State], float]], float]
PrimitiveSpec = Union[Callable[..., Any], Tuple[Any, ...]]
TerminalSpec = Any


@dataclass
class SimulatorConfig:
    prompt: str = "Describe your task for LLM scoring (optional)."
    pop_size: int = 10
    ngen: int = 5
    n_threads: int = 1
    seed: int = 42
    max_depth: int = 5
    cxpb: float = 0.5
    mutpb: float = 0.3
    alpha: float = DEFAULT_ALPHA
    k_select: int = DEFAULT_K_SELECT
    timeout_ms: int = 15000


class SimpleSimulator:
    """
    Minimal simulator wrapper with two modes:
      1) subclass and override load_data()/feature_fns() (optionally cost_fn/extra_*)
      2) pass data_loader/feature_fns/cost_fn/extra_* into the constructor, no subclass needed.
    """

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        *,
        data_loader: Optional[Callable[[], List[Tuple[State, float]]]] = None,
        feature_fns: Optional[Sequence[FeatureFn]] = None,
        cost_fn: Optional[CostFn] = None,
        extra_primitives: Optional[Sequence[PrimitiveSpec]] = None,
        extra_terminals: Optional[Sequence[TerminalSpec]] = None,
    ):
        self.config = config or SimulatorConfig()
        self.result: Optional[EasyRunResult] = None
        self._data_loader = data_loader
        self._feature_fns = feature_fns
        self._user_cost_fn = cost_fn
        self._extra_primitives = extra_primitives
        self._extra_terminals = extra_terminals
        self._cached_data: Optional[List[Tuple[State, float]]] = None

    # ---- Override these two to fit your task OR pass in constructor ----
    def load_data(self) -> List[Tuple[State, float]]:
        """Return list of (state, target) pairs."""
        if self._data_loader is None:
            raise NotImplementedError("Implement load_data() or provide data_loader.")
        if self._cached_data is None:
            self._cached_data = list(self._data_loader())
        return self._cached_data

    def feature_fns(self) -> Sequence[FeatureFn]:
        """Return feature functions that read from state and output float."""
        if self._feature_fns is None:
            raise NotImplementedError("Implement feature_fns() or provide feature_fns.")
        return tuple(self._feature_fns)

    # ---- Optional override if you need custom cost ----
    def cost_fn(self, pf: Callable[[State], float]) -> float:
        if self._user_cost_fn is not None:
            return self._user_cost_fn(pf)
        data = self.load_data()
        return sum((pf(s) - t) ** 2 for s, t in data) / len(data)

    # ---- Optional: expose custom primitives/constants without editing GP internals ----
    def extra_primitives(self) -> Sequence[PrimitiveSpec]:
        """Return extra primitives to add to the primitive set (or empty)."""
        return self._extra_primitives or ()

    def extra_terminals(self) -> Sequence[TerminalSpec]:
        """Return extra terminals/constants to add to the primitive set (or empty)."""
        return self._extra_terminals or ()

    # ---- No need to change below for most users ----
    def run(self) -> EasyRunResult:
        cfg = self.config
        self.result = quick_start(
            prompt=cfg.prompt,
            cost_fn=self.cost_fn,
            feature_fns=self.feature_fns(),
            state_type=self._infer_state_type(),
            extra_primitives=self.extra_primitives(),
            extra_terminals=self.extra_terminals(),
            pop_size=cfg.pop_size,
            ngen=cfg.ngen,
            n_threads=cfg.n_threads,
            seed=cfg.seed,
            max_depth=cfg.max_depth,
            cxpb=cfg.cxpb,
            mutpb=cfg.mutpb,
            alpha=cfg.alpha,
            k_select=cfg.k_select,
            timeout_ms=cfg.timeout_ms,
        )
        return self.result

    def best_pf(self, index: int = 0) -> Callable[[State], float]:
        if not self.result:
            raise ValueError("Call run() first.")
        return gp.compile(expr=self.result.hof[index], pset=self.result.pset)

    def _infer_state_type(self):
        data = self.load_data()
        if not data:
            return object
        sample_state, _ = data[0]
        return sample_state.__class__


# Backward-compatible aliases: one class covers both subclass and injection usage
FunctionalSimulator = SimpleSimulator
SimulatorRunner = SimpleSimulator
SimulatorTemplate = SimpleSimulator


__all__ = ["SimulatorConfig", "SimpleSimulator", "FunctionalSimulator", "SimulatorRunner", "SimulatorTemplate"]
