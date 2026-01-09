#!/usr/bin/env python3
"""tlagp public API: beginner-friendly surfaces with descriptive names."""

# High-level runners (auto pset for single-state tasks)
from .quickstart import EasyRunResult, quick_start
# Custom pset runner
from .runner import run_gp_simple
# Simulation-oriented helpers
from .simulate import (
    FunctionalSimulator,
    SimulatorConfig,
    SimulatorRunner,
    SimulatorTemplate,
    SimpleSimulator,
)
# Optional LLM helpers (environment-driven; safe to ignore)
from .llm import build_llm_client, compose_system_prompt, llm_score_branch
from .gp import add_basic_primitives
from .operators import (
    best_slice_by_llm,
    mate_llm_biased,
    mate_nonllm_subtree,
    mut_llm_guarded,
    eval_with_llm_shaping,
)

# Preferred descriptive names (aliases)
GpAutoResult = EasyRunResult
gp_run_with_pset = run_gp_simple

make_llm_client = build_llm_client
make_llm_prompt = compose_system_prompt

__all__ = [
    # Recommended names
    "quick_start",
    "GpAutoResult",
    "gp_run_with_pset",
    "SimulatorTemplate",
    "SimulatorRunner",
    "SimulatorConfig",
    "make_llm_client",
    "make_llm_prompt",
    "llm_score_branch",
    "add_basic_primitives",
    "best_slice_by_llm",
    "mate_llm_biased",
    "mate_nonllm_subtree",
    "mut_llm_guarded",
    "eval_with_llm_shaping",
    # Legacy/compat names
    "EasyRunResult",
    "run_gp_simple",
    "SimpleSimulator",
    "FunctionalSimulator",
    "build_llm_client",
    "compose_system_prompt",
]
