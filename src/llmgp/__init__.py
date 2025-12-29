#!/usr/bin/env python3
"""llmgp public API: beginner-friendly surfaces with descriptive names."""

# High-level runners (auto pset for single-state tasks)
from .easy import EasyRunResult, easy_run, quick_start
# Custom pset runner
from .runner import run_gp_simple
# Simulation-oriented helpers
from .simulate import SimulatorConfig, SimpleSimulator, FunctionalSimulator
# Optional LLM helpers (environment-driven; safe to ignore)
from .llm import build_llm_client, compose_system_prompt, llm_score_branch

# Preferred descriptive names (aliases)
gp_run_auto = quick_start
GpAutoResult = EasyRunResult
gp_run_with_pset = run_gp_simple
SimulatorTemplate = SimpleSimulator
SimulatorRunner = FunctionalSimulator
make_llm_client = build_llm_client
make_llm_prompt = compose_system_prompt

__all__ = [
    # Recommended names
    "quick_start",
    "gp_run_auto",
    "GpAutoResult",
    "gp_run_with_pset",
    "SimulatorTemplate",
    "SimulatorRunner",
    "SimulatorConfig",
    "make_llm_client",
    "make_llm_prompt",
    "llm_score_branch",
    # Legacy/compat names
    "easy_run",
    "EasyRunResult",
    "run_gp_simple",
    "SimpleSimulator",
    "FunctionalSimulator",
    "build_llm_client",
    "compose_system_prompt",
]
