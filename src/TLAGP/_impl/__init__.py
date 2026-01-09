#!/usr/bin/env python3
"""
Legacy compatibility shim. New code should import from tlagp.core/operators/gp/runners.
"""
from ..core import (
    DEFAULT_ALPHA,
    DEFAULT_K_SELECT,
    SYSTEM_PROMPT,
    build_llm_client,
    compose_system_prompt,
    llm_score_branch,
)
from ..operators import best_slice_by_llm, eval_with_llm_shaping, mate_llm_biased, mate_nonllm_subtree, mut_llm_guarded
from ..gp import add_basic_primitives, extract_subtree_indices_and_trees, extract_subtrees, pick_deep_k_slices, swap_slices_inplace
from ..runners import EasyRunResult, quick_start

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_K_SELECT",
    "SYSTEM_PROMPT",
    "compose_system_prompt",
    "build_llm_client",
    "llm_score_branch",
    "eval_with_llm_shaping",
    "best_slice_by_llm",
    "mate_llm_biased",
    "mate_nonllm_subtree",
    "mut_llm_guarded",
    "quick_start",
    "EasyRunResult",
    "add_basic_primitives",
    "extract_subtree_indices_and_trees",
    "extract_subtrees",
    "pick_deep_k_slices",
    "swap_slices_inplace",
]
