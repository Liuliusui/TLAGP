#!/usr/bin/env python3
"""Lightweight evolutionary loops used by tlagp runners."""
from typing import Any, Iterable, Optional, Sequence, Tuple

from deap import algorithms as _deap_algorithms


def ea_simple(
    population: Sequence[Any],
    toolbox: Any,
    cxpb: float,
    mutpb: float,
    ngen: int,
    *,
    stats: Optional[Any] = None,
    halloffame: Optional[Any] = None,
    verbose: bool = True,
) -> Tuple[Iterable[Any], Any]:
    """
    Thin wrapper around DEAP's eaSimple so the evolution loop is centralized.
    """
    if not hasattr(toolbox, "map"):
        toolbox.register("map", map)
    return _deap_algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=halloffame,
        verbose=verbose,
    )


__all__ = ["ea_simple"]
