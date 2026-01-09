#!/usr/bin/env python3
"""Public facade for generic GP runner."""

from .runners.gp_runner import run_gp_simple

__all__ = ["run_gp_simple"]
