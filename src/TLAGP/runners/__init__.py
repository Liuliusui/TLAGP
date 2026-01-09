#!/usr/bin/env python3
"""Entry points for running GP with minimal wiring."""

from .quickstart import EasyRunResult, quick_start
from .gp_runner import run_gp_simple

__all__ = ["quick_start", "EasyRunResult", "run_gp_simple"]
