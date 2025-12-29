#!/usr/bin/env python3
"""Public entry point for beginner-friendly GP runs."""

from ._impl.easy import EasyRunResult, easy_run, quick_start

__all__ = ["quick_start", "easy_run", "EasyRunResult"]
