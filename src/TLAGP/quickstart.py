#!/usr/bin/env python3
"""Public entry point for beginner-friendly GP runs."""

from .runners.quickstart import EasyRunResult, quick_start

__all__ = ["quick_start", "EasyRunResult"]
