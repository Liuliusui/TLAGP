#!/usr/bin/env python3
"""Domain-specific prompt for the Container Relocation Problem (CRP)."""

CRP_SYSTEM_PROMPT = """
Domain: Container Relocation Problem (bay with multiple stacks and tiers). Each subtree computes a priority index to choose a destination stack for relocating a blocking container.

Bay state features available in the primitive set (examples):
  - HLA(bay): current stack height at the candidate destination.
  - SPA(bay): spare tiers at that destination (n_tiers - height).
  - TOPPRI(bay): priority of the top container at the destination (inf if empty).
  - SH/EMP/CUR(bay, seq): float versions of height, spare tiers, and current target id.
  - DUR(bay, seq): estimated travel distance plus fixed cost.
  - RI/AVG/REM(bay, seq): blocking counts and statistics.
  - NEXT/DIFF/EMPTY/WL/NL/DSM(bay, seq): indicators for next target, priority gaps, emptiness, well-/non-ordered counts, and depth of first lower-priority blocker.

Objective: prioritize expressions that lower total relocations and avoid deadlocks. Favor moves that free the current target, preserve well-ordered stacks, and keep smaller priorities above larger ones. Penalize expressions that ignore bay state, divide by zero, or behave like constants.
"""

__all__ = ["CRP_SYSTEM_PROMPT"]
