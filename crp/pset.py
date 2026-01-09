#!/usr/bin/env python3
"""Primitive set builder for the Container Relocation Problem (CRP)."""
import random

from deap import gp

from crp.bay import Bay
from tlagp import add_basic_primitives


def build_crp_pset():
    pset = gp.PrimitiveSetTyped("PF", [Bay, list], float)
    add_basic_primitives(pset)
    pset.addPrimitive(lambda bay: bay.h[bay.last_dst], [Bay], int, name="HLA")
    pset.addPrimitive(lambda bay: bay.n_tiers - bay.h[bay.last_dst], [Bay], int, name="SPA")
    pset.addPrimitive(
        lambda bay: bay.qlt[bay.last_dst][bay.h[bay.last_dst] - 1] if bay.h[bay.last_dst] > 0 else float("inf"),
        [Bay],
        float,
        name="TOPPRI",
    )
    pset.addPrimitive(lambda bay, seq: bay.h[bay.last_dst], [Bay, list], float, name="SH")
    pset.addPrimitive(lambda bay, seq: bay.n_tiers - bay.h[bay.last_dst], [Bay, list], float, name="EMP")
    pset.addPrimitive(lambda bay, seq: float(seq[0]), [Bay, list], float, name="CUR")

    def DUR(bay, seq):
        return abs(bay.crane_pos - bay.current_stack) + abs(bay.current_stack - bay.last_dst) + 30.0

    pset.addPrimitive(DUR, [Bay, list], float, name="DUR")

    def RI(bay, seq):
        return float(sum(1 for c in bay.pri[bay.last_dst][: bay.h[bay.last_dst]] if c < seq[0]))

    pset.addPrimitive(RI, [Bay, list], float, name="RI")

    def AVG(bay, seq):
        arr = bay.pri[bay.last_dst][: bay.h[bay.last_dst]]
        return float(sum(arr) / len(arr)) if arr else 0.0

    pset.addPrimitive(AVG, [Bay, list], float, name="AVG")

    def REM(bay, seq):
        return float(len(bay.pri[bay.last_dst][: bay.h[bay.last_dst]]))

    pset.addPrimitive(REM, [Bay, list], float, name="REM")

    def NEXT(bay, seq):
        return 1.0 if len(seq) > 1 and seq[1] in bay.pri[bay.last_dst][: bay.h[bay.last_dst]] else 0.0

    pset.addPrimitive(NEXT, [Bay, list], float, name="NEXT")

    def DIFF(bay, seq):
        return float(min(bay.pri[bay.last_dst][: bay.h[bay.last_dst]], default=seq[0]) - seq[0])

    pset.addPrimitive(DIFF, [Bay, list], float, name="DIFF")

    def EMPTY(bay, seq):
        return 1.0 if bay.h[bay.last_dst] == 0 else 0.0

    pset.addPrimitive(EMPTY, [Bay, list], float, name="EMPTY")

    def WL(bay, seq):
        arr = bay.pri[bay.last_dst][: bay.h[bay.last_dst]]
        return float(sum(1 for i, c in enumerate(arr) if all(x <= c for x in arr[i + 1 :])))

    pset.addPrimitive(WL, [Bay, list], float, name="WL")

    def NL(bay, seq):
        arr = bay.pri[bay.last_dst][: bay.h[bay.last_dst]]
        return float(sum(1 for i, c in enumerate(arr) if any(x > c for x in arr[i + 1 :])))

    pset.addPrimitive(NL, [Bay, list], float, name="NL")
    pset.addPrimitive(
        lambda bay, seq: next((float(i + 1) for i, c in enumerate(bay.pri[bay.last_dst][: bay.h[bay.last_dst]]) if c < seq[0]), 0.0),
        [Bay, list],
        float,
        name="DSM",
    )
    pset.addPrimitive(lambda seq: seq, [list], list, name="id_seq")
    pset.addPrimitive(lambda bay: bay, [Bay], Bay, name="id_bay")
    pset.addEphemeralConstant("RND", random.random, float)
    pset.renameArguments(ARG0="bay", ARG1="seq")
    return pset


__all__ = ["build_crp_pset"]
