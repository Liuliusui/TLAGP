#!/usr/bin/env python3
import glob
import os
import random
import sys
from functools import partial
from pathlib import Path

from deap import algorithms, base, creator, gp, tools
from multiprocessing.dummy import Pool as ThreadPool

# Ensure repo root on sys.path so imports work when run from crp/ directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tlagp import (
    DEFAULT_ALPHA,
    DEFAULT_K_SELECT,
    compose_system_prompt,
    build_llm_client,
    llm_score_branch,
    eval_with_llm_shaping,
    mate_llm_biased,
    mut_llm_guarded,
)

from crp.prompt import CRP_SYSTEM_PROMPT
from crp.pset import build_crp_pset
from crp.read_data import load_instance_from_dat
from crp.simulate import apply_relocation_scheme
from util.util import save_pickle

def main(seed=42, pop_size=50, ngen=100, n_threads=8):
    random.seed(seed)
    pset = build_crp_pset()
    timeout_ms = int(os.getenv("LLM_TIMEOUT_MS", "15000"))
    client = build_llm_client(timeout_ms=timeout_ms)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    if not hasattr(creator, "FitnessMinLLM"):
        creator.create("FitnessMinLLM", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualLLM"):
        creator.create("IndividualLLM", gp.PrimitiveTree, fitness=creator.FitnessMinLLM)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.IndividualLLM, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=7)

    prompt = compose_system_prompt(CRP_SYSTEM_PROMPT)
    scorer = partial(llm_score_branch, system_prompt=prompt, client=client) if client else partial(llm_score_branch, system_prompt=prompt)
    toolbox.register("mate", mate_llm_biased, k_select=DEFAULT_K_SELECT, scorer=scorer)
    toolbox.register("mutate", mut_llm_guarded, expr=toolbox.expr, pset=pset, k_select=DEFAULT_K_SELECT, scorer=scorer)

    data_dir = ROOT / "crp"
    train_paths = glob.glob(str(data_dir / "clean" / "*.dat"))
    instances = [load_instance_from_dat(p) for p in train_paths]
    val_paths = glob.glob(str(data_dir / "validation" / "*.dat"))
    val_instances = [load_instance_from_dat(p) for p in val_paths] if val_paths else []

    def cost_fn(pf):
        # Evaluate a compiled priority function on all training instances.
        return apply_relocation_scheme(instances, "RE", pf)

    toolbox.register(
        "evaluate",
        eval_with_llm_shaping,
        cost_fn=cost_fn,
        pset=pset,
        alpha=DEFAULT_ALPHA,
        k_select=DEFAULT_K_SELECT,
        scorer=scorer,
    )

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(5)

    def test_perf(_):
        best = hof[0]
        sim = apply_relocation_scheme(val_instances, "RE", gp.compile(expr=best, pset=pset), max_steps=5000)
        return sim[0] if isinstance(sim, (tuple, list)) else sim

    stats = tools.Statistics(lambda ind: (getattr(ind, "sim_score", None), ind.fitness.values[0]))
    stats.register("min_sim", lambda vs: min(v[0] for v in vs if v[0] is not None))
    stats.register("min_shaped", lambda vs: min(v[1] for v in vs))
    stats.register("test", test_perf)

    with ThreadPool(processes=n_threads) as pool:
        toolbox.register("map", pool.map)
        pop, log = algorithms.eaSimple(
            pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=ngen, stats=stats, halloffame=hof, verbose=True
        )

    print("Best individuals:")
    for best in hof:
        print(best, "sim=", getattr(best, "sim_score", None), "shaped=", best.fitness.values[0])

    os.makedirs("crp_results", exist_ok=True)
    tag = f"s{seed}_p{pop_size}_g{ngen}_ksel{DEFAULT_K_SELECT}_a{DEFAULT_ALPHA}"
    out_fname = os.path.join("crp_results", f"crp_{tag}")
    save_pickle(
        {
            "population": pop,
            "logbook": log,
            "halloffame": hof,
            "random_state": random.getstate(),
            "params": {"seed": seed, "pop_size": pop_size, "ngen": ngen, "n_threads": n_threads},
        },
        out_fname,
    )
    print(f"[INFO] Results saved to {out_fname}.pkl")

if __name__ == "__main__":
    main()
