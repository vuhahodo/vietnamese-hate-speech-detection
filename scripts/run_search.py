"""Run the budgeted searches (resumable, parallel over (algorithm, seed) jobs).

Example:
    python scripts/run_search.py --task t1 --budget 1000 --seeds 0-9 --workers 8
Each job writes results/search_raw/<task>/<algo>_s<seed>.npz (archive of every proposal).
The test split is never touched here.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

from vispam.search.algorithms import ALGORITHMS, run_algorithm  # noqa: E402
from vispam.search.fitness import BudgetedProblem, SearchContext  # noqa: E402

_CTX: SearchContext | None = None


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def _init_worker(task: str) -> None:
    global _CTX
    _CTX = SearchContext(task)


def _job(args: tuple) -> str:
    task, algo, seed, budget, out_dir = args
    out = Path(out_dir) / f"{algo}_s{seed}.npz"
    if out.exists():
        return f"skip {out.name}"
    problem = BudgetedProblem(_CTX.cv_score, _CTX.n_features, budget)
    t0 = time.time()
    run_algorithm(algo, problem, seed)
    wall = time.time() - t0
    x = np.array([a[0] for a in problem.archive])
    f1 = np.array([a[1] for a in problem.archive])
    nf = np.array([a[2] for a in problem.archive])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x, f1=f1, nf=nf, wall=wall, eval_seconds=problem.eval_seconds,
                        n_calls=problem.n_calls, n_unique=len(problem.cache))
    return (f"done {out.name}: calls={problem.n_calls} unique={len(problem.cache)} "
            f"best_cv={f1.max():.4f} wall={wall:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2"], default="t1")
    ap.add_argument("--algos", default=",".join(ALGORITHMS))
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "search_raw"))
    a = ap.parse_args()

    out_dir = Path(a.out) / a.task
    jobs = [(a.task, algo, seed, a.budget, str(out_dir))
            for seed, algo in itertools.product(_parse_seeds(a.seeds), a.algos.split(","))]
    print(f"{len(jobs)} jobs, budget={a.budget}, workers={a.workers}", flush=True)
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(a.task,)) as pool:
        for msg in pool.imap_unordered(_job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
