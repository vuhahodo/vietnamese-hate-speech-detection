"""Search algorithms working on [0, 1]^D vectors with a shared evaluation budget.

Proposed (multi-objective, objectives: -cv_macro_f1, log10(n_features)):
    nsga2   NSGA-II (pymoo)
    mode    multi-objective DE: DE/rand/1/bin variation + non-dominated sorting / crowding survival
Reference (as in the inherited pipeline and standard practice):
    hho         Harris Hawks Optimisation on -cv_macro_f1 only (single objective, as in
                Al-Zoubi et al. 2023 and Anees et al. 2025)
    ga          real-coded GA on -cv_macro_f1 only
    ga_scalar   the same GA on a weighted sum -f1 + LAMBDA * (log10(n_features) - 2)
    random      uniform random search
    grid        coarse structured grid, shuffled and truncated at the budget

Every algorithm stops by ``BudgetExhausted`` raised from the problem, so all of them use the
same number of fitness proposals. Quality is judged on the non-dominated set of the archive.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from vispam.search.fitness import BudgetedProblem, BudgetExhausted
from vispam.search.space import DIM, encode

LAMBDA_SCALAR = 0.03
POP_SIZE = 30


def objectives(f1: float, nf: int) -> np.ndarray:
    """Minimisation form used for dominance and hypervolume."""
    return np.array([-f1, math.log10(max(nf, 1))])


# ---------------------------------------------------------------- non-dominated utilities
def nondominated_sort(f: np.ndarray) -> list[np.ndarray]:
    """Fast non-dominated sorting for minimisation; returns a list of index arrays (fronts)."""
    n = len(f)
    dominates = (f[:, None, :] <= f[None, :, :]).all(-1) & (f[:, None, :] < f[None, :, :]).any(-1)
    dom_count = dominates.sum(0)
    fronts, remaining = [], np.ones(n, bool)
    while remaining.any():
        cur = np.where(remaining & (dom_count == 0))[0]
        if len(cur) == 0:  # numerical safety
            cur = np.where(remaining)[0]
        fronts.append(cur)
        remaining[cur] = False
        dom_count = dom_count - dominates[cur].sum(0)
        dom_count[cur] = -1
    return fronts


def crowding_distance(f: np.ndarray) -> np.ndarray:
    n, m = f.shape
    d = np.zeros(n)
    if n <= 2:
        d[:] = np.inf
        return d
    for j in range(m):
        order = np.argsort(f[:, j], kind="stable")
        rng_ = f[order[-1], j] - f[order[0], j]
        d[order[0]] = d[order[-1]] = np.inf
        if rng_ <= 0:
            continue
        d[order[1:-1]] += (f[order[2:], j] - f[order[:-2], j]) / rng_
    return d


def survive(x: np.ndarray, f: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """NSGA-II environmental selection: fill by fronts, break the last front by crowding."""
    chosen: list[int] = []
    for front in nondominated_sort(f):
        if len(chosen) + len(front) <= n:
            chosen.extend(front.tolist())
        else:
            dist = crowding_distance(f[front])
            keep = front[np.argsort(-dist, kind="stable")[: n - len(chosen)]]
            chosen.extend(keep.tolist())
            break
    idx = np.array(chosen)
    return x[idx], f[idx]


def pareto_mask(f: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(f), bool)
    mask[nondominated_sort(f)[0]] = True
    return mask


# ---------------------------------------------------------------- multi-objective
def _eval_pop(problem: BudgetedProblem, x: np.ndarray) -> np.ndarray:
    return np.array([objectives(*problem(xi)) for xi in x])


def run_mode(problem: BudgetedProblem, rng: np.random.Generator, pop_size: int = POP_SIZE,
             F: float = 0.5, CR: float = 0.9) -> None:
    """Multi-objective DE: DE/rand/1/bin trials, then NSGA-II style survival on parents + trials."""
    x = rng.random((pop_size, DIM))
    f = _eval_pop(problem, x)
    while True:
        trials = np.empty_like(x)
        for i in range(pop_size):
            r = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            mutant = x[r[0]] + F * (x[r[1]] - x[r[2]])
            # reflect into bounds
            mutant = np.where(mutant < 0, -mutant, mutant)
            mutant = np.where(mutant > 1, 2 - mutant, mutant)
            mutant = np.clip(mutant, 0.0, 1.0)
            cross = rng.random(DIM) < CR
            cross[rng.integers(DIM)] = True
            trials[i] = np.where(cross, mutant, x[i])
        ft = _eval_pop(problem, trials)
        x, f = survive(np.vstack([x, trials]), np.vstack([f, ft]), pop_size)


def run_nsga2(problem: BudgetedProblem, rng: np.random.Generator, pop_size: int = POP_SIZE) -> None:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    class _P(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=DIM, n_obj=2, xl=np.zeros(DIM), xu=np.ones(DIM))

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = objectives(*problem(x))

    algo = NSGA2(pop_size=pop_size, sampling=FloatRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
    minimize(_P(), algo, termination=("n_eval", problem.budget),
             seed=int(rng.integers(2**31 - 1)), verbose=False)


# ---------------------------------------------------------------- single-objective references
def _scalar(f1: float, nf: int, lam: float) -> float:
    return -f1 + lam * (math.log10(max(nf, 1)) - 2.0)


def run_ga(problem: BudgetedProblem, rng: np.random.Generator, pop_size: int = POP_SIZE,
           lam: float = 0.0, p_cx: float = 0.9, sigma: float = 0.1, elite: int = 2) -> None:
    """Real-coded GA: tournament(3), BLX-0.3 crossover, Gaussian mutation (1/D per gene), elitism."""
    def fit(xi):
        return _scalar(*problem(xi), lam)

    x = rng.random((pop_size, DIM))
    fv = np.array([fit(xi) for xi in x])
    while True:
        order = np.argsort(fv)
        new = [x[i].copy() for i in order[:elite]]
        while len(new) < pop_size:
            pa = x[min(rng.integers(pop_size, size=3), key=lambda i: fv[i])]
            pb = x[min(rng.integers(pop_size, size=3), key=lambda i: fv[i])]
            if rng.random() < p_cx:
                lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
                span = hi - lo
                child = rng.uniform(lo - 0.3 * span, hi + 0.3 * span)
            else:
                child = pa.copy()
            mut = rng.random(DIM) < 1.0 / DIM
            child = np.where(mut, child + rng.normal(0, sigma, DIM), child)
            new.append(np.clip(child, 0.0, 1.0))
        elite_x, elite_f = x[order[:elite]], fv[order[:elite]]
        new = np.array(new)
        new_f = np.concatenate([elite_f, [fit(xi) for xi in new[elite:]]])
        x, fv = new, new_f


def _levy(dim: int, rng: np.random.Generator, beta: float = 1.5) -> np.ndarray:
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u, v = rng.normal(0, sigma, dim), rng.normal(0, 1, dim)
    return 0.01 * u / np.abs(v) ** (1 / beta)


def run_hho(problem: BudgetedProblem, rng: np.random.Generator, pop_size: int = POP_SIZE) -> None:
    """Harris Hawks Optimisation (Heidari et al. 2019) on -cv_macro_f1 (single objective)."""
    def fit(xi):
        return _scalar(*problem(xi), 0.0)

    n_iter = max(1, problem.budget // pop_size)
    x = rng.random((pop_size, DIM))
    fv = np.array([fit(xi) for xi in x])
    best = int(np.argmin(fv))
    rabbit, rabbit_f = x[best].copy(), fv[best]
    for t in range(n_iter):
        mean = x.mean(0)
        for i in range(pop_size):
            e0 = 2 * rng.random() - 1
            e = 2 * e0 * (1 - t / n_iter)
            jump = 2 * (1 - rng.random())
            xi = x[i]
            if abs(e) >= 1:  # exploration
                if rng.random() >= 0.5:
                    xr = x[rng.integers(pop_size)]
                    new = xr - rng.random() * np.abs(xr - 2 * rng.random() * xi)
                else:
                    new = (rabbit - mean) - rng.random() * (rng.random(DIM))
                new = np.clip(new, 0.0, 1.0)
                fn = fit(new)
                x[i], fv[i] = new, fn
            else:  # exploitation
                r = rng.random()
                if r >= 0.5 and abs(e) >= 0.5:  # soft besiege
                    new = (rabbit - xi) - e * np.abs(jump * rabbit - xi)
                elif r >= 0.5:  # hard besiege
                    new = rabbit - e * np.abs(rabbit - xi)
                else:  # besiege with progressive rapid dives
                    base = xi if abs(e) >= 0.5 else mean
                    y = np.clip(rabbit - e * np.abs(jump * rabbit - base), 0.0, 1.0)
                    fy = fit(y)
                    if fy < fv[i]:
                        x[i], fv[i] = y, fy
                        continue
                    z = np.clip(y + rng.random(DIM) * _levy(DIM, rng), 0.0, 1.0)
                    fz = fit(z)
                    if fz < fv[i]:
                        x[i], fv[i] = z, fz
                    continue
                new = np.clip(new, 0.0, 1.0)
                fn = fit(new)
                x[i], fv[i] = new, fn
            if fv[i] < rabbit_f:
                rabbit, rabbit_f = x[i].copy(), fv[i]
    # leftover budget (dives use extra evaluations): nothing to do, BudgetExhausted ends the run


def run_random(problem: BudgetedProblem, rng: np.random.Generator, **_) -> None:
    while True:
        problem(rng.random(DIM))


def _grid_points() -> list[np.ndarray]:
    levels = (0.0, 0.25, 0.5, 1.0)
    pts = []
    for keep in np.array(np.meshgrid(*[levels] * 4)).reshape(4, -1).T:
        if not keep.any():
            continue
        for model, cs in (("svc", (-1.0, 0.0, 1.0)), ("lr", (0.0, 1.0))):
            for log_c in cs:
                pts.append(encode(tuple(keep), None, log_c, model))
    return pts


def run_grid(problem: BudgetedProblem, rng: np.random.Generator, **_) -> None:
    pts = _grid_points()
    for i in rng.permutation(len(pts)):
        problem(pts[i])
    # grid exhausted before the budget: remaining budget is unused (reported by n_calls)


ALGORITHMS: dict[str, Callable] = {
    "nsga2": run_nsga2,
    "mode": run_mode,
    "hho": run_hho,
    "ga": lambda p, r, **kw: run_ga(p, r, lam=0.0, **kw),
    "ga_scalar": lambda p, r, **kw: run_ga(p, r, lam=LAMBDA_SCALAR, **kw),
    "random": run_random,
    "grid": run_grid,
}


def run_algorithm(name: str, problem: BudgetedProblem, seed: int) -> None:
    rng = np.random.default_rng(seed)
    try:
        ALGORITHMS[name](problem, rng)
    except BudgetExhausted:
        pass
