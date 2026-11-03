"""Figures for the T1 search: pooled Pareto fronts and hypervolume convergence."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vispam.search.algorithms import objectives, pareto_mask  # noqa: E402
from vispam.search.metrics import hypervolume  # noqa: E402

RAW = ROOT / "results" / "search_raw" / "t1"
OUT = ROOT / "docs" / "report" / "figures"
OKABE = {"mode": "#0072B2", "nsga2": "#009E73", "hho": "#D55E00", "random": "#999999",
         "grid": "#CC79A7", "ga_scalar": "#E69F00", "ga": "#000000"}
LABEL = {"mode": "MODE (DE)", "nsga2": "NSGA-II", "hho": "HHO", "random": "Random",
         "grid": "Grid", "ga_scalar": "GA + penalty", "ga": "GA (F1 only)"}
ALGOS = list(OKABE)


def load(algo):
    return [np.load(p) for p in sorted(RAW.glob(f"{algo}_s[0-9]*.npz"))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # 1. pooled Pareto fronts (all seeds)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for a in ALGOS:
        pts = np.vstack([np.c_[d["nf"], d["f1"]] for d in load(a)])
        obj = np.array([objectives(f, n) for n, f in pts])
        front = pts[pareto_mask(obj)]
        front = front[np.argsort(front[:, 0])]
        ax.step(front[:, 0], front[:, 1], where="post", color=OKABE[a], lw=1.8, label=LABEL[a])
    ax.set_xscale("log")
    ax.set_xlim(1e3, 2e5)
    ax.set_ylim(0.80, 0.865)
    ax.set_xlabel("number of features (log scale)")
    ax.set_ylabel("5-fold CV macro F1 (train)")
    ax.set_title("Pooled non-dominated sets over 10 seeds (T1, 500 evaluations each)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "pareto_t1.png", dpi=160)
    plt.close(fig)
    # 2. hypervolume convergence
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    steps = np.arange(30, 501, 10)
    for a in ALGOS:
        curves = []
        for d in load(a):
            o = np.array([objectives(f, n) for f, n in zip(d["f1"], d["nf"])])
            curves.append([hypervolume(o[:s]) for s in steps])
        c = np.array(curves)
        m, s = c.mean(0), c.std(0)
        ax.plot(steps, m, color=OKABE[a], lw=1.8, label=LABEL[a])
        ax.fill_between(steps, m - s, m + s, color=OKABE[a], alpha=0.12, lw=0)
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("hypervolume of evaluated points")
    ax.set_title("Convergence, mean +/- sd over 10 seeds")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "hypervolume_convergence_t1.png", dpi=160)
    plt.close(fig)
    print("saved", [p.name for p in OUT.glob("*.png")])


if __name__ == "__main__":
    main()
