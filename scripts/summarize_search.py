"""Summarise search archives: hypervolume, best CV, dev score of the knee, statistical tests.

    python scripts/summarize_search.py --task t1
Reads results/search_raw/<task>/*.npz, writes results/search_summary_<task>.csv and
results/search_stats_<task>.json. Dev is used for model-selection diagnostics only; test is
never touched here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from vispam.search.fitness import SearchContext  # noqa: E402
from vispam.search.metrics import hypervolume  # noqa: E402
from vispam.search.space import decode  # noqa: E402
from vispam.search.algorithms import objectives, pareto_mask  # noqa: E402

BEST_WITHIN = 0.005  # knee rule: fewest features whose CV F1 is within 0.005 of the run's best


def knee_index(f1: np.ndarray, nf: np.ndarray) -> int:
    best = f1.max()
    ok = np.where(f1 >= best - BEST_WITHIN)[0]
    return int(ok[np.argmin(nf[ok])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="t1")
    ap.add_argument("--raw", default=str(ROOT / "results" / "search_raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--no-dev", action="store_true", help="skip dev scoring of knee/best configs")
    a = ap.parse_args()

    files = sorted((Path(a.raw) / a.task).glob("*.npz"))
    ctx = None if a.no_dev else SearchContext(a.task)
    rows = []
    for fpath in files:
        algo, seed = fpath.stem.rsplit("_s", 1)
        d = np.load(fpath)
        f1, nf, x = d["f1"], d["nf"], d["x"]
        obj = np.array([objectives(p, q) for p, q in zip(f1, nf)])
        front = np.unique(obj[pareto_mask(obj)], axis=0)
        i_best, i_knee = int(np.argmax(f1)), knee_index(f1, nf)
        row = dict(algo=algo, seed=int(seed), calls=int(d["n_calls"]), unique=int(d["n_unique"]),
                   hv=hypervolume(obj), front_size=len(front), best_cv=float(f1[i_best]),
                   best_cv_nfeat=int(nf[i_best]), knee_cv=float(f1[i_knee]),
                   knee_nfeat=int(nf[i_knee]), wall_s=float(d["wall"]))
        if ctx is not None:
            row["best_dev"] = ctx.dev_score(decode(x[i_best]))
            row["knee_dev"] = ctx.dev_score(decode(x[i_knee]))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["algo", "seed"])
    out_dir = Path(a.out_dir)
    df.to_csv(out_dir / f"search_summary_{a.task}.csv", index=False)

    # statistics: paired by seed
    piv = df.pivot(index="seed", columns="algo", values="hv").dropna()
    stats_out: dict = {"n_seeds": int(len(piv)), "algos": list(piv.columns),
                       "hv_mean": piv.mean().round(4).to_dict(), "hv_std": piv.std().round(4).to_dict()}
    if len(piv) >= 3 and piv.shape[1] >= 3:
        fr = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
        stats_out["friedman"] = {"stat": float(fr.statistic), "p": float(fr.pvalue)}
    ranks = piv.rank(axis=1, ascending=False).mean()
    stats_out["mean_rank_hv"] = ranks.round(3).to_dict()
    wil = {}
    for ref in ("nsga2", "mode"):
        if ref not in piv:
            continue
        for other in piv.columns:
            if other == ref or len(piv) < 5:
                continue
            diff = piv[ref] - piv[other]
            p = 1.0 if (diff == 0).all() else float(stats.wilcoxon(piv[ref], piv[other]).pvalue)
            wil[f"{ref}_vs_{other}"] = {"mean_diff_hv": float(diff.mean()), "p": p}
    k = len(wil)  # Holm-Bonferroni within this family
    order = sorted(wil, key=lambda n: wil[n]["p"])
    running = 0.0
    for i, n in enumerate(order):
        running = max(running, min(1.0, wil[n]["p"] * (k - i)))
        wil[n]["p_holm"] = running
    stats_out["wilcoxon_hv"] = wil
    for col in ("best_cv", "knee_cv", "best_dev", "knee_dev", "knee_nfeat"):
        if col in df:
            stats_out[f"{col}_mean"] = df.groupby("algo")[col].mean().round(4).to_dict()
    (out_dir / f"search_stats_{a.task}.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    print(df.groupby("algo")[[c for c in ("hv", "best_cv", "best_dev", "knee_cv", "knee_dev", "knee_nfeat", "front_size") if c in df]].mean().round(4).to_string())
    print(json.dumps({k: v for k, v in stats_out.items() if k in ("friedman", "mean_rank_hv")}, indent=2))


if __name__ == "__main__":
    main()
