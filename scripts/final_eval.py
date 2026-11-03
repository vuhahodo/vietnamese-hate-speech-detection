"""One-shot test evaluation of the configurations fixed on train/dev (task T1).

Candidates (all fixed before the test split is read):
  BASE   LogisticRegression C=1, block B2 only (the VeriReview baseline)
  FULL   LinearSVC C=0.1, all four blocks, no selection (best dev among hand-picked references)
  EA     knee configuration of the MODE run with the highest knee dev score
All are fitted on the training split only. Writes results/final_test_t1.json and
results/final_test_t1_preds.npz. Run it once; it refuses to overwrite existing output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vispam.evaluate import bootstrap_ci, mcnemar, paired_bootstrap_diff, per_class_f1  # noqa: E402
from vispam.search.fitness import SearchContext  # noqa: E402
from vispam.search.space import decode, encode  # noqa: E402

OUT_JSON = ROOT / "results" / "final_test_t1.json"
OUT_NPZ = ROOT / "results" / "final_test_t1_preds.npz"
BEST_WITHIN = 0.005


def ea_knee_vector() -> tuple[np.ndarray, int]:
    summary = pd.read_csv(ROOT / "results" / "search_summary_t1.csv")
    mode = summary[summary.algo == "mode"]
    seed = int(mode.loc[mode.knee_dev.idxmax(), "seed"])
    d = np.load(ROOT / "results" / "search_raw" / "t1" / f"mode_s{seed}.npz")
    f1, nf = d["f1"], d["nf"]
    ok = np.where(f1 >= f1.max() - BEST_WITHIN)[0]
    return d["x"][ok[np.argmin(nf[ok])]], seed


def main() -> None:
    if OUT_JSON.exists():
        raise SystemExit(f"{OUT_JSON.name} exists: the test split is scored once. Delete it deliberately to redo.")
    x_ea, ea_seed = ea_knee_vector()
    cands = {
        "BASE": decode(encode((0, 1, 0, 0), None, 0.0, "lr")),
        "FULL": decode(encode((1, 1, 1, 1), None, -1.0, "svc")),
        "EA": decode(x_ea),
    }
    ctx = SearchContext("t1", load_test=True)
    y = ctx.y_test
    preds = {k: ctx.predict(c, "test") for k, c in cands.items()}

    out: dict = {"n_test": int(len(y)), "ea_source": f"mode seed {ea_seed} (best knee dev)", "models": {}}
    for k, c in cands.items():
        point, lo, hi = bootstrap_ci(y, preds[k])
        out["models"][k] = {
            "n_features": ctx.n_features(c), "keep": c.keep, "weights": c.weights,
            "c": c.c, "classifier": c.model,
            "macro_f1": point, "ci95": [lo, hi],
            "f1_per_class": per_class_f1(y, preds[k]).tolist(),
        }
    out["comparisons"] = {}
    for a, b in (("FULL", "BASE"), ("EA", "BASE"), ("EA", "FULL")):
        diff, lo, hi, p = paired_bootstrap_diff(y, preds[a], preds[b])
        out["comparisons"][f"{a}_minus_{b}"] = {
            "macro_f1_diff": diff, "ci95": [lo, hi], "bootstrap_p": p,
            "mcnemar": {k: (v if not isinstance(v, np.generic) else v.item())
                        for k, v in mcnemar(y, preds[a], preds[b]).items()},
        }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    np.savez_compressed(OUT_NPZ, y=y, **preds)
    print(json.dumps({k: (v["macro_f1"], v["ci95"], v["n_features"]) for k, v in out["models"].items()}, indent=1))
    print(json.dumps({k: (v["macro_f1_diff"], v["ci95"], v["bootstrap_p"]) for k, v in out["comparisons"].items()}, indent=1))


if __name__ == "__main__":
    main()
