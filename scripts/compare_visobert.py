"""Pair ViSoBERT test predictions with the classical models (T1). Writes results/final_test_t1_visobert.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from vispam.evaluate import bootstrap_ci, mcnemar, paired_bootstrap_diff, per_class_f1  # noqa: E402

out_path = ROOT / "results" / "final_test_t1_visobert.json"
c = np.load(ROOT / "results" / "final_test_t1_preds.npz")
p = np.load(ROOT / "results" / "visobert_test_preds_t1.npz")
y = c["y"]
assert (y == p["y"]).all(), "test label order differs"
pb = p["pred"]
point, lo, hi = bootstrap_ci(y, pb)
res = {"visobert": {"macro_f1": point, "ci95": [lo, hi], "f1_per_class": per_class_f1(y, pb).tolist(),
                   "ms_per_review_batched_cpu": float(p["ms_per_review_batched"])}, "vs": {}}
ph = np.load(ROOT / "results" / "phobert_test_preds_t1.npz")["pred"]
c = dict(c)
c["PHOBERT"] = ph
for k in ("EA", "FULL", "BASE", "PHOBERT"):
    d, l, h, bp = paired_bootstrap_diff(y, pb, c[k])
    m = mcnemar(y, pb, c[k])
    res["vs"][f"VISOBERT_minus_{k}"] = {"macro_f1_diff": d, "ci95": [l, h], "bootstrap_p": bp,
                                       "mcnemar": {a: (b.item() if isinstance(b, np.generic) else b) for a, b in m.items()}}
out_path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
print(json.dumps(res, indent=1, default=float))
