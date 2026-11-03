"""Ensemble of PhoBERT and ViSoBERT probabilities for T1: weight and threshold chosen on dev, scored once on test.

Dev probabilities are computed here (models loaded from local directories); test probabilities come from
results/phobert_test_preds_t1.npz and results/visobert_test_preds_t1.npz. Test is read once, for the fixed (weight,
threshold) pair selected on dev. Writes results/dev_probs_t1.npz and results/final_test_t1_ensemble.json.

    python scripts/ensemble_text_models.py --phobert <dir> --visobert <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from vispam.data import load_splits  # noqa: E402
from vispam.evaluate import bootstrap_ci, macro_f1, mcnemar, paired_bootstrap_diff  # noqa: E402


def probs(model_dir, texts, segment: bool, batch: int = 32) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=not segment)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    if segment:
        from underthesea import word_tokenize
        texts = [word_tokenize(t, format="text") for t in texts]
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phobert", type=Path, required=True)
    ap.add_argument("--visobert", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    out_json = ROOT / "results" / "final_test_t1_ensemble.json"
    if out_json.exists():
        raise SystemExit("final_test_t1_ensemble.json exists: test is scored once; delete deliberately to redo")
    dev = load_splits()[0]["dev"]
    y_dev = dev["label"].to_numpy()
    dev_npz = ROOT / "results" / "dev_probs_t1.npz"
    if dev_npz.exists():
        z = np.load(dev_npz)
        p_ph, p_vi = z["phobert"], z["visobert"]
    else:
        p_ph = probs(a.phobert, dev["text"].tolist(), segment=True)
        p_vi = probs(a.visobert, dev["text"].tolist(), segment=False)
        np.savez_compressed(dev_npz, y=y_dev, phobert=p_ph, visobert=p_vi)
    best = None
    for w in np.linspace(0, 1, 11):
        for th in np.linspace(0.2, 0.8, 61):
            f = macro_f1(y_dev, ((w * p_ph + (1 - w) * p_vi) >= th).astype(int))
            if best is None or f > best[0] + 1e-12:
                best = (f, float(w), float(th))
    dev_f1, w, th = best
    single = {"phobert@0.5": macro_f1(y_dev, (p_ph >= 0.5).astype(int)), "visobert@0.5": macro_f1(y_dev, (p_vi >= 0.5).astype(int)),
              "avg@0.5": macro_f1(y_dev, ((p_ph + p_vi) / 2 >= 0.5).astype(int))}
    t_ph = np.load(ROOT / "results" / "phobert_test_preds_t1.npz")
    t_vi = np.load(ROOT / "results" / "visobert_test_preds_t1.npz")
    assert (t_ph["y"] == t_vi["y"]).all()
    y = t_ph["y"]
    pred = ((w * t_ph["prob"] + (1 - w) * t_vi["prob"]) >= th).astype(int)
    point, lo, hi = bootstrap_ci(y, pred)
    res = {"selected_on_dev": {"weight_phobert": w, "threshold": th, "dev_macro_f1": dev_f1, "reference_dev": single},
           "test": {"macro_f1": point, "ci95": [lo, hi], "n": int(len(y))}, "vs": {}}
    for name, other in (("PHOBERT", t_ph["pred"]), ("VISOBERT", t_vi["pred"])):
        d, l, h, bp = paired_bootstrap_diff(y, pred, other)
        res["vs"][f"ENSEMBLE_minus_{name}"] = {"macro_f1_diff": d, "ci95": [l, h], "bootstrap_p": bp,
                                              "mcnemar": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in mcnemar(y, pred, other).items()}}
    np.savez_compressed(ROOT / "results" / "ensemble_test_preds_t1.npz", y=y, pred=pred,
                        prob=w * t_ph["prob"] + (1 - w) * t_vi["prob"])
    out_json.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
