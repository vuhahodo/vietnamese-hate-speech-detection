"""Stack the 4 trained models (TF-IDF+LogReg baseline, PhoBERT class-weighted CE,
PhoBERT focal loss, XLM-R) into one meta-classifier, and separately measure whether
adding the LLM-extracted structured features (see llm_feature_extraction.py) helps
on top of that -- both validated on the real held-out `test.csv`, not just dev CV
(the earlier Haiku experiment in combine_llm_features.py only ever validated on dev).

Inputs (produced by the "Ensemble artifacts" cell appended to
NLP/notebook/vihsd-phobert.ipynb, downloaded via `kaggle kernels output`):
    <probs-dir>/probs_{baseline,phobert_ce,phobert_focal,xlmr}_{dev,test}.csv
    each with columns: p_CLEAN, p_OFFENSIVE, p_HATE, label_id, text_clean
    (row order == the original dev.csv/test.csv row order)

Optional LLM features (see llm_feature_extraction.py), joined by row_index:
    NLP/data/llm_features{_<slug>}_{dev,test}.jsonl

Usage:
    python NLP/scripts/build_ensemble.py --probs-dir NLP/data/ensemble_probs
    python NLP/scripts/build_ensemble.py --probs-dir NLP/data/ensemble_probs --llm-slug sonnet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent
DATA_DIR = FINAL_DIR / "data"

MODEL_NAMES = ["baseline", "phobert_ce", "phobert_focal", "xlmr"]
LABEL_NAMES = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}
PROB_COLS = ["p_CLEAN", "p_OFFENSIVE", "p_HATE"]


def load_model_probs(probs_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Returns (X_ensemble, y, per_model_probs) for one split, asserting all 4 model
    CSVs agree on row count and true label per row (they must all originate from the
    same dev_df/test_df, in the same order, from the same notebook run)."""
    per_model: dict[str, np.ndarray] = {}
    y_ref = None
    for name in MODEL_NAMES:
        path = probs_dir / f"probs_{name}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} -- download kernel output first "
                f"(kaggle kernels output hongnhungnguyenthi/vihsd-phobert-hate-speech)"
            )
        df = pd.read_csv(path)
        per_model[name] = df[PROB_COLS].values
        if y_ref is None:
            y_ref = df["label_id"].values
        elif not np.array_equal(y_ref, df["label_id"].values):
            raise ValueError(f"{path} label_id column doesn't match the other models -- row order mismatch?")
    X = np.column_stack([per_model[name] for name in MODEL_NAMES])
    return X, y_ref, per_model


def sample_original_indices(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Reproduces llm_feature_extraction.py's build_sample() row selection/shuffle exactly,
    but WITHOUT its final `.reset_index(drop=True)` -- so the returned array maps
    row_index (0..N-1, as stored in the llm_features_*.jsonl) back to the original
    dev.csv/test.csv row position. Must stay in lockstep with build_sample() there."""
    rare = df[df["label_id"].isin([1, 2])]
    clean_pool = df[df["label_id"] == 0]
    n_clean = min(len(clean_pool), int(len(rare) * 1.25))
    clean_sample = clean_pool.sample(n=n_clean, random_state=seed)
    sample = pd.concat([rare, clean_sample]).sample(frac=1, random_state=seed)
    return sample.index.values


def load_llm_features(slug: str, split: str) -> pd.DataFrame | None:
    suffix = f"_{slug}" if slug else ""
    path = DATA_DIR / f"llm_features{suffix}_{split}.jsonl"
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "error" in rec:
                continue
            rows.append(rec)
    df = pd.DataFrame(rows)
    # Haiku's original run misspelled this key in most responses -- coalesce both
    # spellings so older `llm_features_dev.jsonl` (no slug) still loads cleanly.
    if "explicit_threat_or_dehumatizing" in df.columns:
        df["explicit_threat_or_dehumanizing"] = df["explicit_threat_or_dehumanizing"].fillna(
            df["explicit_threat_or_dehumatizing"]
        )
    df["llm_suggested_label_id"] = df["llm_suggested_label"].map(LABEL_IDS)
    df = df.dropna(subset=[
        "targets_individual_or_group", "explicit_threat_or_dehumanizing", "severity", "llm_suggested_label_id",
    ]).reset_index(drop=True)
    return df


def macro_f1_cv(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[float, np.ndarray]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=3)
    preds = cross_val_predict(clf, X, y, cv=cv)
    return f1_score(y, preds, average="macro"), preds


def train_eval(X_train, y_train, X_test, y_test, seed: int = 42):
    clf = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=3)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return f1_score(y_test, preds, average="macro"), preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs-dir", default=str(DATA_DIR / "ensemble_probs"))
    parser.add_argument("--llm-slug", default="sonnet", help="filename slug used by llm_feature_extraction.py --model (empty string = original Haiku dev-only run)")
    args = parser.parse_args()
    probs_dir = Path(args.probs_dir)

    print("=" * 70)
    print("1) Reference point: XLM-R alone (argmax), full dev + full test")
    print("=" * 70)
    X_dev, y_dev, per_model_dev = load_model_probs(probs_dir, "dev")
    X_test, y_test, per_model_test = load_model_probs(probs_dir, "test")

    xlmr_dev_f1 = f1_score(y_dev, per_model_dev["xlmr"].argmax(axis=1), average="macro")
    xlmr_test_f1 = f1_score(y_test, per_model_test["xlmr"].argmax(axis=1), average="macro")
    print(f"  XLM-R alone -- dev macro-F1:  {xlmr_dev_f1:.4f}")
    print(f"  XLM-R alone -- test macro-F1: {xlmr_test_f1:.4f}")

    print("\n" + "=" * 70)
    print("2) 4-model ensemble (baseline + PhoBERT-CE + PhoBERT-focal + XLM-R)")
    print("=" * 70)
    ensemble_cv_f1, _ = macro_f1_cv(X_dev, y_dev)
    print(f"  Ensemble -- 5-fold CV on dev:          {ensemble_cv_f1:.4f}")
    ensemble_test_f1, ensemble_test_preds = train_eval(X_dev, y_dev, X_test, y_test)
    print(f"  Ensemble -- trained on full dev, evaluated on FULL TEST (real held-out): {ensemble_test_f1:.4f}")
    print(f"  Delta vs XLM-R alone on test: {ensemble_test_f1 - xlmr_test_f1:+.4f}")
    print(classification_report(y_test, ensemble_test_preds, target_names=list(LABEL_NAMES.values()), digits=3))

    result = {
        "xlmr_alone_dev_f1": xlmr_dev_f1,
        "xlmr_alone_test_f1": xlmr_test_f1,
        "ensemble_cv_dev_f1": ensemble_cv_f1,
        "ensemble_test_f1": ensemble_test_f1,
    }

    llm_dev = load_llm_features(args.llm_slug, "dev")
    llm_test = load_llm_features(args.llm_slug, "test")
    if llm_dev is not None and llm_test is not None:
        print("\n" + "=" * 70)
        print(f"3) + LLM features (slug={args.llm_slug!r}) -- only on the sampled subset that has them")
        print("=" * 70)

        def llm_feat_matrix(df: pd.DataFrame) -> np.ndarray:
            return np.column_stack([
                df["targets_individual_or_group"].astype(int).values,
                df["explicit_threat_or_dehumanizing"].astype(int).values,
                df["severity"].astype(int).values,
                df["llm_suggested_label_id"].values,
            ])

        dev_orig_idx = sample_original_indices(pd.read_csv(DATA_DIR / "dev.csv"))
        test_orig_idx = sample_original_indices(pd.read_csv(DATA_DIR / "test.csv"))
        dev_idx = dev_orig_idx[llm_dev["row_index"].values]
        test_idx = test_orig_idx[llm_test["row_index"].values]

        X_dev_sub = X_dev[dev_idx]
        X_test_sub = X_test[test_idx]
        y_dev_sub = y_dev[dev_idx]
        y_test_sub = y_test[test_idx]

        assert np.array_equal(y_dev_sub, llm_dev["true_label_id"].values), "dev row_index misaligned with llm_features"
        assert np.array_equal(y_test_sub, llm_test["true_label_id"].values), "test row_index misaligned with llm_features"

        X_dev_sub_llm = np.column_stack([X_dev_sub, llm_feat_matrix(llm_dev)])
        X_test_sub_llm = np.column_stack([X_test_sub, llm_feat_matrix(llm_test)])

        ensemble_sub_f1, _ = train_eval(X_dev_sub, y_dev_sub, X_test_sub, y_test_sub)
        ensemble_llm_f1, ensemble_llm_preds = train_eval(X_dev_sub_llm, y_dev_sub, X_test_sub_llm, y_test_sub)

        print(f"  Ensemble alone, same subset -- test macro-F1: {ensemble_sub_f1:.4f}")
        print(f"  Ensemble + LLM features    -- test macro-F1: {ensemble_llm_f1:.4f}")
        print(f"  Delta from adding LLM features: {ensemble_llm_f1 - ensemble_sub_f1:+.4f}")
        print(classification_report(y_test_sub, ensemble_llm_preds, target_names=list(LABEL_NAMES.values()), digits=3))

        result.update({
            "llm_subset_n_dev": len(dev_idx),
            "llm_subset_n_test": len(test_idx),
            "ensemble_subset_test_f1": ensemble_sub_f1,
            "ensemble_plus_llm_test_f1": ensemble_llm_f1,
            "delta_from_llm_features": ensemble_llm_f1 - ensemble_sub_f1,
        })
    else:
        print(f"\n(Skipping LLM-feature comparison -- llm_features{'_' + args.llm_slug if args.llm_slug else ''}_{{dev,test}}.jsonl not found yet.)")

    out_path = DATA_DIR / "ensemble_ablation_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
