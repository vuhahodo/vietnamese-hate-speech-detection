"""Does the Claude-extracted feature (targets/threat/severity/llm_suggested_label)
actually improve macro-F1 over the existing classifier alone?

Combines the existing model's per-class probabilities (from backend/model/baseline.joblib,
or a PhoBERT checkpoint if MODEL_DIR is given) with the LLM-derived structured features from
llm_feature_extraction.py, trains a small secondary classifier on top, and reports macro-F1
with vs without the LLM features on a held-out split of the SAME sampled subset (can't compare
on rows without LLM features, since those were only extracted for the stratified sample).

Usage:
    python NLP/scripts/combine_llm_features.py --split dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent
DATA_DIR = FINAL_DIR / "data"
LABEL_NAMES = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}


def load_llm_features(split: str) -> pd.DataFrame:
    path = DATA_DIR / f"llm_features_{split}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "error" in rec:
                continue  # a handful of rows may have failed JSON parsing even after retries
            rows.append(rec)
    df = pd.DataFrame(rows)
    # Haiku spelled this key "explicit_threat_or_dehumatizing" (dropped the "n") in the
    # overwhelming majority of responses (1074/1084) despite the prompt spelling it correctly -
    # coalesce both variants into one canonical column rather than losing those rows to NaN.
    if "explicit_threat_or_dehumatizing" in df.columns:
        df["explicit_threat_or_dehumanizing"] = df["explicit_threat_or_dehumanizing"].fillna(
            df["explicit_threat_or_dehumatizing"]
        )
    df["llm_suggested_label_id"] = df["llm_suggested_label"].map(LABEL_IDS)

    before = len(df)
    df = df.dropna(subset=[
        "targets_individual_or_group", "explicit_threat_or_dehumanizing", "severity", "llm_suggested_label_id",
    ]).reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with still-missing fields after key-typo normalization.")
    return df


def get_baseline_probs(texts: list[str]) -> np.ndarray:
    bundle = joblib.load(FINAL_DIR / "backend" / "model" / "baseline.joblib")
    X = bundle["vectorizer"].transform(texts)
    return bundle["clf"].predict_proba(X)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    df = load_llm_features(args.split)
    print(f"Loaded {len(df)} rows with LLM features (label dist: {dict(df['true_label_id'].value_counts())})")

    model_probs = get_baseline_probs(df["free_text"].tolist())
    y = df["true_label_id"].values

    llm_feats = np.column_stack([
        df["targets_individual_or_group"].astype(int).values,
        df["explicit_threat_or_dehumanizing"].astype(int).values,
        df["severity"].astype(int).values,
        df["llm_suggested_label_id"].values,
    ])

    X_without_llm = model_probs
    X_with_llm = np.column_stack([model_probs, llm_feats])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def macro_f1_cv(X, y):
        clf = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
        preds = cross_val_predict(clf, X, y, cv=cv)
        return f1_score(y, preds, average="macro"), preds

    f1_without, preds_without = macro_f1_cv(X_without_llm, y)
    f1_with, preds_with = macro_f1_cv(X_with_llm, y)

    # also report the raw baseline-model-alone macro-F1 (argmax of its own probs, no re-ranker)
    f1_model_alone = f1_score(y, model_probs.argmax(axis=1), average="macro")
    # and the LLM's own suggested label alone, as a reference point
    f1_llm_alone = f1_score(y, df["llm_suggested_label_id"].values, average="macro")

    print("\n=== Macro-F1 comparison (5-fold CV, same held-out subset each time) ===")
    print(f"  baseline model alone (argmax, no re-ranker):      {f1_model_alone:.4f}")
    print(f"  re-ranker on model-probs only (no LLM features):  {f1_without:.4f}")
    print(f"  re-ranker on model-probs + LLM features:          {f1_with:.4f}")
    print(f"  LLM suggested_label alone (no model, reference):  {f1_llm_alone:.4f}")
    print(f"\n  Delta from adding LLM features: {f1_with - f1_without:+.4f}")

    print("\n=== classification_report: re-ranker WITHOUT LLM features ===")
    print(classification_report(y, preds_without, target_names=list(LABEL_NAMES.values()), digits=3))
    print("\n=== classification_report: re-ranker WITH LLM features ===")
    print(classification_report(y, preds_with, target_names=list(LABEL_NAMES.values()), digits=3))

    summary = {
        "n_rows": len(df),
        "macro_f1_model_alone": f1_model_alone,
        "macro_f1_reranker_without_llm": f1_without,
        "macro_f1_reranker_with_llm": f1_with,
        "macro_f1_llm_alone": f1_llm_alone,
        "delta_from_llm_features": f1_with - f1_without,
    }
    out_path = DATA_DIR / f"llm_feature_ablation_{args.split}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
