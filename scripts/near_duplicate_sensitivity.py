"""Near-duplicate sensitivity analysis on test set (T5).

Evaluates:
- Maximum cosine similarity between each test sample and all train samples in B2 TF-IDF space.
- Number of test samples with max cosine >= 0.90 (compared with ~71/3971 in VeriReview).
- Cosine similarity distribution across percentiles.
- Macro F1 of T4.6 baseline (LR + B2) on all test vs test with near-duplicates removed (mask cosine < 0.90).
- 95% bootstrap confidence intervals for both.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vispam.data import load_splits
from vispam.evaluate import bootstrap_ci, macro_f1, per_class_f1
from vispam.features import get_feature_block

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_max_cosine_by_chunks(
    X_test: sparse.csr_matrix,
    X_train: sparse.csr_matrix,
    chunk_size: int = 250,
) -> np.ndarray:
    """Compute maximum cosine similarity to train set for each test sample.

    Both X_test and X_train are assumed to be L2-normalized CSR matrices.
    Processes in chunks to maintain low peak memory usage.
    """
    n_test = X_test.shape[0]
    max_sims = np.empty(n_test, dtype=np.float32)

    # Transpose train once
    X_train_t = X_train.T.tocsc()

    for start in range(0, n_test, chunk_size):
        end = min(n_test, start + chunk_size)
        sub_test = X_test[start:end]
        # (chunk_size, n_train) sparse dot product
        sims = sub_test.dot(X_train_t)
        # Take max along columns (axis 1)
        chunk_max = sims.max(axis=1).toarray().flatten()
        max_sims[start:end] = chunk_max

    return max_sims


def main() -> None:
    t0 = time.time()
    splits, audit = load_splits()

    train_texts = splits["train"]["text"].tolist()
    y_train = splits["train"]["label"].to_numpy()

    test_texts = splits["test"]["text"].tolist()
    y_test = splits["test"]["label"].to_numpy()

    n_test = len(test_texts)
    n_train = len(train_texts)

    print("==================== NEAR-DUPLICATE SENSITIVITY ANALYSIS (T5) ====================")
    print(f"Train samples: {n_train}, Test samples: {n_test}")

    b2 = get_feature_block("b2")
    X_train = b2.fit_transform(train_texts)
    X_test = b2.transform(test_texts)

    print(f"B2 TF-IDF shape: train {X_train.shape}, test {X_test.shape}")
    print("Computing max cosine similarity in chunks...")
    max_cosines = compute_max_cosine_by_chunks(X_test, X_train, chunk_size=250)

    # Threshold analysis
    threshold = 0.90
    near_dups_mask = max_cosines >= threshold
    count_near_dups = int(np.sum(near_dups_mask))
    kept_mask = ~near_dups_mask
    count_kept = int(np.sum(kept_mask))

    print(f"\nTest samples with max cosine >= {threshold:.2f}: {count_near_dups}/{n_test} ({count_near_dups / n_test * 100:.2f}%)")

    # Percentiles
    percentiles_to_calc = [10, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = {
        f"p{p}": round(float(np.percentile(max_cosines, p)), 4)
        for p in percentiles_to_calc
    }
    print(f"Cosine percentiles: {percentile_values}")

    # Re-evaluate T4.6 baseline (LR with B2)
    print("\nTraining T4.6 baseline (LR + B2, balanced, C=1.0) on full train set...")
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    preds_all = clf.predict(X_test)

    # 1. Macro F1 on ALL test samples
    mf1_all = macro_f1(y_test, preds_all)
    pcf1_all = per_class_f1(y_test, preds_all)
    _, low_all, high_all = bootstrap_ci(y_test, preds_all, n=1000, seed=42)

    # 2. Macro F1 on FILTERED test samples (removing near-duplicates >= 0.90)
    preds_kept = preds_all[kept_mask]
    y_test_kept = y_test[kept_mask]

    mf1_filtered = macro_f1(y_test_kept, preds_kept)
    pcf1_filtered = per_class_f1(y_test_kept, preds_kept)
    _, low_filt, high_filt = bootstrap_ci(y_test_kept, preds_kept, n=1000, seed=42)

    delta_mf1 = mf1_filtered - mf1_all

    print(f"\n--- EVALUATION RESULTS ---")
    print(f"All Test ({n_test} samples):")
    print(f"  Macro F1: {mf1_all:.4f} [95% CI: {low_all:.4f}, {high_all:.4f}]")
    print(f"  Per class F1: [non-spam: {pcf1_all[0]:.4f}, spam: {pcf1_all[1]:.4f}]")

    print(f"\nFiltered Test ({count_kept} samples, {count_near_dups} dropped):")
    print(f"  Macro F1: {mf1_filtered:.4f} [95% CI: {low_filt:.4f}, {high_filt:.4f}]")
    print(f"  Per class F1: [non-spam: {pcf1_filtered[0]:.4f}, spam: {pcf1_filtered[1]:.4f}]")
    print(f"Delta (Filtered - All): {delta_mf1:+.4f}")

    duration = time.time() - t0

    result_data = {
        "analysis": "near_duplicate_sensitivity",
        "threshold": threshold,
        "test_total": n_test,
        "near_duplicate_count": count_near_dups,
        "near_duplicate_ratio": round(count_near_dups / n_test, 4),
        "kept_test_count": count_kept,
        "expected_near_duplicates_verireview": "~71/3971",
        "cosine_distribution_percentiles": percentile_values,
        "baseline_model": "LogisticRegression(C=1.0, class_weight='balanced')",
        "all_test": {
            "sample_count": n_test,
            "macro_f1": round(mf1_all, 4),
            "f1_non_spam": round(float(pcf1_all[0]), 4),
            "f1_spam": round(float(pcf1_all[1]), 4),
            "bootstrap_ci_95": [round(low_all, 4), round(high_all, 4)],
        },
        "filtered_test": {
            "sample_count": count_kept,
            "macro_f1": round(mf1_filtered, 4),
            "f1_non_spam": round(float(pcf1_filtered[0]), 4),
            "f1_spam": round(float(pcf1_filtered[1]), 4),
            "bootstrap_ci_95": [round(low_filt, 4), round(high_filt, 4)],
        },
        "delta_macro_f1": round(delta_mf1, 4),
        "execution_seconds": round(duration, 2),
    }

    out_file = RESULTS_DIR / "near_duplicate_sensitivity.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(f"\nResults successfully written to {out_file} (elapsed: {duration:.2f}s)")


if __name__ == "__main__":
    main()

