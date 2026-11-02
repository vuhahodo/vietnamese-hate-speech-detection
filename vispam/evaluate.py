"""Evaluation protocol for spam classification tasks.

Implements macro F1, per-class F1, confusion matrix, percentile bootstrap CI,
paired bootstrap difference, McNemar test with exact/chi2 dispatch,
and CPU latency measurement.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import confusion_matrix, f1_score


def macro_f1(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> float:
    """Return macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_f1(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Return F1 score for each class."""
    return f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)


def confusion(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Return confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def bootstrap_ci(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = macro_f1,
    n: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute point estimate and percentile bootstrap confidence interval.

    Returns:
        (point_estimate, low, high)
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: {len(yt)} vs {len(yp)}")
    if len(yt) == 0:
        raise ValueError("Empty input sequences")

    point = float(metric(yt, yp))
    rng = np.random.default_rng(seed)
    n_samples = len(yt)
    scores = np.empty(n, dtype=float)

    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        scores[i] = metric(yt[idx], yp[idx])

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(scores, alpha * 100))
    high = float(np.percentile(scores, (1.0 - alpha) * 100))
    return point, low, high


def paired_bootstrap_diff(
    y_true: Sequence[int] | np.ndarray,
    pred_a: Sequence[int] | np.ndarray,
    pred_b: Sequence[int] | np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = macro_f1,
    n: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float, float]:
    """Compute paired bootstrap difference (model A - model B).

    Returns:
        (diff_point, low, high, p_value)
    """
    yt = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)
    if not (len(yt) == len(pa) == len(pb)):
        raise ValueError("All input arrays must have the same length")
    if len(yt) == 0:
        raise ValueError("Empty input sequences")

    diff_point = float(metric(yt, pa) - metric(yt, pb))
    rng = np.random.default_rng(seed)
    n_samples = len(yt)
    diffs = np.empty(n, dtype=float)

    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        diffs[i] = metric(yt[idx], pa[idx]) - metric(yt[idx], pb[idx])

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(diffs, alpha * 100))
    high = float(np.percentile(diffs, (1.0 - alpha) * 100))

    # Two-sided empirical p-value relative to 0
    p_val = float(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    p_val = min(1.0, max(0.0, p_val))
    return diff_point, low, high, p_val


def mcnemar(
    y_true: Sequence[int] | np.ndarray,
    pred_a: Sequence[int] | np.ndarray,
    pred_b: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Perform McNemar's test on predictions of model A vs model B against ground truth.

    Uses exact binomial test when discordant pairs (b + c) < 25;
    otherwise uses chi-square test with Edwards' continuity correction.

    Returns:
        {"statistic": float, "p_value": float, "b": int, "c": int, "test": "exact" | "chi2"}
    """
    yt = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)
    if not (len(yt) == len(pa) == len(pb)):
        raise ValueError("Input arrays must have equal lengths")

    correct_a = (pa == yt)
    correct_b = (pb == yt)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_disc = b + c

    if n_disc == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "test": "exact"}

    if n_disc < 25:
        # Exact two-sided binomial test with p = 0.5
        res = binomtest(b, n=n_disc, p=0.5, alternative="two-sided")
        return {
            "statistic": float(abs(b - c)),
            "p_value": float(res.pvalue),
            "b": b,
            "c": c,
            "test": "exact",
        }
    else:
        # Chi-square with Edwards' continuity correction
        num = max(0.0, abs(b - c) - 1.0) ** 2
        stat = float(num / n_disc)
        p_val = float(chi2.sf(stat, df=1))
        return {
            "statistic": stat,
            "p_value": p_val,
            "b": b,
            "c": c,
            "test": "chi2",
        }


def latency_ms(
    predict_fn: Callable[[Any], Any],
    texts: Any,
    repeats: int = 3,
) -> float:
    """Measure mean prediction latency per review in milliseconds on CPU.

    Warms up first, then runs over repeated trials.
    Supports list/sequence of texts or sparse feature matrices.
    """
    n_items = texts.shape[0] if hasattr(texts, "shape") else len(texts)
    if n_items == 0:
        return 0.0

    # Warmup run (discarded)
    warmup_n = min(n_items, 50)
    predict_fn(texts[:warmup_n])

    total_time = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(texts)
        total_time += time.perf_counter() - t0

    return float((total_time / repeats / n_items) * 1000.0)
