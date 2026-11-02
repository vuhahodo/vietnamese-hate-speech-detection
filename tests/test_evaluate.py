"""Tests for vispam.evaluate module."""
from __future__ import annotations

import numpy as np
import pytest

from vispam.evaluate import (
    bootstrap_ci,
    confusion,
    latency_ms,
    macro_f1,
    mcnemar,
    paired_bootstrap_diff,
    per_class_f1,
)


def test_macro_and_per_class_f1():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    # Class 0: TP=1, FP=0, FN=1 -> F1 = 2*1/(2*1 + 0 + 1) = 2/3 = 0.6667
    # Class 1: TP=2, FP=1, FN=0 -> F1 = 2*2/(2*2 + 1 + 0) = 4/5 = 0.8
    # Macro F1 = (2/3 + 0.8) / 2 = 1.4667 / 2 = 0.7333
    scores = per_class_f1(y_true, y_pred)
    assert len(scores) == 2
    assert np.isclose(scores[0], 2.0 / 3.0)
    assert np.isclose(scores[1], 0.8)
    assert np.isclose(macro_f1(y_true, y_pred), (2.0 / 3.0 + 0.8) / 2.0)


def test_confusion_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    mat = confusion(y_true, y_pred)
    assert mat.shape == (2, 2)
    assert mat[0, 0] == 1
    assert mat[0, 1] == 1
    assert mat[1, 0] == 0
    assert mat[1, 1] == 2


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.integers(0, 2, size=100)

    point, low, high = bootstrap_ci(y_true, y_pred, n=200, seed=42)
    assert low <= point <= high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_paired_bootstrap_diff():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    pred_a = rng.integers(0, 2, size=100)
    # Identical predictions should have diff = 0
    diff, low, high, p_val = paired_bootstrap_diff(y_true, pred_a, pred_a, n=100, seed=42)
    assert np.isclose(diff, 0.0)
    assert np.isclose(low, 0.0)
    assert np.isclose(high, 0.0)
    assert np.isclose(p_val, 1.0)


def test_mcnemar_exact_small_n():
    # Construct case where b=3, c=7 -> n_disc = 10 < 25 -> exact test
    # y_true = all 1s
    # Model A correct on 3 where Model B wrong (b=3)
    # Model B correct on 7 where Model A wrong (c=7)
    y_true = np.ones(20, dtype=int)
    pred_a = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    pred_b = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    res = mcnemar(y_true, pred_a, pred_b)
    assert res["test"] == "exact"
    assert res["b"] == 3
    assert res["c"] == 7
    # Binomial test with k=3, n=10, p=0.5: two-sided p-value is 2 * binom.cdf(3, 10, 0.5) = 2 * 0.171875 = 0.34375
    assert np.isclose(res["p_value"], 0.34375)


def test_mcnemar_chi2_large_n():
    # Construct case where b=10, c=30 -> n_disc = 40 >= 25 -> chi2 test
    y_true = np.ones(50, dtype=int)
    pred_a = np.zeros(50, dtype=int)
    pred_b = np.zeros(50, dtype=int)

    # b = 10: a correct, b wrong
    pred_a[:10] = 1
    pred_b[:10] = 0

    # c = 30: a wrong, b correct
    pred_a[10:40] = 0
    pred_b[10:40] = 1

    res = mcnemar(y_true, pred_a, pred_b)
    assert res["test"] == "chi2"
    assert res["b"] == 10
    assert res["c"] == 30
    # stat = (|10 - 30| - 1)^2 / 40 = 19^2 / 40 = 361 / 40 = 9.025
    assert np.isclose(res["statistic"], 9.025)
    assert res["p_value"] < 0.01


def test_mcnemar_zero_discordant():
    y_true = [0, 1, 0, 1]
    pred = [0, 1, 0, 1]
    res = mcnemar(y_true, pred, pred)
    assert res["b"] == 0
    assert res["c"] == 0
    assert res["p_value"] == 1.0


def test_latency_ms():
    def dummy_predict(batch):
        return [0] * len(batch)

    texts = ["đây là review số một", "review số hai"]
    lat = latency_ms(dummy_predict, texts, repeats=2)
    assert isinstance(lat, float)
    assert lat >= 0.0

