"""Fitness evaluation for the search: 5-fold stratified CV macro F1 on the training split.

Design notes (evaluation hygiene):
  * Only the training split is used here. Dev is touched only by ``SearchContext.dev_score``
    after a search has finished; test is never touched in this module.
  * Chi-square rankings are computed inside each CV fold (labels of the held-out fold are
    never used for selection). The TF-IDF vocabularies and IDF weights are fitted once on
    the full training split (unsupervised), which is a minor, documented simplification.
  * Block B4 is log-scaled here (raw counts reach 128 and make LinearSVC converge very
    slowly); the baselines in results/baselines_*.csv used the unscaled counts.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
from scipy import sparse
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from vispam.data import load_splits
from vispam.features import get_feature_block
from vispam.search.space import BLOCKS, Config, decode, n_features

LABEL_COL = {"t1": "label", "t2": "spam_type"}
B4_MAX_COUNT = 128.0


def _scale_b4(x: sparse.csr_matrix) -> sparse.csr_matrix:
    x = x.copy()
    x.data = np.minimum(np.log1p(x.data) / np.log1p(B4_MAX_COUNT), 1.0)
    return x


def _chi2_order(x: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, _ = chi2(x, y)
    return np.argsort(-np.nan_to_num(score), kind="stable")


class SearchContext:
    """Holds precomputed, rank-ordered CSC matrices so one evaluation is only slicing + fitting."""

    def __init__(self, task: str = "t1", n_folds: int = 5, fold_seed: int = 0,
                 verbose: bool = False, load_test: bool = False) -> None:
        # load_test=True only for the one-shot final evaluation (scripts/final_eval.py)
        t0 = time.time()
        self.task = task
        splits, _ = load_splits()
        col = LABEL_COL[task]
        train, dev = splits["train"], splits["dev"]
        self.y = train[col].to_numpy()
        self.y_dev = dev[col].to_numpy()
        texts, dev_texts = train["text"].tolist(), dev["text"].tolist()
        test_texts = splits["test"]["text"].tolist() if load_test else None
        self.y_test = splits["test"][col].to_numpy() if load_test else None

        full, dev_m, test_m = {}, {}, {}
        for name in BLOCKS:
            block = get_feature_block(name)
            xt = block.fit_transform(texts).tocsr()
            xd = block.transform(dev_texts).tocsr()
            xs = block.transform(test_texts).tocsr() if load_test else None
            if name == "b4":
                xt, xd = _scale_b4(xt), _scale_b4(xd)
                xs = _scale_b4(xs) if load_test else None
            full[name], dev_m[name], test_m[name] = xt, xd, xs
        self.block_dims = {b: full[b].shape[1] for b in BLOCKS}

        self.folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=fold_seed)
                          .split(texts, self.y))
        # per fold: rank-ordered CSC matrices for the fitting rows (a) and held-out rows (b)
        self._fold = []
        for a, b in self.folds:
            xa, xb = {}, {}
            for name in BLOCKS:
                order = _chi2_order(full[name][a], self.y[a])
                xa[name] = full[name][a][:, order].tocsc()
                xb[name] = full[name][b][:, order].tocsc()
            self._fold.append((xa, xb))
        # full-train ranking for the final dev scoring
        self._full_tr, self._full_dev, self._full_test = {}, {}, {}
        for name in BLOCKS:
            order = _chi2_order(full[name], self.y)
            self._full_tr[name] = full[name][:, order].tocsc()
            self._full_dev[name] = dev_m[name][:, order].tocsc()
            if load_test:
                self._full_test[name] = test_m[name][:, order].tocsc()
        if verbose:
            print(f"SearchContext({task}) ready in {time.time() - t0:.1f}s; dims={self.block_dims}")

    # ------------------------------------------------------------------ helpers
    def _assemble(self, cfg: Config, mats: dict[str, sparse.csc_matrix]) -> sparse.csr_matrix:
        parts = []
        for name, keep, w in zip(BLOCKS, cfg.keep, cfg.weights):
            if keep <= 0.0:
                continue
            k = max(1, int(round(keep * self.block_dims[name])))
            m = mats[name][:, :k]
            parts.append(m if w == 1.0 else m * w)
        return sparse.hstack(parts, format="csr")

    @staticmethod
    def _make_model(cfg: Config):
        if cfg.model == "svc":
            return LinearSVC(C=cfg.c, class_weight="balanced", tol=1e-2, max_iter=300, random_state=0)
        return LogisticRegression(C=cfg.c, class_weight="balanced", solver="liblinear", tol=1e-2,
                                  max_iter=100, random_state=0)

    # ------------------------------------------------------------------ scoring
    def cv_score(self, cfg: Config) -> float:
        """Mean macro F1 over the CV folds (the quantity the searches optimise)."""
        scores = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for (a, b), (xa, xb) in zip(self.folds, self._fold):
                model = self._make_model(cfg)
                model.fit(self._assemble(cfg, xa), self.y[a])
                pred = model.predict(self._assemble(cfg, xb))
                scores.append(f1_score(self.y[b], pred, average="macro"))
        return float(np.mean(scores))

    def dev_score(self, cfg: Config) -> float:
        """Macro F1 on dev of a model fitted on the full training split (post-search only)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._make_model(cfg)
            model.fit(self._assemble(cfg, self._full_tr), self.y)
            pred = model.predict(self._assemble(cfg, self._full_dev))
        return float(f1_score(self.y_dev, pred, average="macro"))

    def predict(self, cfg: Config, split: str = "dev") -> np.ndarray:
        """Fit on the full training split and predict ``split`` ("dev" or "test")."""
        mats = {"dev": self._full_dev, "test": self._full_test}[split]
        if not mats:
            raise RuntimeError("test split not loaded; create the context with load_test=True")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._make_model(cfg)
            model.fit(self._assemble(cfg, self._full_tr), self.y)
            return model.predict(self._assemble(cfg, mats))

    def n_features(self, cfg: Config) -> int:
        return n_features(cfg, self.block_dims)


class BudgetExhausted(Exception):
    pass


class BudgetedProblem:
    """Counts every proposal (cache hits included) against the evaluation budget.

    Objectives returned: (cv_macro_f1 to maximise, n_features to minimise). The full history
    of evaluated points is kept in ``archive`` so every algorithm, single- or
    multi-objective, is judged by the same non-dominated set of what it evaluated.
    """

    def __init__(self, score_fn: Callable[[Config], float], feat_fn: Callable[[Config], int],
                 budget: int) -> None:
        self.score_fn, self.feat_fn, self.budget = score_fn, feat_fn, budget
        self.n_calls = 0
        self.cache: dict = {}
        self.archive: list[tuple[np.ndarray, float, int]] = []
        self.eval_seconds = 0.0

    def __call__(self, x) -> tuple[float, int]:
        if self.n_calls >= self.budget:
            raise BudgetExhausted
        self.n_calls += 1
        cfg = decode(x)
        key = cfg.key()
        if key not in self.cache:
            t0 = time.time()
            self.cache[key] = (self.score_fn(cfg), self.feat_fn(cfg))
            self.eval_seconds += time.time() - t0
        f1, nf = self.cache[key]
        self.archive.append((np.clip(np.asarray(x, dtype=float), 0.0, 1.0), f1, nf))
        return f1, nf
