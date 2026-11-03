"""End-to-end spam pipeline (raw text -> score) for a decoded search configuration.

It reproduces ``SearchContext.predict`` exactly (same blocks, B4 log-scaling, chi-square
ranking on the training split, block weights, classifier), but starts from raw strings so it
can be pickled, timed end to end and shipped.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from vispam.features import get_feature_block
from vispam.search.fitness import SearchContext, _chi2_order, _scale_b4
from vispam.search.space import BLOCKS, Config


class SpamPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._parts: list[tuple] = []  # (block, kept columns, weight, name)
        self.model = None

    def fit(self, texts, y) -> "SpamPipeline":
        y = np.asarray(y)
        self._parts = []
        for name, keep, w in zip(BLOCKS, self.cfg.keep, self.cfg.weights):
            if keep <= 0.0:
                continue
            block = get_feature_block(name, cache_dir=None)
            x = block.fit_transform(texts).tocsr()
            if name == "b4":
                x = _scale_b4(x)
            k = max(1, int(round(keep * x.shape[1])))
            cols = _chi2_order(x, y)[:k]
            self._parts.append((block, cols, w, name))
        self.model = SearchContext._make_model(self.cfg)
        self.model.fit(self._features(texts), y)
        return self

    def _features(self, texts) -> sparse.csr_matrix:
        mats = []
        for block, cols, w, name in self._parts:
            x = block.transform(texts).tocsr()
            if name == "b4":
                x = _scale_b4(x)
            x = x[:, cols]
            mats.append(x if w == 1.0 else x * w)
        return sparse.hstack(mats, format="csr")

    def score(self, texts) -> np.ndarray:
        """Signed margin; larger means more likely spam (class 1)."""
        return self.model.decision_function(self._features(texts))

    def predict(self, texts) -> np.ndarray:
        return self.model.predict(self._features(texts))

    @property
    def n_features(self) -> int:
        return int(sum(len(c) for _, c, _, _ in self._parts))
