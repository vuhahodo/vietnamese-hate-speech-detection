"""Search space: decode a vector in [0, 1]^D into a valid pipeline configuration.

Genes (D = 10):
    0-3  keep ratio of blocks B1..B4 (below ``OFF_THRESHOLD`` the block is switched off)
    4-7  multiplicative weight of blocks B1..B4 (feature-block weights, the block-level
         analogue of the per-feature weights of Al-Zoubi et al., IEEE Access 2023)
    8    log10 of the regularisation constant C
    9    classifier: LinearSVC (< 0.5) or LogisticRegression (>= 0.5)

All decoded values are quantised so that the evaluation cache is effective and so that
every vector maps to a valid configuration (at least one block is always active).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLOCKS = ("b1", "b2", "b3", "b4")
N_BLOCKS = len(BLOCKS)
DIM = 10

OFF_THRESHOLD = 0.15
KEEP_LEVELS = 20  # keep fraction is quantised to multiples of 1/20
MIN_KEEP = 1.0 / KEEP_LEVELS
LOG_C_RANGE = (-2.0, 1.0)  # C in [0.01, 10]
WEIGHT_EXP_RANGE = 2.0  # block weight in [2^-2, 2^2]

GENE_NAMES = (
    "keep_b1", "keep_b2", "keep_b3", "keep_b4",
    "w_b1", "w_b2", "w_b3", "w_b4",
    "log_c", "model",
)


@dataclass(frozen=True)
class Config:
    """A decoded pipeline configuration (hashable, used as cache key)."""

    keep: tuple[float, ...]  # fraction of features kept per block, 0.0 = block off
    weights: tuple[float, ...]  # multiplicative weight per block
    log_c: float  # log10(C)
    model: str  # "svc" or "lr"

    @property
    def c(self) -> float:
        return float(10.0 ** self.log_c)

    @property
    def active(self) -> tuple[str, ...]:
        return tuple(b for b, k in zip(BLOCKS, self.keep) if k > 0.0)

    def key(self) -> tuple:
        return (self.keep, self.weights, self.log_c, self.model)


def _decode_keep(r: float) -> float:
    if r < OFF_THRESHOLD:
        return 0.0
    frac = (r - OFF_THRESHOLD) / (1.0 - OFF_THRESHOLD)
    frac = round(frac * KEEP_LEVELS) / KEEP_LEVELS
    return float(min(1.0, max(MIN_KEEP, frac)))


def decode(x) -> Config:
    """Map any vector in [0, 1]^D (values are clipped) to a valid ``Config``."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if x.shape != (DIM,):
        raise ValueError(f"expected a vector of length {DIM}, got {x.shape}")
    keep = [_decode_keep(v) for v in x[:N_BLOCKS]]
    if not any(k > 0.0 for k in keep):
        keep[int(np.argmax(x[:N_BLOCKS]))] = 0.5
    weights = []
    for v in x[N_BLOCKS:2 * N_BLOCKS]:
        e = round((v - 0.5) * 2.0 * WEIGHT_EXP_RANGE * 2.0) / 2.0  # step 0.5 in the exponent
        weights.append(float(2.0 ** e))
    lo, hi = LOG_C_RANGE
    log_c = round((lo + (hi - lo) * x[8]) * 4.0) / 4.0
    model = "svc" if x[9] < 0.5 else "lr"
    return Config(tuple(keep), tuple(weights), float(log_c), model)


def encode(keep, weights=None, log_c: float = 0.0, model: str = "svc") -> np.ndarray:
    """Inverse of ``decode`` for structured points (used by grid search and tests)."""
    x = np.zeros(DIM)
    for i, k in enumerate(keep):
        x[i] = 0.0 if k <= 0.0 else OFF_THRESHOLD + (1.0 - OFF_THRESHOLD) * k
    weights = weights if weights is not None else (1.0,) * N_BLOCKS
    for i, w in enumerate(weights):
        x[N_BLOCKS + i] = 0.5 + np.log2(w) / (2.0 * WEIGHT_EXP_RANGE)
    lo, hi = LOG_C_RANGE
    x[8] = (log_c - lo) / (hi - lo)
    x[9] = 0.25 if model == "svc" else 0.75
    return np.clip(x, 0.0, 1.0)


def n_features(cfg: Config, block_dims: dict[str, int]) -> int:
    """Number of columns the configuration feeds to the classifier."""
    total = 0
    for b, k in zip(BLOCKS, cfg.keep):
        if k > 0.0:
            total += max(1, int(round(k * block_dims[b])))
    return total
