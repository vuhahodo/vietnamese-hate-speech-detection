"""Search-quality metrics computed from an archive of evaluated points."""

from __future__ import annotations

import numpy as np

from vispam.search.algorithms import objectives, pareto_mask

# Reference point in minimisation space (-macro_F1, log10 n_features). Chosen once, before any
# result was seen: F1 0.5 is far below any sensible model, 10^5.3 > all 149k features.
REF_POINT = np.array([-0.5, 5.3])


def hypervolume(points: np.ndarray, ref: np.ndarray = REF_POINT) -> float:
    """Exact 2-D hypervolume (area dominated by ``points`` and bounded by ``ref``)."""
    pts = points[(points < ref).all(1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[pareto_mask(pts)]
    pts = pts[np.argsort(pts[:, 0])]  # ascending in f1 => descending in f2 on a front
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in pts:
        hv += (ref[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return float(hv)


def archive_objectives(archive) -> np.ndarray:
    return np.array([objectives(f1, nf) for _, f1, nf in archive])


def hv_trace(archive, step: int = 50) -> list[tuple[int, float]]:
    """Hypervolume of the archive after every ``step`` evaluations."""
    f = archive_objectives(archive)
    out = [(n, hypervolume(f[:n])) for n in range(step, len(f) + 1, step)]
    if not out or out[-1][0] != len(f):
        out.append((len(f), hypervolume(f)))
    return out


def front_indices(archive) -> np.ndarray:
    """Indices of unique non-dominated archive entries (duplicates by objective vector removed)."""
    f = archive_objectives(archive)
    _, first = np.unique(f, axis=0, return_index=True)
    idx = np.sort(first)
    return idx[pareto_mask(f[idx])]
