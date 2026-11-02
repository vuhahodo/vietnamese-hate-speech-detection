import numpy as np

from vispam.search.algorithms import (
    ALGORITHMS, nondominated_sort, pareto_mask, run_algorithm, survive,
)
from vispam.search.fitness import BudgetedProblem
from vispam.search.metrics import REF_POINT, hypervolume
from vispam.search.space import BLOCKS, DIM, decode, encode, n_features

DIMS = {"b1": 1000, "b2": 5000, "b3": 36, "b4": 80}


def test_decode_always_valid():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        cfg = decode(rng.random(DIM))
        assert any(k > 0 for k in cfg.keep)
        assert all(k == 0 or 0.05 <= k <= 1.0 for k in cfg.keep)
        assert all(0.25 <= w <= 4.0 for w in cfg.weights)
        assert 0.01 <= cfg.c <= 10.0
        assert cfg.model in ("svc", "lr")
        assert n_features(cfg, DIMS) >= 1


def test_decode_all_off_is_repaired_and_out_of_range_clipped():
    cfg = decode(np.array([0.0, 0.1, 0.05, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    assert cfg.active == ("b2",)
    cfg2 = decode(np.full(DIM, 7.0))
    assert cfg2.keep == (1.0,) * len(BLOCKS) and cfg2.model == "lr"


def test_encode_decode_roundtrip():
    cfg = decode(encode((0.0, 0.5, 1.0, 0.25), (1.0, 2.0, 1.0, 0.5), -1.0, "lr"))
    assert cfg.keep == (0.0, 0.5, 1.0, 0.25)
    assert cfg.weights == (1.0, 2.0, 1.0, 0.5)
    assert cfg.log_c == -1.0 and cfg.model == "lr"


def test_nondominated_sort_and_survival():
    f = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 1.0], [3.0, 4.0], [4.0, 5.0]])
    fronts = nondominated_sort(f)
    assert sorted(fronts[0].tolist()) == [0, 1, 2]
    assert pareto_mask(f).tolist() == [True, True, True, False, False]
    x = np.arange(10, dtype=float).reshape(5, 2)
    xs, fs = survive(x, f, 3)
    assert len(xs) == 3 and {tuple(r) for r in fs.tolist()} == {(1.0, 5.0), (2.0, 3.0), (3.0, 1.0)}


def test_hypervolume_known_value():
    pts = np.array([[-0.9, 3.0], [-0.8, 2.0]])
    ref = np.array([-0.5, 5.3])
    expected = (0.4 * (5.3 - 3.0)) + (0.3 * (3.0 - 2.0)) - 0.0  # staircase
    # staircase: sorted by f1 ascending: (-0.9,3.0) then (-0.8,2.0)
    expected = (ref[0] + 0.9) * (5.3 - 3.0) + (ref[0] + 0.8) * (3.0 - 2.0)
    assert abs(hypervolume(pts, ref) - expected) < 1e-12
    assert hypervolume(np.array([[-0.4, 3.0]]), ref) == 0.0
    assert hypervolume(pts[:1], REF_POINT) < hypervolume(pts, REF_POINT)


def _toy_problem(budget):
    # F1 rises with the kept fraction of b2 and b1 with diminishing returns; cost = #features.
    def score(cfg):
        k = dict(zip(BLOCKS, cfg.keep))
        return 0.6 + 0.2 * np.sqrt(k["b2"]) + 0.1 * np.sqrt(k["b1"]) + 0.02 * (k["b3"] > 0)

    return BudgetedProblem(score, lambda cfg: n_features(cfg, DIMS), budget)


def test_every_algorithm_respects_budget_and_is_reproducible():
    for name in ALGORITHMS:
        p1, p2 = _toy_problem(150), _toy_problem(150)
        run_algorithm(name, p1, seed=3)
        run_algorithm(name, p2, seed=3)
        assert p1.n_calls <= 150
        if name != "grid":
            assert p1.n_calls >= 100, name
        a1 = np.array([a[0] for a in p1.archive])
        a2 = np.array([a[0] for a in p2.archive])
        assert np.allclose(a1, a2), name
