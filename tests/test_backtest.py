"""Tests for the back-test metric functions (no model fitting required)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mpp.backtest import (
    _calibration_table,
    _fit_elo_multinomial,
    _predict_elo_multinomial,
    brier_3way,
    compare_predictors,
    holm,
    log_loss_3way,
    outcome_index,
    paired_bootstrap,
    per_match_losses,
    rps_3way,
    summarise,
)


def test_outcome_index():
    assert outcome_index(2, 0) == 0  # home win
    assert outcome_index(1, 1) == 1  # draw
    assert outcome_index(0, 3) == 2  # away win


def test_log_loss_perfect_vs_uniform():
    # Perfect, confident predictions → log-loss ≈ 0.
    probs = np.array([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005]])
    outcomes = np.array([0, 1])
    assert log_loss_3way(probs, outcomes) < 0.01
    # Uniform predictions → log-loss = ln(3).
    uniform = np.full((2, 3), 1 / 3)
    assert abs(log_loss_3way(uniform, outcomes) - np.log(3)) < 1e-9


def test_brier_bounds():
    onehot = np.array([[1.0, 0.0, 0.0]])
    assert brier_3way(onehot, np.array([0])) == 0.0
    worst = np.array([[0.0, 0.0, 1.0]])
    assert abs(brier_3way(worst, np.array([0])) - 2.0) < 1e-9


def test_rps_rewards_ordinal_closeness():
    # True outcome = away win (2). Predicting the draw (adjacent) should beat
    # predicting a home win (far), even though both are wrong as argmax.
    outcomes = np.array([2])
    near = np.array([[0.0, 1.0, 0.0]])  # all mass on draw
    far = np.array([[1.0, 0.0, 0.0]])   # all mass on home
    assert rps_3way(near, outcomes) < rps_3way(far, outcomes)


def test_summarise_keys():
    probs = np.full((3, 3), 1 / 3)
    out = summarise(probs, np.array([0, 1, 2]))
    assert set(out) >= {"n", "log_loss", "brier", "rps", "accuracy"}
    assert out["n"] == 3


def test_elo_multinomial_monotonic():
    # Synthetic: big positive rating diff → home wins; big negative → away wins.
    rng = np.random.default_rng(0)
    dr = rng.normal(0, 200, 2000)
    # Probabilistic outcomes driven by dr.
    p_home = 1 / (1 + np.exp(-dr / 150))
    outcomes = np.where(rng.random(2000) < p_home, 0, 2)
    weights = np.ones(2000)
    params = _fit_elo_multinomial(dr, outcomes, weights)
    strong_home = _predict_elo_multinomial(params, np.array([400.0]))[0]
    strong_away = _predict_elo_multinomial(params, np.array([-400.0]))[0]
    assert strong_home[0] > strong_home[2]  # home favoured when dr >> 0
    assert strong_away[2] > strong_away[0]  # away favoured when dr << 0


def test_calibration_table_structure():
    probs = np.array([[0.8, 0.1, 0.1], [0.5, 0.3, 0.2], [0.4, 0.35, 0.25]])
    outcomes = np.array([0, 0, 1])
    table = _calibration_table(probs, outcomes)
    assert all({"bin", "n", "pred_win_rate", "actual_win_rate", "gap"} <= set(r) for r in table)


# ---------------------------------------------------------------- significance --

def test_per_match_losses_average_to_aggregate_metrics():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet([2, 1, 2], size=50)
    outcomes = rng.integers(0, 3, size=50)
    per = per_match_losses(probs, outcomes)
    assert all(len(v) == 50 for v in per.values())
    assert np.isclose(per["log_loss"].mean(), log_loss_3way(probs, outcomes))
    assert np.isclose(per["brier"].mean(), brier_3way(probs, outcomes))
    assert np.isclose(per["rps"].mean(), rps_3way(probs, outcomes))


def test_paired_bootstrap_identical_predictors_is_null():
    loss = np.random.default_rng(0).exponential(1.0, 300)
    res = paired_bootstrap(loss, loss, n_boot=500)
    assert res["mean_diff"] == 0.0
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
    assert res["p_value"] == 1.0


def test_paired_bootstrap_detects_consistent_improvement():
    rng = np.random.default_rng(0)
    base = rng.exponential(1.0, 400)
    better = base - 0.05 + rng.normal(0, 0.01, 400)  # A always ~0.05 better
    res = paired_bootstrap(better, base, n_boot=2000)
    assert res["ci_high"] < 0
    assert res["p_value"] < 0.01


def test_pairing_beats_unpaired_noise():
    # Huge shared match-level noise, tiny but consistent edge: only pairing sees it.
    rng = np.random.default_rng(0)
    shared = rng.exponential(1.0, 500)
    a = shared + rng.normal(0, 0.01, 500)
    b = shared + 0.01 + rng.normal(0, 0.01, 500)
    assert paired_bootstrap(a, b, n_boot=2000)["ci_high"] < 0


def test_cluster_bootstrap_counts_clusters():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=120), rng.normal(size=120)
    res = paired_bootstrap(a, b, clusters=np.repeat(np.arange(12), 10), n_boot=500)
    assert res["n_units"] == 12 and res["n"] == 120


def test_holm_matches_hand_computation():
    # p sorted: 0.01, 0.02, 0.04 → ×3, ×2, ×1 → 0.03, 0.04, 0.04 (monotone)
    assert np.allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert holm([0.9, 0.8]) == [1.0, 1.0]
    assert holm([]) == []


def test_compare_predictors_structure():
    rng = np.random.default_rng(0)
    n = 60
    frame = {"outcome": rng.integers(0, 3, n), "window": np.repeat(np.arange(6), 10)}
    for name in ("x", "y", "elo"):
        p = rng.dirichlet([1, 1, 1], size=n)
        frame.update({f"{name}_home": p[:, 0], f"{name}_draw": p[:, 1], f"{name}_away": p[:, 2]})
    rows = compare_predictors(pd.DataFrame(frame), ["x", "y", "elo"], n_boot=200)
    assert len(rows) == 3 * 3  # 3 pairs × 3 metrics
    assert {"mean_diff", "ci_iid", "ci_cluster", "p_cluster_holm", "mde_80"} <= set(rows[0])
    assert all(r["p_cluster_holm"] >= r["p_cluster"] for r in rows)
