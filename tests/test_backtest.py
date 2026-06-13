"""Tests for the back-test metric functions (no model fitting required)."""
from __future__ import annotations

import numpy as np

from mpp.backtest import (
    _calibration_table,
    _fit_elo_multinomial,
    _predict_elo_multinomial,
    brier_3way,
    log_loss_3way,
    outcome_index,
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
