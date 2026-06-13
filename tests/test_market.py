"""Tests for the market layer (de-vig, blend, value bets)."""
from __future__ import annotations

import numpy as np

from mpp.market import (
    blend_3way,
    blend_matrix,
    devig,
    market_probs_from_points,
    matrix_to_3way,
    value_bets,
)


def test_devig_sums_to_one_and_removes_margin():
    # Odds with a clear margin: raw implied = 0.5 + 0.25 + 0.25 = 1.0... use a real overround.
    out = devig({"home": 1.5, "draw": 4.0, "away": 6.0})
    assert out is not None
    assert abs(out["home"] + out["draw"] + out["away"] - 1.0) < 1e-9
    # Overround = 1/1.5 + 1/4 + 1/6 = 0.6667+0.25+0.1667 = 1.0833 > 1 (the vig)
    assert out["overround"] > 1.0
    # Favourite stays the favourite after de-vig.
    assert out["home"] > out["draw"] and out["home"] > out["away"]


def test_devig_rejects_unfilled():
    assert devig(None) is None
    assert devig({"home": 0.0, "draw": 0.0, "away": 0.0}) is None
    assert devig({"home": 1.0, "draw": 3.0, "away": 5.0}) is None  # odd == 1.0 invalid
    assert devig({"_date": "x", "home": 2.0}) is None  # missing draw/away


def test_blend_3way_endpoints_and_bounds():
    model = {"home": 0.8, "draw": 0.1, "away": 0.1}
    market = {"home": 0.5, "draw": 0.3, "away": 0.2}
    # w=0 → pure model ; w=1 → pure market.
    assert abs(blend_3way(model, market, 0.0)["home"] - 0.8) < 1e-9
    assert abs(blend_3way(model, market, 1.0)["home"] - 0.5) < 1e-9
    mid = blend_3way(model, market, 0.5)
    assert 0.5 < mid["home"] < 0.8
    assert abs(sum(mid.values()) - 1.0) < 1e-9


def test_blend_matrix_matches_target_marginals_and_preserves_shape():
    rng = np.random.default_rng(1)
    m = rng.random((9, 9))
    m /= m.sum()
    target = {"home": 0.6, "draw": 0.25, "away": 0.15}
    b = blend_matrix(m, target)
    got = matrix_to_3way(b)
    for k in ("home", "draw", "away"):
        assert abs(got[k] - target[k]) < 1e-9
    assert abs(b.sum() - 1.0) < 1e-9
    # Texture preserved: within the home region, the rank order of cells is unchanged
    # (constant rescaling can't reorder cells).
    home_mask = np.tril(np.ones((9, 9), bool), -1)
    assert np.argmax(m[home_mask]) == np.argmax(b[home_mask])


def test_value_bet_positive_when_model_beats_odds():
    # Model gives home 60%; odds 2.0 imply 50% → EV = 0.6*2 - 1 = +0.2 (value).
    model = {"home": 0.6, "draw": 0.2, "away": 0.2}
    vb = value_bets(model, {"home": 2.0, "draw": 4.0, "away": 5.0})
    home = next(v for v in vb if v["outcome"] == "home")
    assert home["value"] is True
    assert abs(home["ev"] - 0.2) < 1e-9
    assert abs(home["kelly"] - 0.2 / 1.0) < 1e-9  # edge/(odds-1) = 0.2/1.0


def test_market_probs_from_points_constant_cancels():
    # Points ∝ cote : doubler tous les points doit donner les mêmes probabilités
    # (la constante du barème s'annule à la renormalisation).
    p1 = market_probs_from_points({"home": 50, "draw": 125, "away": 150})
    p2 = market_probs_from_points({"home": 100, "draw": 250, "away": 300})
    for k in ("home", "draw", "away"):
        assert abs(p1[k] - p2[k]) < 1e-9
    assert abs(sum(p1[k] for k in ("home", "draw", "away")) - 1.0) < 1e-9
    # Moins de points = proba implicite plus haute (le favori a le moins de points).
    assert p1["home"] > p1["away"]
    assert p1["overround"] is None  # pas de marge récupérable depuis les points
    assert market_probs_from_points(None) is None
    assert market_probs_from_points({"home": 0, "draw": 0, "away": 0}) is None


def test_value_bet_negative_when_model_below_odds():
    # Model 40% on home, odds 2.0 (need >50%) → EV = 0.4*2 - 1 = -0.2 (no value).
    model = {"home": 0.4, "draw": 0.3, "away": 0.3}
    vb = value_bets(model, {"home": 2.0, "draw": 3.0, "away": 3.0})
    home = next(v for v in vb if v["outcome"] == "home")
    assert home["value"] is False
    assert home["ev"] < 0
