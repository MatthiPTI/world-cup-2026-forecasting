"""Tests for BayesianHierarchicalModel.

All tests use inference="map" (optimisation-based point estimate) so they complete
in a few seconds without needing NUTS chains or draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mpp.bayesian_model import BayesianHierarchicalModel
from mpp.model import DixonColesModel  # noqa: F401 (used in shrinkage test)

# --------------------------------------------------------------------------- helpers


def _mixed_conf_matches(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic matches across several real team names (covering multiple confederations)."""
    rng = np.random.default_rng(seed)
    # mix of UEFA, CONMEBOL, CONCACAF, CAF, AFC teams so confederation hierarchy activates
    teams = [
        "France", "Germany", "Spain",          # UEFA
        "Brazil", "Argentina",                  # CONMEBOL
        "Mexico", "United States",              # CONCACAF
        "Senegal", "Morocco",                   # CAF
        "Japan", "South Korea",                 # AFC
    ]
    rows = []
    for _ in range(n):
        i, j = rng.choice(len(teams), 2, replace=False)
        rows.append({
            "home_team": teams[i],
            "away_team": teams[j],
            "home_score": int(rng.poisson(1.5)),
            "away_score": int(rng.poisson(1.0)),
            "neutral": bool(rng.random() > 0.5),
            "weight": float(np.exp(-rng.uniform(0, 3))),
        })
    return pd.DataFrame(rows)


def _shrinkage_matches() -> pd.DataFrame:
    """Scenario that exposes confederation-schedule bias.

    UEFA teams (France, Germany, Spain) play many balanced games among themselves.
    Curaçao (CONCACAF) only plays Solomon Islands and Samoa (OFC) and scores 5–0 every time.
    No cross-confederation games exist, so only the Bayesian prior can calibrate Curaçao's
    absolute strength relative to the UEFA block.

    Expected outcome:
    - Dixon-Coles: Curaçao gets inflated attack because Solomon Islands / Samoa get
      extremely negative defense, making Curaçao look elite.
    - Bayesian MAP: CONCACAF confederation is anchored near the global prior mean (no
      evidence linking it to UEFA), so Curaçao's absolute attack is shrunk toward average.
    """
    rows = []
    # UEFA block: 40 balanced matchdays among 3 teams
    uefa = ["France", "Germany", "Spain"]
    for _ in range(40):
        for i, ta in enumerate(uefa):
            for j, tb in enumerate(uefa):
                if i < j:
                    rows.append({
                        "home_team": ta, "away_team": tb,
                        "home_score": 1, "away_score": 1,
                        "neutral": True, "weight": 1.0,
                    })
    # Curaçao dominating two OFC minnows — no overlap with UEFA teams
    for _ in range(50):
        rows.append({"home_team": "Curaçao", "away_team": "Solomon Islands",
                     "home_score": 5, "away_score": 0, "neutral": True, "weight": 1.0})
        rows.append({"home_team": "Curaçao", "away_team": "Samoa",
                     "home_score": 5, "away_score": 0, "neutral": True, "weight": 1.0})
        rows.append({"home_team": "Solomon Islands", "away_team": "Samoa",
                     "home_score": 0, "away_score": 0, "neutral": True, "weight": 1.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def fitted_bayes() -> BayesianHierarchicalModel:
    return BayesianHierarchicalModel(inference="map").fit(_mixed_conf_matches())


# --------------------------------------------------------------------------- interface tests


def test_fit_produces_params(fitted_bayes):
    assert len(fitted_bayes._attack) == 11
    assert len(fitted_bayes._defense) == 11
    assert len(fitted_bayes.teams) == 11


def test_score_matrix_shape(fitted_bayes):
    m = fitted_bayes.score_matrix("France", "Germany")
    assert m.shape == (9, 9)


def test_score_matrix_sums_to_one(fitted_bayes):
    m = fitted_bayes.score_matrix("Brazil", "Argentina")
    assert abs(m.sum() - 1.0) < 0.01


def test_predict_match_keys(fitted_bayes):
    pred = fitted_bayes.predict_match("Spain", "Japan")
    for key in ("team_a", "team_b", "predicted_score", "xg_a", "xg_b",
                "win_a_prob", "draw_prob", "win_b_prob"):
        assert key in pred


def test_predict_probabilities_sum_to_one(fitted_bayes):
    pred = fitted_bayes.predict_match("France", "Brazil")
    total = pred["win_a_prob"] + pred["draw_prob"] + pred["win_b_prob"]
    assert abs(total - 1.0) < 0.02


def test_predicted_score_format(fitted_bayes):
    pred = fitted_bayes.predict_match("Germany", "Mexico")
    parts = pred["predicted_score"].split("-")
    assert len(parts) == 2
    assert all(p.isdigit() for p in parts)


def test_team_strengths_columns(fitted_bayes):
    df = fitted_bayes.team_strengths()
    assert "team" in df.columns
    assert "attack" in df.columns
    assert "defense" in df.columns
    assert len(df) == 11


def test_team_strengths_sorted_by_attack(fitted_bayes):
    df = fitted_bayes.team_strengths()
    assert list(df["attack"]) == sorted(df["attack"], reverse=True)


def test_adjustments_applied(fitted_bayes):
    pred_base = fitted_bayes.predict_match("France", "Germany")
    adj = {"France": {"attack": 0.5}}
    pred_adj = fitted_bayes.predict_match("France", "Germany", adjustments=adj)
    assert pred_adj["xg_a"] < pred_base["xg_a"]


def test_global_boost_applied(fitted_bayes):
    pred_base = fitted_bayes.predict_match("France", "Germany")
    pred_boost = fitted_bayes.predict_match("France", "Germany", global_boost=1.5)
    assert pred_boost["xg_a"] > pred_base["xg_a"]
    assert pred_boost["xg_b"] > pred_base["xg_b"]


# --------------------------------------------------------------------------- shrinkage test


# --------------------------------------------------------------------------- weight normalisation


def test_weight_normalization_mean():
    """_normalize_weights should produce a vector with mean exactly 1."""
    from mpp.bayesian_model import _normalize_weights
    rng = np.random.default_rng(0)
    w = rng.uniform(0.001, 0.5, size=200)
    normed = _normalize_weights(w)
    assert abs(normed.mean() - 1.0) < 1e-10


def test_weight_normalization_preserves_shape():
    """Relative ordering of weights is preserved after normalization."""
    from mpp.bayesian_model import _normalize_weights
    w = np.array([0.01, 0.02, 0.005, 0.03])
    normed = _normalize_weights(w)
    assert normed[3] > normed[1] > normed[0] > normed[2]


def test_weight_normalization_prevents_collapse():
    """With very small (realistic time-decay) weights, normalization prevents prior
    domination: the spread in attack ratings should be non-trivial."""
    matches = _mixed_conf_matches().copy()
    # Simulate deep-history decay: weights << 1
    matches["weight"] = matches["weight"] * 0.001
    model = BayesianHierarchicalModel(inference="map")
    model.fit(matches)
    attack_range = float(model._attack.max() - model._attack.min())
    assert attack_range > 0.05, (
        f"Attack range collapsed to {attack_range:.4f} — normalization may not be working."
    )


# --------------------------------------------------------------------------- Elo anchoring


def _dominance_matches() -> pd.DataFrame:
    """France dominates all opponents; others play balanced games."""
    rng = np.random.default_rng(99)
    teams = ["France", "Germany", "Spain", "Brazil", "Argentina", "Mexico", "United States"]
    rows = []
    for opp in teams[1:]:
        for _ in range(20):
            rows.append({
                "home_team": "France", "away_team": opp,
                "home_score": int(rng.poisson(3.5)),
                "away_score": int(rng.poisson(0.4)),
                "neutral": True, "weight": 1.0,
            })
            rows.append({
                "home_team": opp, "away_team": "France",
                "home_score": int(rng.poisson(0.4)),
                "away_score": int(rng.poisson(3.5)),
                "neutral": True, "weight": 1.0,
            })
    for _ in range(10):
        for i in range(1, len(teams)):
            for j in range(i + 1, len(teams)):
                rows.append({
                    "home_team": teams[i], "away_team": teams[j],
                    "home_score": int(rng.poisson(1.3)),
                    "away_score": int(rng.poisson(1.3)),
                    "neutral": True, "weight": 1.0,
                })
    return pd.DataFrame(rows)


def test_elo_anchoring_top_team():
    """With data showing France dominating and Elo confirming, France should have
    the highest attack; the Elo-anchored model should not reduce this dominance."""
    matches = _dominance_matches()
    elo_ratings = {
        "France": 2100.0, "Germany": 1100.0, "Spain": 1500.0,
        "Brazil": 1480.0, "Argentina": 1490.0,
        "Mexico": 1350.0, "United States": 1330.0,
    }
    model_elo = BayesianHierarchicalModel(inference="map")
    model_elo.fit(matches, elo_ratings=elo_ratings)

    top = model_elo.team_strengths().iloc[0]["team"]
    assert top == "France", f"Expected France as top team with Elo anchoring, got {top}"


def test_elo_anchoring_gap_not_reduced():
    """France–Germany attack gap should be at least as large with Elo anchoring."""
    matches = _dominance_matches()
    elo_ratings = {t: 1500.0 for t in
                   ["France", "Germany", "Spain", "Brazil", "Argentina", "Mexico", "United States"]}
    elo_ratings["France"] = 2100.0
    elo_ratings["Germany"] = 1000.0

    model_elo = BayesianHierarchicalModel(inference="map")
    model_elo.fit(matches, elo_ratings=elo_ratings)

    model_base = BayesianHierarchicalModel(inference="map")
    model_base.fit(matches)

    fi_e = model_elo._team_idx["France"]
    gi_e = model_elo._team_idx["Germany"]
    fi_b = model_base._team_idx["France"]
    gi_b = model_base._team_idx["Germany"]

    gap_elo = model_elo._attack[fi_e] - model_elo._attack[gi_e]
    gap_base = model_base._attack[fi_b] - model_base._attack[gi_b]

    assert gap_elo >= gap_base - 0.05, (
        f"Elo anchoring should not reduce the France–Germany gap. "
        f"Elo gap={gap_elo:.3f}, Base gap={gap_base:.3f}"
    )


def test_no_elo_backward_compatible():
    """fit() without elo_ratings should still work (backward compatibility)."""
    model = BayesianHierarchicalModel(inference="map")
    model.fit(_mixed_conf_matches())
    assert len(model._attack) == 11


# --------------------------------------------------------------------------- shrinkage test


def test_confederation_shrinkage():
    """Bayesian MAP shrinks Curaçao's attack gap vs average more than Dixon-Coles does."""
    matches = _shrinkage_matches()

    # Minimal regularisation for DC so it goes to the raw data signal
    dc = DixonColesModel(reg=0.05)
    dc.fit(matches)

    bayes = BayesianHierarchicalModel(inference="map")
    bayes.fit(matches)

    dc_str = dc.team_strengths().set_index("team")
    bay_str = bayes.team_strengths().set_index("team")

    # How much above average is Curaçao in each model?
    dc_gap = float(dc_str.loc["Curaçao", "attack"] - dc_str["attack"].mean())
    bay_gap = float(bay_str.loc["Curaçao", "attack"] - bay_str["attack"].mean())

    assert bay_gap < dc_gap, (
        f"Expected Bayesian shrinkage on Curaçao: "
        f"bayes gap={bay_gap:.3f} should be < dc gap={dc_gap:.3f}"
    )
