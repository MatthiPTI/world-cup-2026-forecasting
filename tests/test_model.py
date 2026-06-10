"""Sanity tests for the Dixon-Coles model."""
import numpy as np
import pandas as pd
import pytest

from mpp.model import DixonColesModel


def _dummy_matches(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["France", "Germany", "Spain", "Brazil", "Argentina", "England"]
    rows = []
    for _ in range(n):
        i, j = rng.choice(len(teams), 2, replace=False)
        rows.append({
            "home_team": teams[i],
            "away_team": teams[j],
            "home_score": int(rng.poisson(1.5)),
            "away_score": int(rng.poisson(1.0)),
            "neutral": bool(rng.random() > 0.5),
            "weight": float(np.exp(-rng.uniform(0, 5))),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_model():
    return DixonColesModel().fit(_dummy_matches())


def test_params_set(fitted_model):
    assert fitted_model.params is not None
    assert len(fitted_model.teams) == 6


def test_score_matrix_shape(fitted_model):
    matrix = fitted_model.score_matrix("France", "Germany")
    assert matrix.shape == (9, 9)


def test_score_matrix_sums_to_one(fitted_model):
    matrix = fitted_model.score_matrix("France", "Germany")
    assert abs(matrix.sum() - 1.0) < 0.01


def test_predict_probabilities_sum_to_one(fitted_model):
    pred = fitted_model.predict_match("Spain", "Brazil")
    total = pred["win_a_prob"] + pred["draw_prob"] + pred["win_b_prob"]
    assert abs(total - 1.0) < 0.02


def test_predict_score_format(fitted_model):
    pred = fitted_model.predict_match("Argentina", "England")
    parts = pred["predicted_score"].split("-")
    assert len(parts) == 2
    assert all(p.isdigit() for p in parts)


def test_team_strengths_shape(fitted_model):
    df = fitted_model.team_strengths()
    assert len(df) == 6
    assert "attack" in df.columns and "defense" in df.columns
