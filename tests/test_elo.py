"""Tests for Elo computation."""
from __future__ import annotations

import pandas as pd
import pytest

from mpp.elo import compute_elo, get_team_ratings


def _make_matches(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_winner_gains_elo():
    """A team that wins all matches should end up with a rating above 1500."""
    matches = _make_matches([
        {"date": "2024-01-01", "home_team": "France", "away_team": "Iraq",
         "home_score": 3, "away_score": 0, "tournament": "Friendly"},
        {"date": "2024-01-08", "home_team": "France", "away_team": "Iraq",
         "home_score": 2, "away_score": 1, "tournament": "Friendly"},
        {"date": "2024-01-15", "home_team": "France", "away_team": "Iraq",
         "home_score": 4, "away_score": 0, "tournament": "Friendly"},
    ])
    ratings = compute_elo(matches)
    assert ratings["France"] > 1500, "Consistent winner should exceed initial 1500"
    assert ratings["Iraq"] < 1500, "Consistent loser should fall below initial 1500"


def test_elo_is_zero_sum_ish():
    """Total rating shift after games is bounded (not necessarily exactly zero-sum
    due to initial-rating effects, but winner+loser changes should be close)."""
    matches = _make_matches([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B",
         "home_score": 1, "away_score": 0, "tournament": "Friendly"},
    ])
    ratings = compute_elo(matches)
    # Both start at 1500; one match means A gained roughly what B lost
    gained = ratings["A"] - 1500
    lost = 1500 - ratings["B"]
    assert abs(gained - lost) < 0.01, "Elo shifts should be equal and opposite"


def test_elo_propagates_inter_confederation():
    """Elo separates teams transitively via a common opponent chain.

    France beats Brazil many times; Brazil beats Iraq many times (same K factor
    so the chain is symmetric). France has never played Iraq but should end up
    with a higher rating.
    """
    rows = []
    for i, d in enumerate(range(1, 10)):
        rows.append({
            "date": f"2024-01-{d:02d}",
            "home_team": "France", "away_team": "Brazil",
            "home_score": 2, "away_score": 0, "tournament": "Friendly",
        })
        rows.append({
            "date": f"2024-02-{d:02d}",
            "home_team": "Brazil", "away_team": "Iraq",
            "home_score": 2, "away_score": 0, "tournament": "Friendly",
        })
    matches = _make_matches(rows)
    ratings = compute_elo(matches)
    # France never played Iraq; chain propagation should place France > Iraq
    assert ratings["France"] > ratings["Iraq"], (
        f"France ({ratings['France']:.1f}) should exceed Iraq ({ratings['Iraq']:.1f}) "
        "via the chain France>Brazil>Iraq"
    )


def test_world_cup_k_higher_than_friendly():
    """A World Cup win should change Elo more than a friendly win."""
    base = {
        "date": "2024-01-01", "home_team": "A", "away_team": "B",
        "home_score": 1, "away_score": 0,
    }
    wc = compute_elo(_make_matches([{**base, "tournament": "FIFA World Cup"}]))
    fr = compute_elo(_make_matches([{**base, "tournament": "Friendly"}]))
    assert wc["A"] - 1500 > fr["A"] - 1500, "WC should shift Elo more than a friendly"


def test_goal_diff_multiplier():
    """Larger goal differences should produce larger Elo shifts."""
    base = {
        "date": "2024-01-01", "home_team": "A", "away_team": "B",
        "tournament": "Friendly",
    }
    r1 = compute_elo(_make_matches([{**base, "home_score": 1, "away_score": 0}]))
    r3 = compute_elo(_make_matches([{**base, "home_score": 3, "away_score": 0}]))
    assert r3["A"] - 1500 > r1["A"] - 1500


def test_draw_at_home_penalises_home_team():
    """With home advantage, a HOME draw between equal teams means the home side
    underperformed its expectation → it loses a few points (WFE behaviour)."""
    matches = _make_matches([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B",
         "home_score": 1, "away_score": 1, "tournament": "Friendly"},
    ])
    ratings = compute_elo(matches)
    assert ratings["A"] < 1500, "Home draw should cost the home team points"
    assert ratings["B"] > 1500


def test_draw_on_neutral_keeps_ratings_near_initial():
    """On a neutral venue there is no home advantage: a draw between equal
    teams barely moves the ratings."""
    matches = _make_matches([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B",
         "home_score": 1, "away_score": 1, "tournament": "Friendly",
         "neutral": True},
    ])
    ratings = compute_elo(matches)
    assert abs(ratings["A"] - 1500) < 1e-9
    assert abs(ratings["B"] - 1500) < 1e-9


def test_home_win_gains_less_than_away_win():
    """Winning at home is expected → smaller Elo gain than winning away."""
    base = {
        "date": "2024-01-01", "home_score": 1, "away_score": 0,
        "tournament": "Friendly",
    }
    home_win = compute_elo(_make_matches([
        {**base, "home_team": "A", "away_team": "B"},
    ]))
    away_win = compute_elo(_make_matches([
        {**base, "home_team": "B", "away_team": "A", "home_score": 0, "away_score": 1},
    ]))
    gain_home = home_win["A"] - 1500
    gain_away = away_win["A"] - 1500
    assert gain_away > gain_home, "Away win must be rewarded more than home win"


def test_home_advantage_zero_restores_symmetry():
    """home_advantage=0 reproduces the venue-blind behaviour."""
    matches = _make_matches([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B",
         "home_score": 1, "away_score": 1, "tournament": "Friendly"},
    ])
    ratings = compute_elo(matches, home_advantage=0.0)
    assert abs(ratings["A"] - 1500) < 1e-9
    assert abs(ratings["B"] - 1500) < 1e-9


def test_no_tournament_column_defaults_k():
    """compute_elo handles DataFrames without a tournament column."""
    matches = pd.DataFrame([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B",
         "home_score": 2, "away_score": 0},
    ])
    ratings = compute_elo(matches)
    assert "A" in ratings and "B" in ratings


def test_get_team_ratings_internal():
    """get_team_ratings(source='internal') delegates to compute_elo."""
    matches = _make_matches([
        {"date": "2024-01-01", "home_team": "France", "away_team": "Iraq",
         "home_score": 2, "away_score": 0, "tournament": "Friendly"},
    ])
    ratings = get_team_ratings(source="internal", matches=matches)
    assert isinstance(ratings, dict)
    assert "France" in ratings


def test_get_team_ratings_unknown_source():
    with pytest.raises(ValueError, match="Unknown rating source"):
        get_team_ratings(source="external_nonexistent")
