"""World Football Elo ratings computed from historical match data.

Elo propagates inter-confederation strength via intercontinental matches,
solving the calibration problem that pure within-confederation data cannot.
"""
from __future__ import annotations

import pandas as pd

_INITIAL = 1500.0

# Checked in order; first match wins.
_TOURNAMENT_K_RULES: list[tuple[str, float]] = [
    ("FIFA World Cup", 2.0),
    ("Copa América", 1.75),
    ("UEFA Euro", 1.75),
    ("Africa Cup of Nations", 1.75),
    ("AFC Asian Cup", 1.75),
    ("FIFA Confederations Cup", 1.5),
    ("World Cup qualification", 1.5),
    ("Nations League", 1.25),
    ("Championship qualification", 1.25),
    ("Friendly", 0.5),
]
_DEFAULT_K_MULT = 1.0


def _tournament_k(tournament: str) -> float:
    for key, mult in _TOURNAMENT_K_RULES:
        if key in tournament:
            return mult
    return _DEFAULT_K_MULT


def _goal_diff_mult(gd: int) -> float:
    """World Football Elo goal-difference multiplier."""
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def compute_elo(
    matches: pd.DataFrame,
    k_base: float = 20.0,
    home_advantage: float = 100.0,
) -> dict[str, float]:
    """Compute World Football Elo ratings chronologically.

    Args:
        matches: DataFrame with columns: date, home_team, away_team,
                 home_score, away_score. Optional columns: tournament
                 (used to determine K importance multiplier), neutral
                 (home advantage is skipped on neutral venues).
        k_base: Base K-factor before importance and goal-diff multipliers.
        home_advantage: Elo points added to the home team's rating in the
            expected-score formula (World Football Elo standard: 100).
            Without it, home teams systematically gain rating they don't
            deserve, biasing teams that host often.

    Returns:
        dict mapping team name → final Elo rating.
    """
    ratings: dict[str, float] = {}
    has_tournament = "tournament" in matches.columns
    has_neutral = "neutral" in matches.columns
    df = matches.sort_values("date").reset_index(drop=True)

    for _, row in df.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        hs, aws_ = int(row["home_score"]), int(row["away_score"])
        tournament = str(row["tournament"]) if has_tournament else ""
        is_neutral = bool(row["neutral"]) if has_neutral else False

        r_h = ratings.get(home, _INITIAL)
        r_a = ratings.get(away, _INITIAL)

        ha = 0.0 if is_neutral else home_advantage
        e_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + ha)) / 400.0))

        w_h = 1.0 if hs > aws_ else (0.0 if hs < aws_ else 0.5)
        w_a = 1.0 - w_h

        gd = abs(hs - aws_)
        k = k_base * _tournament_k(tournament) * _goal_diff_mult(gd)

        ratings[home] = r_h + k * (w_h - e_h)
        ratings[away] = r_a + k * (w_a - (1.0 - e_h))

    return ratings


def get_team_ratings(source: str = "internal", **kwargs) -> dict[str, float]:
    """Return team → Elo rating dict.

    Extension point: add branches here to pull from external APIs or
    ranking systems without changing callers.

    Args:
        source: "internal" (default) — compute from historical matches.
        **kwargs: For source="internal", requires matches=<DataFrame>.
    """
    if source == "internal":
        matches = kwargs["matches"]
        return compute_elo(matches)
    raise ValueError(f"Unknown rating source: {source!r}")
