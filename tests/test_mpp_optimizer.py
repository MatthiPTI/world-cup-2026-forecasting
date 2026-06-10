"""Tests for MPP optimizer: scoring, EV computation, ×2 optimal stopping."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from mpp.mpp_optimizer import (
    EXACT_BONUS,
    MPP_EXACT,
    MPP_RESULT,
    compute_markets,
    compute_match_ev,
    decide_double,
    load_state,
    mpp_points,
    run_optimizer,
    save_state,
)

# ─────────────────────────────── helpers ──────────────────────────────────────

def _poisson_matrix(la: float, lb: float, n: int = 9) -> np.ndarray:
    g = np.arange(n)
    m = np.outer(poisson.pmf(g, la), poisson.pmf(g, lb))
    return m / m.sum()


def _fake_prediction(
    date: str, team_a: str, team_b: str, mpp_ev: float,
    points_manquants: bool = False,
) -> dict:
    return {
        "date": date, "team_a": team_a, "team_b": team_b,
        "mpp_ev": mpp_ev, "group": "A",
        "win_a_prob": 0.5, "draw_prob": 0.25, "win_b_prob": 0.25,
        "xg_a": 1.5, "xg_b": 1.0,
        "best_mpp_score": "1-0",
        "mpp_points_manquants": points_manquants,
    }


# ─────────────────────────────── mpp_points ───────────────────────────────────

def test_exact_score_points():
    assert mpp_points((2, 1), (2, 1)) == MPP_EXACT


def test_correct_result_win_a():
    assert mpp_points((2, 0), (3, 1)) == MPP_RESULT


def test_correct_result_draw():
    assert mpp_points((1, 1), (2, 2)) == MPP_RESULT


def test_correct_result_win_b():
    assert mpp_points((0, 2), (1, 3)) == MPP_RESULT


def test_wrong_result():
    assert mpp_points((2, 0), (0, 1)) == 0


# ─────────────────────────────── compute_match_ev ─────────────────────────────

def test_best_score_maximises_ev_brute_force():
    """With match_points, compute_match_ev maximises EV — verified by brute force."""
    rng = np.random.default_rng(7)
    matrix = rng.dirichlet(np.ones(81)).reshape(9, 9)
    match_points = {"home": 60, "draw": 120, "away": 150}

    result = compute_match_ev(matrix, match_points=match_points)

    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())

    max_ev_bf = -np.inf
    for a in range(9):
        for b in range(9):
            if a > b:
                pts, prob = 60.0, p_home
            elif a == b:
                pts, prob = 120.0, p_draw
            else:
                pts, prob = 150.0, p_away
            ev_ab = pts * prob + EXACT_BONUS * float(matrix[a, b])
            if ev_ab > max_ev_bf:
                max_ev_bf = ev_ab

    assert abs(result["ev"] - max_ev_bf) < 1e-6


def test_ev_is_positive():
    m = _poisson_matrix(1.5, 1.0)
    result = compute_match_ev(m, match_points={"home": 50, "draw": 120, "away": 150})
    assert result["ev"] > 0


def test_value_score_can_beat_modal_on_clear_favourite():
    """Optimal EV should always beat a clearly wrong prediction."""
    matrix = _poisson_matrix(3.5, 0.4)
    match_points = {"home": 38, "draw": 155, "away": 210}
    result = compute_match_ev(matrix, match_points=match_points)

    p_away = float(np.triu(matrix, 1).sum())
    # EV of a clearly bad prediction: predict heavy away-win in a walkover
    ev_bad = 210.0 * p_away + EXACT_BONUS * float(matrix[0, 5])
    assert result["ev"] > ev_bad, "Optimal score should outperform a random bad prediction"


# ─────────────────────────────── compute_markets ──────────────────────────────

def test_markets_sum_consistency():
    m = _poisson_matrix(1.5, 1.2)
    mkts = compute_markets(m)
    assert 0 < mkts["over_1_5"] <= 1.0
    assert mkts["over_1_5"] >= mkts["over_2_5"] >= mkts["over_3_5"]
    assert 0 < mkts["btts"] <= 1.0


def test_markets_top5_sorted():
    m = _poisson_matrix(1.5, 1.2)
    mkts = compute_markets(m)
    probs = [s["prob"] for s in mkts["top5_scores"]]
    assert probs == sorted(probs, reverse=True)


def test_over_1_5_high_scoring():
    """In a high-scoring match (λ=4,4), over-1.5 should be very high."""
    m = _poisson_matrix(4.0, 4.0)
    mkts = compute_markets(m)
    assert mkts["over_1_5"] > 0.95


def test_btts_high_when_both_attack():
    m = _poisson_matrix(3.0, 3.0)
    mkts = compute_markets(m)
    assert mkts["btts"] > 0.85


# ──────────────────────── compute_match_ev (new formula) ──────────────────────

def test_outsider_with_high_points_beats_favourite():
    """An outsider outcome with very high cote can yield better EV than the favourite.

    λ_a=1.5, λ_b=0.8 → home wins ~55%. But away_pts=300 makes EV(away) dominate.
    """
    matrix = _poisson_matrix(1.5, 0.8)
    match_points = {"home": 50, "draw": 100, "away": 300}
    result = compute_match_ev(matrix, match_points=match_points)
    # p_away ≈ 0.20; EV_away_zone ≈ 300×0.20 + 20×modal_away >> 50×0.55
    assert result["issue"] == "away", (
        "away_pts=300 should dominate even with lower P(away)"
    )
    assert result["est_value"] is True, "Away is not the most probable → value flag"


def test_exact_bonus_selects_modal_within_chosen_issue():
    """The +20 exact bonus selects the most probable score within the winning issue.

    Without the bonus, all home-win cells share the same EV = pts_home × p_home,
    so argmax would pick the first cell (lowest flat index: [1,0]).
    With +20, [2,0] (higher M) outscores [1,0] → deterministic and correct.
    """
    matrix = np.zeros((9, 9))
    matrix[2, 0] = 0.35   # 2-0: most likely home win (flat index 18)
    matrix[1, 0] = 0.15   # 1-0: less likely home win (flat index 9, comes first)
    matrix[1, 1] = 0.20   # draw
    matrix[0, 1] = 0.30   # away win
    matrix /= matrix.sum()

    match_points = {"home": 100, "draw": 50, "away": 50}
    result = compute_match_ev(matrix, match_points=match_points)
    # p_home=0.50 → EV[2,0]=100×0.50+20×0.35=57 > EV[1,0]=100×0.50+20×0.15=53
    assert result["issue"] == "home"
    assert result["score"] == "2-0", "Most probable home-win score chosen via +20 bonus"


def test_fallback_returns_modal_score_when_no_points():
    """Without match_points, the modal score (argmax of matrix) is returned."""
    matrix = _poisson_matrix(1.5, 1.0)
    modal_i, modal_j = np.unravel_index(int(matrix.argmax()), matrix.shape)

    result = compute_match_ev(matrix)  # no match_points

    assert result["points_manquants"] is True
    assert result["points_issue"] is None
    assert result["score"] == f"{modal_i}-{modal_j}", "Fallback must return modal score"
    assert result["ev"] > 0  # modal probability is always positive


def test_all_zero_points_treated_as_missing():
    """A placeholder entry {home:0, draw:0, away:0} falls back to modal score."""
    matrix = _poisson_matrix(1.5, 1.0)
    modal_i, modal_j = np.unravel_index(int(matrix.argmax()), matrix.shape)

    result = compute_match_ev(matrix, match_points={"home": 0, "draw": 0, "away": 0})

    assert result["points_manquants"] is True, "All-zero entry must trigger fallback"
    assert result["score"] == f"{modal_i}-{modal_j}"


def test_underscore_keys_stripped_from_match_points():
    """Metadata keys like _date and _group are ignored; real points still apply."""
    matrix = _poisson_matrix(1.5, 0.8)
    match_points_raw = {
        "_date": "2026-06-12",
        "_group": "A",
        "home": 50,
        "draw": 100,
        "away": 300,
    }
    result_with_meta = compute_match_ev(matrix, match_points=match_points_raw)
    result_clean = compute_match_ev(matrix, match_points={"home": 50, "draw": 100, "away": 300})

    assert result_with_meta["score"] == result_clean["score"]
    assert abs(result_with_meta["ev"] - result_clean["ev"]) < 1e-9
    assert result_with_meta["points_manquants"] is False


# ─────────────────────────────── decide_double ────────────────────────────────

def _state_fresh():
    return {"double_used": False, "double_match": None, "double_date": None}


def test_double_already_used():
    state = {"double_used": True, "double_match": "A vs B", "double_date": "2026-06-11"}
    result = decide_double([], "2026-06-11", state)
    assert result["recommendation"] == "ALREADY_USED"


def test_double_save_when_better_future():
    """Should SAVE when future EVs are much higher than today's."""
    preds = [
        _fake_prediction("2026-06-11", "A", "B", mpp_ev=2.0),
        _fake_prediction("2026-06-15", "C", "D", mpp_ev=3.8),
        _fake_prediction("2026-06-15", "E", "F", mpp_ev=3.9),
        _fake_prediction("2026-06-16", "G", "H", mpp_ev=3.7),
        _fake_prediction("2026-06-16", "I", "J", mpp_ev=3.6),
        _fake_prediction("2026-06-17", "K", "L", mpp_ev=3.5),
    ]
    result = decide_double(preds, "2026-06-11", _state_fresh())
    assert result["recommendation"] == "SAVE"


def test_double_use_when_best_today():
    """Should USE when today's EV equals or exceeds all future EVs."""
    preds = [
        _fake_prediction("2026-06-11", "France", "Iraq", mpp_ev=3.9),
        _fake_prediction("2026-06-15", "A", "B", mpp_ev=2.5),
        _fake_prediction("2026-06-16", "C", "D", mpp_ev=2.3),
    ]
    result = decide_double(preds, "2026-06-11", _state_fresh())
    assert result["recommendation"] == "USE"
    assert "France" in result["match_key"]


def test_double_failsafe_last_round():
    """On the last date, always USE even if today's EV seems low."""
    preds = [
        _fake_prediction("2026-06-27", "A", "B", mpp_ev=2.1),
        _fake_prediction("2026-06-27", "C", "D", mpp_ev=1.9),
    ]
    result = decide_double(preds, "2026-06-27", _state_fresh())
    assert result["recommendation"] == "USE"


def test_no_today_matches():
    preds = [_fake_prediction("2026-06-20", "A", "B", mpp_ev=3.0)]
    result = decide_double(preds, "2026-06-11", _state_fresh())
    assert result["recommendation"] == "SAVE"


# ──────────────────── decide_double : unités mixtes (cotes manquantes) ────────


def test_double_ignores_unscored_matches():
    """Les matchs sans cotes (EV = probabilité ~0.2) ne doivent pas être comparés
    aux matchs avec cotes (EV en points ~50-150)."""
    preds = [
        # Aujourd'hui : cotes saisies, EV en points
        _fake_prediction("2026-06-11", "France", "Iraq", mpp_ev=80.0),
        # Futur : cotes saisies, EV en points plus faible
        _fake_prediction("2026-06-15", "A", "B", mpp_ev=40.0),
        # Futur : SANS cotes — son EV=0.95 (une probabilité) ne doit pas être
        # interprété comme "EV très faible" dans le percentile
        _fake_prediction("2026-06-16", "C", "D", mpp_ev=0.95, points_manquants=True),
    ]
    result = decide_double(preds, "2026-06-11", _state_fresh())
    assert result["recommendation"] == "USE", (
        "80 pts aujourd'hui > 40 pts futurs : le match non coté (0.95) ne doit "
        "pas polluer la comparaison"
    )


def test_double_all_unscored_defers():
    """Sans aucune cote saisie, la décision ×2 est reportée (SAVE)."""
    preds = [
        _fake_prediction("2026-06-11", "A", "B", mpp_ev=0.25, points_manquants=True),
        _fake_prediction("2026-06-15", "C", "D", mpp_ev=0.18, points_manquants=True),
    ]
    result = decide_double(preds, "2026-06-11", _state_fresh())
    assert result["recommendation"] == "SAVE"
    assert "Aucune cote" in result["note"]


def test_double_saves_when_future_has_no_cotes_yet():
    """Aujourd'hui coté mais futur non coté (hors dernière journée) → SAVE,
    on ne brûle pas le jeton sans pouvoir comparer."""
    preds = [
        _fake_prediction("2026-06-11", "France", "Iraq", mpp_ev=80.0),
        _fake_prediction("2026-06-15", "A", "B", mpp_ev=0.2, points_manquants=True),
        _fake_prediction("2026-06-16", "C", "D", mpp_ev=0.2, points_manquants=True),
    ]
    result = decide_double(preds, "2026-06-11", _state_fresh())
    assert result["recommendation"] == "SAVE"


def test_double_failsafe_still_fires_on_true_last_day():
    """Dernière journée réelle + match coté aujourd'hui → USE (failsafe)."""
    preds = [
        _fake_prediction("2026-06-27", "A", "B", mpp_ev=55.0),
        _fake_prediction("2026-06-27", "C", "D", mpp_ev=0.2, points_manquants=True),
    ]
    result = decide_double(preds, "2026-06-27", _state_fresh())
    assert result["recommendation"] == "USE"


# ──────────────────── run_optimizer : persistance du jeton ────────────────────


def test_run_optimizer_does_not_persist_without_confirm():
    """Une recommandation USE ne doit PAS consommer le jeton sans --confirm-double."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        preds = [
            _fake_prediction("2026-06-11", "France", "Iraq", mpp_ev=80.0),
            _fake_prediction("2026-06-15", "A", "B", mpp_ev=40.0),
        ]
        _, decision = run_optimizer(preds, "2026-06-11", state_path)
        assert decision["recommendation"] == "USE"
        assert not state_path.exists(), "Sans confirmation, l'état ne doit pas être écrit"
        assert load_state(state_path)["double_used"] is False


def test_run_optimizer_persists_with_confirm():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        preds = [
            _fake_prediction("2026-06-11", "France", "Iraq", mpp_ev=80.0),
            _fake_prediction("2026-06-15", "A", "B", mpp_ev=40.0),
        ]
        _, decision = run_optimizer(preds, "2026-06-11", state_path, confirm_double=True)
        assert decision["recommendation"] == "USE"
        state = load_state(state_path)
        assert state["double_used"] is True
        assert "France" in state["double_match"]


# ─────────────────────────────── state persistence ────────────────────────────

def test_state_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        state = {"double_used": True, "double_match": "France vs Iraq", "double_date": "2026-06-22"}
        save_state(state, path)
        loaded = load_state(path)
        assert loaded == state


def test_load_state_creates_default_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nonexistent.json"
        state = load_state(path)
        assert state["double_used"] is False
        assert state["double_match"] is None


def test_run_optimizer_marks_double():
    """run_optimizer should mark the double_reco column on the recommended match."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        preds = [
            _fake_prediction("2026-06-11", "France", "Iraq", mpp_ev=3.9),
            _fake_prediction("2026-06-15", "A", "B", mpp_ev=2.0),
        ]
        enriched, decision = run_optimizer(preds, "2026-06-11", state_path)
        uses = [p for p in enriched if p.get("double_reco") == "OUI"]
        assert len(uses) == 1
        assert "France" in uses[0]["team_a"]
