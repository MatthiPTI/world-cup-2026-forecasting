"""MPP (Mon Petit Prono) score optimizer and ×2 token manager.

Real MPP scoring (cote-based):
  - Points for the correct RESULT depend on the cote for that outcome and vary per match.
    Example: Mexico vs South Africa → home=49 / draw=125 / away=148.
  - +EXACT_BONUS (20 pts) added if the exact score is also correct.
  - Wrong result: 0.

Match cotes are read from data/mpp_points.json:
  { "<team_a> vs <team_b>": {"home": <int>, "draw": <int>, "away": <int>}, ... }
When a fixture is absent from that file, the module falls back to the modal score.

The ×2 token doubles the points on one match per tournament (unique usage).
Optimal stopping: use it on the match with highest expected value, but only
if that EV exceeds the 90th-percentile of future-match EVs (or it's the last round).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MPP_EXACT = 4
MPP_RESULT = 2
MPP_WRONG = 0
EXACT_BONUS = 20


def mpp_points(
    predicted: tuple[int, int],
    actual: tuple[int, int],
    exact_pts: int = MPP_EXACT,
    result_pts: int = MPP_RESULT,
) -> int:
    """Points earned when prediction is (pa, pb) and actual result is (aa, ab)."""
    pa, pb = predicted
    aa, ab = actual
    if (pa, pb) == (aa, ab):
        return exact_pts
    pred_sign = (pa > pb) - (pa < pb)
    act_sign = (aa > ab) - (aa < ab)
    return result_pts if pred_sign == act_sign else MPP_WRONG


def compute_match_ev(
    matrix: np.ndarray,
    match_points: dict | None = None,
    exact_bonus: int = EXACT_BONUS,
) -> dict:
    """Return EV-optimal MPP prediction for a score probability matrix.

    When match_points is provided, uses the real MPP scoring formula:
        EV[a,b] = match_points[outcome(a,b)] × P(outcome) + exact_bonus × M[a,b]
    so a high-points outsider can outperform a low-points favourite.

    When match_points is None, falls back to the modal score (argmax of matrix)
    and sets points_manquants=True.

    Args:
        matrix: n×n array where M[i,j] = P(home scores i, away scores j).
        match_points: {"home": int, "draw": int, "away": int} MPP points per outcome.
        exact_bonus: Bonus points added when exact score is correct (default 20).

    Returns dict:
        score          "a-b" best predicted score string
        ev             expected value of that prediction
        issue          "home" | "draw" | "away"
        points_issue   int points for chosen outcome (None in fallback)
        proba_issue    P(chosen outcome)
        est_value      True if chosen outcome is NOT the most probable
        points_manquants  True when called without match_points
    """
    n = matrix.shape[0]
    g = np.arange(n)
    pred_a, pred_b = np.meshgrid(g, g, indexing="ij")

    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())

    # Sanitize: remove reference metadata (_date, _group, …) then treat any entry
    # whose home/draw/away are all ≤ 0 (unfilled placeholder) as absent → fallback.
    if match_points is not None:
        match_points = {k: v for k, v in match_points.items() if not k.startswith("_")}
        if not (float(match_points.get("home", 0)) > 0
                and float(match_points.get("draw", 0)) > 0
                and float(match_points.get("away", 0)) > 0):
            match_points = None

    if match_points is None:
        # Fallback: modal score (highest-probability cell)
        best_flat = int(matrix.argmax())
        bi, bj = np.unravel_index(best_flat, matrix.shape)
        issue = "home" if bi > bj else ("draw" if bi == bj else "away")
        proba_issue = p_home if issue == "home" else (p_draw if issue == "draw" else p_away)
        return {
            "score": f"{bi}-{bj}",
            "ev": float(matrix[bi, bj]),
            "issue": issue,
            "points_issue": None,
            "proba_issue": round(proba_issue, 4),
            "est_value": False,
            "points_manquants": True,
        }

    pts_h = float(match_points["home"])
    pts_d = float(match_points["draw"])
    pts_a = float(match_points["away"])

    pts_grid = np.empty((n, n))
    pts_grid[pred_a > pred_b] = pts_h
    pts_grid[pred_a == pred_b] = pts_d
    pts_grid[pred_a < pred_b] = pts_a

    p_outcome_grid = np.empty((n, n))
    p_outcome_grid[pred_a > pred_b] = p_home
    p_outcome_grid[pred_a == pred_b] = p_draw
    p_outcome_grid[pred_a < pred_b] = p_away

    ev_grid = pts_grid * p_outcome_grid + exact_bonus * matrix

    best_flat = int(ev_grid.argmax())
    bi, bj = np.unravel_index(best_flat, ev_grid.shape)
    ev = float(ev_grid[bi, bj])

    issue = "home" if bi > bj else ("draw" if bi == bj else "away")
    proba_issue = p_home if issue == "home" else (p_draw if issue == "draw" else p_away)
    points_issue = int(pts_h if issue == "home" else (pts_d if issue == "draw" else pts_a))

    most_probable = max(
        [("home", p_home), ("draw", p_draw), ("away", p_away)], key=lambda x: x[1]
    )[0]

    return {
        "score": f"{bi}-{bj}",
        "ev": ev,
        "issue": issue,
        "points_issue": points_issue,
        "proba_issue": round(proba_issue, 4),
        "est_value": issue != most_probable,
        "points_manquants": False,
    }


def compute_markets(matrix: np.ndarray) -> dict:
    """Derive bet markets from a score probability matrix.

    Returns:
        over_1_5, over_2_5, over_3_5: P(total goals > threshold)
        btts: P(both teams score ≥ 1)
        top5_scores: list of {score, prob} sorted by probability desc
    """
    n = matrix.shape[0]
    g = np.arange(n)
    ia, jb = np.meshgrid(g, g, indexing="ij")
    total = ia + jb

    over_1_5 = float(matrix[total > 1].sum())
    over_2_5 = float(matrix[total > 2].sum())
    over_3_5 = float(matrix[total > 3].sum())
    btts = float(matrix[1:, 1:].sum())

    flat = matrix.ravel()
    top5_idx = np.argsort(flat)[-5:][::-1]
    top5 = []
    for idx in top5_idx:
        i, j = np.unravel_index(int(idx), matrix.shape)
        top5.append({"score": f"{i}-{j}", "prob": round(float(matrix[i, j]), 4)})

    return {
        "over_1_5": round(over_1_5, 3),
        "over_2_5": round(over_2_5, 3),
        "over_3_5": round(over_3_5, 3),
        "btts": round(btts, 3),
        "top5_scores": top5,
    }


def load_state(state_path: Path) -> dict:
    """Load ×2 token state from JSON, initialising if absent."""
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    return {"double_used": False, "double_match": None, "double_date": None}


def save_state(state: dict, state_path: Path) -> None:
    state_path.parent.mkdir(exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def decide_double(
    predictions: list[dict],
    reference_date: str,
    state: dict,
    threshold_pct: float = 90.0,
) -> dict:
    """Optimal stopping decision for the ×2 token.

    Args:
        predictions: All fixture predictions with mpp_ev computed.
        reference_date: "YYYY-MM-DD" string for today.
        state: Current token state dict.
        threshold_pct: Percentile of future EVs used as the use-threshold.

    Returns:
        dict with keys: match_key, recommendation ("USE"/"SAVE"/"ALREADY_USED"),
        note (human-readable explanation).
    """
    if state.get("double_used"):
        match = state.get("double_match", "?")
        date = state.get("double_date", "?")
        return {
            "match_key": None,
            "recommendation": "ALREADY_USED",
            "note": f"Jeton ×2 déjà utilisé sur {match} ({date}).",
        }

    # Garde-fou unités mixtes : sans cotes saisies, mpp_ev est une PROBABILITÉ
    # (~0.2, fallback score modal) alors qu'avec cotes c'est des POINTS (~50-150).
    # Mélanger les deux casse la logique de percentile → on ne compare que les
    # matchs dont les points-cote sont réellement saisis.
    scored = [p for p in predictions if not p.get("mpp_points_manquants", False)]
    n_unscored_future = sum(
        1 for p in predictions
        if p.get("mpp_points_manquants", False) and p.get("date", "") > reference_date
    )

    if not scored:
        return {
            "match_key": None,
            "recommendation": "SAVE",
            "note": ("Aucune cote saisie dans mpp_points.json — décision ×2 reportée "
                     "(les EV fallback ne sont pas comparables)."),
        }

    today = [p for p in scored if p.get("date") == reference_date]
    future = [p for p in scored if p.get("date", "") > reference_date]

    if not today:
        best_future = max(future, key=lambda p: p["mpp_ev"], default=None)
        target = (f"{best_future['team_a']} vs {best_future['team_b']} "
                  f"({best_future['date']}, EV={best_future['mpp_ev']:.2f})"
                  if best_future else "aucun match futur")
        return {
            "match_key": None,
            "recommendation": "SAVE",
            "note": f"Pas de match aujourd'hui. Cible projetée : {target}.",
        }

    best_today = max(today, key=lambda p: p["mpp_ev"])
    ev_today = best_today["mpp_ev"]
    mk = f"{best_today['team_a']} vs {best_today['team_b']}"

    # Failsafe : dernière journée du calendrier COMPLET (pas seulement des matchs
    # avec cotes saisies, sinon on déclencherait USE trop tôt).
    all_dates = sorted({p["date"] for p in predictions})
    last_date = all_dates[-1] if all_dates else reference_date
    is_last_round = reference_date == last_date

    if is_last_round:
        return {
            "match_key": mk,
            "recommendation": "USE",
            "note": f"Dernière journée — jeton ×2 placé sur {mk} (EV={ev_today:.2f}, failsafe).",
        }

    if not future:
        # Des matchs futurs existent mais aucun n'a de cotes saisies : impossible
        # de comparer. On conserve le jeton plutôt que de le brûler à l'aveugle.
        return {
            "match_key": mk,
            "recommendation": "SAVE",
            "note": (f"EV aujourd'hui={ev_today:.2f} mais aucune cote saisie pour les "
                     f"{n_unscored_future} matchs futurs — saisir les cotes à venir dans "
                     "mpp_points.json pour activer la comparaison."),
        }

    future_evs = np.array([p["mpp_ev"] for p in future])
    threshold = float(np.percentile(future_evs, threshold_pct))
    max_future_ev = float(future_evs.max())

    if ev_today >= threshold or ev_today >= max_future_ev:
        return {
            "match_key": mk,
            "recommendation": "USE",
            "note": (f"EV={ev_today:.2f} ≥ seuil {threshold_pct:.0f}e percentile "
                     f"({threshold:.2f}) — utiliser le ×2 sur {mk}."),
        }

    best_future = max(future, key=lambda p: p["mpp_ev"])
    bf = (
        f"{best_future['team_a']} vs {best_future['team_b']}"
        f" ({best_future['date']}, EV={best_future['mpp_ev']:.2f})"
    )
    return {
        "match_key": mk,
        "recommendation": "SAVE",
        "note": (f"EV aujourd'hui={ev_today:.2f} < seuil {threshold:.2f}. "
                 f"Conserver le jeton. Cible projetée : {bf}."),
    }


def run_optimizer(
    predictions: list[dict],
    reference_date: str,
    state_path: Path,
    confirm_double: bool = False,
) -> tuple[list[dict], dict]:
    """Enrich predictions with MPP data and apply ×2 decision.

    Args:
        predictions: List of match dicts already containing mpp_ev and best_mpp_score.
        reference_date: "YYYY-MM-DD" string.
        state_path: Path to mpp_state.json.
        confirm_double: When True AND the decision is USE, persist double_used in
            the state file. Default False so that simply (re)running predictions
            never burns the token — pass --confirm-double once the ×2 has really
            been played in the app.

    Returns:
        (enriched_predictions, double_decision)
    """
    state = load_state(state_path)
    decision = decide_double(predictions, reference_date, state)

    # Persist state ONLY on explicit confirmation : une simple recommandation
    # (ou un run de test) ne doit jamais consommer le jeton.
    if decision["recommendation"] == "USE":
        if confirm_double:
            state["double_used"] = True
            state["double_match"] = decision["match_key"]
            state["double_date"] = reference_date
            save_state(state, state_path)
        else:
            decision["note"] += (
                " [Recommandation seulement — état non enregistré. Une fois le ×2 "
                "réellement joué dans l'app, relancer avec --confirm-double.]"
            )

    for pred in predictions:
        mk = f"{pred['team_a']} vs {pred['team_b']}"
        if (decision["recommendation"] == "USE"
                and pred.get("date") == reference_date
                and mk == decision["match_key"]):
            pred["double_reco"] = "OUI"
            pred["double_note"] = decision["note"]
        elif decision["recommendation"] == "ALREADY_USED":
            pred["double_reco"] = "UTILISÉ"
            pred["double_note"] = decision["note"]
        else:
            pred["double_reco"] = "NON"
            pred["double_note"] = ""

    return predictions, decision
