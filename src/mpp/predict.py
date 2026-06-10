"""Generate World Cup 2026 group stage score predictions.

Usage:
    uv run mpp-predict                              # Dixon-Coles (default)
    uv run mpp-predict --model bayes                # Bayesian hierarchical (NUTS)
    uv run mpp-predict --model bayes --bayes-inference map  # fast MAP (debug only)
    uv run mpp-predict --refresh                    # re-download data before predicting
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mpp.data import download_data, load_all_matches, load_matches
from mpp.elo import get_team_ratings
from mpp.fixtures import TEAM_NAME_MAP, WC2026_FIXTURES
from mpp.model import DixonColesModel
from mpp.mpp_optimizer import (
    compute_markets,
    compute_match_ev,
    run_optimizer,
)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"
PROJECT_ROOT = Path(__file__).parent.parent.parent
ADJUSTMENTS_FILE = PROJECT_ROOT / "adjustments.json"
STATE_FILE = OUTPUT_DIR / "mpp_state.json"
POINTS_FILE = OUTPUT_DIR / "mpp_points.json"
MATRICES_FILE = OUTPUT_DIR / "score_matrices.json"


def _load_fixture_points() -> dict:
    """Load per-fixture MPP cote-points from data/mpp_points.json (optional).

    Returns an empty dict if the file is absent; callers fall back to modal score.
    """
    if not POINTS_FILE.exists():
        return {}
    with open(POINTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _load_adjustments() -> tuple[dict, float]:
    """Return (team_adjustments, global_scoring_boost)."""
    if not ADJUSTMENTS_FILE.exists():
        return {}, 1.0
    with open(ADJUSTMENTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    global_boost = float(data.get("_global_scoring_boost", 1.0))
    team_adj = {
        team: {k: v for k, v in adj.items() if k != "note"}
        for team, adj in data.items()
        if isinstance(adj, dict) and not team.startswith("_")
    }
    return team_adj, global_boost


def _print_adjustments(global_boost: float, adjustments: dict) -> None:
    if not ADJUSTMENTS_FILE.exists():
        return
    with open(ADJUSTMENTS_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    if global_boost == 1.0 and not adjustments:
        return
    print("Ajustements manuels actifs :")
    if global_boost != 1.0:
        print(f"  [global] scoring_boost×{global_boost}  (48 équipes — écarts de niveau accrus)")
    for team, adj in raw.items():
        if team.startswith("_") or not isinstance(adj, dict):
            continue
        parts = [f"{k}×{v}" for k, v in adj.items() if k not in ("note",)]
        note = adj.get("note", "")
        print(f"  {team}: {', '.join(parts)}" + (f"  ({note})" if note else ""))
    print()


def _build_model(model_type: str, bayes_inference: str, dc_rho: float = 0.0):
    if model_type == "bayes":
        from mpp.bayesian_model import BayesianHierarchicalModel
        return BayesianHierarchicalModel(
            draws=1000, tune=1000, chains=4, inference=bayes_inference, rho=dc_rho
        )
    return DixonColesModel(reg=0.3)


def _run_sanity_checks(model, results: list[dict]) -> None:
    """Print guard-rails after Bayesian inference to detect collapse."""
    print("\n" + "=" * 60)
    print("Garde-fous (sanity checks) :")

    # 1. Average expected goals across all fixtures
    xg_avg = np.mean([r["xg_a"] + r["xg_b"] for r in results])
    ok = 2.4 <= xg_avg <= 2.9
    flag = "OK" if ok else "WARNING"
    print(f"  [{flag}] Buts moyens attendus : {xg_avg:.2f}  (cible 2.4–2.9)")

    # 2. Favorite probability on a large mismatch (France vs Iraq)
    iraq_pred = next(
        (r for r in results
         if r.get("team_a") in ("France",) and r.get("team_b") in ("Iraq",)),
        None,
    )
    if iraq_pred:
        fav = iraq_pred["win_a_prob"]
        ok2 = fav >= 0.80
        flag2 = "OK" if ok2 else "WARNING — collapse persiste!"
        print(f"  [{flag2}] France vs Iraq win prob : {fav:.1%}  (cible >80%)")
    else:
        print("  [INFO] France vs Iraq non trouvé dans les fixtures.")

    # 3. Attack posterior spread
    att_std = float(model._attack_std.mean())
    att_range = float(model._attack.max() - model._attack.min())
    ok3 = att_range > 0.3
    flag3 = "OK" if ok3 else "WARNING — variance collapse!"
    print(f"  [{flag3}] Écart-type moyen (posterior) : {att_std:.3f}  |  "
          f"Range attaques : {att_range:.3f}  (cible range >0.3)")
    print("=" * 60 + "\n")


def _flatten_top5(top5: list[dict]) -> dict:
    out = {}
    for k in range(1, 6):
        if k - 1 < len(top5):
            out[f"score_{k}"] = top5[k - 1]["score"]
            out[f"prob_{k}"] = top5[k - 1]["prob"]
        else:
            out[f"score_{k}"] = ""
            out[f"prob_{k}"] = 0.0
    return out


def save_outputs(results: list[dict]) -> None:
    """Write predictions.json, predictions.csv and mpp_pronos.csv.

    Shared by mpp-predict (full model run) and mpp-optimize (optimizer-only rerun
    from saved score matrices).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    out_json = OUTPUT_DIR / "predictions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    out_csv = OUTPUT_DIR / "predictions.csv"
    flat_rows = []
    for r in results:
        row = {
            "group": r["group"],
            "date": r["date"],
            "team_a": r["team_a"],
            "team_b": r["team_b"],
            "predicted_score": r["predicted_score"],
            "xg_a": r["xg_a"],
            "xg_b": r["xg_b"],
            "win_a_prob": r["win_a_prob"],
            "draw_prob": r["draw_prob"],
            "win_b_prob": r["win_b_prob"],
            "over_1_5": r["over_1_5"],
            "over_2_5": r["over_2_5"],
            "over_3_5": r["over_3_5"],
            "btts": r["btts"],
        }
        row.update(_flatten_top5(r.get("top5_scores", [])))
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    out_mpp = OUTPUT_DIR / "mpp_pronos.csv"
    mpp_rows = []
    for r in results:
        mpp_rows.append({
            "group": r["group"],
            "date": r["date"],
            "team_a": r["team_a"],
            "team_b": r["team_b"],
            "best_mpp_score": r["best_mpp_score"],
            "mpp_ev": r["mpp_ev"],
            "mpp_ev_safe": r.get("mpp_ev_safe", ""),
            "issue_choisie": r.get("mpp_issue", ""),
            "points_issue": r.get("mpp_points_issue", ""),
            "proba_issue": r.get("mpp_proba_issue", ""),
            "est_value": "OUI" if r.get("mpp_est_value") else "NON",
            "win_a_prob": r["win_a_prob"],
            "draw_prob": r["draw_prob"],
            "win_b_prob": r["win_b_prob"],
            "xg_a": r["xg_a"],
            "xg_b": r["xg_b"],
            "double_reco": r.get("double_reco", "NON"),
            "double_note": r.get("double_note", ""),
        })
    pd.DataFrame(mpp_rows).to_csv(out_mpp, index=False)

    print(
        f"Saved {len(results)} predictions → "
        f"{out_json.name}, {out_csv.name}, {out_mpp.name}\n"
    )


def apply_match_ev(pred: dict, ev_result: dict) -> None:
    """Copy compute_match_ev fields onto a prediction dict (shared with optimize)."""
    pred["best_mpp_score"] = ev_result["score"]
    pred["mpp_ev"] = round(ev_result["ev"], 3)
    pred["mpp_ev_safe"] = round(ev_result["ev_safe"], 3)
    pred["mpp_issue"] = ev_result["issue"]
    pred["mpp_points_issue"] = ev_result["points_issue"]
    pred["mpp_proba_issue"] = ev_result["proba_issue"]
    pred["mpp_est_value"] = ev_result["est_value"]
    pred["mpp_points_manquants"] = ev_result["points_manquants"]


def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 predictions")
    parser.add_argument("--refresh", action="store_true", help="Re-download match data")
    parser.add_argument(
        "--model",
        choices=["dixoncoles", "bayes"],
        default="dixoncoles",
        help="Prediction model (default: dixoncoles)",
    )
    parser.add_argument(
        "--bayes-inference",
        choices=["nuts", "advi", "map"],
        default="nuts",
        help="Inference method when --model bayes (default: nuts)",
    )
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Reference date YYYY-MM-DD for ×2 optimizer (default: today)",
    )
    parser.add_argument(
        "--confirm-double",
        action="store_true",
        help="Enregistre l'utilisation du jeton ×2 dans mpp_state.json "
             "(à passer UNE FOIS, après avoir réellement joué le ×2 dans l'app)",
    )
    parser.add_argument(
        "--dc-rho",
        type=float,
        default=0.0,
        help="Correction Dixon-Coles des scores bas pour le modèle bayésien "
             "(ex: -0.1 augmente 0-0/1-1, diminue 1-0/0-1 ; défaut 0.0 = off, "
             "à calibrer par back-test)",
    )
    parser.add_argument(
        "--value-margin",
        type=float,
        default=1.15,
        help="Garde anti-variance : un pick outsider n'est retenu que si son EV "
             "dépasse value_margin × EV du pick sûr (défaut 1.15 = +15%%). "
             "1.0 = EV pure.",
    )
    args = parser.parse_args()

    if args.model == "bayes" and args.bayes_inference == "map":
        warnings.warn(
            "MAP effondre les variances hiérarchiques — debug uniquement. "
            "Utilisez --bayes-inference nuts pour des prédictions fiables.",
            stacklevel=1,
        )

    reference_date = (
        date.fromisoformat(args.reference_date)
        if args.reference_date
        else date.today()
    )

    if args.refresh:
        download_data(force=True)

    adjustments, global_boost = _load_adjustments()
    _print_adjustments(global_boost, adjustments)

    fixture_points = _load_fixture_points()
    if fixture_points:
        print(f"Points-cote chargés pour {len(fixture_points)} fixture(s) depuis mpp_points.json\n")
    else:
        print("Aucun mpp_points.json — mode fallback (score modal) pour l'optimiseur.\n")

    # ---- Elo ratings from full history ----
    print("Computing Elo ratings from full historical dataset...")
    elo_raw = load_all_matches(min_year=1960, reference_date=reference_date)
    elo_ratings = get_team_ratings(source="internal", matches=elo_raw)
    print(f"  {len(elo_ratings)} teams rated. "
          f"Top 5: {sorted(elo_ratings, key=elo_ratings.get, reverse=True)[:5]}\n")

    # ---- Training data ----
    if args.model == "bayes":
        matches = load_matches(
            min_year=2016,
            reference_date=reference_date,
            include_friendlies=True,
            friendly_weight=0.3,
            min_matches_per_team=35,
        )
    else:
        matches = load_matches(
            min_year=2016,
            reference_date=reference_date,
            include_friendlies=False,
            min_matches_per_team=35,
        )
    print(f"Loaded {len(matches)} matches for training.\n")

    model = _build_model(args.model, args.bayes_inference, dc_rho=args.dc_rho)
    if args.dc_rho != 0.0 and args.model == "bayes":
        print(f"Correction Dixon-Coles active : rho={args.dc_rho}\n")

    if args.model == "bayes":
        from mpp.bayesian_model import BayesianHierarchicalModel
        assert isinstance(model, BayesianHierarchicalModel)
        model.fit(matches, elo_ratings=elo_ratings)
    else:
        model.fit(matches)

    print(f"\nTop 10 teams by attack rating ({args.model}):")
    print(model.team_strengths().head(10).to_string(index=False))
    print()

    results = []
    matrices_out: dict[str, list] = {}
    missing_teams: set[str] = set()

    for fixture in WC2026_FIXTURES:
        ta_display = fixture["team_a"]
        tb_display = fixture["team_b"]
        ta = TEAM_NAME_MAP.get(ta_display, ta_display)
        tb = TEAM_NAME_MAP.get(tb_display, tb_display)

        if ta not in model._team_idx:
            missing_teams.add(ta)
            continue
        if tb not in model._team_idx:
            missing_teams.add(tb)
            continue

        pred = model.predict_match(ta, tb, adjustments=adjustments, global_boost=global_boost)
        matrix = model.score_matrix(ta, tb, adjustments=adjustments, global_boost=global_boost)

        markets = compute_markets(matrix)
        match_key = f"{ta_display} vs {tb_display}"
        mp = fixture_points.get(match_key)
        ev_result = compute_match_ev(matrix, match_points=mp, value_margin=args.value_margin)

        pred["group"] = fixture["group"]
        pred["date"] = fixture["date"]
        pred["team_a"] = ta_display
        pred["team_b"] = tb_display
        pred.update(markets)
        apply_match_ev(pred, ev_result)

        matrices_out[match_key] = np.round(matrix, 8).tolist()
        results.append(pred)

    if missing_teams:
        print(
            f"WARNING: {len(missing_teams)} team(s) not found in training data: "
            f"{sorted(missing_teams)}"
        )
        print("Those fixtures were skipped. Check TEAM_NAME_MAP in fixtures.py.\n")

    # ---- Sanity checks (Bayesian only) ----
    if args.model == "bayes":
        _run_sanity_checks(model, results)

    # ---- MPP optimizer ----
    ref_str = reference_date.isoformat()
    results, double_decision = run_optimizer(
        results, ref_str, STATE_FILE, confirm_double=args.confirm_double
    )

    print(f"Jeton ×2 : {double_decision['recommendation']}")
    print(f"  {double_decision['note']}\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- Save score matrices (pour mpp-optimize : recalcul sans refit) ----
    with open(MATRICES_FILE, "w", encoding="utf-8") as f:
        json.dump(matrices_out, f)
    print(f"Score matrices sauvegardées → {MATRICES_FILE.name} "
          "(mpp-optimize permet de recalculer les picks sans relancer le modèle)")

    save_outputs(results)
    print("=" * 70)

    current_group = None
    for r in results:
        if r["group"] != current_group:
            current_group = r["group"]
            print(f"\n  Group {current_group}")
            print("  " + "-" * 60)
        ta, tb = r["team_a"], r["team_b"]
        score = r["predicted_score"]
        mpp = r["best_mpp_score"]
        dbl = " ×2" if r.get("double_reco") == "OUI" else ""
        print(
            f"  {r['date']}  {ta:>22} vs {tb:<22}"
            f"  {score}  "
            f"({r['win_a_prob']:.0%} / {r['draw_prob']:.0%} / {r['win_b_prob']:.0%})"
            f"  MPP:{mpp}{dbl}"
        )

    print()


if __name__ == "__main__":
    main()
