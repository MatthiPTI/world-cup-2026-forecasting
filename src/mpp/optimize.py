"""Recalcule les picks MPP depuis les matrices sauvegardées — sans refit du modèle.

Workflow quotidien :
    1. Mettre à jour data/mpp_points.json (nouvelles cotes depuis l'app)
    2. uv run --env-file .env mpp-optimize          # ~1 seconde
    3. Recharger le site (Cmd+R)

Le refit complet (mpp-predict --model bayes) n'est nécessaire que lorsque de
nouveaux RÉSULTATS doivent être intégrés au modèle (typiquement après chaque
journée), pas à chaque saisie de cotes.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np

from mpp.mpp_optimizer import compute_match_ev, run_optimizer
from mpp.predict import (
    MATRICES_FILE,
    OUTPUT_DIR,
    POINTS_FILE,
    STATE_FILE,
    apply_match_ev,
    save_outputs,
)

PREDICTIONS_FILE = OUTPUT_DIR / "predictions.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalcule les picks MPP (EV cote-based) sans relancer le modèle"
    )
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Date de référence YYYY-MM-DD pour le jeton ×2 (défaut : aujourd'hui)",
    )
    parser.add_argument(
        "--confirm-double",
        action="store_true",
        help="Enregistre l'utilisation du jeton ×2 (après l'avoir réellement joué)",
    )
    parser.add_argument(
        "--value-margin",
        type=float,
        default=1.15,
        help="Garde anti-variance (défaut 1.15 ; 1.0 = EV pure)",
    )
    args = parser.parse_args()

    if not MATRICES_FILE.exists() or not PREDICTIONS_FILE.exists():
        raise SystemExit(
            "Matrices ou prédictions absentes — lance d'abord un run complet :\n"
            "  uv run --env-file .env mpp-predict --model bayes"
        )

    reference_date = (
        date.fromisoformat(args.reference_date) if args.reference_date else date.today()
    )

    with open(MATRICES_FILE, encoding="utf-8") as f:
        matrices = json.load(f)
    with open(PREDICTIONS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    points = {}
    if POINTS_FILE.exists():
        with open(POINTS_FILE, encoding="utf-8") as f:
            points = json.load(f)

    n_scored = 0
    for pred in results:
        match_key = f"{pred['team_a']} vs {pred['team_b']}"
        if match_key not in matrices:
            continue
        matrix = np.array(matrices[match_key])
        ev_result = compute_match_ev(
            matrix, match_points=points.get(match_key), value_margin=args.value_margin
        )
        apply_match_ev(pred, ev_result)
        if not ev_result["points_manquants"]:
            n_scored += 1

    print(f"{n_scored}/{len(results)} matchs avec cotes — value_margin={args.value_margin}")

    ref_str = reference_date.isoformat()
    results, decision = run_optimizer(
        results, ref_str, STATE_FILE, confirm_double=args.confirm_double
    )
    print(f"Jeton ×2 : {decision['recommendation']}")
    print(f"  {decision['note']}\n")

    save_outputs(results)

    value_picks = [r for r in results if r.get("mpp_est_value")]
    if value_picks:
        print("Picks VALUE retenus (EV outsider ≥ marge) :")
        for r in value_picks:
            print(f"  {r['date']}  {r['team_a']} vs {r['team_b']}: "
                  f"{r['best_mpp_score']} ({r['mpp_issue']}, "
                  f"{r['mpp_points_issue']} pts, P={r['mpp_proba_issue']:.0%}, "
                  f"EV={r['mpp_ev']:.1f} vs safe {r['mpp_ev_safe']:.1f})")


if __name__ == "__main__":
    main()
