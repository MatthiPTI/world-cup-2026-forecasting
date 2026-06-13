"""Couche marché : cotes du bookmaker → probabilités, blend avec le modèle, et edge.

Deux usages, volontairement séparés (voir AskUserQuestion 2026-06-13) :

  1. BLEND (pour le MPP) — on convertit les cotes en probabilités « marché »
     dé-vigées, puis on les mélange avec celles du modèle. Le marché price déjà la
     force inter-confédération que nos données internes captent mal, donc le blend
     corrige le biais de calendrier résiduel (le problème Curaçao) et améliore la
     calibration. On rescale la matrice de score pour coller aux marges 1/N/2
     blendées tout en gardant la « texture » des scores du modèle (ex. si home gagne,
     le 2-1 reste le plus probable). L'optimiseur MPP tourne ensuite sur cette matrice.

  2. EDGE (pour le pari réel) — on mesure où le modèle PUR diverge du marché. C'est
     volontairement le modèle non-blendé : si on blendait d'abord puis mesurait
     l'edge vs le marché, on raboterait mécaniquement son propre avantage. Une issue
     est « value » si P_modèle × cote > 1 (espérance positive en pariant à cette cote).

⚠ Le poids du blend n'est PAS encore validé par back-test (le dataset historique n'a
pas de cotes). Défaut 0.6 (léger penchant vers le marché, plus fin que nous), réglable
via --market-weight. À calibrer dès qu'on aura des cotes historiques.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mpp.mpp_optimizer import compute_match_ev

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MATRICES_FILE = DATA_DIR / "score_matrices.json"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"
POINTS_FILE = DATA_DIR / "mpp_points.json"
ODDS_FILE = DATA_DIR / "mpp_odds.json"

_OUTCOMES = ("home", "draw", "away")


# ------------------------------------------------------------------- de-vig --

def devig(odds: dict | None) -> dict | None:
    """Cotes décimales 1/N/2 → probabilités dé-vigées (méthode proportionnelle).

    La somme des probas implicites brutes (1/cote) dépasse 1 : c'est la marge du
    bookmaker (l'« overround »/vig). On la retire en renormalisant à 1.

    Renvoie {"home","draw","away","overround"} ou None si les cotes sont absentes
    ou invalides (toute cote ≤ 1 → entrée non remplie).
    """
    if odds is None:
        return None
    clean = {k: float(v) for k, v in odds.items() if not k.startswith("_")}
    try:
        oh, od, oa = clean["home"], clean["draw"], clean["away"]
    except KeyError:
        return None
    if min(oh, od, oa) <= 1.0:
        return None
    raw = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    overround = float(raw.sum())
    p = raw / overround
    return {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2]),
            "overround": overround}


def market_probs_from_points(points: dict | None) -> dict | None:
    """Probabilités marché DÉRIVÉES des points MPP (point ∝ cote).

    Les points MPP sont ~proportionnels à la cote, donc 1/point ∝ 1/cote ∝ proba
    implicite. La constante de proportionnalité s'annule à la renormalisation : on
    obtient donc les probas marché sans connaître le barème exact. Pratique pour le
    BLEND quand les cotes décimales ne sont pas saisies.

    ⚠ Ne donne PAS les cotes réelles → pas d'EV/Kelly (value bets) : ça exige de
    vraies cotes décimales, et de toute façon le MPP n'est pas un site de pari.
    Renvoie {"home","draw","away","overround":None} ou None si points absents/≤0.
    """
    if points is None:
        return None
    clean = {k: float(v) for k, v in points.items() if not k.startswith("_")}
    try:
        ph, pd_, pa = clean["home"], clean["draw"], clean["away"]
    except KeyError:
        return None
    if min(ph, pd_, pa) <= 0:
        return None
    raw = np.array([1.0 / ph, 1.0 / pd_, 1.0 / pa])
    p = raw / raw.sum()
    return {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2]), "overround": None}


# -------------------------------------------------------------------- blend --

def matrix_to_3way(matrix: np.ndarray) -> dict:
    """Marges 1/N/2 d'une matrice de score M[i,j]=P(home=i, away=j)."""
    return {
        "home": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "away": float(np.triu(matrix, 1).sum()),
    }


def blend_3way(model: dict, market: dict, market_weight: float) -> dict:
    """Mélange linéaire : (1-w)·modèle + w·marché, renormalisé à 1."""
    w = float(np.clip(market_weight, 0.0, 1.0))
    b = {k: (1.0 - w) * model[k] + w * market[k] for k in _OUTCOMES}
    s = sum(b.values())
    return {k: v / s for k, v in b.items()}


def blend_matrix(matrix: np.ndarray, blended_3way: dict) -> np.ndarray:
    """Rescale les cellules de la matrice pour que ses marges 1/N/2 == blended_3way.

    Chaque région (victoire home / nul / victoire away) est multipliée par un facteur
    constant pour atteindre la proba blendée, ce qui préserve la distribution
    conditionnelle des scores DANS chaque issue (la « texture » du modèle).
    """
    m = matrix.astype(float).copy()
    n = m.shape[0]
    g = np.arange(n)
    a, b = np.meshgrid(g, g, indexing="ij")
    masks = {"home": a > b, "draw": a == b, "away": a < b}
    cur = matrix_to_3way(m)
    for k in _OUTCOMES:
        if cur[k] > 0:
            m[masks[k]] *= blended_3way[k] / cur[k]
    total = m.sum()
    if total > 0:
        m /= total
    return m


# --------------------------------------------------------------------- edge --

def value_bets(model_3way: dict, odds: dict) -> list[dict]:
    """Edge du modèle PUR vs les cotes offertes.

    Pour chaque issue : EV par mise unitaire = P_modèle × cote − 1 (value si > 0),
    et fraction de Kelly = edge / (cote − 1). On compare au P implicite BRUT (1/cote),
    pas au dé-vigé, car on parie à la cote réelle (vig incluse).
    """
    clean = {k: float(v) for k, v in odds.items() if not k.startswith("_")}
    out = []
    for k in _OUTCOMES:
        o = clean[k]
        p = float(model_3way[k])
        ev = p * o - 1.0
        out.append({
            "outcome": k,
            "odds": round(o, 2),
            "model_prob": round(p, 4),
            "implied_prob": round(1.0 / o, 4),
            "ev": round(ev, 4),
            "kelly": round(ev / (o - 1.0), 4) if o > 1.0 else 0.0,
            "value": ev > 0.0,
        })
    return out


# ------------------------------------------------------------------- report --

def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_report(
    matrices: dict,
    predictions: list,
    odds_all: dict,
    points_all: dict | None,
    market_weight: float,
    value_margin: float,
    min_ev: float = 0.05,
    min_prob: float = 0.10,
) -> list[dict]:
    """Construit le rapport marché match par match.

    Un value bet n'est retenu (best_value) que s'il dépasse min_ev ET min_prob.
    Le garde-fou min_prob est essentiel : un « +EV » sur un outsider à faible
    probabilité (ex. Curaçao à 6 % coté 34) vient presque toujours du modèle qui
    SURévalue la longue cote — notre défaut connu — pas d'un vrai avantage. Sous
    min_prob, le marché (qui price l'inter-confédération) est quasi toujours plus juste.
    """
    points_all = points_all or {}
    rows: list[dict] = []
    for pred in predictions:
        key = f"{pred['team_a']} vs {pred['team_b']}"
        if key not in matrices:
            continue
        matrix = np.array(matrices[key], dtype=float)
        model_3way = matrix_to_3way(matrix)
        pts = points_all.get(key)
        odds_entry = odds_all.get(key)

        # Source du marché : vraies cotes en priorité (seules elles permettent l'edge),
        # sinon repli sur les points MPP (blend uniquement).
        market = devig(odds_entry)
        source = "odds" if market is not None else ""
        if market is None:
            market = market_probs_from_points(pts)
            source = "points" if market is not None else ""

        row = {
            "date": pred.get("date", ""),
            "match": key,
            "model_home": round(model_3way["home"], 3),
            "model_draw": round(model_3way["draw"], 3),
            "model_away": round(model_3way["away"], 3),
            "market_source": source,
        }

        pick_model = compute_match_ev(matrix, match_points=pts, value_margin=value_margin)
        row["pick_model"] = pick_model["score"]

        if market is None:
            row.update({
                "mkt_home": None, "mkt_draw": None, "mkt_away": None, "overround": None,
                "blend_home": None, "blend_draw": None, "blend_away": None,
                "pick_blended": pick_model["score"], "value_bets": [],
                "best_value": None,
            })
            rows.append(row)
            continue

        blended = blend_3way(model_3way, market, market_weight)
        b_matrix = blend_matrix(matrix, blended)
        pick_blended = compute_match_ev(b_matrix, match_points=pts, value_margin=value_margin)

        # Value bets : seulement avec de VRAIES cotes décimales (EV/Kelly impossibles
        # depuis les seuls points MPP).
        vbets: list[dict] = []
        best = None
        if source == "odds":
            vbets = value_bets(model_3way, odds_entry)
            candidates = [
                v for v in vbets
                if v["value"] and v["ev"] >= min_ev and v["model_prob"] >= min_prob
            ]
            best = max(candidates, key=lambda v: v["kelly"]) if candidates else None

        row.update({
            "mkt_home": round(market["home"], 3),
            "mkt_draw": round(market["draw"], 3),
            "mkt_away": round(market["away"], 3),
            "overround": round(market["overround"], 4) if market.get("overround") else None,
            "blend_home": round(blended["home"], 3),
            "blend_draw": round(blended["draw"], 3),
            "blend_away": round(blended["away"], 3),
            "pick_blended": pick_blended["score"],
            "value_bets": vbets,
            "best_value": best,
        })
        rows.append(row)
    return rows


def _print_report(rows: list[dict], market_weight: float) -> None:
    with_market = [r for r in rows if r["market_source"]]
    n_odds = sum(1 for r in with_market if r["market_source"] == "odds")
    n_points = sum(1 for r in with_market if r["market_source"] == "points")
    print("\n" + "=" * 80)
    print(f"  COUCHE MARCHÉ — {len(with_market)}/{len(rows)} matchs avec marché "
          f"({n_odds} cotes, {n_points} via points) | poids marché = {market_weight}")
    print("=" * 80)

    if not with_market:
        print("\n  Aucune cote ni point — renseigne data/mpp_odds.json (cotes décimales)")
        print("  ou data/mpp_points.json (points MPP) puis relance `mpp-market`.\n")
        return

    print("\n  Probabilités — modèle → marché → blend, pick blendé  (src: o=cotes, p=points)")
    print(f"  {'match':<32}{'src':>4}{'modèle (H/N/A)':>17}{'marché':>17}{'blend':>17}{'pick':>7}")
    for r in with_market:
        src = "o" if r["market_source"] == "odds" else "p"
        m = f"{r['model_home']:.0%}/{r['model_draw']:.0%}/{r['model_away']:.0%}"
        k = f"{r['mkt_home']:.0%}/{r['mkt_draw']:.0%}/{r['mkt_away']:.0%}"
        bl = f"{r['blend_home']:.0%}/{r['blend_draw']:.0%}/{r['blend_away']:.0%}"
        print(f"  {r['match']:<32}{src:>4}{m:>17}{k:>17}{bl:>17}{r['pick_blended']:>7}")

    # Value bets (edge du modèle pur vs marché — uniquement matchs avec vraies cotes)
    vbs = []
    for r in with_market:
        if r["best_value"]:
            vbs.append((r["match"], r["best_value"]))
    print("\n  ⚑ VALUE BETS (modèle pur vs cotes ; filtrés EV≥5 % et P_mod≥10 %)")
    if not vbs:
        print("    Aucun value bet fiable — le modèle ne bat pas le marché ici.")
    else:
        print(f"  {'match':<34}{'issue':>7}{'cote':>7}{'P_mod':>8}{'EV':>8}{'Kelly':>8}")
        for match, v in sorted(vbs, key=lambda x: -x[1]["kelly"]):
            print(f"  {match:<34}{v['outcome']:>7}{v['odds']:>7.2f}"
                  f"{v['model_prob']:>8.0%}{v['ev']:>+8.1%}{v['kelly']:>8.1%}")
        print("\n  Triés par Kelly (taille de mise risque-ajustée). En pratique : ¼ Kelly.")
        print("  Rappel : un +EV sur une longue cote à faible P_mod est filtré — c'est")
        print("  presque toujours le modèle qui se trompe sur la queue, pas un vrai edge.")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Couche marché : blend modèle/cotes + détection de value bets."
    )
    p.add_argument("--market-weight", type=float, default=0.6,
                   help="Poids du marché dans le blend [0-1] (défaut 0.6).")
    p.add_argument("--value-margin", type=float, default=1.15,
                   help="Garde anti-variance des picks (défaut 1.15 ; 1.0 = EV pure).")
    p.add_argument("--min-ev", type=float, default=0.05,
                   help="EV minimale pour retenir un value bet (défaut 0.05 = +5 %).")
    p.add_argument("--min-prob", type=float, default=0.10,
                   help="Proba modèle minimale pour un value bet (défaut 0.10). "
                        "Filtre les artefacts de longue cote (modèle peu fiable sur la queue).")
    args = p.parse_args()

    matrices = load_json(MATRICES_FILE)
    predictions = load_json(PREDICTIONS_FILE)
    if matrices is None or predictions is None:
        raise SystemExit(
            "Matrices/prédictions absentes — lance d'abord :\n"
            "  uv run --env-file .env mpp-predict --model bayes"
        )
    odds_all = load_json(ODDS_FILE) or {}
    points_all = load_json(POINTS_FILE) or {}

    rows = build_report(matrices, predictions, odds_all, points_all,
                        args.market_weight, args.value_margin,
                        min_ev=args.min_ev, min_prob=args.min_prob)
    _print_report(rows, args.market_weight)

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "market_report.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )
    flat = [{k: v for k, v in r.items() if k != "value_bets"} for r in rows]
    for r in flat:
        bv = r.pop("best_value", None)
        r["value_outcome"] = bv["outcome"] if bv else ""
        r["value_ev"] = bv["ev"] if bv else ""
    pd.DataFrame(flat).to_csv(DATA_DIR / "market_report.csv", index=False)
    print(f"  → {DATA_DIR/'market_report.json'}")
    print(f"  → {DATA_DIR/'market_report.csv'}\n")


if __name__ == "__main__":
    main()
