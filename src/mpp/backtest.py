"""Walk-forward back-test for the MPP prediction models.

Why this exists
---------------
Every knob in this project (manual adjustments, the global scoring boost, the
time-decay xi, the Elo prior strength, rho…) is currently tuned by eye. Without
out-of-sample validation we cannot tell whether a change helps or hurts — and the
profitability goal is unreachable without a number to optimise.

This module retrains the model on data strictly *before* each checkpoint date and
scores its predictions on the matches that came *after*, so nothing the model saw
in training is ever evaluated (no leakage). It reports:

  * Probabilistic accuracy on the 1/N/2 market: log-loss, multiclass Brier, and
    RPS (the ranked probability score, the standard metric for ordered football
    outcomes). Lower is better for all three.
  * Hit rate of the most likely outcome, and exact-score hit rate (relevant to MPP).
  * A favourite-calibration table (is the model over/under-confident?).
  * A per-confederation bias table — the headline diagnostic for the
    strength-of-schedule problem: if CONCACAF shows a positive bias (predicted win
    rate > actual) and CONMEBOL a negative one, the model is over-rating soft
    schedules exactly as suspected. Re-run with --apply-adjustments to see whether
    adjustments.json actually reduces that bias.

Baselines (a model is only worth keeping if it beats these):
  * base-rate — predict the training-set frequency of home/draw/away every time.
  * elo — a multinomial logistic on the Elo rating difference (a genuine, cheap
    forecaster). If our model can't beat plain Elo, the extra machinery isn't earning
    its keep.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mpp.confederations import get_confederation
from mpp.data import DATA_DIR, WARMUP_TOURNAMENTS, load_all_matches, load_matches
from mpp.elo import compute_elo
from mpp.model import DixonColesModel

# Outcome encoding: 0 = home win, 1 = draw, 2 = away win (ordered by goal difference).
# Friendlies and pre-tournament warm-ups have rotated squads → excluded from evaluation.
_NON_REPRESENTATIVE = {"Friendly"} | WARMUP_TOURNAMENTS


# --------------------------------------------------------------------- metrics --

def outcome_index(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def log_loss_3way(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log-likelihood of the realised 1/N/2 outcome. Lower is better."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(outcomes)), outcomes])))


def brier_3way(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs the one-hot outcome."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def rps_3way(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score for ordered outcomes (H > D > A). Lower is better.

    RPS = 1/(K-1) * sum_{k=1..K-1} (cum_pred_k - cum_actual_k)^2, averaged over matches.
    The standard skill metric for football result forecasts; rewards probability mass
    placed *near* the true ordinal category, not just on it.
    """
    cum_pred = np.cumsum(probs, axis=1)[:, :-1]  # drop last (always 1)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    cum_actual = np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_pred - cum_actual) ** 2, axis=1) / (probs.shape[1] - 1)))


def summarise(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    return {
        "n": int(len(outcomes)),
        "log_loss": round(log_loss_3way(probs, outcomes), 4),
        "brier": round(brier_3way(probs, outcomes), 4),
        "rps": round(rps_3way(probs, outcomes), 4),
        "accuracy": round(float(np.mean(probs.argmax(axis=1) == outcomes)), 4),
    }


# ---------------------------------------------------------------- Elo baseline --

def _fit_elo_multinomial(
    dr: np.ndarray, outcomes: np.ndarray, weights: np.ndarray, l2: float = 1e-3
) -> np.ndarray:
    """Fit a weighted 3-class softmax of outcome on [1, dr/400].

    Returns a (2, 2) parameter array (classes home & draw; away is the reference).
    """
    x = np.column_stack([np.ones_like(dr), dr / 400.0])  # (n, 2)
    n_feat = x.shape[1]

    def neg_ll(flat: np.ndarray) -> float:
        w = flat.reshape(2, n_feat)
        logits = np.column_stack([x @ w[0], x @ w[1], np.zeros(len(x))])
        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        p = ex / ex.sum(axis=1, keepdims=True)
        ll = np.log(np.clip(p[np.arange(len(outcomes)), outcomes], 1e-12, 1.0))
        return -float(np.dot(weights, ll)) + l2 * float(np.sum(flat ** 2))

    res = minimize(neg_ll, np.zeros(2 * n_feat), method="L-BFGS-B")
    return res.x.reshape(2, n_feat)


def _predict_elo_multinomial(params: np.ndarray, dr: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones_like(dr), dr / 400.0])
    logits = np.column_stack([x @ params[0], x @ params[1], np.zeros(len(x))])
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ back-test --

def _checkpoints(start: date, end: date, freq_days: int) -> list[date]:
    out, cur = [], start
    while cur < end:
        out.append(cur)
        cur += timedelta(days=freq_days)
    return out


def run_backtest(
    start: date,
    end: date,
    *,
    model_type: str = "dixoncoles",
    refit_freq_days: int = 45,
    min_year_train: int = 2016,
    include_friendlies_train: bool = True,
    friendly_weight: float = 0.3,
    min_matches_per_team: int = 35,
    reg: float = 0.1,
    rho: float = 0.0,
    global_boost: float = 1.0,
    adjustments: dict | None = None,
    elo_min_year: int = 2008,
    bayes_inference: str = "map",
    max_goals: int = 8,
) -> dict:
    """Run the walk-forward back-test and return a results dict."""
    print(
        f"Back-test {model_type} | {start} → {end} | refit every {refit_freq_days}d"
        + (" | + adjustments" if adjustments else "")
    )

    # Evaluation pool: competitive matches only (friendlies have rotated squads and
    # are not representative of what we predict).
    eval_pool = load_all_matches(min_year=min_year_train, reference_date=end)
    eval_pool = eval_pool[~eval_pool["tournament"].isin(_NON_REPRESENTATIVE)].copy()
    eval_pool["edate"] = eval_pool["date"].dt.date

    records: list[dict] = []
    skipped = 0
    checkpoints = _checkpoints(start, end, refit_freq_days)

    for i, ckpt in enumerate(checkpoints):
        nxt = checkpoints[i + 1] if i + 1 < len(checkpoints) else end
        window = eval_pool[(eval_pool["edate"] > ckpt) & (eval_pool["edate"] <= nxt)]
        if window.empty:
            continue

        # --- train on data strictly before the checkpoint ---
        train = load_matches(
            min_year=min_year_train,
            reference_date=ckpt,
            include_friendlies=include_friendlies_train,
            friendly_weight=friendly_weight,
            min_matches_per_team=min_matches_per_team,
        )
        if len(train) < 200:
            continue

        # Elo ratings as of the checkpoint (used by the model anchor and the baseline).
        elo_hist = load_all_matches(min_year=elo_min_year, reference_date=ckpt)
        elo = compute_elo(elo_hist)

        if model_type == "bayes":
            from mpp.bayesian_model import BayesianHierarchicalModel

            model = BayesianHierarchicalModel(inference=bayes_inference, rho=rho)
            model.fit(train, elo_ratings=elo)
        else:
            model = DixonColesModel(reg=reg)
            model.fit(train)
        known = set(model.teams)

        # Elo-baseline multinomial, trained on the same window's outcomes.
        tr_h = train["home_team"].to_numpy()
        tr_a = train["away_team"].to_numpy()
        tr_dr = np.array(
            [
                elo.get(h, 1500.0) + (0.0 if neu else 100.0) - elo.get(a, 1500.0)
                for h, a, neu in zip(tr_h, tr_a, train["neutral"].to_numpy())
            ]
        )
        tr_out = np.array(
            [outcome_index(hs, as_) for hs, as_ in zip(train["home_score"], train["away_score"])]
        )
        elo_params = _fit_elo_multinomial(tr_dr, tr_out, train["weight"].to_numpy())

        # Base-rate baseline: weighted training frequency of H/D/A.
        w = train["weight"].to_numpy()
        base_rate = np.array([w[tr_out == k].sum() for k in (0, 1, 2)])
        base_rate = base_rate / base_rate.sum()

        # --- score every eval match in this window ---
        for _, m in window.iterrows():
            h, a = str(m["home_team"]), str(m["away_team"])
            if h not in known or a not in known:
                skipped += 1
                continue
            hs, as_ = int(m["home_score"]), int(m["away_score"])
            neu = bool(m["neutral"])
            out = outcome_index(hs, as_)

            mat = model.score_matrix(
                h, a, max_goals=max_goals, adjustments=adjustments,
                global_boost=global_boost, neutral=neu,
            )
            p_home = float(np.tril(mat, -1).sum())
            p_draw = float(np.trace(mat))
            p_away = float(np.triu(mat, 1).sum())
            gh, ga = np.unravel_index(mat.argmax(), mat.shape)

            dr = elo.get(h, 1500.0) + (0.0 if neu else 100.0) - elo.get(a, 1500.0)
            elo_p = _predict_elo_multinomial(elo_params, np.array([dr]))[0]

            records.append({
                "date": str(m["edate"]),
                "home": h, "away": a,
                "conf_home": get_confederation(h), "conf_away": get_confederation(a),
                "neutral": neu,
                "home_score": hs, "away_score": as_, "outcome": out,
                "exact_actual": f"{hs}-{as_}", "exact_pred": f"{gh}-{ga}",
                "m_home": p_home, "m_draw": p_draw, "m_away": p_away,
                "e_home": float(elo_p[0]), "e_draw": float(elo_p[1]), "e_away": float(elo_p[2]),
                "b_home": float(base_rate[0]), "b_draw": float(base_rate[1]),
                "b_away": float(base_rate[2]),
            })

    if not records:
        raise RuntimeError("No matches scored — check the date window and data availability.")

    df = pd.DataFrame(records)
    outcomes = df["outcome"].to_numpy()
    model_p = df[["m_home", "m_draw", "m_away"]].to_numpy()
    elo_p = df[["e_home", "e_draw", "e_away"]].to_numpy()
    base_p = df[["b_home", "b_draw", "b_away"]].to_numpy()

    metrics = {
        "model": summarise(model_p, outcomes),
        "elo_baseline": summarise(elo_p, outcomes),
        "base_rate": summarise(base_p, outcomes),
    }
    metrics["model"]["exact_score_rate"] = round(
        float((df["exact_pred"] == df["exact_actual"]).mean()), 4
    )

    return {
        "config": {
            "model": model_type, "start": str(start), "end": str(end),
            "refit_freq_days": refit_freq_days, "global_boost": global_boost,
            "rho": rho, "reg": reg, "adjustments_applied": bool(adjustments),
            "n_scored": len(df), "n_skipped_unknown_team": skipped,
        },
        "metrics": metrics,
        "calibration": _calibration_table(model_p, outcomes),
        "confederation_bias": _confederation_bias(df),
        "frame": df,
    }


def _calibration_table(probs: np.ndarray, outcomes: np.ndarray) -> list[dict]:
    """Reliability of the model's favourite: predicted vs realised hit rate by bin."""
    fav_p = probs.max(axis=1)
    fav_correct = (probs.argmax(axis=1) == outcomes).astype(float)
    bins = [0.33, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (fav_p >= lo) & (fav_p < hi)
        if mask.sum() == 0:
            continue
        table.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": int(mask.sum()),
            "pred_win_rate": round(float(fav_p[mask].mean()), 3),
            "actual_win_rate": round(float(fav_correct[mask].mean()), 3),
            "gap": round(float(fav_p[mask].mean() - fav_correct[mask].mean()), 3),
        })
    return table


def _confederation_bias(df: pd.DataFrame) -> list[dict]:
    """Per-confederation: model-predicted win rate vs actual win rate.

    bias > 0  → model over-rates this confederation (predicts more wins than happen);
                this is the strength-of-schedule symptom for soft confederations.
    bias < 0  → model under-rates it (tough schedules, e.g. CONMEBOL).
    """
    rows = []
    for _, r in df.iterrows():
        rows.append({"conf": r["conf_home"], "pred": r["m_home"], "won": int(r["outcome"] == 0)})
        rows.append({"conf": r["conf_away"], "pred": r["m_away"], "won": int(r["outcome"] == 2)})
    cb = pd.DataFrame(rows)
    out = []
    for conf, g in cb.groupby("conf"):
        out.append({
            "confederation": conf,
            "team_appearances": int(len(g)),
            "pred_win_rate": round(float(g["pred"].mean()), 3),
            "actual_win_rate": round(float(g["won"].mean()), 3),
            "bias": round(float(g["pred"].mean() - g["won"].mean()), 3),
        })
    return sorted(out, key=lambda x: -x["bias"])


# ------------------------------------------------------------------- printing --

def _print_report(res: dict) -> None:
    c = res["config"]
    print("\n" + "=" * 70)
    print(f"  BACK-TEST — {c['model']}  ({c['n_scored']} matchs notés, "
          f"{c['n_skipped_unknown_team']} ignorés équipe inconnue)")
    print("=" * 70)

    print("\n  Métriques probabilistes (1/N/2) — plus bas = mieux, sauf accuracy")
    print(f"  {'prédicteur':<14}{'n':>6}{'log_loss':>10}{'brier':>9}{'rps':>9}{'accuracy':>10}")
    for name, m in res["metrics"].items():
        print(f"  {name:<14}{m['n']:>6}{m['log_loss']:>10}{m['brier']:>9}"
              f"{m['rps']:>9}{m['accuracy']:>10}")
    print(f"\n  Taux de score exact (modèle) : "
          f"{res['metrics']['model']['exact_score_rate']:.1%}")

    print("\n  Calibration du favori (pred vs réel ; gap>0 = trop confiant)")
    print(f"  {'bin':<12}{'n':>6}{'pred':>8}{'réel':>8}{'gap':>8}")
    for row in res["calibration"]:
        print(f"  {row['bin']:<12}{row['n']:>6}{row['pred_win_rate']:>8}"
              f"{row['actual_win_rate']:>8}{row['gap']:>8}")

    print("\n  ⚑ Biais par confédération (bias>0 = surévaluée = calendrier mou)")
    print(f"  {'conf':<12}{'app.':>6}{'pred':>8}{'réel':>8}{'biais':>8}")
    for row in res["confederation_bias"]:
        print(f"  {row['confederation']:<12}{row['team_appearances']:>6}"
              f"{row['pred_win_rate']:>8}{row['actual_win_rate']:>8}{row['bias']:>8}")
    print()


# ------------------------------------------------------------------------ CLI --

def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward back-test for MPP models.")
    p.add_argument("--model", choices=["dixoncoles", "bayes"], default="dixoncoles")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default: end - 540d)")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--refit-freq", type=int, default=45, help="Days between refits.")
    p.add_argument("--global-boost", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--reg", type=float, default=0.1, help="Dixon-Coles L2 strength.")
    p.add_argument("--apply-adjustments", action="store_true",
                   help="Load adjustments.json and apply during the back-test "
                        "(to measure whether the manual fixes actually help).")
    p.add_argument("--bayes-inference", choices=["map", "advi", "nuts"], default="map")
    args = p.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=540)

    adjustments = None
    boost = args.global_boost
    if args.apply_adjustments:
        adj_path = Path(__file__).parent.parent.parent / "adjustments.json"
        raw = json.loads(adj_path.read_text())
        boost = float(raw.get("_global_scoring_boost", boost))
        adjustments = {k: v for k, v in raw.items() if not k.startswith("_")}

    res = run_backtest(
        start, end,
        model_type=args.model,
        refit_freq_days=args.refit_freq,
        rho=args.rho,
        reg=args.reg,
        global_boost=boost,
        adjustments=adjustments,
        bayes_inference=args.bayes_inference,
    )
    _print_report(res)

    DATA_DIR.mkdir(exist_ok=True)
    frame = res.pop("frame")
    (DATA_DIR / "backtest_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    frame.to_csv(DATA_DIR / "backtest_predictions.csv", index=False)
    print(f"  → {DATA_DIR/'backtest_metrics.json'}")
    print(f"  → {DATA_DIR/'backtest_predictions.csv'}\n")


if __name__ == "__main__":
    main()
