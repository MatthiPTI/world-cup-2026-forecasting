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

All predictors are scored in ONE pass, on the same checkpoints and the same matches,
so their per-match losses are paired. Differences are then tested with a paired
bootstrap (see `paired_bootstrap`): a 0.001-nat gap in mean log-loss means nothing
until we know how much it moves from one resample of the evaluation set to the next.

Known limitations (documented rather than hidden):
  * Every predictor — Elo included — is frozen between checkpoints: ratings are not
    updated match by match inside an evaluation window. Same information set for all,
    but a live Elo would be a somewhat stronger baseline.
  * The Elo multinomial link is fitted on ratings that already include the training
    results (in-sample features → probably slightly over-confident baseline). This
    is not test leakage: training data are always <= checkpoint < evaluated matches.
  * Matches involving a team below the `min_matches_per_team` threshold are skipped
    for every predictor alike.
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


def _onehot(outcomes: np.ndarray, k: int) -> np.ndarray:
    onehot = np.zeros((len(outcomes), k))
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return onehot


def per_match_losses(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, np.ndarray]:
    """Per-match log-loss, Brier and RPS (each an array of length n). Lower is better.

    Kept per match (not just averaged) so that two predictors scored on the same
    matches can be compared pairwise — see `paired_bootstrap`.
    """
    n, k = probs.shape
    p = np.clip(probs, 1e-12, 1.0)
    onehot = _onehot(outcomes, k)
    cum_pred = np.cumsum(probs, axis=1)[:, :-1]  # drop last (always 1)
    cum_actual = np.cumsum(onehot, axis=1)[:, :-1]
    return {
        "log_loss": -np.log(p[np.arange(n), outcomes]),
        "brier": np.sum((probs - onehot) ** 2, axis=1),
        "rps": np.sum((cum_pred - cum_actual) ** 2, axis=1) / (k - 1),
    }


def log_loss_3way(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log-likelihood of the realised 1/N/2 outcome. Lower is better."""
    return float(np.mean(per_match_losses(probs, outcomes)["log_loss"]))


def brier_3way(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs the one-hot outcome."""
    return float(np.mean(per_match_losses(probs, outcomes)["brier"]))


def rps_3way(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score for ordered outcomes (H > D > A). Lower is better.

    RPS = 1/(K-1) * sum_{k=1..K-1} (cum_pred_k - cum_actual_k)^2, averaged over matches.
    The standard skill metric for football result forecasts; rewards probability mass
    placed *near* the true ordinal category, not just on it.
    """
    return float(np.mean(per_match_losses(probs, outcomes)["rps"]))


def summarise(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    return {
        "n": int(len(outcomes)),
        "log_loss": round(log_loss_3way(probs, outcomes), 4),
        "brier": round(brier_3way(probs, outcomes), 4),
        "rps": round(rps_3way(probs, outcomes), 4),
        "accuracy": round(float(np.mean(probs.argmax(axis=1) == outcomes)), 4),
    }


# ---------------------------------------------------------------- significance --

def paired_bootstrap(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    clusters: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap of the mean per-match loss difference d = loss_a - loss_b.

    d < 0 means predictor A is better. Pairing matters: both predictors face the same
    matches, so most of the match-to-match noise (a 5-1 upset hurts everyone) cancels
    in d. Bootstrapping each predictor's mean separately would ignore that and give
    intervals several times too wide.

    clusters: optional label per match (here: the refit window). When given, whole
        clusters are resampled instead of single matches. Matches scored by the same
        fit share its parameter errors, so their d's are correlated; the i.i.d.
        bootstrap would then understate the variance. Few clusters (< ~10) make this
        interval unreliable — report both.

    p_value: two-sided, by inverting the percentile interval (the smallest alpha at
        which the interval would exclude 0).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    if clusters is None:
        n_units = n
        boot = np.empty(n_boot)
        for start in range(0, n_boot, 1000):  # chunked: n_boot × n indices can be large
            stop = min(start + 1000, n_boot)
            boot[start:stop] = d[rng.integers(0, n, size=(stop - start, n))].mean(axis=1)
    else:
        _, inv = np.unique(np.asarray(clusters), return_inverse=True)
        sums = np.bincount(inv, weights=d)
        counts = np.bincount(inv).astype(float)
        n_units = len(sums)
        pick = rng.integers(0, n_units, size=(n_boot, n_units))
        boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)

    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p_value = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p_value,
        "n": int(n),
        "n_units": int(n_units),
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj.tolist()


def minimum_detectable_effect(d: np.ndarray, power_z: float = 0.84, alpha_z: float = 1.96):
    """Smallest mean difference detectable with 80 % power at 5 % two-sided (i.i.d.)."""
    return float((alpha_z + power_z) * np.std(d, ddof=1) / np.sqrt(len(d)))


def compare_predictors(
    frame: pd.DataFrame,
    predictors: list[str],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    min_clusters: int = 5,
) -> list[dict]:
    """All pairwise paired-bootstrap tests between `predictors`, for each metric.

    Holm correction is applied per metric across the pairs (the family is "which
    predictors differ"), on the cluster-bootstrap p-values (the conservative ones).
    With fewer than `min_clusters` refit windows (e.g. a single tournament) the
    cluster bootstrap is meaningless — with 2 clusters it can return p = 0 — so it is
    skipped (None) and Holm falls back to the i.i.d. p-values.
    """
    outcomes = frame["outcome"].to_numpy()
    windows = frame["window"].to_numpy()
    losses = {
        p: per_match_losses(frame[[f"{p}_home", f"{p}_draw", f"{p}_away"]].to_numpy(), outcomes)
        for p in predictors
    }
    pairs = [(a, b) for i, a in enumerate(predictors) for b in predictors[i + 1:]]
    use_clusters = len(np.unique(windows)) >= min_clusters
    rows: list[dict] = []
    for metric in ("log_loss", "brier", "rps"):
        block = []
        for a, b in pairs:
            la, lb = losses[a][metric], losses[b][metric]
            iid = paired_bootstrap(la, lb, n_boot=n_boot, seed=seed)
            clu = (paired_bootstrap(la, lb, clusters=windows, n_boot=n_boot, seed=seed)
                   if use_clusters else None)
            block.append({
                "metric": metric, "a": a, "b": b,
                "mean_diff": round(iid["mean_diff"], 5),
                "ci_iid": [round(iid["ci_low"], 5), round(iid["ci_high"], 5)],
                "p_iid": round(iid["p_value"], 4),
                "ci_cluster": [round(clu["ci_low"], 5), round(clu["ci_high"], 5)] if clu else None,
                "p_cluster": round(clu["p_value"], 4) if clu else None,
                "n": iid["n"], "n_clusters": int(len(np.unique(windows))),
                "mde_80": round(minimum_detectable_effect(la - lb), 5),
            })
        key = "p_cluster" if use_clusters else "p_iid"
        for row, adj in zip(block, holm([r[key] for r in block])):
            row["p_holm"] = round(adj, 4)
            row["holm_on"] = key
        rows.extend(block)
    return rows


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


# "bayes_noncentered" = the original Bayesian parameterization, kept to measure whether
# fixing its NUTS divergences changed predictive performance.
FITTED_MODELS = ("dixoncoles", "bayes", "bayes_noncentered")
BASELINES = ("elo", "base_rate")


def _fit_model(name: str, train: pd.DataFrame, elo: dict, *, reg: float, rho: float,
               bayes_inference: str, bayes_kwargs: dict | None):
    if name.startswith("bayes"):
        from mpp.bayesian_model import BayesianHierarchicalModel

        param = "noncentered" if name == "bayes_noncentered" else None  # None = auto
        model = BayesianHierarchicalModel(inference=bayes_inference, rho=rho,
                                          parameterization=param, **(bayes_kwargs or {}))
        return model.fit(train, elo_ratings=elo)
    return DixonColesModel(reg=reg).fit(train)


def run_backtest(
    start: date,
    end: date,
    *,
    models: tuple[str, ...] | list[str] = ("dixoncoles",),
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
    bayes_inference: str = "nuts",
    bayes_kwargs: dict | None = None,
    max_goals: int = 8,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Run the walk-forward back-test for `models` + the Elo and base-rate baselines.

    Every predictor is scored on exactly the same matches (a match is kept only if
    all fitted models know both teams), so per-match losses are paired and the
    significance tests in `compare_predictors` are valid.
    """
    models = list(dict.fromkeys(models))  # dedupe, keep order
    has_bayes = any(m.startswith("bayes") for m in models)
    unknown = set(models) - set(FITTED_MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}; choose from {FITTED_MODELS}")
    print(
        f"Back-test {'+'.join(models)} vs Elo | {start} → {end} | refit every "
        f"{refit_freq_days}d" + (f" | bayes={bayes_inference}" if has_bayes else "")
        + (" | + adjustments" if adjustments else "")
    )

    # Evaluation pool: competitive matches only (friendlies have rotated squads and
    # are not representative of what we predict).
    eval_pool = load_all_matches(min_year=min_year_train, reference_date=end)
    eval_pool = eval_pool[~eval_pool["tournament"].isin(_NON_REPRESENTATIVE)].copy()
    eval_pool["edate"] = eval_pool["date"].dt.date

    records: list[dict] = []
    fit_diagnostics: list[dict] = []
    skipped = 0
    checkpoints = _checkpoints(start, end, refit_freq_days)

    for i, ckpt in enumerate(checkpoints):
        nxt = checkpoints[i + 1] if i + 1 < len(checkpoints) else end
        window = eval_pool[(eval_pool["edate"] > ckpt) & (eval_pool["edate"] <= nxt)]
        if window.empty:
            continue

        # --- train on data up to the checkpoint (evaluated matches are all after it) ---
        train = load_matches(
            min_year=min_year_train,
            reference_date=ckpt,
            include_friendlies=include_friendlies_train,
            friendly_weight=friendly_weight,
            min_matches_per_team=min_matches_per_team,
        )
        if len(train) < 200:
            continue

        # Elo ratings as of the checkpoint (used by the Bayesian anchor and the baseline).
        elo_hist = load_all_matches(min_year=elo_min_year, reference_date=ckpt)
        elo = compute_elo(elo_hist)

        fitted = {}
        for name in models:
            fitted[name] = _fit_model(name, train, elo, reg=reg, rho=rho,
                                      bayes_inference=bayes_inference,
                                      bayes_kwargs=bayes_kwargs)
            diag = getattr(fitted[name], "diagnostics", None)
            if diag:
                fit_diagnostics.append({"window": i, "checkpoint": str(ckpt),
                                        "model": name, **diag})
        known = set.intersection(*(set(m.teams) for m in fitted.values()))

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

            rec = {
                "date": str(m["edate"]), "window": i, "tournament": m["tournament"],
                "home": h, "away": a,
                "conf_home": get_confederation(h), "conf_away": get_confederation(a),
                "neutral": neu,
                "home_score": hs, "away_score": as_, "outcome": outcome_index(hs, as_),
                "exact_actual": f"{hs}-{as_}",
            }
            for name, model in fitted.items():
                mat = model.score_matrix(
                    h, a, max_goals=max_goals, adjustments=adjustments,
                    global_boost=global_boost, neutral=neu,
                )
                gh, ga = np.unravel_index(mat.argmax(), mat.shape)
                rec[f"{name}_home"] = float(np.tril(mat, -1).sum())
                rec[f"{name}_draw"] = float(np.trace(mat))
                rec[f"{name}_away"] = float(np.triu(mat, 1).sum())
                rec[f"{name}_exact"] = f"{gh}-{ga}"

            dr = elo.get(h, 1500.0) + (0.0 if neu else 100.0) - elo.get(a, 1500.0)
            elo_p = _predict_elo_multinomial(elo_params, np.array([dr]))[0]
            rec.update({"elo_home": float(elo_p[0]), "elo_draw": float(elo_p[1]),
                        "elo_away": float(elo_p[2])})
            rec.update({"base_rate_home": float(base_rate[0]),
                        "base_rate_draw": float(base_rate[1]),
                        "base_rate_away": float(base_rate[2])})
            records.append(rec)

    if not records:
        raise RuntimeError("No matches scored — check the date window and data availability.")

    df = pd.DataFrame(records)
    outcomes = df["outcome"].to_numpy()
    predictors = models + list(BASELINES)

    def probs(name: str) -> np.ndarray:
        return df[[f"{name}_home", f"{name}_draw", f"{name}_away"]].to_numpy()

    metrics = {name: summarise(probs(name), outcomes) for name in predictors}
    for name in models:
        metrics[name]["exact_score_rate"] = round(
            float((df[f"{name}_exact"] == df["exact_actual"]).mean()), 4
        )

    # Significance: fitted models vs each other and vs Elo (base-rate is not a contender).
    contenders = models + ["elo"]

    return {
        "config": {
            "models": models, "start": str(start), "end": str(end),
            "refit_freq_days": refit_freq_days, "global_boost": global_boost,
            "rho": rho, "reg": reg, "adjustments_applied": bool(adjustments),
            "bayes_inference": bayes_inference if has_bayes else None,
            "bayes_kwargs": bayes_kwargs if has_bayes else None,
            "min_matches_per_team": min_matches_per_team,
            "n_scored": len(df), "n_skipped_unknown_team": skipped,
            "n_windows": int(df["window"].nunique()), "n_boot": n_boot, "seed": seed,
        },
        "metrics": metrics,
        "significance": compare_predictors(df, contenders, n_boot=n_boot, seed=seed),
        "calibration": {name: _calibration_table(probs(name), outcomes) for name in contenders},
        "confederation_bias": {name: _confederation_bias(df, name) for name in contenders},
        "fit_diagnostics": fit_diagnostics,
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


def _confederation_bias(df: pd.DataFrame, name: str) -> list[dict]:
    """Per-confederation: predicted win rate vs actual win rate for predictor `name`.

    bias > 0  → model over-rates this confederation (predicts more wins than happen);
                this is the strength-of-schedule symptom for soft confederations.
    bias < 0  → model under-rates it (tough schedules, e.g. CONMEBOL).
    """
    won_home = (df["outcome"] == 0).astype(int)
    won_away = (df["outcome"] == 2).astype(int)
    cb = pd.concat([
        pd.DataFrame({"conf": df["conf_home"], "pred": df[f"{name}_home"], "won": won_home}),
        pd.DataFrame({"conf": df["conf_away"], "pred": df[f"{name}_away"], "won": won_away}),
    ])
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
    print("\n" + "=" * 78)
    print(f"  BACK-TEST — {' + '.join(c['models'])} vs baselines  ({c['n_scored']} matchs notés, "
          f"{c['n_windows']} fenêtres, {c['n_skipped_unknown_team']} ignorés équipe inconnue)")
    print("=" * 78)

    print("\n  Métriques probabilistes (1/N/2) — plus bas = mieux, sauf accuracy")
    print(f"  {'prédicteur':<14}{'n':>6}{'log_loss':>10}{'brier':>9}{'rps':>9}"
          f"{'accuracy':>10}{'exact':>8}")
    for name, m in res["metrics"].items():
        exact = f"{m['exact_score_rate']:.1%}" if "exact_score_rate" in m else ""
        print(f"  {name:<14}{m['n']:>6}{m['log_loss']:>10}{m['brier']:>9}"
              f"{m['rps']:>9}{m['accuracy']:>10}{exact:>8}")

    print("\n  Significativité — bootstrap apparié, diff = A − B (diff<0 → A meilleur)")
    print("  IC 95 % i.i.d. (par match) et par fenêtre de refit ; p Holm sur les p par fenêtre "
          "(sur les p i.i.d. si < 5 fenêtres)")
    print(f"  {'métrique':<9}{'A vs B':<24}{'diff':>9}{'IC iid':>22}{'IC fenêtre':>22}"
          f"{'p Holm':>8}{'MDE':>8}")
    for r in res["significance"]:
        ci_i = f"[{r['ci_iid'][0]:+.4f}, {r['ci_iid'][1]:+.4f}]"
        ci_c = (f"[{r['ci_cluster'][0]:+.4f}, {r['ci_cluster'][1]:+.4f}]"
                if r["ci_cluster"] else "n/a")
        print(f"  {r['metric']:<9}{r['a'] + ' vs ' + r['b']:<24}{r['mean_diff']:>+9.4f}"
              f"{ci_i:>22}{ci_c:>22}{r['p_holm']:>8}{r['mde_80']:>8.4f}")
    if c["n_windows"] < 10:
        print(f"  ⚠ {c['n_windows']} fenêtres seulement : l'IC par fenêtre est peu fiable "
              "(réduire --refit-freq).")

    for name, table in res["calibration"].items():
        print(f"\n  Calibration du favori — {name} (pred vs réel ; gap>0 = trop confiant)")
        print(f"  {'bin':<12}{'n':>6}{'pred':>8}{'réel':>8}{'gap':>8}")
        for row in table:
            print(f"  {row['bin']:<12}{row['n']:>6}{row['pred_win_rate']:>8}"
                  f"{row['actual_win_rate']:>8}{row['gap']:>8}")

    for name, table in res["confederation_bias"].items():
        print(f"\n  ⚑ Biais par confédération — {name} (bias>0 = surévaluée = calendrier mou)")
        print(f"  {'conf':<12}{'app.':>6}{'pred':>8}{'réel':>8}{'biais':>8}")
        for row in table:
            print(f"  {row['confederation']:<12}{row['team_appearances']:>6}"
                  f"{row['pred_win_rate']:>8}{row['actual_win_rate']:>8}{row['bias']:>8}")

    if res["fit_diagnostics"]:
        print("\n  Diagnostics d'échantillonnage (bayes)")
        for d in res["fit_diagnostics"]:
            print(f"  fenêtre {d['window']:>2} ({d['checkpoint']}) : "
                  + ", ".join(f"{k}={v}" for k, v in d.items()
                              if k not in {"window", "checkpoint", "model"}))
    print()


# ------------------------------------------------------------------------ CLI --

def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward back-test for MPP models.")
    p.add_argument("--models", nargs="+", choices=list(FITTED_MODELS), default=["dixoncoles"],
                   help="Fitted models to score (Elo and base-rate baselines are always "
                        "included). Ex. : --models dixoncoles bayes")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default: end - 540d)")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--refit-freq", type=int, default=45, help="Days between refits.")
    p.add_argument("--global-boost", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--reg", type=float, default=0.1, help="Dixon-Coles L2 strength.")
    p.add_argument("--apply-adjustments", action="store_true",
                   help="Load adjustments.json and apply during the back-test "
                        "(to measure whether the manual fixes actually help).")
    p.add_argument("--bayes-inference", choices=["map", "advi", "nuts"], default="nuts")
    p.add_argument("--bayes-chains", type=int, default=4)
    p.add_argument("--bayes-draws", type=int, default=1000)
    p.add_argument("--bayes-tune", type=int, default=1000)
    p.add_argument("--bayes-target-accept", type=float, default=0.8)
    p.add_argument("--n-boot", type=int, default=10_000, help="Bootstrap resamples.")
    p.add_argument("--seed", type=int, default=0, help="Seed (bootstrap + NUTS).")
    p.add_argument("--out-prefix", type=str, default="backtest",
                   help="Output files: data/<prefix>_metrics.json, <prefix>_predictions.csv")
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
        models=args.models,
        refit_freq_days=args.refit_freq,
        rho=args.rho,
        reg=args.reg,
        global_boost=boost,
        adjustments=adjustments,
        bayes_inference=args.bayes_inference,
        bayes_kwargs={
            "chains": args.bayes_chains, "draws": args.bayes_draws, "tune": args.bayes_tune,
            "target_accept": args.bayes_target_accept, "random_seed": args.seed,
        },
        n_boot=args.n_boot,
        seed=args.seed,
    )
    _print_report(res)

    DATA_DIR.mkdir(exist_ok=True)
    frame = res.pop("frame")
    metrics_path = DATA_DIR / f"{args.out_prefix}_metrics.json"
    preds_path = DATA_DIR / f"{args.out_prefix}_predictions.csv"
    metrics_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    frame.to_csv(preds_path, index=False)
    print(f"  → {metrics_path}")
    print(f"  → {preds_path}\n")


def report_from_predictions(
    frame: pd.DataFrame,
    *,
    tournaments: list[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Recompute metrics + significance from a saved predictions frame (no refit).

    Used for sub-population analyses (e.g. only the World Cup) and to regenerate the
    README tables. Fitted models are detected from the `<name>_exact` columns.
    """
    if "tournament" not in frame.columns:  # files written before the column existed
        from mpp.data import RESULTS_FILE

        res = pd.read_csv(RESULTS_FILE, usecols=["date", "home_team", "away_team", "tournament"])
        res = res.rename(columns={"home_team": "home", "away_team": "away"})
        frame = frame.merge(res, on=["date", "home", "away"], how="left")
    if tournaments:
        frame = frame[frame["tournament"].isin(tournaments)]
    if frame.empty:
        raise RuntimeError("No match left after filtering.")

    models = [c[: -len("_exact")] for c in frame.columns if c.endswith("_exact")]
    outcomes = frame["outcome"].to_numpy()
    metrics = {}
    for name in models + list(BASELINES):
        cols = [f"{name}_home", f"{name}_draw", f"{name}_away"]
        metrics[name] = summarise(frame[cols].to_numpy(), outcomes)
    return {
        "n": len(frame),
        "n_windows": int(frame["window"].nunique()),
        "tournaments": tournaments,
        "metrics": metrics,
        "significance": compare_predictors(frame, models + ["elo"], n_boot=n_boot, seed=seed),
    }


def main_report() -> None:
    p = argparse.ArgumentParser(
        description="Metrics + paired bootstrap from a saved back-test predictions CSV."
    )
    p.add_argument("predictions", help="e.g. data/backtest_full_predictions.csv")
    p.add_argument("--tournament", action="append", default=None,
                   help="Keep only this tournament (repeatable), e.g. 'FIFA World Cup'.")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rep = report_from_predictions(pd.read_csv(args.predictions), tournaments=args.tournament,
                                  n_boot=args.n_boot, seed=args.seed)
    print(f"\n  {rep['n']} matchs, {rep['n_windows']} fenêtres"
          + (f" — {', '.join(rep['tournaments'])}" if rep["tournaments"] else ""))
    print(f"\n  {'prédicteur':<20}{'log_loss':>10}{'brier':>9}{'rps':>9}{'accuracy':>10}")
    for name, m in rep["metrics"].items():
        print(f"  {name:<20}{m['log_loss']:>10}{m['brier']:>9}{m['rps']:>9}{m['accuracy']:>10}")
    print(f"\n  {'métrique':<9}{'A vs B':<34}{'diff':>9}{'IC iid':>22}{'IC fenêtre':>22}"
          f"{'p Holm':>8}{'MDE':>8}")
    for r in rep["significance"]:
        ci_i = f"[{r['ci_iid'][0]:+.4f}, {r['ci_iid'][1]:+.4f}]"
        ci_c = (f"[{r['ci_cluster'][0]:+.4f}, {r['ci_cluster'][1]:+.4f}]"
                if r["ci_cluster"] else "n/a")
        print(f"  {r['metric']:<9}{r['a'] + ' vs ' + r['b']:<34}{r['mean_diff']:>+9.4f}"
              f"{ci_i:>22}{ci_c:>22}{r['p_holm']:>8}{r['mde_80']:>8.4f}")
    print()


if __name__ == "__main__":
    main()
