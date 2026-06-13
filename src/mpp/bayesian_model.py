"""Bayesian hierarchical Poisson model for football score prediction.

Implements the same public interface as DixonColesModel so predict.py can swap models.
Partial pooling by confederation corrects the strength-of-schedule bias that inflates
parameters for teams (e.g. Curaçao) that play mostly within weak confederations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    import pymc as pm
    import pytensor.tensor as pt
    _HAS_PYMC = True
except ImportError:
    _HAS_PYMC = False

from mpp.confederations import get_confederation


def _f(x: object) -> float:
    """Robustly extract a float from a numpy scalar / 0-d array / list."""
    return float(np.asarray(x).ravel()[0])


def _a(x: object) -> np.ndarray:
    """Robustly extract a 1-D numpy array from a MAP result value."""
    return np.asarray(x).ravel()


def dixon_coles_correction(
    matrix: np.ndarray, lambda_a: float, lambda_b: float, rho: float
) -> np.ndarray:
    """Apply the Dixon-Coles low-score adjustment to an independent-Poisson matrix.

    Independent Poisson underestimates draws and low scores. The DC tau factors
    reweight the four low-score cells (typical rho ≈ -0.05 à -0.15) :

        tau(0,0) = 1 - lambda_a*lambda_b*rho
        tau(1,0) = 1 + lambda_b*rho
        tau(0,1) = 1 + lambda_a*rho
        tau(1,1) = 1 - rho

    With rho < 0 : 0-0 et 1-1 montent, 1-0 et 0-1 descendent.
    The matrix is renormalized to sum to 1. rho=0 returns the matrix unchanged.
    """
    if rho == 0.0:
        return matrix
    out = matrix.copy()
    out[0, 0] *= max(1.0 - lambda_a * lambda_b * rho, 0.0)
    out[1, 0] *= max(1.0 + lambda_b * rho, 0.0)
    out[0, 1] *= max(1.0 + lambda_a * rho, 0.0)
    out[1, 1] *= max(1.0 - rho, 0.0)
    out /= out.sum()
    return out


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Mean-normalize weights so their average equals 1.

    Raw exp(-xi*days) weights are all ≤1, so their sum is much smaller than
    len(weights). This makes pm.Potential's weighted log-likelihood tiny relative
    to the prior mass, causing priors to dominate and all teams to collapse toward
    the global mean. Normalizing restores effective sample size while preserving
    the relative time-decay shape.
    """
    return weights * len(weights) / weights.sum()


class BayesianHierarchicalModel:
    """Hierarchical Bayesian Poisson model with confederation-level partial pooling.

    Model structure (non-centered parameterization):

        tau_att, tau_def  ~ HalfNormal(1)          # between-confederation spread
        sigma_att, sigma_def ~ HalfNormal(0.5)     # within-confederation spread

        conf_att_nc[c] ~ Normal(0, 1)              # confederation attack (non-centered)
        conf_att[c]    = tau_att * (conf_att_nc - mean(conf_att_nc))  # zero-sum

        att_raw[i] ~ Normal(0, 1)                  # team offset (non-centered)

        [if elo_ratings provided]
        beta_att, beta_def ~ HalfNormal(1)         # how much to trust Elo
        elo_z[i]           = (elo[i] - mean) / std # standardised Elo z-score

        attack[i]  = beta_att*elo_z[i] + conf_att[c(i)] + sigma_att*att_raw[i] − global_mean
        defense[i] = beta_def*elo_z[i] + conf_def[c(i)] + sigma_def*def_raw[i] − global_mean

        mu        ~ Normal(0, 1)                   # global log-goals intercept
        home_adv  ~ Normal(0.2, 0.2)               # applied only for non-neutral venues

        lambda_home = exp(mu + attack[home] - defense[away] + home_adv * !neutral)
        lambda_away = exp(mu + attack[away] - defense[home])

    Weighted likelihood via pm.Potential (weights are mean-normalized to restore ESS).
    """

    def __init__(
        self,
        draws: int = 500,
        tune: int = 500,
        chains: int = 2,
        inference: str = "nuts",
        rho: float = 0.0,
    ) -> None:
        """
        Args:
            draws: Posterior draws per chain (NUTS/ADVI).
            tune: Tuning steps (NUTS only).
            chains: Parallel chains (NUTS only).
            inference: "nuts" (proper Bayesian, slow), "advi" (variational, medium),
                       or "map" (point estimate, fast — debug only, collapses variances).
            rho: Dixon-Coles low-score correction applied post-hoc to score
                matrices (0.0 = off ; typiquement -0.05 à -0.15, à calibrer par
                back-test). Negative rho increases P(0-0) and P(1-1).
        """
        if not _HAS_PYMC:
            raise ImportError("PyMC is required: uv add pymc arviz")
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.inference = inference
        self.rho = rho

        self.teams: list[str] = []
        self._team_idx: dict[str, int] = {}
        self._conf_list: list[str] = []
        self._conf_idx: np.ndarray = np.array([], dtype=int)

        # Posterior mean (and std) attack/defense per team — populated by fit()
        self._attack: np.ndarray = np.array([])
        self._defense: np.ndarray = np.array([])
        self._attack_std: np.ndarray = np.array([])
        self._defense_std: np.ndarray = np.array([])
        self._mu: float = 0.0
        self._home_adv: float = 0.2
        self._elo_z: np.ndarray | None = None

        self._idata = None

    # ------------------------------------------------------------------ fit --

    def fit(
        self,
        matches: pd.DataFrame,
        elo_ratings: dict[str, float] | None = None,
    ) -> "BayesianHierarchicalModel":
        """Fit on a DataFrame with columns: home_team, away_team, home_score,
        away_score, neutral, weight.

        Args:
            matches: Training data from load_matches().
            elo_ratings: Optional dict of team → Elo rating (e.g. from compute_elo).
                When provided, adds an Elo-anchored prior: teams with high Elo get
                higher attack and defense priors, propagating inter-confederation
                strength without requiring direct cross-confederation matches.
        """
        if self.inference == "map":
            print(
                "WARNING: MAP effondre les variances hiérarchiques — debug uniquement. "
                "Utilisez --bayes-inference nuts pour des prédictions fiables."
            )

        self.teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self._team_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        conf_labels = [get_confederation(t) for t in self.teams]
        self._conf_list = sorted(set(conf_labels))
        _conf_to_idx = {c: i for i, c in enumerate(self._conf_list)}
        self._conf_idx = np.array([_conf_to_idx[c] for c in conf_labels], dtype=int)
        n_confs = len(self._conf_list)

        home_idx = matches["home_team"].map(self._team_idx).values.astype(int)
        away_idx = matches["away_team"].map(self._team_idx).values.astype(int)
        home_goals = matches["home_score"].values.astype(int)
        away_goals = matches["away_score"].values.astype(int)
        neutral = matches["neutral"].values.astype(bool)
        conf_idx = self._conf_idx

        # Normalize weights: raw exp(-xi*days) values are all ≤1, so their sum <<
        # len(weights). This makes pm.Potential negligible vs the prior, pulling all
        # teams toward the global mean. Normalization restores effective sample size.
        weights = matches["weight"].values.astype(float)
        weights = _normalize_weights(weights)

        # Compute Elo z-scores if ratings are provided
        if elo_ratings is not None:
            elo_vals = np.array([elo_ratings.get(t, 1500.0) for t in self.teams])
            elo_mean = float(elo_vals.mean())
            elo_std = float(max(elo_vals.std(), 1e-6))
            self._elo_z = (elo_vals - elo_mean) / elo_std
        else:
            self._elo_z = None

        print(
            f"Fitting Bayesian hierarchical model: {len(matches)} matches, "
            f"{n_teams} teams, {n_confs} confederations ({self.inference})"
            + (" + Elo anchoring" if elo_ratings else "") + "..."
        )

        with pm.Model() as pymc_model:
            # ---- Confederation-level hyperpriors ----
            tau_att = pm.HalfNormal("tau_att", sigma=1.0)
            tau_def = pm.HalfNormal("tau_def", sigma=1.0)

            # Non-centered confederation means; zero-sum enforced by centering
            conf_att_nc = pm.Normal("conf_att_nc", 0, 1, shape=n_confs)
            conf_def_nc = pm.Normal("conf_def_nc", 0, 1, shape=n_confs)
            conf_att = pm.Deterministic(
                "conf_att", tau_att * (conf_att_nc - pt.mean(conf_att_nc))
            )
            conf_def = pm.Deterministic(
                "conf_def", tau_def * (conf_def_nc - pt.mean(conf_def_nc))
            )

            # ---- Team-level (non-centered) ----
            sigma_att = pm.HalfNormal("sigma_att", sigma=0.5)
            sigma_def = pm.HalfNormal("sigma_def", sigma=0.5)
            att_raw = pm.Normal("att_raw", 0, 1, shape=n_teams)
            def_raw = pm.Normal("def_raw", 0, 1, shape=n_teams)

            # ---- Elo anchoring (optional) ----
            # beta_att/beta_def ~ HalfNormal(1): model learns how much to trust Elo.
            # Positive constraint: high Elo → better attack AND harder to score against.
            if self._elo_z is not None:
                elo_z_pt = pt.as_tensor_variable(self._elo_z.astype("float64"))
                beta_att = pm.HalfNormal("beta_att", sigma=1.0)
                beta_def = pm.HalfNormal("beta_def", sigma=1.0)
                elo_att_contrib = beta_att * elo_z_pt
                elo_def_contrib = beta_def * elo_z_pt
            else:
                elo_att_contrib = pt.zeros(n_teams)
                elo_def_contrib = pt.zeros(n_teams)

            # Global sum-to-zero: absorbs the grand mean into mu (identifiability)
            attack_unc = elo_att_contrib + conf_att[conf_idx] + sigma_att * att_raw
            attack = pm.Deterministic("attack", attack_unc - pt.mean(attack_unc))

            defense_unc = elo_def_contrib + conf_def[conf_idx] + sigma_def * def_raw
            defense = pm.Deterministic("defense", defense_unc - pt.mean(defense_unc))

            # ---- Global parameters ----
            mu = pm.Normal("mu", 0.0, 1.0)
            home_adv = pm.Normal("home_adv", 0.2, 0.2)

            # ---- Expected goals ----
            adv = pt.where(neutral, 0.0, home_adv)
            lambda_home = pt.exp(mu + attack[home_idx] - defense[away_idx] + adv)
            lambda_away = pt.exp(mu + attack[away_idx] - defense[home_idx])

            # ---- Weighted Poisson log-likelihood via pm.Potential ----
            # pm.Potential lets us apply per-match weights (time-decay from load_matches).
            home_logp = pm.logp(pm.Poisson.dist(mu=lambda_home), home_goals)
            away_logp = pm.logp(pm.Poisson.dist(mu=lambda_away), away_goals)
            pm.Potential("weighted_ll", pt.sum(weights * (home_logp + away_logp)))

        if self.inference == "map":
            with pymc_model:
                map_result = pm.find_MAP(progressbar=True)
            self._extract_map_params(map_result, conf_idx)

        elif self.inference == "advi":
            with pymc_model:
                approx = pm.fit(n=20_000, progressbar=True)
                self._idata = approx.sample(self.draws)
            self._extract_posterior_params()

        else:  # nuts (default)
            with pymc_model:
                self._idata = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    progressbar=True,
                )
            self._extract_posterior_params()

        self._print_top_attacks()
        return self

    # ------------------------------------------------------- param extraction --

    def _extract_map_params(self, map_result: dict, conf_idx: np.ndarray) -> None:
        """Compute attack/defense from MAP free-variable values."""
        tau_att_val = _f(map_result["tau_att"])
        tau_def_val = _f(map_result["tau_def"])
        conf_att_nc_val = _a(map_result["conf_att_nc"])
        conf_def_nc_val = _a(map_result["conf_def_nc"])
        sigma_att_val = _f(map_result["sigma_att"])
        sigma_def_val = _f(map_result["sigma_def"])
        att_raw_val = _a(map_result["att_raw"])
        def_raw_val = _a(map_result["def_raw"])

        conf_att_val = tau_att_val * (conf_att_nc_val - conf_att_nc_val.mean())
        conf_def_val = tau_def_val * (conf_def_nc_val - conf_def_nc_val.mean())

        if self._elo_z is not None:
            beta_att_val = _f(map_result["beta_att"])
            beta_def_val = _f(map_result["beta_def"])
            elo_att = beta_att_val * self._elo_z
            elo_def = beta_def_val * self._elo_z
        else:
            elo_att = np.zeros(len(self.teams))
            elo_def = np.zeros(len(self.teams))

        attack_unc = elo_att + conf_att_val[conf_idx] + sigma_att_val * att_raw_val
        defense_unc = elo_def + conf_def_val[conf_idx] + sigma_def_val * def_raw_val

        self._attack = attack_unc - attack_unc.mean()
        self._defense = defense_unc - defense_unc.mean()
        self._attack_std = np.zeros(len(self.teams))
        self._defense_std = np.zeros(len(self.teams))
        self._mu = _f(map_result["mu"])
        self._home_adv = _f(map_result["home_adv"])

    def _extract_posterior_params(self) -> None:
        """Compute posterior means and stds from NUTS/ADVI InferenceData."""
        post = self._idata.posterior
        # attack/defense are Deterministics: shape (n_chains, n_draws, n_teams)
        att = post["attack"].values.reshape(-1, len(self.teams))
        def_ = post["defense"].values.reshape(-1, len(self.teams))

        self._attack = att.mean(axis=0)
        self._defense = def_.mean(axis=0)
        self._attack_std = att.std(axis=0)
        self._defense_std = def_.std(axis=0)
        self._mu = float(post["mu"].values.mean())
        self._home_adv = float(post["home_adv"].values.mean())

    # ----------------------------------------------------------- predictions --

    def _expected_goals(
        self,
        team_a: str,
        team_b: str,
        adjustments: dict | None = None,
        global_boost: float = 1.0,
        neutral: bool = True,
    ) -> tuple[float, float]:
        ia = self._team_idx[team_a]
        ib = self._team_idx[team_b]
        la = float(np.exp(self._mu + self._attack[ia] - self._defense[ib]))
        lb = float(np.exp(self._mu + self._attack[ib] - self._defense[ia]))

        if not neutral:
            la *= float(np.exp(self._home_adv))

        if adjustments:
            adj_a = adjustments.get(team_a, {})
            adj_b = adjustments.get(team_b, {})
            la *= adj_a.get("attack", 1.0) * adj_b.get("defense", 1.0)
            lb *= adj_b.get("attack", 1.0) * adj_a.get("defense", 1.0)

        la *= global_boost
        lb *= global_boost
        return la, lb

    def score_matrix(
        self,
        team_a: str,
        team_b: str,
        max_goals: int = 8,
        adjustments: dict | None = None,
        global_boost: float = 1.0,
        neutral: bool = True,
    ) -> np.ndarray:
        """Return (max_goals+1)×(max_goals+1) probability matrix P(a=i, b=j)."""
        la, lb = self._expected_goals(team_a, team_b, adjustments, global_boost, neutral=neutral)
        g = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(g, la), poisson.pmf(g, lb))
        matrix = np.maximum(matrix, 0.0)
        matrix /= matrix.sum()
        return dixon_coles_correction(matrix, la, lb, self.rho)

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        adjustments: dict | None = None,
        global_boost: float = 1.0,
        neutral: bool = True,
    ) -> dict:
        """Return prediction dict (same structure as DixonColesModel.predict_match)."""
        matrix = self.score_matrix(
            team_a, team_b, adjustments=adjustments, global_boost=global_boost, neutral=neutral
        )
        win_a = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        win_b = float(np.triu(matrix, 1).sum())

        ga, gb = np.unravel_index(matrix.argmax(), matrix.shape)
        la, lb = self._expected_goals(
            team_a, team_b, adjustments, global_boost=global_boost, neutral=neutral
        )

        return {
            "team_a": team_a,
            "team_b": team_b,
            "predicted_score": f"{ga}-{gb}",
            "xg_a": round(la, 2),
            "xg_b": round(lb, 2),
            "win_a_prob": round(win_a, 3),
            "draw_prob": round(draw, 3),
            "win_b_prob": round(win_b, 3),
        }

    def team_strengths(self) -> pd.DataFrame:
        """Return DataFrame of posterior attack/defense ratings (with std if available)."""
        df = pd.DataFrame({
            "team": self.teams,
            "attack": self._attack.round(3),
            "defense": self._defense.round(3),
        })
        if self._attack_std.max() > 0:
            df["attack_std"] = self._attack_std.round(3)
            df["defense_std"] = self._defense_std.round(3)
        return df.sort_values("attack", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ misc --

    def _print_top_attacks(self) -> None:
        top = sorted(range(len(self.teams)), key=lambda i: -self._attack[i])[:5]
        has_std = self._attack_std.max() > 0
        print("Done. Top 5 by posterior mean attack:")
        for i in top:
            std_str = f" ±{self._attack_std[i]:.3f}" if has_std else ""
            conf = get_confederation(self.teams[i])
            print(f"  {self.teams[i]:<25} att={self._attack[i]:.3f}{std_str}  [{conf}]")
