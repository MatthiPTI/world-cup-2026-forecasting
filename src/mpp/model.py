"""Dixon-Coles Poisson model for predicting football scores."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return k * np.log(np.maximum(lam, 1e-10)) - lam - gammaln(k + 1)


def _tau_vectorized(
    hs: np.ndarray, aws: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float
) -> np.ndarray:
    """Dixon-Coles low-score correction applied to full match arrays."""
    tau = np.ones(len(hs))
    tau[(hs == 0) & (aws == 0)] = 1 - lh[(hs == 0) & (aws == 0)] * la[(hs == 0) & (aws == 0)] * rho
    tau[(hs == 0) & (aws == 1)] = 1 + lh[(hs == 0) & (aws == 1)] * rho
    tau[(hs == 1) & (aws == 0)] = 1 + la[(hs == 1) & (aws == 0)] * rho
    tau[(hs == 1) & (aws == 1)] = 1 - rho
    return tau


class DixonColesModel:
    """Dixon-Coles model fitted by maximum weighted log-likelihood.

    Parameter vector layout: [attack_0..N-1, defense_0..N-1, mu, rho, home_adv]
    - attack_i / defense_i: team strength parameters (centered so mean attack = 0)
    - mu: overall scoring intercept
    - rho: Dixon-Coles low-score correlation (typically small negative)
    - home_adv: log-scale home advantage (applied only to non-neutral matches)
    """

    def __init__(self, reg: float = 0.1) -> None:
        """
        Args:
            reg: L2 regularization strength on attack/defense params.
                 Pulls extreme estimates toward zero, reducing the strength-of-schedule
                 bias for teams that played mostly weak opposition (e.g. CONCACAF minnows).
                 Higher = more shrinkage toward average. Default 0.1.
        """
        self.reg = reg
        self.params: Optional[np.ndarray] = None
        self.teams: list[str] = []
        self._team_idx: dict[str, int] = {}

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        """Fit on a DataFrame with columns: home_team, away_team, home_score,
        away_score, neutral, weight."""
        all_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self.teams = all_teams
        self._team_idx = {t: i for i, t in enumerate(all_teams)}
        N = len(all_teams)

        home_idx = matches["home_team"].map(self._team_idx).values
        away_idx = matches["away_team"].map(self._team_idx).values
        hs = matches["home_score"].values.astype(float)
        aws = matches["away_score"].values.astype(float)
        weights = matches["weight"].values
        neutral = matches["neutral"].values.astype(bool)

        # Initial vector: attacks=0, defenses=0, mu=0, rho=-0.1, home_adv=0.25
        x0 = np.zeros(2 * N + 3)
        x0[2 * N + 1] = -0.1
        x0[2 * N + 2] = 0.25

        def neg_ll(params: np.ndarray) -> float:
            attack = params[:N]
            defense = params[N : 2 * N]
            mu = params[2 * N]
            rho = float(np.clip(params[2 * N + 1], -0.99, 0.99))
            home_adv = params[2 * N + 2]

            adv = np.where(neutral, 0.0, home_adv)
            lh = np.exp(mu + attack[home_idx] - defense[away_idx] + adv)
            la = np.exp(mu + attack[away_idx] - defense[home_idx])

            tau = _tau_vectorized(hs.astype(int), aws.astype(int), lh, la, rho)
            if np.any(tau <= 1e-10):
                return 1e10

            ll = np.log(tau) + _poisson_logpmf(hs, lh) + _poisson_logpmf(aws, la)
            # L2 regularization: penalize extreme attack/defense values
            l2 = self.reg * (np.sum(attack ** 2) + np.sum(defense ** 2))
            return -float(np.dot(weights, ll)) + l2

        bounds = (
            [(-3.0, 3.0)] * N   # attack
            + [(-3.0, 3.0)] * N  # defense
            + [(-3.0, 3.0)]      # mu
            + [(-0.99, 0.99)]    # rho
            + [(0.0, 2.0)]       # home_adv
        )

        print(f"Fitting Dixon-Coles model: {len(matches)} matches, {N} teams...")
        result = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 1000, "ftol": 1e-9})

        p = result.x
        # Center attacks so their mean is absorbed into mu (identifiability)
        mean_att = p[:N].mean()
        p[:N] -= mean_att
        p[2 * N] += mean_att
        self.params = p
        print(f"Done. Weighted log-likelihood: {-result.fun:.1f}  rho={p[2*N+1]:.3f}  home_adv={p[2*N+2]:.3f}")
        return self

    def _expected_goals(
        self,
        team_a: str,
        team_b: str,
        adjustments: dict | None = None,
        global_boost: float = 1.0,
        neutral: bool = True,
    ) -> tuple[float, float]:
        """Return (xG_a, xG_b), with optional manual adjustments.

        adjustments keys per team:
          "attack"  (float) — multiplier on goals scored by that team (0.8 = -20%)
          "defense" (float) — multiplier on goals conceded by that team (1.2 = +20% for opponent)
        global_boost: applied equally to both teams (e.g. 1.15 for a high-mismatch tournament)
        neutral: if False, team_a is treated as the home side and the fitted home
            advantage is applied to its expected goals. WC2026 fixtures stay neutral;
            the back-test passes the real venue flag of each historical match.
        """
        N = len(self.teams)
        mu = self.params[2 * N]
        ia = self._team_idx[team_a]
        ib = self._team_idx[team_b]
        la = float(np.exp(mu + self.params[ia] - self.params[N + ib]))
        lb = float(np.exp(mu + self.params[ib] - self.params[N + ia]))

        if not neutral:
            la *= float(np.exp(self.params[2 * N + 2]))

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
        """Return (max_goals+1)×(max_goals+1) probability matrix.
        matrix[i, j] = P(team_a scores i, team_b scores j).
        """
        N = len(self.teams)
        rho = float(self.params[2 * N + 1])
        la, lb = self._expected_goals(
            team_a, team_b, adjustments, global_boost=global_boost, neutral=neutral
        )

        g = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(g, la), poisson.pmf(g, lb))

        # Dixon-Coles correction for {0,1} × {0,1} cells
        matrix[0, 0] *= max(1 - la * lb * rho, 1e-10)
        matrix[0, 1] *= 1 + la * rho
        matrix[1, 0] *= 1 + lb * rho
        matrix[1, 1] *= max(1 - rho, 1e-10)

        matrix = np.maximum(matrix, 0)
        matrix /= matrix.sum()
        return matrix

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        adjustments: dict | None = None,
        global_boost: float = 1.0,
        neutral: bool = True,
    ) -> dict:
        """Return prediction dict for team_a vs team_b (neutral venue by default)."""
        matrix = self.score_matrix(
            team_a, team_b, adjustments=adjustments, global_boost=global_boost, neutral=neutral
        )
        win_a = float(np.tril(matrix, -1).sum())   # team_a goals > team_b goals
        draw = float(np.trace(matrix))
        win_b = float(np.triu(matrix, 1).sum())    # team_b goals > team_a goals

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
        """Return a DataFrame of attack/defense ratings for all teams."""
        N = len(self.teams)
        return pd.DataFrame({
            "team": self.teams,
            "attack": self.params[:N].round(3),
            "defense": self.params[N : 2 * N].round(3),
        }).sort_values("attack", ascending=False).reset_index(drop=True)
