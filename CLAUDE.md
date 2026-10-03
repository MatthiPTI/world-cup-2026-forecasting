# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / sync dependencies
uv sync

# Generate predictions — Dixon-Coles (default)
uv run --env-file .env mpp-predict

# Generate predictions — Bayesian hierarchical (MAP = rapide, nuts = propre)
uv run --env-file .env mpp-predict --model bayes --bayes-inference map
uv run --env-file .env mpp-predict --model bayes

# Confirmer l'utilisation réelle du jeton ×2 (sinon jamais persisté)
uv run --env-file .env mpp-predict --model bayes --confirm-double

# Correction Dixon-Coles des nuls (opt-in, à calibrer par back-test)
uv run --env-file .env mpp-predict --model bayes --dc-rho -0.1

# Recalculer les picks après mise à jour des cotes (SANS refit du modèle, ~1 s)
uv run --env-file .env mpp-optimize
uv run --env-file .env mpp-optimize --value-margin 1.0   # EV pure, sans garde anti-variance

# Couche marché — blend modèle/cotes + détection de value bets (SANS refit)
# Probas marché : cotes décimales data/mpp_odds.json EN PRIORITÉ, sinon repli sur les
# points data/mpp_points.json (la constante du barème s'annule au dé-vig).
uv run --env-file .env mpp-market                        # blend + value bets ; MET À JOUR les picks du site
uv run --env-file .env mpp-market --report-only          # rapport seul, ne touche pas predictions.json
uv run --env-file .env mpp-market --market-weight 0.4    # plus de poids au modèle
uv run --env-file .env mpp-market --min-prob 0.15        # value bets plus conservateurs
# Sorties : data/market_report.csv/json + (par défaut) predictions.json & mpp_pronos.csv blendés.
# Blend = picks tirés vers le marché (anti-biais calendrier). Edge = modèle PUR vs cotes (EV/Kelly).
# Workflow : mpp-predict --model bayes  →  mpp-market  →  recharger le site.

# Site local de comparaison (modèle vs mes pronos vs collègues)
uv run --env-file .env mpp-site

# Back-test walk-forward (validation out-of-sample — la boucle de mesure)
uv run --env-file .env mpp-backtest                          # Dixon-Coles vs Elo, 18 derniers mois
uv run --env-file .env mpp-backtest --models dixoncoles bayes   # comparaison complète (NUTS, ~15 min)
uv run --env-file .env mpp-backtest --apply-adjustments      # mesure l'impact de adjustments.json
uv run --env-file .env mpp-backtest --start 2023-01-01 --refit-freq 60
uv run --env-file .env mpp-backtest --models bayes --bayes-inference map   # rapide, debug
# Tous les modèles sont notés en UN passage sur les mêmes matchs (pertes appariées).
# Sorties : data/<prefix>_metrics.json + data/<prefix>_predictions.csv (--out-prefix, défaut backtest)
# Métriques : log-loss / Brier / RPS / accuracy / score exact vs baselines (Elo, base-rate)
# Significativité : bootstrap apparié (par match + par fenêtre de refit), Holm sur les paires
# Diagnostic clé : biais par confédération (bias>0 = surévaluée = calendrier mou)
# Diagnostics NUTS par fenêtre (divergences, R-hat, ESS, E-BFMI) dans fit_diagnostics

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_bar

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

> **Note Python 3.14 + uv + macOS** : uv marque tout le contenu de `.venv` comme
> `UF_HIDDEN` (macOS), et Python 3.14 ignore désormais les fichiers `.pth` cachés.
> Résultat : l'install éditable ne trouve pas `mpp` sans `--env-file .env`.
> `.env` est local (git-ignoré) : le créer avec `cp .env.example .env`, lancer depuis la racine.
>
> **Note PyTensor + macOS 27** : clang ≥ 21 rejette le drapeau `-ld64` que PyTensor ajoute
> → toute compilation C échoue (`library 'd64' not found`), donc tout le bayésien.
> `.env.example` pointe `PYTENSOR_FLAGS` vers `scripts/clang++-no-ld64`, qui retire ce drapeau.

## Architecture

- `pyproject.toml` — project metadata, dependencies, and tool config (ruff, pytest)
- `.env.example` — template for the local, git-ignored `.env` (`PYTHONPATH=src` + PyTensor compiler shim)
- `scripts/clang++-no-ld64` — clang++ shim stripping PyTensor's `-ld64` flag (macOS 27)
- `src/mpp/confederations.py` — mapping statique équipe → confédération FIFA (utilisé par le modèle bayésien)
- `src/mpp/bayesian_model.py` — modèle Poisson hiérarchique bayésien (PyMC), même interface que DixonColesModel
- `uv.lock` — pinned dependency lockfile; commit this file
- `src/mpp/` — prediction engine package
  - `data.py` — downloads & preprocesses historical international results (martj42/international_results)
  - `model.py` — Dixon-Coles Poisson model (fit + predict)
  - `fixtures.py` — WC 2026 group stage fixtures (72 matches, groups A–L)
  - `predict.py` — entry point: trains model, applies adjustments, outputs CSV + JSON
- `adjustments.json` — manual team adjustments (see below)
- `tests/` — pytest test suite
- `data/` — downloaded data and prediction outputs (git-ignored)

## Ajustements manuels

Editer `adjustments.json` avant chaque journée pour corriger le modèle (blessures, rotations) :

```json
{
  "France": {"attack": 0.8, "note": "Mbappé blessé"},
  "Spain": {"defense": 1.15, "note": "Défense fragilisée"}
}
```

- `attack` (défaut 1.0) : multiplie les buts espérés de cette équipe
- `defense` (défaut 1.0) : multiplie les buts que l'adversaire marque contre elle
- Équipes absentes = modèle pur, aucune modification

## Conventions

- `data/` est git-ignoré — ne jamais commiter les fichiers de données brutes
- Dependencies are managed exclusively through `uv` (`uv add <pkg>`, `uv remove <pkg>`)
- Les noms d'équipes dans `adjustments.json` doivent correspondre aux noms du dataset historique
  (voir `TEAM_NAME_MAP` dans `src/mpp/fixtures.py` pour les équivalences)
