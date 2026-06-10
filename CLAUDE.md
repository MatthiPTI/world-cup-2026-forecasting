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
> Fix permanent (optionnel) : ajouter `export UV_ENV_FILE=/path/to/MPP/.env`
> dans `~/.zshrc`, puis `source ~/.zshrc`. Après ça, `uv run mpp-predict` fonctionnera directement.

## Architecture

- `pyproject.toml` — project metadata, dependencies, and tool config (ruff, pytest)
- `.env` — sets absolute `PYTHONPATH` so `uv run --env-file .env` finds the `mpp` package (Python 3.14 + uv + macOS workaround)
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
