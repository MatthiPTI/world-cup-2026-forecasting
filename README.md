# MPP: forecasting the 2026 World Cup, honestly

A football forecasting engine I built for the 2026 World Cup group stage. It compares a
**Bayesian hierarchical Poisson model** (PyMC/NUTS), a **Dixon-Coles** model and an
**Elo** baseline. All three are evaluated in one **walk-forward back-test**, and the gaps
between them are tested with **paired bootstrap tests**.

The headline result is a negative one, and I report it as such:

> __RESULTS_TLDR__

Most of the work here went into measuring things properly: the leakage-free back-test,
the paired significance tests, and the sampler diagnostics. That work is what made the
negative result clear.

---

## Contents

1. [The problem](#the-problem)
2. [Models](#models)
3. [Evaluation methodology](#evaluation-methodology)
4. [Results](#results)
5. [Diagnosing the NUTS divergences](#diagnosing-the-nuts-divergences)
6. [Limitations and next steps](#limitations-and-next-steps)
7. [Reproducing](#reproducing)
8. [Repository layout](#repository-layout)

---

## The problem

International football is a **sparse, poorly connected comparison graph**. Teams play
mostly inside their own confederation, so a team like Curaçao (CONCACAF) can look strong
because it beats weak regional opponents. Few matches connect it to European or South
American sides, so a per-team model cannot tell "strong team" apart from "soft schedule".

In the target application, a prediction game scored on outcome and exact score, the
cost of getting this wrong is concrete. The model has to price inter-confederation
matches (Germany vs Curaçao, USA vs Paraguay) that its training data barely covers.

## Models

All models output a full **score matrix** `P(home = i, away = j)`. The 1/X/2 probabilities,
over/under and exact scores are all derived from it.

| Model | What it is | Role |
|---|---|---|
| **Elo** ([elo.py](src/mpp/elo.py)) | World Football Elo (K by tournament importance, goal-difference multiplier, +100 home), mapped to 1/X/2 by a multinomial logistic on the rating difference | Baseline to beat |
| **Dixon-Coles** ([model.py](src/mpp/model.py)) | Poisson attack/defence per team with the Dixon-Coles low-score correction, fitted by time-weighted maximum likelihood with L2 | Classical benchmark |
| **Bayesian hierarchical** ([bayesian_model.py](src/mpp/bayesian_model.py)) | Poisson log-linear model with partial pooling by confederation and an Elo-informed prior, sampled with NUTS | Main model |

The Bayesian model:

```
attack_i  = β_att · elo_z(i) + conf_att[c(i)] + att_dev_i        (centred to sum to zero)
defence_i = β_def · elo_z(i) + conf_def[c(i)] + def_dev_i
conf_att ~ ZeroSumNormal(τ_att)        att_dev_i ~ Normal(0, σ_att)
β ~ HalfNormal(1)   τ ~ HalfNormal(1)   σ ~ HalfNormal(0.5)

log λ_home = μ + attack_home − defence_away + home_adv · 1[not neutral]
goals ~ Poisson(λ), likelihood weighted by exp(−ξ · days_ago), weights mean-normalised
```

- **Confederation pooling** shrinks teams from weak, isolated confederations toward
  their group mean. That is the structural answer to the soft-schedule problem.
- **The Elo prior** carries information from friendlies and intercontinental matches,
  so the likelihood does not have to rediscover it. `β` learns how far to trust Elo.
- **Mean-normalised time weights.** Raw decay weights (all ≤ 1) shrink the effective
  sample size inside `pm.Potential`. The prior then dominates and every team collapses
  toward average (an early bug: France ≈ Iraq). Rescaling the weights to mean 1 keeps
  the decay shape and restores the information content.

A separate **market layer** ([market.py](src/mpp/market.py)) removes the bookmaker
margin from the odds and blends the result with the model. It is not part of the
back-test, because the dataset has no historical odds.

## Evaluation methodology

[backtest.py](src/mpp/backtest.py), command `mpp-backtest`.

**Walk-forward, no leakage.** Every 45 days (a *checkpoint*), all models are refitted on
matches dated up to that checkpoint. They are then scored on the competitive matches
played before the next checkpoint. Friendlies and pre-tournament warm-up events are used
in training (down-weighted) but excluded from evaluation, because squads are rotated.

**Metrics** on the 1/X/2 outcome: log-loss (primary), multiclass Brier, and the ranked
probability score (RPS, the standard ordinal metric for football). Accuracy and
exact-score hit rate are reported but not used to choose between models.

**One pass, paired losses.** All predictors are scored on *exactly the same matches*,
with the same checkpoints and training sets. A match is kept only if every fitted model
knows both teams. This makes the per-match losses comparable pairwise.

**Paired bootstrap.** For predictors A and B, `d_k = loss_A(k) − loss_B(k)` and I
bootstrap the mean of `d`. Pairing matters a lot. Upsets hurt every model at once
(per-match log-losses of the Bayesian model and Elo correlate at ≈ 0.9), so most of the
noise cancels in `d`. Unpaired intervals on each model's mean would be several times
wider.

- **Two resampling schemes.** The first resamples matches (i.i.d.). The second
  resamples whole **refit windows** (cluster bootstrap): matches scored by the same fit
  share its parameter error, so their `d_k` are correlated, and the i.i.d. interval is
  probably too narrow. Both are reported.
- **p-values** come from inverting the percentile interval. **Holm** correction is
  applied across the model pairs of each metric.
- **Minimum detectable effect** (80 % power, 5 % two-sided) is reported next to each
  difference, so a non-significant result can be read as "too small to matter at this
  n" rather than "no evidence either way".

## Results

__RESULTS_SECTION__

## Diagnosing the NUTS divergences

__NUTS_SECTION__

## Limitations and next steps

- **Plug-in predictions.** Score matrices use the *posterior mean* of attack and defence,
  not the posterior predictive (an average of matrices over draws). The predictive
  version is more honest about parameter uncertainty and could improve log-loss on
  thinly observed teams.
- **Frozen predictors.** Every predictor, Elo included, is frozen between checkpoints.
  This is fair, since they all share the same information set, but a live Elo updated
  match by match would be a stronger baseline.
- **In-sample Elo link.** The Elo → 1/X/2 logistic is fitted on ratings that already
  include the training results, so the baseline is probably slightly over-confident.
  This is not test leakage.
- **The Bayesian model uses Elo as its prior**, so it is not an independent competitor
  to Elo. That partly explains the 0.9 loss correlation and the small gap.
- **Selection.** Matches involving a team with fewer than 35 matches since 2016 are
  skipped for every predictor alike.
- **No historical odds.** Comparing against the closing line is the real test for
  betting use, and it is not possible with this dataset. The market-blend weight
  (0.6) is therefore not validated.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync
cp .env.example .env        # see below
uv run --env-file .env pytest

# Full comparison (NUTS for every refit window, ~30 min on a laptop)
uv run --env-file .env mpp-backtest --models dixoncoles bayes bayes_noncentered \
    --start 2024-12-20 --end 2026-08-31 --out-prefix backtest_full

# World Cup 2026 group-stage predictions
uv run --env-file .env mpp-predict --model bayes
```

Why `.env`:

- **Python 3.14 + uv on macOS.** uv marks `.venv` files as hidden and Python 3.14 skips
  hidden `.pth` files, so the editable install is invisible. `PYTHONPATH=src` fixes it.
- **macOS 27 / clang ≥ 21.** PyTensor adds a `-ld64` linker flag that the new clang
  rejects, which breaks every C compilation. `PYTENSOR_FLAGS` points to
  [scripts/clang++-no-ld64](scripts/clang++-no-ld64), a two-line shim that drops it.
  Remove that line on Linux or older macOS.

Data: [martj42/international_results](https://github.com/martj42/international_results)
(CC0), downloaded on first run into `data/` (git-ignored).

## Repository layout

```
src/mpp/
  data.py            loading, time-decay weights, friendly down-weighting
  elo.py             World Football Elo
  model.py           Dixon-Coles
  bayesian_model.py  hierarchical Bayesian model + NUTS diagnostics
  confederations.py  team → confederation map
  backtest.py        walk-forward back-test, paired bootstrap, Holm
  market.py          bookmaker de-vig, model/market blend, value bets
  mpp_optimizer.py   expected-points pick optimiser for the prediction game
  fixtures.py        the 72 group-stage fixtures
  predict.py         CLI: fit + predict the World Cup
tests/               pytest suite (no network; Bayesian tests use MAP or tiny NUTS runs)
```

## License

MIT for the code. The match data is CC0 (martj42/international_results).
