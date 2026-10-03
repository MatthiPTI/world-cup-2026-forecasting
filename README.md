# MPP: forecasting the 2026 World Cup, honestly

A football forecasting engine I built for the 2026 World Cup group stage. It compares a
**Bayesian hierarchical Poisson model** (PyMC/NUTS), a **Dixon-Coles** model and an
**Elo** baseline. All three are evaluated in one **walk-forward back-test**, and the gaps
between them are tested with **paired bootstrap tests**.

Results in one paragraph:

> On 852 out-of-sample competitive matches (Dec 2024 to Aug 2026, including the 104 matches
> of the 2026 World Cup), the Bayesian model **beats Dixon-Coles significantly** (−0.023
> nats of log-loss, Holm-adjusted p = 0.012). It is also **ahead of a plain Elo baseline
> (−0.011 nats), but that gap is not statistically significant**: the sample could only
> have detected gaps of about 0.03 nats. Fixing the sampler's divergences made the
> posterior trustworthy, but it **did not change predictive accuracy** (|Δ| < 0.001 nats).

Most of the work here went into measuring things properly: the leakage-free back-test,
the paired significance tests, and the sampler diagnostics. Without that work, the
earlier claim "the Bayesian model beats Elo" (a 0.001-nat gap) would have gone out
unchecked.

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

Window: checkpoints every 45 days from 2024-12-20 to 2026-08-31, giving 13 refit windows
and **852 scored matches** (77 skipped because a team was below the data threshold).
The Bayesian model was fitted with NUTS (4 chains × 1000 draws, 1000 tuning steps) at
every checkpoint. Lower is better for all three scores.

| Predictor | Log-loss | Brier | RPS | Accuracy |
|---|---:|---:|---:|---:|
| **Bayesian hierarchical** | **0.8156** | **0.4778** | **0.1588** | **0.630** |
| Elo | 0.8264 | 0.4844 | 0.1616 | 0.627 |
| Dixon-Coles | 0.8383 | 0.4912 | 0.1645 | 0.627 |
| Base rate (constant H/D/A) | 1.0509 | 0.6347 | 0.2339 | 0.470 |

**Are the gaps real?** Paired bootstrap on per-match log-loss (10,000 resamples). Δ = A − B,
negative means A is better. Holm correction is applied across all model pairs.

| A vs B | Δ log-loss | 95 % CI (matches) | 95 % CI (refit windows) | Holm p | MDE |
|---|---:|---|---|---:|---:|
| Bayesian vs Dixon-Coles | −0.0226 | [−0.032, −0.014] | [−0.045, −0.006] | **0.012** | 0.013 |
| Bayesian vs Elo | −0.0108 | [−0.033, +0.011] | [−0.038, +0.019] | 1.0 | 0.031 |
| Dixon-Coles vs Elo | +0.0118 | [−0.011, +0.035] | [−0.021, +0.043] | 1.0 | 0.033 |

Brier and RPS give the same verdicts: Bayesian beats Dixon-Coles (Holm p = 0.005 and
0.020), and every other gap is non-significant.

How to read this:

- **The Bayesian model beats Dixon-Coles**, under both resampling schemes and on all three
  scores. Partial pooling and the Elo prior are doing real work: Dixon-Coles, fitted per
  team by maximum likelihood, overfits teams with few informative matches.
- **Bayesian vs Elo is undecided, not a tie.** The point estimate favours the Bayesian
  model by 0.011 nats, but at n = 852 only differences of about 0.03 nats could be
  detected reliably. Saying "it beats Elo" would overclaim. Saying "it is no better than
  Elo" would also overclaim. The honest statement is that this sample cannot separate
  them.
- **Calibration** is good: in every favourite-probability bin, the Bayesian model's
  predicted and realised win rates differ by at most 0.04. The per-confederation bias,
  the direct measure of the soft-schedule problem, is at most 0.016 in absolute value.

**World Cup 2026 only** (104 matches, the target application):

| Predictor | Log-loss | Brier | RPS | Accuracy |
|---|---:|---:|---:|---:|
| **Bayesian hierarchical** | **0.871** | **0.513** | **0.165** | **0.625** |
| Elo | 0.910 | 0.542 | 0.179 | 0.587 |
| Dixon-Coles | 0.936 | 0.563 | 0.188 | 0.567 |

Same ranking, larger gaps. Bayesian vs Dixon-Coles is significant (Δ = −0.065, CI
[−0.106, −0.023], Holm p = 0.023). Bayesian vs Elo (Δ = −0.039, CI [−0.115, +0.036]) is
not: 104 matches can only detect gaps of about 0.11 nats. These 104 matches fall into
just 2 refit windows, so only the per-match bootstrap is meaningful here. The code skips
the cluster bootstrap below 5 clusters, because with 2 clusters it returns nonsense
(including p = 0).

Every number above can be regenerated without refitting:

```bash
uv run --env-file .env mpp-backtest-report data/backtest_full_predictions.csv
uv run --env-file .env mpp-backtest-report data/backtest_full_predictions.csv --tournament "FIFA World Cup"
```

## Diagnosing the NUTS divergences

The first version of the Bayesian model sampled with warnings. At the latest checkpoint
it produced **46 divergent transitions out of 1000 (4.6 %)** and R-hat up to 1.022.

**Where were the divergences?** I located each divergent draw within the marginal
distribution of every hyperparameter. They were **not** at small `τ`, where the classic
hierarchical funnel puts them. They sat at **large `τ`** (median rank: 90th percentile
for `τ_def`, and 98th for `τ_att` once `target_accept` was raised).

**Why.** The original model was *non-centred*: `conf_att = τ · z`, with `z ~ N(0, 1)`.
That is the right choice when groups have little data. Here every confederation has
thousands of matches, so `conf_att` is pinned by the likelihood, and `z = conf_att / τ`
has to move in exact inverse proportion to `τ`. The result is a curved ridge in
`(τ, z)` that NUTS cannot follow with a single step size. A second, smaller issue: the
mean of `z` was identified only by the prior (its posterior sd was 0.41 = 1/√6, exactly
the prior value). That is a flat direction the sampler wasted effort on.

**The fix** is a centred parameterisation: `conf_att ~ ZeroSumNormal(τ)` and
`att_dev ~ Normal(0, σ)`. `ZeroSumNormal` removes the redundant direction as well.

| Latest checkpoint, 4 chains × 1000 | Divergences | max R-hat | min ESS (bulk) | E-BFMI | Time |
|---|---:|---:|---:|---:|---:|
| Non-centred, `target_accept` 0.8 | 46 / 1000 (2 chains) | 1.022 | 219 | 0.59 | 34 s |
| Non-centred, `target_accept` 0.9 | 36 | 1.007 | 546 | 0.61 | 78 s |
| Non-centred, `target_accept` 0.95 | 2 | 1.006 | 532 | 0.62 | 135 s |
| **Centred** (`target_accept` 0.8) | **0** | **1.006** | **1080** | **0.95** | **48 s** |

Raising `target_accept` only hides the problem, at about 3× the cost. Changing the
geometry removes it. Across all 13 back-test refits, the centred model had **0
divergences** in every window, R-hat ≤ 1.012, ESS ≥ 569 and E-BFMI ≥ 0.87. The
non-centred one had between 64 and 708 divergences per window out of 4000 draws, and in
its worst window R-hat reached 1.32 with an ESS of 9.

**Did it matter for prediction?** I kept the old version as a separate back-test model
(`bayes_noncentered`) so the two could be compared on identical matches. Δ log-loss =
−0.0002, with a CI of [−0.0007, +0.0002]: the two are indistinguishable. The plug-in
predictions use posterior *means* of attack and defence, and those means were robust
even when the chains mixed badly. The fix matters for anything that relies on the
**posterior spread**: parameter uncertainty, posterior-predictive score matrices, and
the "how much does the model trust Elo" coefficient `β`.

**One trap.** The centred form is wrong for MAP. Its joint density is unbounded as
`σ → 0` (the funnel tip), so the optimiser collapses every team to average. The default
parameterisation therefore depends on the inference method: centred for NUTS,
non-centred for MAP and ADVI. A regression test pins this.

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

# Full comparison: one NUTS fit per Bayesian variant per refit window (slow)
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
scripts/             clang++ shim for PyTensor on macOS 27
```

## License

MIT for the code. The match data is CC0 (martj42/international_results).
