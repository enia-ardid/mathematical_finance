# Core model and retained simulation experiment

The project simulates daily log-returns of the S&P 500 index and eleven sector ETFs. The core recurrence and the original simulation experiment remain available through `--mode simulation`. This page describes that retained experiment. The [forecasting workflow](forecasting.md) uses the same core with observed history, exact conditional moments and validation-based calibration; its equations and evaluation protocol are documented separately.

The core does **not** implement a verified fractional Black–Scholes equation or estimate a time-varying volatility process. The distinction matters: the derivation in the accompanying report does not yield the implemented update. See the [scientific review](scientific_review.md).

## Data and chronological split

Prices are aligned by date, then the last 1,644 prices (or all available prices if fewer) are used. Log-returns are `log(P[t]) - log(P[t-1])`; missing prices are never forward-filled. Leading gaps before an ETF's first observation are trimmed, but interior gaps, duplicates, nonpositive prices and unsorted dates fail validation. The data loader takes the intersection of observed index and ETF dates; one step means one retained trading observation, not one calendar day.

For `T` returns, the first `floor(0.7*T)` returns form training. The initial price and these training returns give `split + 1` training prices. The first held-out price is at price-row `split + 1`. All estimates, sector ordering, coupling candidates, drift adjustments and scale factors use training data only. Perturbing every held-out value must leave the fitted parameters and simulated paths unchanged; a regression test enforces this for both models.

## Discrete dynamics

At unit time and sector spacing, the preserved recurrence is

$$r_{n,j}=\mu_{n,j}+\sigma_{n,j}z_{n,j}+\kappa(Lr_{n-1})_j
  +\sum_{k=0}^{n-1}w_{k,j}(r_{n-1-k,j}-c_{n-1-k,j}),$$

$$w_{0,j}=1,\qquad w_{k,j}=\left(1-\frac{\alpha_j+1}{k}\right)w_{k-1,j}.$$

The innovations `z` are independent standard normals before coupling. The direct index model uses `c = 0` and `κ = 0`; the sector model uses `c = μ`. The initial simulated return is the first observed training return. These GL binomial weights give a history-dependent discrete filter; they do not make this recurrence a GL differential-equation solver. No time-step refinement or continuous-time noise interpretation is claimed.

Daily `μ` and `σ` are the training log-return mean and sample standard deviation. They parameterize the forcing term; neither is necessarily the mean or standard deviation of the resulting simulated returns. Hurst estimates use the corrected rescaled-range estimator in `nolds.hurst_rs`, with its RANSAC fit and a fixed estimation seed of 1729. The direct model sets `α = clip(H, .05, .95)`; sectors retain `α = clip(.5 + .6*(H-.5), .05, .95)`. This mapping and shrinkage are heuristics, not identified fractional orders. See the [estimator documentation](https://cschoel.github.io/nolds/nolds.html#nolds.hurst_rs).

Sector order comes from average linkage on training correlation distances `1 - ρ`. The order provides a one-dimensional modeling lattice, not an optimal economic geography. `L` is a centered second difference with reflected ghost nodes: the first boundary uses `2*(r[1]-r[0])`, and the last uses `2*(r[-2]-r[-1])`. A single asset has zero diffusion.

Coupling candidates are zero, half and all of `κ_base = 0.5 * max(0, tanh(mean(atanh(ρ))))`, using off-diagonal training correlations clipped just inside ±1. Selection minimizes training price RMSE over five seeds. This retains the notebook's training criterion, including its dependence on ETF price units; it is not validation-based tuning or a causal estimate of contagion. The fit is evaluated once on held-out data.

The index model retains the retrospective training scenario of 20 February–30 April 2020 (`μ × .8`, `σ × 4`). Sectors retain the final notebook's default of no COVID adjustment. The index adjustment is applied only to training dates. It is not a forecast of the event and does not affect the supplied 2021–2026 CSV experiment.

## Price calibration and evaluation

Returns accumulate into log-prices. Each seed receives a constant adjustment to its simulated returns so that its training-end log-price matches the final observed training price. A positive least-squares scale in price levels is then fitted on training prices and applied to the entire path. These are substantive calibration steps from the final notebook, not consequences of fractional calculus. Scaling can move the training endpoint away from the observed value again. Test paths continue the calibrated simulated history; they are not rolling one-step forecasts and are not conditioned on all observed training returns as lagged state.

The default ensemble averages **prices** from ten seeds (`2 + 37*i` for the index; `3 + 37*i` for sectors). The coupling grid uses the first five sector seeds. An ensemble's metric differs from the average metric of its individual members. Ten paths give coarse Monte Carlo quantiles: the 10th–90th percentile envelope describes these simulations, not a calibrated confidence or prediction interval.

RMSE, MAE, R², bias, and sMAPE are evaluated on identical held-out dates. Here sMAPE is `200 * mean(abs(pred-actual)/(abs(pred)+abs(actual)))`, with a zero contribution when both values are zero. R² is undefined for a constant target. Baselines start at the last observed training price: a constant price and an exponential path using the training mean log-return. The latter is a log-drift reference, not the expected price of a lognormal random walk (which would also include the variance correction).

The ETF comparison uses an illustrative buy-and-hold portfolio:

$$V_t=V_0\sum_j w_j\frac{P_{t,j}}{P_{0,j}},\qquad \sum_jw_j=1.$$

Each simulated basket is rebased at the last training date before averaging. Weights are matched by ticker and mean initial capital allocations. They are not dated S&P 500 sector weights; the basket is not an index replication. Adjusted ETF series can include distributions while `^GSPC` is a price index, so their comparison is descriptive. The data source's adjustment convention must be known before making a stronger claim.

## Limits

This retained simulation experiment has one chronological split and no untouched external test set. There is no trading strategy, transaction-cost model or evidence here of profitability. The historical period and modeling choices were already inspected during the original project. Correcting implementation leakage does not undo that research-level selection. Clipping α and choosing a moderate κ do not prove stability. Gaussian innovations, constant estimated parameters and a fixed sector order do not establish heavy tails, volatility clustering or causal shock propagation.

Full history is retained: simulation costs `O(seeds × assets × T²)` operations and `O(seeds × assets × T)` storage for saved paths. Recursive weight generation does not remove the convolution over past returns. Numerical tests check implementation properties; they do not validate financial or continuous-time assumptions.
