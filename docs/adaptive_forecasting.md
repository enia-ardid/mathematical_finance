# Adaptive forecast calibration

This extension keeps `temporal_memory.py` and `forecasting.py` unchanged. It learns a **direct forecast head** around the original memory forecasts. This is a different prediction layer from both the simulator and the earlier nonnegative blend. It can learn a negative correction, shrink weak signals, and select a simpler reference when past forecasts favor it.

The purpose is to test whether the memory signal adds useful information. A separately fitted version without memory is evaluated on identical targets. Baseline selection, negative results and the contribution of the forecast head must remain visible.

## Features and targets

At observed origin $o$, let the target for asset $j$ and horizon $h$ be its simple price change:

$$y_{o,h,j}=\frac{P_{o+h,j}}{P_{o,j}}-1,\qquad h\in\{1,5,21\}.$$

The log-drift reference is $b_{o,h,j}=\exp(h\widehat\mu_{o,j})-1$. The core supplies expected prices using the [unchanged conditional-moment equations](forecasting.md). Define memory features relative to drift:

$$x^{\mathrm{memory},s}_{o,h,j}
=\frac{\widehat P^{\mathrm{memory},s}_{o+h,j}}{P_{o,j}}-1-b_{o,h,j}.$$

For the index, the only coupling scale is $s=0$. For sectors, use $s\in\{0,0.5,1\}$ times the original correlation-based coupling scale. These are predictors of the external head, not three newly modified simulators. The reported raw-memory reference uses $s=0$ for the index and $s=0.5$ for sectors, declared before inspecting this experiment.

Control features are the latest log-return, trailing mean log-returns over 5, 21 and 63 observations, and

$$x^{\mathrm{vol}}_{o,j}=\log\left(\frac{\widehat\sigma_{20,o,j}}{\widehat\sigma_{252,o,j}}\right).$$

Features use only information observed through $o$. Core parameters and sector order are refitted on expanding history at least 21 observations apart. With five-observation origins, this normally means 25 observations. Parameters are not refitted retrospectively when building historical features. Within the sector panel, a common head is fitted to normalized targets across all assets; the index has its own head.

## Regularized direct forecasts

For each horizon, fit a ridge correction to $y_i-b_i$. Standardize every feature using only the current fitting rows. If $\mathbf z_i$ denotes those standardized features, solve

$$\min_{a,\boldsymbol\beta}\;
\frac1N\sum_{i=1}^{N}\left(y_i-b_i-a-\mathbf z_i^{\mathsf T}\boldsymbol\beta\right)^2
+\eta\lVert\boldsymbol\beta\rVert_2^2.$$

The intercept is unpenalized. With centered predictors, $\widehat a=\overline{y-b}$ and

$$\widehat{\boldsymbol\beta}
=\left(\frac{Z^{\mathsf T}Z}{N}+\eta I\right)^{-1}
\frac{Z^{\mathsf T}(\mathbf y-\mathbf b-\widehat a\mathbf1)}{N}.$$

The code solves the linear system without explicitly forming an inverse. Candidate penalties are $\eta\in\{1,10\}$ and fitting windows are 504 or 1,008 retained observations. At least 40 distinct training origins are required. These are strong shrinkage penalties because the squared-error objective is divided by $N$; they are not directly equivalent to the unnormalized penalty parameter in many libraries.

The point prediction is

$$\widehat P_{o+h,j}=P_{o,j}\left(1+b_{o,h,j}+\widehat a+
\mathbf z_{o,h,j}^{\mathsf T}\widehat{\boldsymbol\beta}\right).$$

A nonfinite or nonpositive candidate price falls back to the drift reference before scoring. The fit audit records this numerical fallback. Unlike the earlier blend constrained to $[0,1]$, ridge coefficients can reverse a signal's direction. This is a learned empirical correction, not a new derivation of the original PDE.

## Labels must mature before fitting or selection

A row from origin $u$ can train the $h$-session head at $o$ only if

$$u+h\le o.$$

This condition matters when targets overlap. A 21-session outcome is unavailable just because its originating date is in the past. The same boundary applies to candidate selection and interval calibration. Tests perturb future prices and check past features, fitted coefficients, chosen candidates and forecasts.

At each new origin, compare the four memory-head candidates, constant price and log-drift using **previously issued** candidate forecasts whose outcomes have matured. Average squared normalized errors over the previous 252 observations, with at least 20 eligible forecast origins. Before that threshold, use drift. Ties favor the simpler candidates. No fitting residuals are substituted for these earlier forecast errors.

Repeat this complete procedure for the four control-only candidates plus the same baselines. This is the **without-memory ablation**. Its comparison with the full method measures the effect of adding memory features and their associated model selection; it is not a proof of causality or a standalone test of a fractional process.

The [recorded finite protocol](results/adaptive_protocol.json) fixes this candidate set. Evaluation uses all complete origins from 2023 onward, reports each calendar year and includes previously inspected 2026 data. These are exploratory retrospective results, not a new untouched test or formal preregistration. The methodology follows [rolling-origin evaluation](https://otexts.com/fpp3/tscv.html); temporal ordering alone does not erase earlier researcher exposure to the data.

## Adaptive uncertainty intervals

An expanding constant variance cannot represent changing volatility. The new scale uses an exponentially weighted second moment of daily returns:

$$v_{o,j}=0.94v_{o-1,j}+0.06r_{o,j}^2,\qquad
s_{o,h,j}=\sqrt{hv_{o,j}}.$$

Initialize with the first observed squared return. At each origin, collect matured errors of the **actual adaptive forecasts previously issued**, normalized by their own historical scale:

$$e_{u,h,j}=\frac{|y_{u,h,j}-\widehat y_{u,h,j}|}{s_{u,h,j}}.$$

Use the previous 252 observations and at least 20 distinct origins. Pool sector errors across assets within each horizon. For $n$ errors, let $q$ be the order statistic at rank $\min(n,\lceil0.8(n+1)\rceil)$. The nominal 80% price interval is

$$\left[
P_{o,j}\max(0,1+\widehat y_{o,h,j}-qs_{o,h,j}),\quad
P_{o,j}(1+\widehat y_{o,h,j}+qs_{o,h,j})
\right].$$

Before enough errors have matured, use $q=\Phi^{-1}(0.9)$. The scale floor avoids division by zero. These empirical intervals do **not** have a claimed distribution-free coverage guarantee: errors are dependent, regimes change and sector observations are correlated. A lower price bound of zero is only a support constraint.

Compare them with expanding-variance Gaussian intervals around the **same adaptive point forecast**, and with the original raw-memory intervals. Report coverage, width and the interval score. With miss probability $\alpha=0.2$, lower bound $\ell$, upper bound $u$ and realized normalized return $y$,

$$\operatorname{IS}_{\alpha}(\ell,u;y)
=(u-\ell)+\frac{2}{\alpha}(\ell-y)\mathbf1_{\{y<\ell\}}
+\frac{2}{\alpha}(y-u)\mathbf1_{\{y>u\}}.$$

Lower scores balance narrow intervals with penalties for missed outcomes; widening every interval is not automatically an improvement. See [Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) for proper scoring rules and interval scores.

## Audit and latest forecasts

```bash
python run_adaptive.py --synthetic --output artifacts/adaptive-demo
python run_adaptive.py --data-dir artifacts/yahoo-through-2026-10-16 --output artifacts/adaptive-market
```

The offline example has 900 synthetic prices, allowing a warm-up for fitting and candidate selection. Market runs retain the entire aligned history. `--report-start` changes the first scored origin, not which future labels a fit may access.

Each run saves all forecasts, per-year metrics, interval scores, selection counts, complete coefficient/scaler audits, input and source-code hashes, and `latest_forecasts.csv`. The latest forecasts are made from the final observed date and have no scored target. Horizons denote trading observations; no unverified future exchange-calendar dates are invented.

Timestamped files under `issued_forecasts/` are created exclusively and retained across reruns. This is a local issuance record, not an independently timestamped public forecast. It supports subsequent evaluation when targets become observable.

Paired error comparisons include a descriptive circular block bootstrap with 2,000 replicates and seed 1729. Blocks have length $\lceil h/5\rceil$ origins, and all sectors at an origin are resampled together. These intervals reflect some local dependence, but do not account fully for regime shifts, researcher selection or historical data revisions. They are not a basis for claiming a reliable trading edge.
