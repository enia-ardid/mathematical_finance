# Forecast diagnostics

These diagnostics use the saved origin-level predictions in `artifacts/external-study-complete/forecasts.csv`. They do not modify, recalibrate or refit forecasts. The [protocol](../paper/diagnostics_protocol.json) was recorded after the earlier study results were known and before these diagnostic calculations. The analysis is exploratory. The [amendment](../paper/diagnostics_protocol_amendment.json) relaxes only a CSV loss-reconstruction check from $10^{-12}$ to $10^{-10}$; all recorded losses and statistical rules are preserved.

## Paired loss tests

For each model and horizon, pair QLIKE losses against GJR on identical asset, origin and target records. Average the differences over assets at each of the 186 ordered origins:

$$d_o=\frac{1}{J}\sum_{j=1}^J\bigl(L_{o,j}^{\rm model}-L_{o,j}^{\rm GJR}\bigr).$$

The two-sided asymptotic Diebold–Mariano statistic is

$$\mathrm{DM}=\frac{\bar d}{\sqrt{\widehat\Omega_L/n}},\qquad
\widehat\Omega_L=\widehat\gamma_0+2\sum_{\ell=1}^L\left(1-\frac{\ell}{L+1}\right)\widehat\gamma_\ell,$$

where the centered autocovariances use divisor $n$. Bartlett HAC lags are four origins for the primary calculation and nine for sensitivity. The largest target overlaps four origin lags because forecasts are issued every five sessions. No small-sample horizon correction is applied. Holm adjustment uses all 18 model–panel–horizon comparisons separately at each lag choice.

All 36 raw p-values exceed 0.05; the smallest is 0.0662 for four lags and 0.0736 for nine. This does not establish equivalence to GJR. Full values are in [diagnostics_dm.csv](../paper/diagnostics_dm.csv).

## Calibration regressions

For each panel and horizon, average normalized forecasts and targets over assets at each origin:

$$x_o=\frac1J\sum_j\frac{\widehat V_{o,h,j}}{h s_{o,j}^2},\qquad
y_o=\frac1J\sum_j\frac{Y_{o,h,j}}{h s_{o,j}^2},\qquad
y_o=\alpha+\beta x_o+e_o.$$

The saved `reference_scale` is already $h s_{o,j}^2$. Ordinary least squares estimates are accompanied by Bartlett HAC standard errors with the $n/(n-2)$ correction. A two-degree-of-freedom Wald test assesses $(\alpha,\beta)=(0,1)$. Holm adjustment uses all 27 model–panel–horizon regressions separately for each lag choice. The slopes and intercepts are diagnostic: they are never applied to the model's predictions.

At a one-session horizon on the country panel, all three methods—refined fractional, fixed ensemble and GJR—reject the joint calibration null after Holm adjustment at both lag choices. No other regression rejects after this adjustment at 5%. The complete coefficients, robust standard errors, $R^2$ and p-values are in [diagnostics_mz.csv](../paper/diagnostics_mz.csv). This is a material calibration limitation shared by the three methods.

## Wins and scope

[Asset counts](../paper/diagnostics_asset_wins.csv) report wins, ties and losses across 186 matched origins for every asset and horizon. A tie has an absolute loss difference no larger than $10^{-12}$. [Panel counts](../paper/diagnostics_panel_wins.csv) separately count assets with lower full-period mean loss than GJR; these drive the bar chart. Win frequency and mean loss need not rank models identically.

The small number of dependent origins, overlapping targets, noisy squared-return proxies and rolling estimation limit asymptotic inference. The model's second-moment conditions do not establish the higher-moment and dependence conditions needed by these tests. Panel aggregation can hide asset-level miscalibration. Neither multiplicity adjustment nor these additional diagnostics removes prior exposure to the retrospective sample.

## Reproduction

From the repository root, with the retained input records and a fresh output directory:

```bash
python run_forecast_diagnostics.py --output artifacts/diagnostics-reproduction
python render_submission_assets.py
```

The runner verifies the protocol chain, source hashes, input hashes, score reconstruction and aggregate agreement before calculation. Five tests compare the HAC regressions and Holm adjustment with independent library results and check pairing and degenerate inputs. The full project suite contains 66 tests.
