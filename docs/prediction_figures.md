# Observed evolution versus model forecasts

The paper includes the full recorded time paths, alongside the aggregate loss tables. These plots use saved forecasts: they do not fit models again, alter parameters or change scores.

## Reading Figure 2

The figure displays **actual mean normalized QLIKE**, with one dot and a four-decimal label for every model, including EWMA in orange. Its six panels cover the index and sector panel at horizons of 1, 5 and 21 observations, on a common horizontal scale. Lower scores are better. Values match the initial-study table.

![Direct QLIKE comparison, including EWMA](../paper/figures/forecast_comparison.png)

## What the prediction curves show

The target in the current paper is the magnitude of future fluctuations, measured through a centered squared-return proxy. It is not the future direction of the price. Both variance forecasts and realized targets are shown in annualized volatility units:

$$
\widehat\sigma^{\mathrm{ann}}_{o,h,j}=100\sqrt{\frac{252}{h}\widehat V_{o,h,j}},
\qquad
\sigma^{\mathrm{proxy}}_{o,h,j}=100\sqrt{\frac{252}{h}Y_{o,h,j}}.
$$

The black curve is the subsequently observed return-based proxy. Blue is the persistent asymmetric fractional model, orange the fixed 50:50 ensemble, and dashed green GJR-GARCH. The square root of a variance forecast is displayed for readability; it is not generally the conditional expectation of volatility. The observed proxy is noisy, especially at one session, and is not latent volatility measured directly.

For Figures 6–10, the date is **the end of the forecast window**. The prediction was issued 1, 5 or 21 observed trading sessions earlier, using only data available then. Lines connect forecasts issued every five observations. The 21-session windows overlap. The initial-study Figure 3 retains its explicitly labeled forecast-origin axis; it belongs to the earlier specification.

![S&P 500 predictions at three horizons](../paper/figures/prediction_index_horizons.png)

The multi-asset summaries average all marginal variances before taking the square root. They show root-mean-square marginal volatility, not portfolio volatility: they omit cross-asset covariance. They are also different from an arithmetic average of asset volatilities.

![Panel risk evolution at five and twenty-one sessions](../paper/figures/prediction_panel_evolution.png)

The appendix includes **all eleven sector ETFs and all eight country ETFs** at 21 sessions, without selecting assets or subperiods according to how closely the model tracks observations. Individual axes use a logarithmic scale with positive inputs and independent limits. This keeps smaller fluctuations visible alongside the largest peaks. Peaks are retained without clipping; no extra temporal smoothing is applied. The graphs complement the reported losses rather than replacing them with visual judgments.

## Regenerate the figures

Run from the repository root **after** the numerical study exporters, which preserve their original figure generators:

```bash
MPLBACKEND=Agg python render_submission_assets.py
```

Then compile the paper and regenerate both archives:

```bash
cd paper
tectonic --reruns 3 main.tex
cd ..
python package_paper.py
```

The [renderer](../render_submission_assets.py) verifies the saved score aggregates and each method's target/date alignment. The [asset manifest](../paper/submission_assets_manifest.json) records the source hash, input hashes, transformations, complete asset lists and output hashes. The frozen numerical-study source files and protocols remain unchanged.


## Memory, zoom panels and dispersion

Figure 1 shows the normalized finite fractional kernel, its effective persistent-memory response for $b\in\{0,0.5,0.9\}$, and the fitted graph edge weights. The middle panel uses the exact convolution recurrence with $d=0.4$ and $M=252$ through lag 252 in the main panel and 504 in the inset; the tail for $b>0$ is geometric. The zero tail at $b=0$ is omitted on the logarithmic axis.

Figures 3 and 6 show the full evaluation period and a separate 2023–2024 zoom. The zoom uses the same saved forecasts, with no refitting or rescaling of their values. Figure 7 adds interquartile bands computed across the displayed asset volatilities separately for each series. The bands describe cross-asset dispersion, not uncertainty about the mean. The RMS central curve need not lie inside the interquartile band.

All eleven manuscript figures use the Okabe–Ito palette, with line styles and hatching where useful. Figure 11 counts assets whose full-period mean QLIKE is lower than GJR's, for each panel, horizon and model. Its denominator is the complete panel: one index, eleven sectors or eight country ETFs. The appendix table instead counts lower-loss forecast origins within each asset, out of 186. These are different summaries and neither treats overlapping observations as independent trials.
