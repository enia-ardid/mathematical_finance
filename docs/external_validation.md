# External-universe validation and a fixed forecast combination

This experiment preserves the persistent asymmetric fractional model exactly and evaluates it on eight previously unevaluated country ETFs. It also adds a direct FIGARCH reference and one optional forecast combination. The earlier simulations, initial variance model and refined model remain separate and reproducible.

## Recorded design

The [protocol](../paper/external_protocol.json) was recorded locally before downloading EWA, EWC, EWG, EWH, EWJ, EWL, EWQ and EWU. The [data guide](data.md) records the source, date range, adjustment convention and input hash. This is a retrospective external-universe replication. It is not an independently registered preregistration, a statistically independent sample or a prospective evaluation: these funds share currency denomination, trading hours and global shocks with the US panel.

All model grids, fitting windows, refit spacing, validation rules and forecast horizons remain unchanged. The test universe is not changed after seeing results. Validation uses only observations available at each origin, and all compared models share each panel's targets, means and scales. The same fixed additions are evaluated on the previous US data, explicitly retaining that analysis's exploratory status.

The [computational amendment](../paper/external_protocol_amendment.json) records a library-interface correction after download. The first attempt completed the fractional forecasts but failed on the first FIGARCH forecast because a strided NumPy panel column was not C-contiguous. The wrapper now explicitly copies residuals into contiguous arrays. The statistical specification is unchanged, and the international fractional-forecast file is byte-for-byte identical across both runs. No completed FIGARCH comparisons existed before the correction.

## Forecast combination

Let the standalone refined model predict cumulative variance $\widehat V^F_{o,h,j}$, and let GJR-GARCH predict $\widehat V^G_{o,h,j}$. The added forecast is

$$
\widehat V^{E}_{o,h,j}=\frac12\widehat V^F_{o,h,j}
 +\frac12\widehat V^G_{o,h,j}.
$$

The weight is fixed at one half for every asset, date and horizon. No new weight is learned or selected on results. The fractional component still uses its positive memory kernel and selected graph diffusion. This is an optional forecast ensemble, not a replacement for the core stochastic model or a new generative recursion.

The combination is positive, and its loss is recomputed:

$$
\mathcal L_E=\log\!\left(\frac{\widehat V^E}{hs_j^2}\right)
 +\frac{Y}{\widehat V^E}.
$$

It is **not** the average of the component losses. Neither a lower loss nor superiority over both components is guaranteed by averaging their predictions.

## FIGARCH reference

The reference uses `arch==7.2.0`, `ZeroMean`, Gaussian innovations and `FIGARCH(p=1, q=1, power=2, truncation=1000)`. Returns are centered and standardized by the same fitted means and scales as all other methods. Each independent asset fit uses the preceding 1,008 returns and the existing refit dates. SLSQP uses the library's default initial estimates, `maxiter=1000` and `ftol=1e-9`.

The fitting backcast and coefficients remain fixed between refits; observed returns update the forecast state. Analytic daily conditional variances are converted back to asset units and summed over 1, 5 and 21 sessions. All optimizer statuses are saved. Nonfinite fits or predictions abort the run; finite nonconverged fits are reported instead of silently replaced. See the [official FIGARCH implementation documentation](https://arch.readthedocs.io/en/latest/univariate/generated/arch.univariate.FIGARCH.html).

The reference's 1,000-lag ARCH approximation differs from this project's normalized 252-lag fractional-integral kernel. They are distinct models; the [related-work note](related_work.md) explains that distinction and the existing network volatility literature.

## Recorded results

Each panel has 186 complete forecast origins and 38 refits. International origins run from 6 January 2023 to 16 September 2026, with targets through 15 October. The US grid runs from 9 January 2023 to 17 September 2026, with targets through 16 October. Comparisons pair methods within each panel; the calendar grids are not identical across panels. All 760 FIGARCH fits converge, as do all 304 new GARCH and 304 new GJR fits.

| International panel | 1 session | 5 sessions | 21 sessions |
| --- | ---: | ---: | ---: |
| Refined fractional | 0.52898 | 0.66359 | 0.67147 |
| Fixed 50:50 ensemble | 0.51307 | 0.65094 | 0.66134 |
| GJR-GARCH | 0.50833 | 0.64958 | 0.66226 |
| FIGARCH | 0.54380 | 0.66713 | 0.67299 |
| GARCH | 0.52875 | 0.65874 | 0.66723 |
| Fractional without graph | 0.52335 | 0.66335 | 0.67013 |
| Exponential kernel | 0.52905 | 0.65263 | 0.66534 |

These are mean normalized QLIKE scores; lower is better. The ensemble improves on the standalone fractional model and FIGARCH at all nine panel–horizon combinations across the US index, US sectors and international funds. It beats GJR in three: sectors at 5/21 sessions and international funds at 21 sessions. **All paired 95% ranges against GJR include zero at both block lengths.** No score-level ratios are interpreted as percentages.

For international one-session forecasts, ensemble minus standalone fractional QLIKE is **−0.01590**, with ten-origin-block range **[−0.03100, −0.00219]**. Ensemble minus FIGARCH is **−0.03073**, with range **[−0.05536, −0.01145]**. The small international 21-session win against GJR is **−0.00092**, with range **[−0.01174, 0.01110]**. These nominal ranges do not establish a general advantage over GJR.

The international standalone model beats EWMA and FIGARCH at each horizon, but loses to GARCH and GJR. The graph ablation is better at all three horizons, although nonzero diffusion was selected at 21 of 38 refits. The exponential ablation is better at 5 and 21 sessions. Validation selection therefore does not demonstrate that either diffusion or fractional weighting improves outer-sample performance universally.

The ensemble improves on the standalone model in 22 of 24 international asset–horizon comparisons; EWH at 5/21 sessions are exceptions. It also worsens pooled international QLIKE at 5/21 sessions in 2024. The secondary normalized RMSE is slightly worse at the pooled international one-session horizon (1.28769 versus 1.28751). These exceptions remain in the results rather than being excluded from the reporting universe.

## Reproduction and outputs

```bash
python download_external.py --output artifacts/yahoo-international-new
python run_external.py --external-dir artifacts/yahoo-international-new \
  --us-data-dir artifacts/yahoo-through-2026-10-16 \
  --us-study artifacts/refinement-study \
  --output artifacts/external-reproduction --publish-assets
```

For the recorded local snapshot, use `artifacts/yahoo-international-through-2026-10-16`. The completed run is stored in `artifacts/external-study-complete`. New downloads can change historical adjusted prices. The runner validates source/data hashes and refuses to overwrite a completed forecast file.

The public record includes [aggregate and annual scores](../paper/external_metrics.csv), [paired comparisons](../paper/external_comparisons.csv), [per-asset scores](../paper/external_asset_metrics.csv) and [metadata](../paper/external_summary.json). Detailed forecasts, fitting audits and raw market data stay in ignored local artifacts. Paired circular bootstraps resample whole asset panels with blocks of 5 or 10 origins, using 2,000 replicates and seed 1729. Their intervals are descriptive and nominal; they do not adjust for all researcher choices, multiple comparisons or nonstationarity.
