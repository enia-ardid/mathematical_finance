# Adaptive model: measured results and limits

The extension uses the same recorded Yahoo Finance snapshot as the preceding review, with common price history from 2018-06-19 through 2026-10-16. Its finite [candidate protocol](results/adaptive_protocol.json) was written before running this extension. All historical data had already been available in the project; this is an **exploratory retrospective evaluation**, not independent confirmation of a new forecasting advantage.

## Evaluation sample

There are **186 common forecast origins** from **2023-01-09 through 2026-09-17**, with targets through **2026-10-16**, for the index and eleven ETFs. Each origin is evaluated at 1, 5 and 21 trading observations. Selection and fitting use matured prior targets only. The end of the sample is scored only where all three horizons are complete.

The origin grid differs from the previous 2026 experiment. Compare methods within this table, not against numbers from a different grid. The full year-by-year results are retained; 2026 is not selected as the only favorable period.

```bash
python run_adaptive.py --data-dir artifacts/yahoo-through-2026-10-16 --output artifacts/adaptive-market
```

The data-download command is in the [earlier execution record](results.md). Inputs remain local; the summary records their hashes. A revised upstream snapshot may change the results.

## Point forecasts

Index origin-normalized RMSE, pooled over the reported years:

| Horizon | Adaptive head | Without memory | Raw memory | Log-drift | Constant price |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 session | 1.0441% | 1.0434% | 1.4374% | 1.0289% | 1.0302% |
| 5 sessions | 1.8549% | 1.8633% | 2.3083% | 1.8526% | 1.8812% |
| 21 sessions | 3.9697% | 4.0201% | 3.9749% | 3.8139% | 4.0424% |

The head reduces raw-memory RMSE by about **27% at one session** and **20% at five sessions**. Memory features improve on the control-only ablation at five and 21 sessions, but not at one session. **The drift reference still has lower pooled error at all three horizons.** The candidate should therefore remain experimental rather than being described as a superior point predictor.

The comparison varies by year:

| Year | 1-session adaptive / drift | 5-session adaptive / drift | 21-session adaptive / drift |
| --- | ---: | ---: | ---: |
| 2023 | 0.8936% / 0.8694% | 1.8220% / 1.7840% | 3.7917% / 3.6686% |
| 2024 | 0.9233% / 0.9138% | 1.5758% / 1.5837% | 3.0995% / 3.0219% |
| 2025 | 1.3854% / 1.3655% | 2.1222% / 2.1388% | 4.8292% / 4.4583% |
| 2026 | 0.8210% / 0.8189% | 1.8607% / 1.8644% | 3.9645% / 4.0213% |

For the sector panel, pooled normalized RMSE is **1.1922%, 2.4176% and 4.8707%** for the adaptive head, compared with **1.1887%, 2.4107% and 4.8328%** for drift. Pooling squares normalized errors before averaging; it is not an arithmetic average of asset RMSEs. The adaptive head improves on the no-memory ablation at all three horizons, but the difference is small.

Selection counts are included rather than implying that memory is always used. For the index, the full head is selected at 57/186, 139/186 and 103/186 origins at horizons 1, 5 and 21, respectively; the remaining forecasts use a baseline. No candidate triggered the nonpositive-price fallback in this run.

The paired block-bootstrap ranges do not show a stable memory advantage across horizons. They are descriptive, not a correction for researcher exposure, all model-selection uncertainty or data revisions. Earlier dates, correlated assets and overlapping targets are not equivalent to independent replications.

## Uncertainty: the clearest improvement

The new nominal **80%** intervals use changing volatility and matured forecast errors. For the index:

| Horizon | Adaptive coverage | Adaptive width | Adaptive interval score | Fixed-Gaussian score | Raw-memory score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 session | 79.03% | 2.5603% | 3.7328% | 3.9649% | 5.0024% |
| 5 sessions | 81.72% | 5.1827% | 6.8180% | 7.9228% | 15.3083% |
| 21 sessions | 79.57% | 10.1211% | 14.2760% | 16.6066% | 21.9119% |

Widths and scores are normalized by origin price. Lower interval scores are better: they penalize both excessive width and missed outcomes. The fixed-Gaussian comparison uses the same adaptive point forecast with expanding historical variance, isolating the interval construction. The raw-memory comparison also has a different center. Adaptive scores improve over the fixed-Gaussian reference by approximately **6%, 14% and 14%** across these horizons.

Coverage near the nominal level in this sample does not guarantee future coverage. Per-year and per-sector coverage is retained in the complete interval file. A single point forecast should not be presented without its horizon and uncertainty.

## Reproduction and forecast issuance

- [Point metrics, including every year and asset](results/adaptive_metrics.csv)
- [Coverage, width and interval scores](results/adaptive_intervals.csv)
- [Candidate-selection counts](results/adaptive_choices.csv)
- [Paired error comparisons](results/adaptive_paired_comparisons.csv)
- [Run summary, source metadata and hashes](results/adaptive_summary.json)
- [Recorded candidate protocol](results/adaptive_protocol.json)

Origin-level predictions and full coefficient/scaler audits are saved locally under `artifacts/adaptive-market/`. `latest_forecasts.csv` contains unscored forecasts issued from the final observed date, with 1-, 5- and 21-session horizons. Every run also writes a new timestamped record under `issued_forecasts/`, preserving earlier issuances across reruns. These are local records, not independently certified timestamps. Future exchange-calendar dates are not guessed.

The original simulator and earlier forecast layer remain unchanged. The [new specification](adaptive_forecasting.md) gives the equations and chronology. All **35 tests pass**, covering future-data independence, matured-label boundaries, signed calibration, interval scoring, scale invariance and latest-forecast handling alongside the earlier regression suite. The market CLI and offline CLI completed successfully. The maintained notebook executed all seven code cells from a fresh kernel; its adaptive predictions, metrics, interval scores, choices and latest forecasts exactly match the offline CLI. The notebook also exports to HTML. These checks used the isolated Python 3.12 environment on macOS; other platforms were not tested.

The justified conclusion is improved raw-model calibration and substantially better uncertainty estimates in this sample. A reliable point-forecasting or trading advantage has **not** been established. Further model decisions should use a newly specified study and subsequently observed data, rather than retuning this history until a favorable score appears.
