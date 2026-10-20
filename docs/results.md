# Execution record

The subsequent [adaptive-model evaluation](adaptive_results.md) reports 186 rolling origins over 2023–2026, a no-memory ablation, empirical uncertainty intervals and latest-forecast issuance. The results below remain the record of the earlier workflows.

These results use Python 3.12.10 and the dependencies in `requirements.txt`. The 17 October forecast extension preserves the core simulator from the 16 October review. Its rolling forecasts and the earlier single-origin simulations are different tasks, recorded separately below.

## Rolling forecasts: January–October 2026

The [forecast protocol](results/forecast_protocol.json) was recorded before inspecting this newly downloaded test period. It fixes horizons, origin spacing, refit spacing, the coupling grid and the calibration rule. This is a working protocol record, not a formal preregistration or a prospective trading trial. No choices were changed in response to the test scores.

The new Yahoo Finance snapshot was downloaded on **2026-10-17**, requesting 2018-01-01 through 2026-10-17 (exclusive). It contains 2,210 price dates before alignment. Common history begins on **2018-06-19** and ends on **2026-10-16**. Development ends on **2026-01-02**; the first permissible test target is **2026-01-05**. This period follows both datasets examined in the earlier review, although all adjusted prices come from a retrospective snapshot.

Internal validation uses 36 expanding origins. The outer test also contains **36 origins**, from 2026-01-02 through 2026-09-15, with targets through 2026-10-14 and eight parameter refits. Origins advance every five observations and all horizons share the same origins. The last incomplete forecast blocks are excluded. Earlier test observations become available to later forecasts, while calibration weights and coupling scales remain frozen.

The index's origin-normalized RMSE is:

| Horizon | Raw memory | Calibrated | Log-drift reference | Constant price | Memory weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 session | 1.2708% | 0.8643% | 0.8676% | 0.8682% | 0.0221 |
| 5 sessions | 1.8006% | 1.6167% | 1.6167% | 1.6500% | 0 |
| 21 sessions | 3.3997% | 3.4738% | 3.4738% | 3.6893% | 0 |

The calibrated one-session forecast reduces raw-memory error by about **32%**, but only about **0.45%** relative to constant-price RMSE. Its skill score relative to constant-price mean squared error is **0.0090**. At five and 21 sessions, validation selects the drift reference entirely: these results do not demonstrate a memory contribution. The raw model performs better than the calibrated forecast at 21 sessions in the outer test; the frozen selection is retained, including this loss.

For sectors, the selected one-session memory weight is **0.0274**, with coupling scale **0.5**; longer horizons select zero memory weight and zero coupling. The calibrated forecast beats raw memory in 11/11, 11/11 and 7/11 sectors at 1, 5 and 21 sessions, respectively. It beats constant price in 7/11, 7/11 and 4/11 sectors. These counts describe this sample; they are not independent significance tests. Long-horizon targets overlap and sectors are correlated.

The [full metrics](results/forecast_2026_metrics.csv), [selected weights](results/forecast_2026_choices.csv), [validation scores](results/forecast_2026_validation.csv) and [run summary](results/forecast_2026_summary.json) provide all results, dates, source metadata, input hashes and the core file hash. The README figure is copied from this run's `forecast_overview.png`. Raw prices, origin-by-origin forecasts and full refit logs remain in ignored local artifacts. A fresh download can change adjusted prices, so exact snapshot reproduction requires retaining those local inputs.

```bash
python download_data.py --output artifacts/yahoo-through-2026-10-16 --start 2018-01-01 --end 2026-10-17
python run_experiment.py --mode forecast --data-dir artifacts/yahoo-through-2026-10-16 --test-start 2026-01-05 --max-prices 10000 --output artifacts/forecast-2026
```

The [forecast specification](forecasting.md) gives the conditional-moment equations, calibration objective and information available at each origin. Forecast means are exact under the fitted Gaussian recurrence; this is not a claim that market prices follow that distribution. No profitable-trading conclusion or general predictive superiority follows from this experiment.

## Offline forecast demonstration

```bash
python run_experiment.py --mode forecast --synthetic --output artifacts/demo
```

The maintained notebook runs this same forecast workflow on 500 synthetic business-date prices generated with seed 17. The inputs have a common Gaussian component and independent sector components, with no imposed long-memory mechanism. They verify execution and make the example available offline; their scores are not evidence of market forecasting skill.

## Retained simulation experiments — 16 October review

The following results use the earlier multi-year continuation from one training origin, with training-only drift and scale calibration. They do not use later observed returns. The command-line default remains `--mode simulation` for compatibility. Seeds, coupling-grid candidates and split fractions were not adjusted after inspecting these held-out results.

### Synthetic simulation

```bash
python run_experiment.py --synthetic --output artifacts/demo
```

The synthetic input has 500 business-date prices, generated with seed 17. It uses the same generator as the forecast example above, but the evaluation task differs.

The index ensemble has RMSE **429.60** versus **122.27** for the log-drift reference on the final 150 returns. Its R² is **−5.1644**. This is an execution example, not a benchmark of financial forecasting skill. Its figure is retained at `docs/figures/synthetic_overview.png`.

### Yahoo Finance snapshot: 2018–2024

The documented downloader completed a fresh request for 2018-01-01 through 2025-01-01 (exclusive). The full download contains 1,761 price dates; the common ETF/index history is trimmed for ETF inception and the last 1,644 aligned prices are used. Training has 1,150 returns ending **2023-01-13**. The 493 held-out returns run from **2023-01-17 to 2024-12-31**.

| Index model | Test RMSE | Test MAE | Test R² |
| --- | ---: | ---: | ---: |
| Memory ensemble, ten fixed seeds | 275.09 | 234.64 | 0.8164 |
| Constant price | 1,086.56 | 881.78 | −1.8640 |
| Log-drift baseline | 706.19 | 556.00 | −0.2098 |

This approximately recovers the reported index score on a new snapshot, but does not validate the manuscript's derivation or reproduce the original data exactly. The sector pipeline uses the corrected training-only ordering, coupling scale and endpoint. The simulated basket's held-out index RMSE is **891.15** (R² **−0.9265**); its RMSE against the observed basket is **949.10** (R² **−1.2075**). A good direct-index score does not imply a good sector reconstruction.

The complete [Yahoo metrics](results/yahoo_metrics.csv), [basket comparisons](results/yahoo_portfolio_metrics.csv) and [run summary](results/yahoo_run_summary.json) include the download timestamp, adjustment setting and hashes. Its figure is retained at `docs/figures/market_overview.png`. Raw downloaded CSVs are retained locally in the ignored artifacts directory, not redistributed. The exact saved snapshot is therefore not bundled for a new clone; a new provider download may differ.

Commands used for this sourced experiment:

```bash
python download_data.py --output artifacts/yahoo-2018-2024 --start 2018-01-01 --end 2025-01-01
python run_experiment.py --data-dir artifacts/yahoo-2018-2024 --output artifacts/yahoo-results
```

### Supplied local CSVs

The files cover 2021-01-05 through 2026-01-02. Their provenance is unresolved, as explained in [data.md](data.md). All 1,255 prices were used; 877 returns form training and 377 form evaluation. Training ends **2024-07-02** and test starts **2024-07-03**.

| Index model | Test RMSE | Test MAE | Test R² |
| --- | ---: | ---: | ---: |
| Memory ensemble, ten fixed seeds | 875.72 | 725.40 | −2.6199 |
| Constant price | 714.50 | 576.51 | −1.4097 |
| Log-drift baseline | 274.51 | 220.09 | 0.6443 |

The memory ensemble underperforms both references on the index. Three of the eleven sector ensembles have lower RMSE than their log-drift references (XLF, XLY, XLRE); all sector ensemble R² values are negative. These results do not support a general advantage from the memory recurrence. This window overlaps the Yahoo experiment and is not an independent untouched test set.

The simulated ETF basket has RMSE **311.07** and R² **0.5432** against the index, using held-out dates and a common last-training anchor. Against the observed ETF basket, its RMSE is **276.28** and R² **0.5552**. These are different targets, reported separately. The basket's illustrative weights and unverified adjustment conventions prevent an index-replication interpretation.

The complete [metrics](results/local_metrics.csv), [portfolio comparisons](results/local_portfolio_metrics.csv) and [run summary](results/local_run_summary.json) record all sectors, dates, parameters, seeds, versions and input hashes. Raw prices remain private. Re-execution with the same local files uses:

```bash
python run_experiment.py --data-dir /path/to/data_finance --output artifacts/local
```

## Historical outputs

The stored pre-review final notebook reports index test R² `0.8163` and a single-path ETF reconstruction score `0.7481`. These are historical outputs, not results of the corrected code. The original market snapshot is not available in the repository, the sector pipeline had test leakage, and the reconstruction score includes training dates. See [scientific_review.md](scientific_review.md) for the exact discrepancies. The historical beta also contains mixed execution states and is not used to validate the maintained implementation.

## What was checked

The 25 tests cover training/test independence, future-price perturbations, deterministic estimation, conditional means and variances, the binomial weights and recurrence/PDE distinction, reflected boundaries, return-date alignment, missing-price rejection, sMAPE, baseline anchoring and capital-weighted portfolio invariance. The simulation CLI completed synthetic, supplied-CSV and Yahoo runs; the forecast CLI completed synthetic and 2026 Yahoo runs. The maintained notebook executed from a fresh kernel with no cell errors and exported to HTML. `pip check` found no broken dependency requirements in the isolated Python 3.12 environment. Installation and execution were checked on macOS; other platforms were not tested.

These checks establish the stated implementation behavior. They do not prove model stability, a continuous-time limit, market-data provenance, statistical significance or profitable trading.
