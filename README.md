# ETF simulation with temporal memory

This project studies how fractional memory and interactions between sectors can model fluctuations in the S&P 500 and eleven sector ETFs. It includes stochastic return simulations, price-forecast experiments and a conditional-variance model for forecasting risk.

The revised model combines **positive fractional-memory weights and diffusion on a sector-correlation graph**, with a persistent variance state and asymmetric responses to negative returns. Its equations, mathematical properties and chronological evaluation are developed in the [research manuscript](paper/main.pdf). The [LaTeX package](paper/latex-source.zip) includes the manuscript, figures, tables and bibliography. The [refined model explanation](docs/refinement.md) connects the equations to the implementation and the earlier specification.

An [external-universe experiment](docs/external_validation.md) retains that model unchanged, evaluates eight international ETFs, and adds FIGARCH plus an optional **fixed 50:50 fractional/GJR forecast ensemble**. Its rules were recorded before the new download. The [related-work comparison](docs/related_work.md) explains the contribution relative to existing network volatility models. A separate [arXiv source archive](paper/arxiv-source.zip) contains only manuscript dependencies for compilation.

The [prediction figure guide](docs/prediction_figures.md) compares observed risk evolution with the recorded model forecasts at several horizons and for every ETF. It also explains the revised Figure 2, where EWMA and all other methods have visible, directly labeled QLIKE scores.

![Volatility-forecast losses across the index, sectors and international ETFs](paper/figures/external_comparison.png)

*186 forecast origins per panel during 2023–2026. Lower loss is better; methods share targets within each panel. The US panels reuse development data. The international panel is a separate universe with rules fixed before download, still using retrospective adjusted prices.*

## Run the research workflow

Use **Python 3.12**. From the repository root, on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_refinement.py --synthetic --output artifacts/refinement-synthetic
python -m unittest discover -s tests -v
```

The example runs offline after installation, using 1,300 explicitly synthetic prices. It saves forecasts, aggregate and annual metrics, paired comparisons, fitting audits and figures. Synthetic execution checks the workflow; it is not evidence of market performance.

To reproduce the market study with the documented local snapshot:

```bash
python run_refinement.py --data-dir artifacts/yahoo-through-2026-10-16 \
  --output artifacts/refinement-study --publish-assets
```

The [data guide](docs/data.md) documents the CSV format, downloader and source limitations. Market prices are not bundled. The [paper guide](paper/README.md) explains the recorded protocol, numerical outputs and LaTeX build. A fresh adjusted-price download may differ from the recorded snapshot.

## Recorded findings

Lower QLIKE means a better variance forecast on the common targets. Pooled scores are:

| Panel | Horizon | Refined fractional | Fixed 50:50 ensemble | GJR-GARCH | FIGARCH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Index | 1 session | 0.5819 | 0.5393 | 0.5251 | 0.6514 |
| Index | 5 sessions | 0.3574 | 0.3274 | 0.3128 | 0.3642 |
| Index | 21 sessions | 0.4977 | 0.4580 | 0.4360 | 0.4764 |
| Sectors | 1 session | 0.6242 | 0.6060 | 0.6053 | 0.6523 |
| Sectors | 5 sessions | 0.5478 | 0.5371 | 0.5401 | 0.5422 |
| Sectors | 21 sessions | 0.6043 | 0.5925 | 0.5931 | 0.5966 |
| International | 1 session | 0.5290 | 0.5131 | 0.5083 | 0.5438 |
| International | 5 sessions | 0.6636 | 0.6509 | 0.6496 | 0.6671 |
| International | 21 sessions | 0.6715 | 0.6613 | 0.6623 | 0.6730 |

The fixed ensemble improves on the standalone fractional model and FIGARCH in all nine pooled comparisons. It beats GJR-GARCH in three, with small margins; all ten-origin-block uncertainty ranges against GJR include zero. The international panel does not confirm a benefit from graph diffusion. The [complete external results](docs/external_validation.md) retain annual deterioration, per-asset exceptions and ablations. All 760 FIGARCH fits converged.

The earlier refinement itself improved the initial fractional specification in all six US comparisons and GARCH in four; GJR-GARCH remained stronger in all six. Those [earlier results](docs/refinement.md) and their [paired comparisons](paper/refinement_comparisons.csv) remain available. Neither successive gains nor external-universe replication establish directional price accuracy or trading profitability.

The model has an explicit positive variance floor, a graph operator that preserves constants and contracts graph energy, and fixed-parameter stationarity and moment properties under stated assumptions. Its multi-step marginal variance forecasts are computed without Monte Carlo error. These mathematical properties do not guarantee predictive superiority, directional price accuracy or trading profitability.

## Earlier simulations and price forecasts

The earlier return model and forecast extensions remain available:

```bash
python run_experiment.py --mode simulation --synthetic --output artifacts/simulation
python run_experiment.py --mode forecast --synthetic --output artifacts/fixed-blend
python run_adaptive.py --synthetic --output artifacts/adaptive-demo
python -m jupyter lab "Fraccional model S&P500 (final version).ipynb"
```

The [maintained notebook](Fraccional%20model%20S%26P500%20%28final%20version%29.ipynb) contains executed synthetic examples of the earlier simulator and adaptive price-forecast extension, followed by the new variance specification and its reproduction command. Change `DATA_DIR` to use local CSVs in the earlier examples.

The [adaptive specification](docs/adaptive_forecasting.md) explains the regularized forecast correction and error-based intervals. Its [execution record](docs/adaptive_results.md) reports improvements in interval scores and coverage; a simple drift forecast remains stronger on pooled point error. The [core forecast specification](docs/forecasting.md), [simulation methodology](docs/methodology.md) and [earlier results](docs/results.md) document the preceding workflows.

## Reproducibility and scope

Features, graph estimates, scaling, fitting and selection use information available at each forecast origin. The US historical periods were inspected during development. The international universe was fixed before its first download, but shares global shocks with the US assets and remains retrospective. Neither experiment is a prospective test. Daily squared-return targets are noisy volatility proxies, and retrospectively adjusted prices are not point-in-time data.

The **53 passing tests** cover future-price perturbations, training-label boundaries, conditional moments, interval scoring, alignment, reproducibility and portfolio aggregation. The variance tests check fractional kernels, graph positivity, energy contraction, asset relabeling, GARCH/GJR filtering, asymmetric normalization, validation alignment and simulated moments. The new checks cover FIGARCH library equivalence on strided panel inputs, correct ensemble scoring and the effective-memory impulse response.

The standalone implementation is [refined_fractional.py](refined_fractional.py), with its experiment in [run_refinement.py](run_refinement.py). [external_validation.py](external_validation.py) and [run_external.py](run_external.py) add the frozen-universe comparison. The first variance specification remains in [fractional_diffusion.py](fractional_diffusion.py), reproduced by [run_paper.py](run_paper.py). The earlier return core is [temporal_memory.py](temporal_memory.py); [forecasting.py](forecasting.py) and [adaptive_forecasting.py](adaptive_forecasting.py) contain its separate forecast layers. The [scientific review](docs/scientific_review.md) explains why the original fractional-PDE derivation did not justify that return recurrence. The variance specifications have their own explicit assumptions and proofs.

## Citation and reuse

Please cite this project using [CITATION.cff](CITATION.cff) and identify the Git commit used. The manuscript has not been submitted to arXiv. No reuse license has yet been selected by the authors; see the [publication notes](docs/publication.md).
