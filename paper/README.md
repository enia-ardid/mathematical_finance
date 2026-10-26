# Research manuscript

`main.tex` contains the manuscript and `main.pdf` the compiled document. The source uses A4 pages, 12-point Latin Modern type and numbered equations. Figures and tables are generated from saved experimental records. `latex-source.zip` contains the sources and dependencies for a conventional LaTeX installation or Overleaf.

The model combines **fractional memory and graph diffusion in conditional variance**. The manuscript proves positivity, graph contraction and fixed-parameter moment stability, and derives the effective kernel of the [persistent asymmetric specification](../docs/refinement.md). Experiments cover the initial model, its refinements, [country ETFs](../docs/external_validation.md), forecast combinations and network benchmarks.

## Build

With [Tectonic](https://tectonic-typesetting.github.io/en-US/):

```bash
cd paper
tectonic --reruns 3 main.tex
```

A conventional alternative is `latexmk -pdf main.tex`. The packages are standard LaTeX; BibTeX uses `references.bib`. The documented build uses Tectonic 0.15.0. Its first run may download fonts and packages.

The fixed pass count avoids a [known bibliography rerun-detection issue in Tectonic 0.15](https://github.com/tectonic-typesetting/tectonic/discussions/1186). The source archive is checked by extracting it and compiling independently; references and the contents page are resolved in the delivered PDF.

After compiling, run `python package_paper.py` from the repository root to create both archives. `latex-source.zip` contains the LaTeX dependencies plus aggregate data and protocol records. `arxiv-source.zip` contains only the main TeX source, referenced tables and figures, bibliography source and compiled `.bbl`. It excludes the output PDF, logs, raw prices and unused figures, following the [arXiv TeX preparation guidance](https://info.arxiv.org/help/submit_tex.html). The archive has been built locally; no arXiv upload or server-side compilation has been performed.

## Reproduce numerical assets

From the repository root, using the existing Python environment:

```bash
python run_paper.py --synthetic --output artifacts/paper-synthetic
python run_paper.py --data-dir artifacts/yahoo-through-2026-10-16 --output artifacts/paper-study --publish-assets
python run_refinement.py --synthetic --output artifacts/refinement-synthetic
python run_refinement.py --data-dir artifacts/yahoo-through-2026-10-16 --output artifacts/refinement-study --publish-assets
python -m unittest discover -s tests -v
```

The [data guide](../docs/data.md) explains how to download a documented snapshot. Exact recorded inputs are retained locally, not redistributed; a new adjusted-price download can differ. The market command exports origin-level predictions and complete fitting audits to ignored artifacts, and copies aggregate metrics, paired comparisons, figures and LaTeX tables into this folder. The synthetic example has independent Gaussian returns without an imposed fractional or volatility mechanism.

After all numerical exporters, run `python render_submission_assets.py` before building the PDF. This regenerates all eleven figures and ten table fragments from saved forecasts and diagnostic records. Figure 2 displays direct QLIKE scores including EWMA; path figures include all three index horizons, panel summaries and every ETF. See the [figure guide](../docs/prediction_figures.md) for units, date alignment and regeneration. The recorded model code, forecasts and losses are unchanged.

- `protocol.json`: candidate grids and evaluation rules recorded before this experiment.
- `summary.json`: source metadata, input/code hashes, dates and optimizer-convergence counts.
- `metrics.csv`: aggregate QLIKE and squared-error diagnostics, including annual results.
- `comparisons.csv`: paired loss differences and nominal block-bootstrap ranges.
- `refinement_*`: separate protocol, metadata and results for the persistent asymmetric follow-up.
- `external_*`: separate external-universe protocol, disclosed computational amendment, metadata, aggregate/annual/asset-level scores and paired comparisons.
- `tables/` and `figures/`: generated manuscript assets.
- `submission_assets_manifest.json`: presentation-source, input and output hashes for every manuscript figure and generated table.
- `prediction_figures_manifest.json`: compatible subset manifest for the six prediction figures.
- `diagnostics_*`: recorded diagnostic rules, a CSV-tolerance amendment, summary and DM, calibration and win-count records.

## Results and interpretation

In the US development study, the first fractional model has lower pooled QLIKE than tuned EWMA in all six panel–horizon comparisons, but loses to GARCH. The persistent asymmetric refinement improves on that fractional model in all six comparisons and on GARCH at one and five sessions in both panels. GJR-GARCH remains stronger in all six cases. Nominal ranges against GARCH include zero. Those years were available during earlier project iterations; these are retrospective exploratory results.

The later fixed 50:50 fractional/GJR ensemble improves on the standalone refined model and FIGARCH in all nine pooled comparisons across the US index, US sectors and eight international ETFs. It beats GJR in three; all nine nominal 95% paired intervals include zero at both block lengths. This establishes neither superiority nor equivalence to GJR. The external panel does not confirm a benefit from graph diffusion. Annual, per-asset and secondary-score exceptions are retained in the [external study](../docs/external_validation.md).

The model restricts multivariate ARCH lag matrices to a shared graph resolvent and fractional temporal profile. The [literature comparison](../docs/related_work.md) explains the relation to FIGARCH, spatial ARCH, network log-ARCH and network GARCH. Its fixed-parameter moment results and its empirical accuracy are assessed separately.

## Reproduce the external study

The external protocol fixes eight international country ETFs before their first download. Keep both protocol JSON files: the amendment records a C-contiguous-array compatibility correction in the FIGARCH wrapper, without changing statistical rules. A fresh adjusted-price snapshot can differ from the original.

```bash
python download_external.py --output artifacts/yahoo-international-new
python run_external.py --external-dir artifacts/yahoo-international-new \
  --us-data-dir artifacts/yahoo-through-2026-10-16 \
  --us-study artifacts/refinement-study \
  --output artifacts/external-reproduction --publish-assets
```

To reproduce the recorded run, replace the external directory with `artifacts/yahoo-international-through-2026-10-16`. The complete local run is in `artifacts/external-study-complete`. The runner verifies source and data hashes, reuses the unchanged US forecasts, and refuses to overwrite a completed result. Further statistical changes require a separate experiment record; the frozen protocol identifies the recorded run.


## Network comparison

The [Network GARCH study](../docs/network_benchmark.md) applies the published first-order recursion to weighted and binary correlation networks, with a pooled control. Its protocol was recorded after earlier results were known and before fitting the new benchmarks. It preserves every reused fractional and reference forecast. All 228 selected fits converge; the complete test suite passes 66 tests.

With the recorded local data and earlier study files, run the following from the repository root with a new output directory:

~~~bash
python run_network_benchmark.py --output artifacts/network-reproduction --publish-assets
~~~

The runner validates the source and snapshot hashes in `network_protocol.json`. A new price snapshot requires a separately recorded protocol, not overwriting this record. Public `network_*` files contain the protocol, summary, annual/pooled metrics, asset metrics and paired comparisons; the manuscript uses `tables/network_scores.tex`.

## Additional diagnostics and presentation checks

The [diagnostic specification](../docs/forecast_diagnostics.md) defines the paired DM tests, Mincer–Zarnowitz regressions and win counts, including HAC lag choices and Holm families. These diagnostics use saved forecasts without recalibration. They were specified after the earlier results were known and are exploratory.

```bash
python run_forecast_diagnostics.py --output artifacts/diagnostics-reproduction
python render_submission_assets.py
python render_submission_assets.py --output artifacts/presentation-reproduction
python verify_submission.py
```

Use a fresh output directory for the diagnostic runner. The presentation renderer needs the retained origin-level records under `artifacts/`; it never fits a model. The [verification record](../docs/verification.md) describes the checks on figures, tables, citations and source archives.
