# Research manuscript

`main.tex` is the English research manuscript; `main.pdf` is the compiled manuscript. The source follows the original report's A4, 12-point Latin Modern typography, title page, contents page, numbered equations, single-column text and running header. Figures and numerical tables are generated from the recorded experiment. `latex-source.zip` contains the LaTeX sources and figure/table dependencies for a conventional LaTeX installation or Overleaf.

This is a **new variance-model specification preserving fractional memory and sector diffusion**, not a claim that the original return recurrence solved a fractional Black–Scholes equation. The earlier implementation remains available. The manuscript supplies proofs of positivity, graph contraction and fixed-parameter moment stability, and an effective-kernel derivation. It includes the first study, a [persistent asymmetric refinement](../docs/refinement.md), and an [external-universe extension](../docs/external_validation.md) with FIGARCH and a fixed forecast combination.

## Build

With [Tectonic](https://tectonic-typesetting.github.io/en-US/):

```bash
cd paper
tectonic --reruns 3 main.tex
```

A conventional alternative is `latexmk -pdf main.tex`. The packages are standard LaTeX; BibTeX uses `references.bib`. Tectonic 0.15.0 was downloaded locally into the ignored `artifacts/paper-tools/` directory for this workspace, without adding a compiler binary to the repository. The first build can download fonts and packages.

The fixed pass count avoids a [known bibliography rerun-detection issue in Tectonic 0.15](https://github.com/tectonic-typesetting/tectonic/discussions/1186). The source archive is checked by extracting it and compiling independently; references and the contents page are resolved in the delivered PDF.

After compiling, run `python package_paper.py` from the repository root to create both archives. `latex-source.zip` contains the LaTeX dependencies plus aggregate data and protocol records for review. `arxiv-source.zip` contains only the main TeX source, referenced tables and figures, bibliography source and compiled `.bbl`. It excludes the output PDF, logs, raw prices and unused figures, following the [arXiv TeX preparation guidance](https://info.arxiv.org/help/submit_tex.html). The archive has been built locally; no arXiv upload or server-side compilation has been performed.

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

## What the paper supports

In the US development study, the first fractional model has lower pooled QLIKE than tuned EWMA in all six panel–horizon comparisons, but loses to GARCH. The persistent asymmetric refinement improves on that fractional model in all six comparisons and on GARCH at one and five sessions in both panels. GJR-GARCH remains stronger in all six cases. Nominal ranges against GARCH include zero. Those years were available during earlier project iterations; these are retrospective exploratory results.

The later fixed 50:50 fractional/GJR ensemble improves on the standalone refined model and FIGARCH in all nine pooled comparisons across the US index, US sectors and eight international ETFs. It beats GJR in three; all nine nominal 95% paired intervals include zero at both block lengths. This establishes neither superiority nor equivalence to GJR. The external panel does not confirm a benefit from graph diffusion. Annual, per-asset and secondary-score exceptions are retained in the [external study](../docs/external_validation.md).

The mathematical reformulation and reproducible benchmark provide a structured assessment of the model. The model is a restricted multivariate ARCH specification; the [related-work comparison](../docs/related_work.md) explains its relationship to FIGARCH, spatial ARCH, network log-ARCH, network GARCH and threshold network GARCH. It does not establish novelty priority, editorial acceptance, universal predictive superiority or trading profitability.

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
