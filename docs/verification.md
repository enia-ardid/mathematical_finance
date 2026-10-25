# Reproduction and document checks

The manuscript assets are generated from the retained prediction records. The numerical models, fitting routines, original protocols and original result CSVs are preserved. Additional forecast diagnostics have their own protocol and computational amendment.

The checked manuscript has 36 pages, eleven figures, nine numbered tables and 22 references. Ten generated table fragments supply the nine tables, because Table 1 has two panels. All 66 tests pass. The arXiv archive contains 26 dependencies; the extended LaTeX archive contains 55 files. Both compile independently with matching PDF text and no LaTeX warnings. The asset check verifies 144 score-table cells and exact regeneration of all figures and table fragments.

## Automated checks

```bash
python -m unittest discover -s tests -v
python render_submission_assets.py
python render_submission_assets.py --output artifacts/presentation-reproduction
python verify_submission.py --regenerated artifacts/presentation-reproduction
```

The suite contains 66 tests. The verifier checks input and source hashes, generated asset hashes, score-table values and best-value formatting, bootstrap differences and intervals, per-asset win tables, reported comparison counts, diagnostic outcomes, citations, cross-references, bibliography order and local documentation links. Separately rendering the eleven figures and ten table fragments must reproduce their bytes exactly. Figures are saved without variable PDF timestamps. No model fit is part of either renderer or verifier.

Score tables use four decimal places and right-aligned numeric columns. Bold indicates the smallest unrounded loss among the displayed models, including ties. Loss differences and intervals use five decimal places. Diagnostic statistics and p-values use three decimal places, with values below 0.001 shown as inequalities. Calibration p-values and win frequencies are not ranked as forecast-quality scores. The full precision is retained in CSV files.

## Manuscript and source archives

Build the manuscript using the commands in the [paper guide](../paper/README.md), then run `python package_paper.py`. Both source archives are extracted into clean directories and compiled independently. Their extracted PDF text must match the main PDF, including resolved citations and table numbers. Build logs are checked for missing references and overfull boxes.

Visual checks cover figure legends, axis units and scale, table alignment, equation superscripts, the displayed finite-memory sum and the contents page. Table 4 is regenerated from the recorded network metrics. Its horizon labels are legitimate numbers; page headers retain normal pagination.

American English is used in the manuscript prose. A local Webster-based American English dictionary is used for spelling review, with technical vocabulary, abbreviations, names and TeX fragments reviewed separately. Bibliographic titles preserve their published spelling.

The [diagnostic guide](forecast_diagnostics.md) records the new statistical findings. No paired DM comparison against GJR rejects at 5%; one-session country-panel calibration rejects for all three models after Holm adjustment at both lag choices. These findings remain part of the manuscript.
