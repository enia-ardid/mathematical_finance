# Publication notes

The current repository presents exploratory simulations and rolling forecasts with a tested offline entry point, an executed notebook, equations in Markdown, explicit assumptions and measured baseline comparisons. It should not be described as a validated fractional-PDE solver or as reproducing the submitted report's numerical claims. The [scientific review](scientific_review.md) states the revisions needed for a new manuscript submission.

The [adaptive extension](adaptive_results.md) improves the raw-model calibration and interval scores in its recorded sample. Its point forecasts do not consistently outperform drift, so it remains an experimental comparison. Neither the earlier blend nor the adaptive head is presented as a proven market predictor.

The later [revised manuscript](../paper/README.md) introduces a different, explicitly defined variance model preserving fractional memory and diffusion. Its [persistent asymmetric refinement](refinement.md) improves the earlier fractional model in all six pooled comparisons and GARCH in four; GJR-GARCH remains stronger in all six. The manuscript includes proofs, regenerated comparisons and LaTeX/PDF files. It remains a research draft for joint-author review; mathematical properties and retrospective average improvements do not imply journal acceptance or independent evidence of predictive superiority.

## Files and history

The maintained notebook replaces stale outputs with a fresh synthetic execution, including the adaptive extension. Its internal Hurst-estimation warnings remain visible, with local environment paths redacted from saved warning text. The beta notebook retains its original code and figures, with an English historical notice and personal filesystem paths redacted from text outputs. Unmodified local backups are kept in the ignored `artifacts/originals/` directory. Git history has not been rewritten.

The raw CSVs, supplied PDF, virtual environment, execution artifacts, local caches and environment files are excluded from the publishable working tree. The included execution record contains aggregate metrics and file hashes, not the raw market CSVs. The supplied original PDF remains outside the repository and was not edited; the new manuscript and its sources are under `paper/`. Original source-code authorship and collaboration are not inferred from the review.

## Remaining decisions

The expanded manuscript adds an external-universe experiment with a specification fixed before download, a direct FIGARCH reference, an optional fixed 50:50 forecast combination, and a more precise comparison with existing network volatility models. See [external validation](external_validation.md) and [related work](related_work.md). The new panel remains retrospective and correlated with US markets; it is not a prospective independent trading test.

For a public portfolio, the corrected implementation and documented limits are suitable to show. The unverified local CSVs must not be presented as a sourced, independently reproducible market dataset. The original PDF is not ready to publish as a validated research paper; its mathematical derivation, results and data provenance require the changes identified in the review.

No license was selected on behalf of the joint authors. Agree on reuse terms before describing the repository as open-source or inviting reuse under a particular license. This does not prevent showing the repository publicly.

For a GitHub social preview, use `paper/figures/external_comparison.png`; it shows the development and external variance-forecast comparisons. Earlier figures are retained for their corresponding execution records. Uploading the image in the repository's social-preview settings is a manual step. Repository visibility and GitHub settings have not been changed.
