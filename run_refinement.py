"""Evaluate the recorded persistence/asymmetry refinement and all earlier references."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from refined_fractional import rolling_refinement
from run_paper import summarize, paired_losses
from temporal_memory import load_prices, synthetic_prices

LABELS = {'fractional_diffusion': 'Original fractional',
          'persistent_symmetric': 'Persistent symmetric',
          'persistent_asymmetric': 'Persistent asymmetric',
          'asymmetric_no_graph': 'Without graph',
          'asymmetric_exponential': 'Exponential kernel',
          'ewma': 'EWMA', 'garch': 'GARCH', 'gjr_garch': 'GJR-GARCH'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--synthetic', action='store_true')
    inputs.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/refinement-study'))
    parser.add_argument('--publish-assets', action='store_true')
    args = parser.parse_args()
    if args.synthetic:
        index, sectors = synthetic_prices(1300)
        start = str(index.index[900].date())
        source, hashes = 'Synthetic iid Gaussian returns; seed 17; 1300 prices.', {}
    else:
        index, sectors = load_prices(args.data_dir)
        start = '2023-01-01'
        manifest = args.data_dir / 'source.json'
        source = json.loads(manifest.read_text()) if manifest.exists() else 'Unverified local provenance.'
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.data_dir / 'raw').glob('*.csv')}
    args.output.mkdir(parents=True, exist_ok=True)
    frames, audits = [], {}
    for panel, prices in [('index', index), ('sectors', sectors)]:
        print(f'Evaluating {panel}...', flush=True)
        forecasts, audit = rolling_refinement(prices, start=start)
        frames.append(forecasts.assign(panel=panel))
        audits[panel] = audit
        print(f'{panel}: {forecasts.origin.nunique()} origins, {len(audit)} refits.', flush=True)
    forecasts = pd.concat(frames, ignore_index=True)
    metrics = summarize(forecasts)
    # Reuse the identical bootstrap calculation, with the new model as its target.
    renamed = forecasts.assign(method=forecasts.method.replace({
        'fractional_diffusion': 'original_fractional', 'persistent_asymmetric': 'fractional_diffusion'}))
    comparisons = paired_losses(renamed)
    comparisons['reference'] = comparisons.reference.replace({'original_fractional': 'fractional_diffusion'})
    code = ['refined_fractional.py', 'run_refinement.py', 'fractional_diffusion.py', 'run_paper.py']
    record = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), source=source, input_sha256=hashes,
                  code_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code},
                  protocol=json.loads(Path('paper/refinement_protocol.json').read_text()),
                  first_origin=str(forecasts.origin.min().date()), last_origin=str(forecasts.origin.max().date()),
                  last_target=str(forecasts.target.max().date()), origins=int(forecasts.origin.nunique()),
                  refits_per_panel={k: len(v) for k, v in audits.items()},
                  nonconverged_gjr_fits=sum(not g['converged'] for a in audits.values() for f in a for g in f['gjr']))
    forecasts.to_csv(args.output / 'forecasts.csv', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    comparisons.to_csv(args.output / 'comparisons.csv', index=False)
    (args.output / 'fit_audit.json').write_text(json.dumps(audits, indent=2, allow_nan=False) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    pooled = metrics[metrics.year == 'all']
    table = pooled.pivot(index='method', columns=['panel', 'horizon'], values='qlike')
    table = table.loc[list(LABELS), [('index', h) for h in [1, 5, 21]] + [('sectors', h) for h in [1, 5, 21]]]
    table.index = [LABELS[m] for m in table.index]
    table.to_latex(args.output / 'scores.tex', float_format='%.4f', multicolumn_format='c')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, panel in zip(axes, ['index', 'sectors']):
        for method, color in [('fractional_diffusion', '#8a9197'), ('persistent_asymmetric', '#b65d25'),
                              ('garch', '#467f7c'), ('gjr_garch', '#243b53')]:
            row = pooled[(pooled.panel == panel) & (pooled.method == method)].set_index('horizon')
            ax.plot([1, 5, 21], row.loc[[1, 5, 21], 'qlike'], marker='o', color=color, label=LABELS[method])
        ax.set(xlabel='Forecast horizon (observations)', ylabel='Mean normalized QLIKE', xticks=[1, 5, 21],
               title='S&P 500' if panel == 'index' else 'Sector panel')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(args.output / f'comparison.{extension}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    if args.publish_assets:
        for name in ['metrics.csv', 'comparisons.csv', 'summary.json']:
            shutil.copyfile(args.output / name, Path('paper') / ('refinement_' + name))
        shutil.copyfile(args.output / 'scores.tex', 'paper/tables/refinement_scores.tex')
        for extension in ['pdf', 'png']:
            shutil.copyfile(args.output / f'comparison.{extension}', f'paper/figures/refinement_comparison.{extension}')
    print(table.round(5).to_string())
    print(f"GJR fits without convergence: {record['nonconverged_gjr_fits']}. Results: {args.output}.")


if __name__ == '__main__':
    main()
