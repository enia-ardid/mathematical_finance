"""Evaluate the frozen fractional specification, FIGARCH and a fixed 50:50 ensemble."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import arch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from external_validation import append_equal_combination, append_figarch
from refined_fractional import rolling_refinement
from run_paper import summarize, paired_losses
from temporal_memory import load_prices

LABELS = {'ewma': 'EWMA', 'garch': 'GARCH', 'gjr_garch': 'GJR-GARCH',
          'figarch': 'FIGARCH', 'persistent_asymmetric': 'Fractional',
          'asymmetric_no_graph': 'Without graph',
          'asymmetric_exponential': 'Exponential kernel',
          'equal_combination': 'Fixed 50:50 ensemble'}


def paired_comparisons(forecasts):
    results = []
    for target in ['persistent_asymmetric', 'equal_combination']:
        renamed = forecasts.assign(method=forecasts.method.replace({
            'fractional_diffusion': 'original_fractional', target: 'fractional_diffusion'}))
        comparison = paired_losses(renamed)
        comparison['reference'] = comparison.reference.replace({'original_fractional': 'fractional_diffusion'})
        results.append(comparison.assign(model=target))
    return pd.concat(results, ignore_index=True)


def publish(output, metrics):
    for name in ['metrics.csv', 'comparisons.csv', 'summary.json', 'asset_metrics.csv']:
        shutil.copyfile(output / name, Path('paper') / ('external_' + name))
    table = metrics[(metrics.year == 'all') & (metrics.panel == 'international')].pivot(
        index='method', columns='horizon', values='qlike').loc[list(LABELS)]
    table.index = [LABELS[m] for m in table.index]
    table.to_latex('paper/tables/external_scores.tex', float_format='%.4f')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    selected = ['persistent_asymmetric', 'equal_combination', 'gjr_garch', 'figarch']
    for ax, panel in zip(axes, ['index', 'sectors', 'international']):
        data = metrics[(metrics.year == 'all') & (metrics.panel == panel)]
        for method in selected:
            row = data[data.method == method].sort_values('horizon')
            ax.plot(range(3), row.qlike, marker='o', label=LABELS[method])
        ax.set(xticks=range(3), xticklabels=['1', '5', '21'], xlabel='Horizon (sessions)', title=panel.title())
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean normalized QLIKE')
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(Path('paper/figures') / ('external_comparison.' + extension), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-dir', type=Path, required=True)
    parser.add_argument('--us-data-dir', type=Path, required=True)
    parser.add_argument('--us-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/external-study'))
    parser.add_argument('--publish-assets', action='store_true')
    args = parser.parse_args()
    protocol_path = Path('paper/external_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    amendment_path = Path('paper/external_protocol_amendment.json')
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else None
    expected_hashes = protocol['code_sha256'].copy()
    if amendment:
        if amendment['original_protocol_sha256'] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
            raise ValueError('Amendment does not refer to this protocol.')
        expected_hashes.update(amendment['corrected_code_sha256'])
    # Refuse accidental post-freeze model or protocol changes.
    for name, digest in expected_hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name} differs from the frozen specification.')
    source = json.loads((args.external_dir / 'source.json').read_text())
    digest = hashlib.sha256((args.external_dir / 'prices.csv').read_bytes()).hexdigest()
    if digest != source['sha256']['prices.csv'] or source['protocol_sha256'] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError('External data or protocol hash mismatch.')
    external = pd.read_csv(args.external_dir / 'prices.csv', index_col='Date', parse_dates=['Date'])
    if list(external.columns) != protocol['external_tickers']:
        raise ValueError('External universe changed.')
    # Reuse recorded US forecasts only when the recorded code and data match.
    us_record = json.loads((args.us_study / 'summary.json').read_text())
    for name, digest in us_record['code_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'US study source mismatch: {name}')
    for name, digest in us_record['input_sha256'].items():
        if hashlib.sha256((args.us_data_dir / 'raw' / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'US study input mismatch: {name}')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'forecasts.csv').exists():
        raise FileExistsError('Use a new output directory to preserve the recorded run.')
    print('Evaluating the frozen fractional models on international ETFs...', flush=True)
    forecasts, audits = rolling_refinement(external, start=protocol['report_start'])
    forecasts = forecasts.assign(panel='international')
    forecasts.to_csv(args.output / 'international_core_forecasts.csv', index=False)
    (args.output / 'international_fit_audit.json').write_text(json.dumps(audits, indent=2, allow_nan=False) + '\n')
    print(f'International: {forecasts.origin.nunique()} origins, {len(audits)} refits. Adding FIGARCH.', flush=True)
    forecasts, fig_audit = append_figarch(external, forecasts, audits)
    frames = [append_equal_combination(forecasts)]
    all_audits = {'international': fig_audit}
    us = pd.read_csv(args.us_study / 'forecasts.csv', parse_dates=['origin', 'target', 'last_fit'])
    us_audits = json.loads((args.us_study / 'fit_audit.json').read_text())
    index, sectors = load_prices(args.us_data_dir)
    for name, prices in [('index', index), ('sectors', sectors)]:
        print(f'Adding FIGARCH and the fixed ensemble to recorded {name} forecasts...', flush=True)
        augmented, audit = append_figarch(prices, us[us.panel == name], us_audits[name])
        frames.append(append_equal_combination(augmented))
        all_audits[name] = audit
    forecasts = pd.concat(frames, ignore_index=True)
    metrics, comparison = summarize(forecasts), paired_comparisons(forecasts)
    assets = forecasts.groupby(['panel', 'asset', 'horizon', 'method']).agg(
        qlike=('qlike', 'mean'), origins=('origin', 'nunique')).reset_index()
    dates = {name: dict(first_origin=str(b.origin.min().date()), last_origin=str(b.origin.max().date()),
                       last_target=str(b.target.max().date()), origins=int(b.origin.nunique()))
             for name, b in forecasts.groupby('panel')}
    record = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol=protocol, amendment=amendment, external_source=source, us_source=us_record,
                  arch_version=arch.__version__, dates=dates,
                  figarch_fits=sum(len(a['fits']) for panel in all_audits.values() for a in panel),
                  nonconverged_figarch_fits=sum(f['convergence_flag'] != 0 for panel in all_audits.values()
                                              for a in panel for f in a['fits'].values()),
                  nonconverged_external_gjr_fits=sum(not f['converged'] for a in audits for f in a['gjr']),
                  nonconverged_external_garch_fits=sum(not f['converged'] for a in audits for f in a['garch']))
    forecasts.to_csv(args.output / 'forecasts.csv', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    comparison.to_csv(args.output / 'comparisons.csv', index=False)
    assets.to_csv(args.output / 'asset_metrics.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    (args.output / 'figarch_fit_audit.json').write_text(json.dumps(all_audits, indent=2, allow_nan=False) + '\n')
    if args.publish_assets:
        publish(args.output, metrics)
    print(metrics[(metrics.year == 'all') & metrics.method.isin(LABELS)].round(5).to_string(index=False))
    print(f"FIGARCH convergence failures: {record['nonconverged_figarch_fits']}/{record['figarch_fits']}.")


if __name__ == '__main__':
    main()
