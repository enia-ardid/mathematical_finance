"""Evaluate the adaptive memory forecast head and record latest forecasts."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_forecasting import adaptive_forecast, build_features, evaluate_adaptive, paired_comparisons
from temporal_memory import load_prices, synthetic_prices


def diagnostic_figure(metrics, intervals, label):
    point = metrics[(metrics.asset == 'SP500') & (metrics.year == 'all')]
    distribution = intervals[(intervals.asset == 'SP500') & (intervals.year == 'all')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(3)
    methods = ['adaptive', 'without_memory', 'raw_memory', 'drift', 'constant']
    colors = ['#bf651d', '#377c81', '#a280b0', '#668c56', '#8c969f']
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = point[point.method == method].sort_values('horizon')
        axes[0].bar(x + (i - 2) * .16, values.relative_RMSE_percent, .15,
                    label=method.replace('_', ' ').capitalize(), color=color)
    for i, method in enumerate(['adaptive', 'fixed_gaussian', 'raw_memory']):
        values = distribution[distribution.method == method].sort_values('horizon')
        axes[1].bar(x + (i - 1) * .22, values.interval_score_percent, .21,
                    label=method.replace('_', ' ').capitalize(), color=colors[[0, 3, 2][i]])
    axes[0].set(title='Point forecasts · lower error is better', ylabel='Origin-normalized RMSE (%)')
    axes[1].set(title='80% intervals · lower score is better', ylabel='Origin-normalized interval score (%)')
    for ax in axes:
        ax.set(xticks=x, xticklabels=['1 session', '5 sessions', '21 sessions'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Temporal memory · adaptive forecast evaluation', x=.07, ha='left', fontsize=16)
    fig.text(.07, .9, label, color='#52616b', fontsize=10)
    fig.text(.07, .025, 'Retrospective results · all methods share origins · interval coverage is measured, not guaranteed',
             fontsize=9, color='#52616b')
    fig.tight_layout(rect=[.02, .06, .99, .87])
    return fig


def export_adaptive_run(index_prices, sector_prices, output, source, report_start, input_hashes=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_forecasts, all_metrics, all_intervals, all_scored, audits = [], [], [], [], {}
    for panel, prices, sector in [('index', index_prices, False), ('sectors', sector_prices, True)]:
        print(f'{panel}: generating causal features and fitting forecast heads...', flush=True)
        features = build_features(prices, sector_model=sector)
        forecasts, audit = adaptive_forecast(features)
        metric, interval, scored = evaluate_adaptive(forecasts, report_start)
        all_forecasts.append(forecasts.assign(panel=panel))
        all_metrics.append(metric.assign(panel=panel))
        all_intervals.append(interval.assign(panel=panel))
        all_scored.append(scored.assign(panel=panel))
        audits[panel] = audit
    forecasts = pd.concat(all_forecasts, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    scored = pd.concat(all_scored, ignore_index=True)
    comparisons = paired_comparisons(scored)
    choices = scored.groupby(['panel', 'horizon', 'year', 'origin', 'selected']).size().reset_index()
    choices = choices.groupby(['panel', 'horizon', 'year', 'selected']).size().rename('origins').reset_index()
    forecasts.to_csv(output / 'forecasts.csv', index=False)
    metrics.to_csv(output / 'metrics.csv', index=False)
    intervals.to_csv(output / 'intervals.csv', index=False)
    comparisons.to_csv(output / 'paired_comparisons.csv', index=False)
    choices.to_csv(output / 'choices.csv', index=False)
    (output / 'fit_audit.json').write_text(json.dumps(audits, indent=2, allow_nan=False) + '\n')
    latest = forecasts[forecasts.origin == forecasts.origin.max()].copy()
    latest['point_price'] = latest.anchor * (1 + latest.adaptive)
    latest['lower_price'] = latest.anchor * (1 + latest.lower)
    latest['upper_price'] = latest.anchor * (1 + latest.upper)
    latest_columns = ['origin', 'asset', 'horizon', 'point_price', 'lower_price', 'upper_price',
                      'selected', 'validation_origins', 'interval_origins']
    latest[latest_columns].to_csv(output / 'latest_forecasts.csv', index=False)
    issued = datetime.now(timezone.utc)
    root = Path(__file__).parent
    summary = {
        'issued_at_utc': issued.isoformat(), 'source': source, 'input_sha256': input_hashes or {},
        'python': platform.python_version(),
        'dependencies': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'nolds', 'scikit-learn']},
        'code_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ['temporal_memory.py', 'forecasting.py', 'adaptive_forecasting.py', 'run_adaptive.py']},
        'protocol': json.loads((root / 'docs/results/adaptive_protocol.json').read_text()),
        'first_price': str(index_prices.index[0].date()), 'last_price': str(index_prices.index[-1].date()),
        'first_scored_origin': str(scored.origin.min().date()), 'last_scored_origin': str(scored.origin.max().date()),
        'last_scored_target': str(scored.target.max().date()), 'scored_origins': int(scored.origin.nunique()),
        'latest_origin': str(latest.origin.max().date()),
        'interpretation': 'Exploratory retrospective evaluation; previously inspected history; no independent validation or trading-profit claim.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    records = latest[latest_columns].copy()
    records['origin'] = records.origin.dt.strftime('%Y-%m-%d')
    archive = output / 'issued_forecasts'
    archive.mkdir(exist_ok=True)
    # Exclusive creation preserves each issuance even if summary files are rerun.
    with (archive / f'{issued:%Y%m%dT%H%M%S%fZ}.json').open('x') as stream:
        json.dump({'summary': summary, 'forecasts': records.to_dict(orient='records')}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    label = f"{summary['first_scored_origin']} to {summary['last_scored_target']} · "
    label += 'synthetic demonstration' if isinstance(source, str) and source.startswith('Synthetic') else 'recorded market snapshot'
    figure = diagnostic_figure(metrics, intervals, label)
    figure.savefig(output / 'adaptive_overview.png', dpi=150, facecolor='white')
    return metrics, intervals, latest[latest_columns], figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', action='store_true')
    source.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/adaptive'))
    parser.add_argument('--report-start', help='First scored origin, YYYY-MM-DD; fitting uses earlier history.')
    args = parser.parse_args()
    hashes = {}
    if args.synthetic:
        sp, sectors = synthetic_prices(900)
        provenance = 'Synthetic Gaussian prices; 900 observations; seed 17; not market data.'
        start = args.report_start or str(sp.index[650].date())
    else:
        sp, sectors = load_prices(args.data_dir)
        manifest = args.data_dir / 'source.json'
        provenance = json.loads(manifest.read_text()) if manifest.exists() else 'Local CSVs; provenance unverified.'
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.data_dir / 'raw').glob('*.csv')}
        start = args.report_start or '2023-01-01'
    metrics, _, latest, figure = export_adaptive_run(sp, sectors, args.output, provenance, start, hashes)
    plt.close(figure)
    print(metrics[(metrics.asset == 'SP500') & (metrics.year == 'all')].round(5).to_string(index=False))
    print('\nLatest forecasts (unscored):')
    print(latest[latest.asset == 'SP500'].round(4).to_string(index=False))
    print(f'\nSaved results, fit audit and append-only forecast issuance to {args.output}.')


if __name__ == '__main__':
    main()
