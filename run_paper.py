"""Reproduce the fractional-memory graph-variance study and manuscript assets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fractional_diffusion import fractional_weights, rolling_variance_forecasts
from temporal_memory import load_prices, synthetic_prices

LABELS = {'historical': 'Historical', 'ewma': 'EWMA', 'garch': 'GARCH',
          'fractional_only': 'Fractional only', 'exponential_diffusion': 'Exponential + graph',
          'fractional_diffusion': 'Fractional + graph'}


def summarize(forecasts):
    data = forecasts.assign(year=forecasts.origin.dt.year.astype(str))
    data = pd.concat([data, data.assign(year='all')])
    return data.groupby(['panel', 'year', 'horizon', 'method']).agg(
        qlike=('qlike', 'mean'), normalized_rmse=('normalized_square_error', lambda x: np.sqrt(x.mean())),
        origins=('origin', 'nunique'), observations=('qlike', 'size')).reset_index()


def paired_losses(forecasts, repetitions=2000, seed=1729):
    rng = np.random.default_rng(seed)
    output = []
    for (panel, horizon), block in forecasts.groupby(['panel', 'horizon']):
        errors = block.groupby(['origin', 'method']).qlike.mean().unstack('method')
        for reference in [c for c in errors if c != 'fractional_diffusion']:
            difference = (errors.fractional_diffusion - errors[reference]).to_numpy()
            n = len(difference)
            for length in (5, 10):
                starts = rng.integers(0, n, (repetitions, int(np.ceil(n / length))))
                index = ((starts[..., None] + np.arange(length)) % n).reshape(repetitions, -1)[:, :n]
                lower, upper = np.quantile(difference[index].mean(axis=1), [.025, .975])
                output.append({'panel': panel, 'horizon': horizon, 'reference': reference,
                               'block_length': length, 'origins': n, 'mean_difference': difference.mean(),
                               'lower_95': lower, 'upper_95': upper})
    return pd.DataFrame(output)


def figures(forecasts, summary, final, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for order in (.2, .4, .6, .8):
        axes[0].loglog(np.arange(1, 253), fractional_weights(order), label=fr'$d={order}$')
    axes[0].set(xlabel='Lag (trading observations)', ylabel='Normalized memory weight', title='Fractional-integral memory')
    axes[0].legend(frameon=False, fontsize=8)
    graph = -final['sectors']['graph'].copy()
    np.fill_diagonal(graph, 0)
    im = axes[1].imshow(graph, cmap='Blues')
    assets = list(forecasts[forecasts.panel == 'sectors'].asset.unique())
    axes[1].set(xticks=np.arange(len(assets)), xticklabels=assets, yticks=np.arange(len(assets)),
                yticklabels=assets, title='Final observed-history sector graph')
    axes[1].tick_params(axis='x', rotation=75, labelsize=7)
    axes[1].tick_params(axis='y', labelsize=7)
    fig.colorbar(im, ax=axes[1], shrink=.8, label='Rescaled edge weight')
    fig.tight_layout()
    fig.savefig(destination / 'kernel_graph.pdf', bbox_inches='tight')
    fig.savefig(destination / 'kernel_graph.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, panel in zip(axes, ['index', 'sectors']):
        table = summary[(summary.panel == panel) & (summary.year == 'all')].pivot(index='method', columns='horizon', values='qlike')
        table = table.loc[list(LABELS)]
        for i, h in enumerate([1, 5, 21]):
            ax.barh(np.arange(len(table)) + (i - 1) * .24, table[h] - table.loc['ewma', h],
                    height=.23, label=f'{h} sessions')
        ax.axvline(0, color='black', lw=.7)
        ax.set(yticks=np.arange(len(table)), yticklabels=[LABELS[m] for m in table.index],
               xlabel='Mean QLIKE difference from EWMA', title='S&P 500' if panel == 'index' else 'Sector panel')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / 'forecast_comparison.pdf', bbox_inches='tight')
    fig.savefig(destination / 'forecast_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    index = forecasts[(forecasts.panel == 'index') & (forecasts.horizon == 21)]
    actual = index[index.method == 'fractional_diffusion']
    ax.plot(actual.origin, np.sqrt(252 * actual.realized_proxy / 21) * 100,
            color='#a3abb1', label='Subsequent squared-return proxy', lw=1.2)
    for name, color in [('fractional_diffusion', '#b65d25'), ('ewma', '#467f7c'), ('garch', '#243b53')]:
        b = index[index.method == name]
        ax.plot(b.origin, 100 * np.sqrt(252 * b.variance / 21), label=LABELS[name], color=color, lw=1)
    ax.set(ylabel='Annualized volatility (%)', xlabel='Forecast origin', title='21-observation index risk forecasts')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(destination / 'variance_paths.pdf', bbox_inches='tight')
    fig.savefig(destination / 'variance_paths.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-dir', type=Path)
    group.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/paper-study'))
    parser.add_argument('--publish-assets', action='store_true', help='Refresh aggregate tables and figures under paper/.')
    args = parser.parse_args()
    if args.synthetic:
        index, sectors = synthetic_prices(1300)
        start = str(index.index[900].date())
        source, hashes = 'Synthetic iid Gaussian return example; 1300 prices; seed 17.', {}
    else:
        index, sectors = load_prices(args.data_dir)
        start = '2023-01-01'
        manifest = args.data_dir / 'source.json'
        source = json.loads(manifest.read_text()) if manifest.exists() else 'Unverified local CSV provenance.'
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.data_dir / 'raw').glob('*.csv')}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frames, audits, final = [], {}, {}
    for panel, prices in [('index', index), ('sectors', sectors)]:
        print(f'Fitting and evaluating {panel}...', flush=True)
        forecast, audit, fitted = rolling_variance_forecasts(prices, start=start)
        frames.append(forecast.assign(panel=panel))
        audits[panel], final[panel] = audit, fitted
    forecasts = pd.concat(frames, ignore_index=True)
    summary, comparison = summarize(forecasts), paired_losses(forecasts)
    forecasts.to_csv(output / 'forecasts.csv', index=False)
    summary.to_csv(output / 'metrics.csv', index=False)
    comparison.to_csv(output / 'comparisons.csv', index=False)
    (output / 'fit_audit.json').write_text(json.dumps(audits, indent=2, allow_nan=False) + '\n')
    nonconverged = sum(not g['converged'] for a in audits.values() for fit in a for g in fit['garch'])
    record = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'source': source, 'input_sha256': hashes,
              'code_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['fractional_diffusion.py', 'run_paper.py']},
              'protocol': json.loads(Path('paper/protocol.json').read_text()),
              'first_price': str(index.index[0].date()), 'last_price': str(index.index[-1].date()),
              'first_origin': str(forecasts.origin.min().date()), 'last_origin': str(forecasts.origin.max().date()),
              'last_target': str(forecasts.target.max().date()), 'origins': int(forecasts.origin.nunique()),
              'refits_per_panel': {k: len(v) for k, v in audits.items()}, 'nonconverged_garch_fits': nonconverged}
    (output / 'summary.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    figures(forecasts, summary, final, output / 'figures')
    if args.publish_assets:
        import shutil
        for name in ['metrics.csv', 'comparisons.csv', 'summary.json']:
            shutil.copyfile(output / name, Path('paper') / name)
        for p in (output / 'figures').iterdir():
            shutil.copyfile(p, Path('paper/figures') / p.name)
        table = summary[summary.year == 'all'].pivot(index='method', columns=['panel', 'horizon'], values='qlike')
        table = table.loc[list(LABELS), [('index', h) for h in [1, 5, 21]] + [('sectors', h) for h in [1, 5, 21]]]
        table.index = [LABELS[m] for m in table.index]
        table.to_latex('paper/tables/main_scores.tex', float_format='%.4f', multicolumn_format='c')
        annual = summary[(summary.method == 'fractional_diffusion') & (summary.year != 'all')].pivot(
            index='year', columns=['panel', 'horizon'], values='qlike')
        annual.to_latex('paper/tables/annual_scores.tex', float_format='%.4f', multicolumn_format='c')
    print(summary[summary.year == 'all'].round(5).to_string(index=False))
    print(f'GARCH fits without convergence: {nonconverged}. Results saved in {output}.')


if __name__ == '__main__':
    main()
