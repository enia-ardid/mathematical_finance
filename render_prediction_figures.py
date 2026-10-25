"""Render manuscript comparisons and observed-versus-forecast paths from saved results.

Presentation only: no fitting, parameter selection, new forecasts or new scores.
Run after the numerical study exporters and before compiling the manuscript.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
import numpy as np
import pandas as pd

from run_paper import LABELS as INITIAL_LABELS

METHODS = {
    'persistent_asymmetric': ('Fractional model', '#0072B2', '-', 1.15),
    'equal_combination': ('Fixed 50:50 ensemble', '#E69F00', '-', 1.4),
    'gjr_garch': ('GJR-GARCH', '#009E73', '--', 1.05),
}
SECTORS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLC']
COUNTRIES = ['EWA', 'EWC', 'EWG', 'EWH', 'EWJ', 'EWL', 'EWQ', 'EWU']
NAMES = {
    'XLB': 'Materials', 'XLE': 'Energy', 'XLF': 'Financials', 'XLI': 'Industrials',
    'XLK': 'Technology', 'XLP': 'Consumer staples', 'XLRE': 'Real estate',
    'XLU': 'Utilities', 'XLV': 'Health care', 'XLY': 'Consumer discretionary',
    'XLC': 'Communication services', 'EWA': 'Australia', 'EWC': 'Canada',
    'EWG': 'Germany', 'EWH': 'Hong Kong', 'EWJ': 'Japan', 'EWL': 'Switzerland',
    'EWQ': 'France', 'EWU': 'United Kingdom',
}


def save(fig, destination, name):
    for suffix in ['pdf', 'png']:
        metadata = {'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {}
        fig.savefig(destination / f'{name}.{suffix}', bbox_inches='tight', dpi=180, metadata=metadata)
    plt.close(fig)


def initial_comparison(metrics, destination):
    """Plot raw normalized QLIKE, including EWMA's actual, nonzero score."""
    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.1))
    names = list(INITIAL_LABELS)
    scores = metrics[metrics.year == 'all']
    low, high = scores.qlike.min(), scores.qlike.max()
    span = high - low
    for row, horizon in enumerate([1, 5, 21]):
        for col, panel in enumerate(['index', 'sectors']):
            ax = axes[row, col]
            subset = scores[(scores.panel == panel) & (scores.horizon == horizon)].set_index('method')
            values = subset.loc[names, 'qlike'].to_numpy()
            for j, (method, value) in enumerate(zip(names, values)):
                color = '#E69F00' if method == 'ewma' else '#0072B2'
                ax.axhline(j, color='#e9edf0', lw=.65, zorder=0)
                ax.plot(value, j, 'o', color=color, markersize=5)
                ax.annotate(f'{value:.4f}', (value, j), xytext=(6, 0),
                            textcoords='offset points', va='center', fontsize=8)
            ax.set(ylim=(len(names) - .4, -.6), xlim=(low - .10 * span, high + .40 * span),
                   yticks=range(len(names)),
                   yticklabels=[INITIAL_LABELS[n] for n in names] if col == 0 else [])
            name = 'S&P 500' if panel == 'index' else 'Sector panel'
            ax.set_title(f'{name} | {horizon} session' + ('s' if horizon != 1 else ''), fontsize=10)
            ax.tick_params(axis='y', length=0, labelsize=8.5)
            ax.tick_params(axis='x', labelsize=8)
            if row == 2:
                ax.set_xlabel('Mean normalized QLIKE', fontsize=9)
    fig.suptitle('Initial specification: lower scores are better\nEWMA is shown in orange with its actual score', fontsize=11, y=.995)
    fig.subplots_adjust(left=.23, right=.98, bottom=.07, top=.91, hspace=.42, wspace=.14)
    save(fig, destination, 'forecast_comparison')


def path_data(forecasts, panel, horizon, asset=None):
    """Return aligned observed and predicted variance, grouped before square root."""
    block = forecasts[(forecasts.panel == panel) & (forecasts.horizon == horizon)
                      & forecasts.method.isin(METHODS)]
    if asset is not None:
        block = block[block.asset == asset]
    if block.empty or block.duplicated(['origin', 'asset', 'method']).any():
        raise ValueError('Empty or duplicated plot inputs.')
    keys = ['origin', 'target', 'asset']
    components = {name: block[block.method == name].set_index(keys).sort_index()
                  for name in METHODS}
    reference = components['persistent_asymmetric']
    for other in components.values():
        pd.testing.assert_index_equal(reference.index, other.index)
        np.testing.assert_array_equal(reference.realized_proxy, other.realized_proxy)
    if not all(block.target > block.origin):
        raise ValueError('Targets must follow their forecast origins.')
    # This is the mean of marginal variances, not an equal-weight portfolio variance.
    result = reference[['realized_proxy']].groupby(level=['origin', 'target']).mean()
    for method, values in components.items():
        result[method] = values.variance.groupby(level=['origin', 'target']).mean()
    if result.isna().any().any() or (result < 0).any().any():
        raise ValueError('Invalid aligned variance series.')
    return 100 * np.sqrt(252 * result / horizon)


def path_axis(ax, series, title, xlabel=False, compact=False, log=False, zoom=False):
    dates = series.index.get_level_values('target')
    ax.plot(dates, series.realized_proxy, color='#000000', lw=1.25,
            alpha=.78, label='Observed return-based proxy', zorder=2)
    for method, (label, color, style, width) in METHODS.items():
        ax.plot(dates, series[method], color=color, ls=style, lw=width,
                label=label, zorder=3)
    ax.set_title(title, loc='left', fontsize=9 if compact else 10)
    if log:
        if (series <= 0).any().any():
            raise ValueError('Log-scale plots require strictly positive observations; no clipping.')
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis='y', color='#e7ebee', lw=.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp(year=dates.min().year, month=1, day=1),
                dates.max() + pd.Timedelta(days=10))
    if zoom:
        ax.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2024-12-31'))
    ax.tick_params(labelsize=8 if compact else 9)
    if xlabel:
        ax.set_xlabel('End of forecast window', fontsize=9)


def legend(fig, y, iqr=False):
    handles = [Line2D([0], [0], color='#000000', lw=1.25, label='Observed return-based proxy')]
    handles += [Line2D([0], [0], color=c, lw=w, ls=s, label=label)
                for label, c, s, w in METHODS.values()]
    if iqr:
        handles.append(Patch(facecolor='#000000', alpha=.10,
                             label='Shading: cross-asset IQR (each series)'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, y),
               ncol=2, frameon=False, fontsize=9, columnspacing=1.5)


def index_paths(forecasts, destination):
    fig, axes = plt.subplots(3, 2, figsize=(9, 7.6), sharex='col')
    for row, horizon in enumerate([1, 5, 21]):
        series = path_data(forecasts, 'index', horizon)
        for col in (0, 1):
            if col:
                subset = series.loc[series.index.get_level_values('target') < '2025-01-01']
            else:
                subset = series
            title = f'{horizon} sessions | ' + ('2023–2024 detail' if col else 'full period')
            path_axis(axes[row, col], subset, title, xlabel=row == 2, compact=True, zoom=bool(col))
        axes[row, 0].set_ylabel('Annualized volatility (%)', fontsize=9)
    legend(fig, .995)
    fig.subplots_adjust(left=.09, right=.985, bottom=.075, top=.895, hspace=.35, wspace=.20)
    save(fig, destination, 'prediction_index_horizons')


def panel_paths(forecasts, destination):
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.4), sharex=True)
    for row, (panel, label) in enumerate([('sectors', 'Sector ETFs'), ('international', 'Country ETFs')]):
        for col, horizon in enumerate([5, 21]):
            path_axis(axes[row, col], path_data(forecasts, panel, horizon),
                      f'{label} | {horizon} sessions', xlabel=row == 1, compact=True)
            block = forecasts[(forecasts.panel == panel) & (forecasts.horizon == horizon)]
            for method, color in [('observed', '#000000')] + [(m, v[1]) for m, v in METHODS.items()]:
                part = block[block.method == ('persistent_asymmetric' if method == 'observed' else method)]
                values = 100*np.sqrt(252*part['realized_proxy' if method == 'observed' else 'variance']/horizon)
                quantiles = part.assign(display=values).groupby('target').display.quantile([.25, .75]).unstack()
                axes[row, col].fill_between(quantiles.index, quantiles[.25], quantiles[.75],
                                           color=color, alpha=.10, linewidth=0, zorder=1)
            if col == 0:
                axes[row, col].set_ylabel('Annualized RMS volatility (%)', fontsize=9)
    legend(fig, .995, iqr=True)
    fig.subplots_adjust(left=.11, right=.985, bottom=.10, top=.80, hspace=.36, wspace=.22)
    save(fig, destination, 'prediction_panel_evolution')


def asset_paths(forecasts, panel, assets, name, destination):
    rows = int(np.ceil(len(assets) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(7.25, rows * 1.8 + 1.0), sharex=True, squeeze=False)
    for i, asset in enumerate(assets):
        ax = axes.flat[i]
        path_axis(ax, path_data(forecasts, panel, 21, asset),
                  f'{asset} | {NAMES[asset]}', xlabel=i // 2 == rows - 1, compact=True, log=True)
        if i % 2 == 0:
            ax.set_ylabel('Annualized vol. (%, log scale)', fontsize=8)
    for ax in list(axes.flat)[len(assets):]:
        ax.set_visible(False)
    legend(fig, .995)
    fig.subplots_adjust(left=.10, right=.985, bottom=.08, top=1 - .78 / fig.get_figheight(),
                        hspace=.38, wspace=.22)
    save(fig, destination, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('artifacts/external-study-complete'))
    parser.add_argument('--paper-dir', type=Path, default=Path('paper'))
    args = parser.parse_args()
    input_path = args.study / 'forecasts.csv'
    forecasts = pd.read_csv(input_path, parse_dates=['origin', 'target', 'last_fit'],
                            float_precision='round_trip')
    initial_path = args.paper_dir / 'metrics.csv'
    initial = pd.read_csv(initial_path)
    final_metrics = pd.read_csv(args.paper_dir / 'external_metrics.csv')
    actual = forecasts.groupby(['panel', 'horizon', 'method']).qlike.mean().sort_index()
    expected = final_metrics[final_metrics.year == 'all'].set_index(['panel', 'horizon', 'method']).qlike.sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    if set(forecasts[forecasts.panel == 'sectors'].asset) != set(SECTORS):
        raise ValueError('The recorded eleven-sector universe is required.')
    if set(forecasts[forecasts.panel == 'international'].asset) != set(COUNTRIES):
        raise ValueError('The recorded eight-country universe is required.')
    destination = args.paper_dir / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    initial_comparison(initial, destination)
    index_paths(forecasts, destination)
    panel_paths(forecasts, destination)
    asset_paths(forecasts, 'sectors', SECTORS[:6], 'prediction_sectors_a', destination)
    asset_paths(forecasts, 'sectors', SECTORS[6:], 'prediction_sectors_b', destination)
    asset_paths(forecasts, 'international', COUNTRIES, 'prediction_countries', destination)
    names = ['forecast_comparison', 'prediction_index_horizons', 'prediction_panel_evolution',
             'prediction_sectors_a', 'prediction_sectors_b', 'prediction_countries']
    record = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Presentation only; frozen forecasts, fitted models and scores unchanged.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [input_path, initial_path, args.paper_dir / 'external_metrics.csv']},
                  figure2='Raw mean normalized QLIKE, identical values to the initial-study table. Lower is better; EWMA shown explicitly.',
                  time_axis='Target-window end. Each prediction was issued horizon trading observations earlier.',
                  transformation='100*sqrt(252*sum_of_daily_variances/horizon); same transformation of the squared-return proxy. Panel variances averaged before square root; no covariance or portfolio interpretation.',
                  scope_of_paths=dict(index_horizons=[1, 5, 21], panel_horizons=[5, 21],
                                      asset_horizon=21, sectors=SECTORS, countries=COUNTRIES,
                                      dates='Full recorded evaluation period, plus 2023-2024 index detail panels.'),
                  log_axes='Asset paths use log y-axes with strictly positive inputs, no floor or clipping.',
                  iqr='Panel bands are cross-asset 25th/75th percentiles, not confidence intervals; central curves are RMS volatility.',
                  palette='Okabe-Ito',
                  outputs_sha256={str(p.relative_to(args.paper_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for n in names for p in [destination / f'{n}.pdf', destination / f'{n}.png']})
    (args.paper_dir / 'prediction_figures_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Rendered six figures: direct QLIKE comparison and observed-versus-forecast paths.')


if __name__ == '__main__':
    main()
