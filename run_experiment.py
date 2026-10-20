"""Run the notebook's experiment offline or on local market-price CSVs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from temporal_memory import (
    PORTFOLIO_WEIGHTS, buy_and_hold, evaluate, load_prices, metrics,
    run_model, synthetic_prices,
)
from forecasting import forecast_metrics, rolling_forecast


def forecast_overview(forecasts, label):
    index = forecasts[forecasts.asset == 'SP500']
    monthly = index[index.horizon == 21]
    table = forecast_metrics(index).xs('SP500', level='asset')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), gridspec_kw={'width_ratios': [1.3, 1]})
    colors = {'calibrated': '#bf651d', 'memory': '#9b72aa', 'drift': '#428273', 'constant': '#8a939b'}
    axes[0].plot(monthly.target, monthly.actual, label='Observed index', color='#243b53', lw=2)
    axes[0].plot(monthly.target, monthly.calibrated, label='Calibrated forecast', color=colors['calibrated'], lw=1.6)
    axes[0].plot(monthly.target, monthly.constant, label='Same-origin constant price', color=colors['constant'], ls='--')
    axes[0].set(title='21-session forecasts · origins every 5 sessions', ylabel='Index level')
    axes[0].tick_params(axis='x', labelrotation=25)
    axes[0].legend(fontsize=8, frameon=False)
    x = np.arange(3)
    for i, method in enumerate(['memory', 'calibrated', 'drift', 'constant']):
        errors = table.xs(method, level='method').relative_RMSE_percent
        axes[1].bar(x + (i - 1.5) * .19, errors, width=.18, label=method.capitalize(), color=colors[method])
    axes[1].set(xticks=x, xticklabels=['1 session', '5 sessions', '21 sessions'],
                title='Forecast error at identical origins', ylabel='Origin-normalized RMSE (%)')
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Temporal memory · conditional forecasts', x=.07, ha='left', fontsize=16)
    fig.text(.07, .91, label, fontsize=10, color='#52616b')
    weight = monthly.memory_weight.iloc[0]
    fig.text(.07, .025, f'21-session memory blend weight: {weight:.3f} · 0 selects the drift baseline · overlapping targets',
             fontsize=9, color='#52616b')
    fig.tight_layout(rect=[.02, .06, .99, .88])
    return fig


def export_forecast_run(index_prices, sector_prices, output, source, label, input_hashes=None, test_start=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    forecasts, validations, selections, fits = [], [], [], {}
    summary = {'source': source, 'input_sha256': input_hashes or {},
               'core_sha256': hashlib.sha256(Path(__file__).with_name('temporal_memory.py').read_bytes()).hexdigest(),
               'python': platform.python_version(),
               'dependencies': {name: importlib.metadata.version(name)
                                for name in ['numpy', 'pandas', 'scipy', 'nolds', 'scikit-learn']},
               'protocol': {'horizons': [1, 5, 21], 'origin_step': 5, 'refit_min_spacing': 21,
                            'validation_fraction': .4, 'validation_origin_step': 21,
                            'hurst_seed': 1729, 'inference': 'exact conditional Gaussian moments'},
               'runs': {}}
    for name, prices, sector in [('index', index_prices, False), ('sectors', sector_prices, True)]:
        forecast, calibration, fit_log = rolling_forecast(prices, sector, test_start=test_start)
        forecasts.append(forecast)
        validations.append(calibration.scores.assign(model=name))
        selections.append(calibration.choices.reset_index().assign(model=name))
        fits[name] = fit_log
        summary['runs'][name] = {
            'first_price': str(prices.index[0].date()), 'last_price': str(prices.index[-1].date()),
            'development_end': str(calibration.development_end.date()),
            'first_origin': str(forecast.origin.min().date()), 'last_origin': str(forecast.origin.max().date()),
            'first_target': str(forecast.target.min().date()), 'last_target': str(forecast.target.max().date()),
            'validation_origins': [str(value.date()) for value in calibration.validation_origins],
            'test_origins': int(forecast.origin.nunique()), 'refits': len(fit_log),
            'choices': calibration.choices.reset_index().to_dict(orient='records'),
        }
    forecasts = pd.concat(forecasts, ignore_index=True)
    table = forecast_metrics(forecasts)
    forecasts.to_csv(output / 'forecasts.csv', index=False)
    table.to_csv(output / 'forecast_metrics.csv')
    pd.concat(validations).to_csv(output / 'validation_scores.csv', index=False)
    pd.concat(selections).to_csv(output / 'calibration_choices.csv', index=False)
    (output / 'forecast_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output / 'forecast_fits.json').write_text(json.dumps(fits, indent=2, allow_nan=False) + '\n')
    fig = forecast_overview(forecasts, label)
    fig.savefig(output / 'forecast_overview.png', dpi=150, facecolor='white')
    return table, forecasts, fig


def overview(index_result, sector_result, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), gridspec_kw={'width_ratios': [1.35, 1]})
    split = index_result.split
    dates = index_result.actual.index[split:]
    actual = index_result.actual.iloc[split:, 0]
    anchor = index_result.actual.iloc[split - 1, 0]
    lower, upper = np.quantile(index_result.paths[:, split:, 0], [.1, .9], axis=0)
    axes[0].fill_between(dates, lower, upper, color='#d17b35', alpha=.17, label='10–90% simulation range')
    axes[0].plot(dates, actual, color='#243b53', label='Observed input', lw=1.7)
    axes[0].plot(dates, index_result.mean.iloc[split:, 0], color='#bf651d', label='Memory ensemble', lw=1.7)
    drift = anchor * np.exp(index_result.parameters['mu'][0] * np.arange(1, len(dates) + 1))
    axes[0].plot(dates, drift, color='#428273', ls='--', label='Log-drift baseline')
    axes[0].set(title='Index paths on held-out dates', ylabel='Index level')
    axes[0].legend(fontsize=8, loc='best', frameon=False)
    axes[0].tick_params(axis='x', labelrotation=25)
    table = evaluate(sector_result)['RMSE'].unstack('model')
    ratio = (table['memory ensemble'] / table['log-drift baseline']).sort_values()
    axes[1].barh(ratio.index, ratio, color=['#428273' if r < 1 else '#bf651d' for r in ratio])
    axes[1].axvline(1, color='#243b53', ls='--', lw=1)
    axes[1].set(title='Sector error relative to baseline', xlabel='RMSE ratio · below 1 is better')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('ETF simulation with temporal memory', fontsize=16, x=.07, ha='left')
    fig.text(.07, .91, label, fontsize=10, color='#52616b')
    fig.text(.07, .025, 'Single chronological split · fixed seeds · simulation ranges are not calibrated confidence intervals',
             fontsize=9, color='#52616b')
    fig.tight_layout(rect=[.02, .06, .99, .88])
    return fig


def portfolio_comparison(index_result, sector_result):
    """Compare baskets on held-out dates, with a common last-training anchor."""
    if not index_result.actual.index.equals(sector_result.actual.index) or index_result.split != sector_result.split:
        raise ValueError('Index and sectors must use identical dates and split.')
    split = sector_result.split
    level = float(index_result.actual.iloc[split - 1, 0])
    observed = buy_and_hold(sector_result.actual.iloc[split - 1:], PORTFOLIO_WEIGHTS, level).iloc[1:]
    baskets = []
    for path in sector_result.paths:
        frame = pd.DataFrame(path[split - 1:], index=sector_result.actual.index[split - 1:],
                             columns=sector_result.actual.columns)
        baskets.append(buy_and_hold(frame, PORTFOLIO_WEIGHTS, level).iloc[1:].to_numpy())
    simulated = np.mean(baskets, axis=0)
    index = index_result.actual.iloc[split:, 0].to_numpy()
    return pd.DataFrame([
        {'comparison': 'simulated basket vs observed basket', **metrics(observed, simulated)},
        {'comparison': 'observed basket vs index', **metrics(index, observed)},
        {'comparison': 'simulated basket vs index', **metrics(index, simulated)},
    ]).set_index('comparison')


def export_run(index_result, sector_result, output, source, label, input_hashes=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.concat([evaluate(index_result), evaluate(sector_result)])
    table.to_csv(output / 'metrics.csv')
    portfolio_comparison(index_result, sector_result).to_csv(output / 'portfolio_metrics.csv')
    summary = {
        'source': source, 'input_sha256': input_hashes or {},
        'python': platform.python_version(),
        'dependencies': {name: importlib.metadata.version(name)
                         for name in ['numpy', 'pandas', 'scipy', 'nolds', 'scikit-learn', 'matplotlib']},
        'method': 'GL-weighted discrete recurrence; not a fractional PDE solver',
        'runs': {},
    }
    for name, result in [('index', index_result), ('sectors', sector_result)]:
        params = result.parameters
        summary['runs'][name] = {
            'first_return': str(result.actual.index[0].date()),
            'last_training_return': str(result.actual.index[result.split - 1].date()),
            'first_test_return': str(result.actual.index[result.split].date()),
            'last_return': str(result.actual.index[-1].date()),
            'training_returns': result.split, 'test_returns': len(result.actual) - result.split,
            'seeds': list(result.seeds), 'hurst_seed': 1729,
            'calibration_seeds': list(result.calibration_seeds),
            'kappa': result.kappa, 'kappa_training_rmse': result.kappa_scores,
            'parameters': {key: value.tolist() if isinstance(value, np.ndarray) else value
                           for key, value in params.items()},
        }
    (output / 'run_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    fig = overview(index_result, sector_result, label)
    fig.savefig(output / 'overview.png', dpi=150, facecolor='white')
    return table, fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', action='store_true')
    source.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/run'))
    parser.add_argument('--max-prices', type=int, default=1644)
    parser.add_argument('--simulations', type=int, default=10)
    parser.add_argument('--mode', choices=['simulation', 'forecast'], default='simulation')
    parser.add_argument('--test-start', help='First permissible target date for forecast mode (YYYY-MM-DD).')
    args = parser.parse_args()
    if args.max_prices < 150:
        parser.error('--max-prices must be at least 150.')
    if args.test_start and args.mode != 'forecast':
        parser.error('--test-start is only used by forecast mode.')
    hashes = {}
    if args.synthetic:
        sp, sectors = synthetic_prices()
        description = 'Synthetic Gaussian prices; 500 observations; seed 17; not market data.'
        label = 'Synthetic demonstration · 11 sectors · no market-performance claim'
    else:
        sp, sectors = load_prices(args.data_dir)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (args.data_dir / 'raw').glob('*.csv')}
        provenance = args.data_dir / 'source.json'
        description = json.loads(provenance.read_text()) if provenance.exists() else 'Local CSVs; provider and adjustment convention unverified.'
        label = (description.get('provider', 'Local CSV experiment') + ' · recorded data snapshot'
                 if isinstance(description, dict) else 'Local CSV experiment · provenance unverified')
    sp, sectors = sp.iloc[-args.max_prices:], sectors.iloc[-args.max_prices:]
    if args.mode == 'forecast':
        table, _, fig = export_forecast_run(sp, sectors, args.output, description, label, hashes, args.test_start)
    else:
        index_result = run_model(sp, n_sim=args.simulations)
        sector_result = run_model(sectors, sector_model=True, n_sim=args.simulations)
        table, fig = export_run(index_result, sector_result, args.output, description, label, hashes)
    plt.close(fig)
    print(table.round(4).to_string())
    print(f'\nSaved {args.mode} results and provenance to {args.output}.')


if __name__ == '__main__':
    main()
