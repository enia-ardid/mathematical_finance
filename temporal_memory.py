"""Discrete ETF return simulations; not a solver for a fractional pricing PDE."""

from dataclasses import dataclass
from pathlib import Path

import nolds
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

SECTORS = ('XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLC')
# Illustrative initial capital allocations, not historical index weights.
PORTFOLIO_WEIGHTS = dict(zip(SECTORS, (.03, .04, .12, .08, .30, .05, .03, .03, .13, .10, .09)))


def validate_prices(prices):
    if not isinstance(prices, pd.DataFrame) or prices.empty or prices.shape[1] == 0:
        raise ValueError('Prices must be a nonempty DataFrame.')
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError('Prices need a valid DatetimeIndex.')
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Price dates must be unique and increasing.')
    if not prices.columns.is_unique:
        raise ValueError('Asset names must be unique.')
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Prices must be finite and positive; missing prices are not filled.')


def load_prices(directory):
    """Read raw CSVs, trim leading inception gaps, then align observed dates."""
    root = Path(directory) / 'raw'
    sp = pd.read_csv(root / 'sp500_close.csv', index_col='Date', parse_dates=['Date'])
    sectors = pd.read_csv(root / 'sector_etfs_close.csv', index_col='Date', parse_dates=['Date'])
    for frame in (sp, sectors):
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError('Raw price dates must be unique and increasing before inception trimming.')
    if list(sp.columns) != ['SP500'] or set(sectors.columns) != set(SECTORS):
        raise ValueError('Expected SP500 and all eleven sector tickers; see docs/data.md.')
    if sectors.isna().all().any():
        raise ValueError('At least one sector has no observations.')
    first_common = max(sectors[col].first_valid_index() for col in SECTORS)
    sectors = sectors.loc[first_common:, list(SECTORS)]
    validate_prices(sp)
    validate_prices(sectors)
    dates = sp.index.intersection(sectors.index).sort_values()
    if len(dates) < 3:
        raise ValueError('Insufficient overlapping price dates.')
    return sp.loc[dates], sectors.loc[dates]


def synthetic_prices(n_prices=500, seed=17):
    """Correlated Gaussian example for offline execution, not empirical evidence."""
    if n_prices < 200:
        raise ValueError('Use at least 200 prices for the Hurst estimator.')
    rng = np.random.RandomState(seed)
    common = rng.normal(0, .008, (n_prices - 1, 1))
    returns = .0003 + common + rng.normal(0, .006, (n_prices - 1, len(SECTORS)))
    levels = 100 * np.exp(np.vstack([np.zeros(len(SECTORS)), returns.cumsum(axis=0)]))
    dates = pd.bdate_range('2020-01-01', periods=n_prices)
    sectors = pd.DataFrame(levels, index=dates, columns=SECTORS)
    index_returns = returns.mean(axis=1)
    sp = pd.DataFrame({'SP500': 3000 * np.exp(np.r_[0., index_returns.cumsum()])}, index=dates)
    return sp, sectors


def grunwald_weights(alpha, length):
    """Coefficients of (1-z)**alpha, including w[0] = 1."""
    if not np.isfinite(alpha) or not 0 < alpha <= 1 or length < 1:
        raise ValueError('Require 0 < alpha <= 1 and a positive length.')
    w = np.ones(length)
    for k in range(1, length):
        w[k] = (1 - (alpha + 1) / k) * w[k - 1]
    return w


def laplacian(values):
    """Unit-spaced lattice with reflected ghost nodes (zero centered gradient)."""
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.zeros_like(values)
    padded = np.pad(values, 1, mode='reflect')
    return padded[:-2] - 2 * values + padded[2:]


def estimate_hurst(values, seed=1729):
    # nolds' RANSAC uses NumPy's global RNG. Isolate it from simulation seeds.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        h = float(nolds.hurst_rs(values, fit='RANSAC'))
    finally:
        np.random.set_state(state)
    if not np.isfinite(h):
        raise ValueError('Hurst estimation failed; check the training returns.')
    return h


def fit_parameters(train_prices, sector_model):
    """All data-dependent choices see training prices only."""
    validate_prices(train_prices)
    returns = np.log(train_prices).diff().iloc[1:]
    if len(returns) < 100 or (returns.std() <= 1e-12).any():
        raise ValueError('Need at least 100 nonconstant training returns per asset.')
    order = list(returns.columns)
    kappa_base = 0.
    if sector_model and len(order) > 1:
        distance = pdist(returns.to_numpy().T, metric='correlation')
        order = list(returns.columns[leaves_list(linkage(distance, method='average'))])
        corr = returns.corr().to_numpy()
        off_diagonal = corr[np.triu_indices(len(order), 1)]
        fisher_mean = np.tanh(np.arctanh(np.clip(off_diagonal, -.999999, .999999)).mean())
        kappa_base = float(.5 * max(0., fisher_mean))
    returns = returns[order]
    h = np.array([estimate_hurst(returns[col].to_numpy()) for col in order])
    shrink = .6 if sector_model else 1.
    return {
        'order': order,
        'hurst': h,
        'alpha': np.clip(.5 + shrink * (h - .5), .05, .95),
        'mu': returns.mean().to_numpy(),
        'sigma': returns.std(ddof=1).to_numpy(),
        'kappa_base': kappa_base,
    }


def simulate_returns(initial, mu_path, sigma_path, alpha, kappa, seed, centered):
    """Original GL-weighted recurrence, at a fixed step of one trading day.

    The plus-sign convolution starts at w[0] on the previous return. This is
    deliberately the project's discrete model, not an isolated GL derivative.
    """
    steps, assets = mu_path.shape
    weights = np.column_stack([grunwald_weights(a, steps) for a in alpha])
    rng = np.random.RandomState(seed)
    returns = np.zeros((steps, assets))
    returns[0] = initial
    for t in range(1, steps):
        past = returns[:t] - mu_path[:t] if centered else returns[:t]
        memory = np.sum(weights[:t] * past[::-1], axis=0)
        returns[t] = (mu_path[t] + sigma_path[t] * rng.normal(size=assets)
                      + kappa * laplacian(returns[t - 1]) + memory)
    if not np.isfinite(returns).all():
        raise FloatingPointError('Nonfinite simulated returns; parameters may be unstable.')
    return returns


def calibrated_path(train_prices, horizon, parameters, kappa, seed, sector_model):
    """Continue a training-calibrated path; no held-out prices are accepted."""
    ordered = train_prices[parameters['order']]
    observed = np.log(ordered).diff().iloc[1:]
    n_train = len(observed)
    steps = n_train + horizon
    mu_path = np.tile(parameters['mu'], (steps, 1))
    sigma_path = np.tile(parameters['sigma'], (steps, 1))
    if not sector_model:
        # Retrospective training scenario from the final notebook, never forecast.
        covid = np.flatnonzero((observed.index >= '2020-02-20') & (observed.index <= '2020-04-30'))
        mu_path[covid] *= .8
        sigma_path[covid] *= 4.
    simulated = simulate_returns(observed.iloc[0].to_numpy(), mu_path, sigma_path,
                                 parameters['alpha'], kappa, seed, sector_model)
    log_start = np.log(ordered.iloc[0].to_numpy())
    cumulative = simulated.cumsum(axis=0)
    adjustment = (np.log(ordered.iloc[-1].to_numpy()) - log_start - cumulative[n_train - 1]) / n_train
    log_levels = log_start + cumulative + np.arange(1, steps + 1)[:, None] * adjustment
    with np.errstate(over='raise', invalid='raise', under='raise'):
        levels = np.exp(log_levels)
    fit_levels = levels[:n_train]
    scale = (fit_levels * ordered.iloc[1:].to_numpy()).sum(axis=0) / (fit_levels**2).sum(axis=0)
    levels *= scale
    if not np.isfinite(levels).all() or (levels <= 0).any():
        raise FloatingPointError('Invalid price paths after training calibration.')
    return levels


@dataclass
class SimulationResult:
    actual: pd.DataFrame
    paths: np.ndarray
    split: int
    parameters: dict
    kappa_scores: dict
    kappa: float
    seeds: tuple
    calibration_seeds: tuple

    @property
    def mean(self):
        return pd.DataFrame(self.paths.mean(axis=0), index=self.actual.index, columns=self.actual.columns)


def run_model(prices, sector_model=False, test_fraction=.3, n_sim=10, n_calibration=5):
    validate_prices(prices)
    if not 0 < test_fraction < 1 or n_sim < 1 or n_calibration < 1:
        raise ValueError('Require 0 < test_fraction < 1 and positive ensemble sizes.')
    split = int((1 - test_fraction) * (len(prices) - 1))
    horizon = len(prices) - 1 - split
    if horizon < 2:
        raise ValueError('Need at least two held-out prices to evaluate R².')
    train = prices.iloc[:split + 1]
    parameters = fit_parameters(train, sector_model)
    start_seed = 3 if sector_model else 2
    seeds = tuple(start_seed + 37 * i for i in range(n_sim))
    grid = [0., .5 * parameters['kappa_base'], parameters['kappa_base']] if sector_model else []
    calibration_seeds = tuple(3 + 37 * i for i in range(n_calibration)) if sector_model else ()
    scores = {}
    for kappa in dict.fromkeys(grid):
        # Selection runs only over the training horizon.
        paths = [calibrated_path(train, 0, parameters, kappa, seed, True)
                 for seed in calibration_seeds]
        error = np.mean(paths, axis=0) - train[parameters['order']].iloc[1:].to_numpy()
        scores[kappa] = float(np.sqrt(np.mean(error**2)))
    best = min(scores, key=scores.get) if scores else 0.
    paths = np.stack([calibrated_path(train, horizon, parameters, best, seed, sector_model)
                      for seed in seeds])
    actual = prices[parameters['order']].iloc[1:]
    return SimulationResult(actual, paths, split, parameters, scores, best, seeds, calibration_seeds)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size < 2:
        raise ValueError('Metrics require matching arrays with at least two observations.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Cannot evaluate nonfinite values.')
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)
    sst = np.sum((actual - actual.mean())**2)
    return {'RMSE': float(np.sqrt(np.mean(error**2))), 'MAE': float(np.mean(np.abs(error))),
            'R2': float(1 - np.sum(error**2) / sst) if sst > 0 else float('nan'),
            'sMAPE_percent': float(200 * smape_terms.mean()), 'bias': float(error.mean())}


def evaluate(result):
    """Identical held-out dates for the ensemble and fixed-origin baselines."""
    rows = []
    lower, upper = np.quantile(result.paths, [.1, .9], axis=0)
    for j, asset in enumerate(result.actual.columns):
        actual = result.actual.iloc[result.split:, j].to_numpy()
        start = result.actual.iloc[result.split - 1, j]
        steps = np.arange(1, len(actual) + 1)
        predictions = {
            'memory ensemble': result.mean.iloc[result.split:, j].to_numpy(),
            'constant price': np.full(len(actual), start),
            'log-drift baseline': start * np.exp(result.parameters['mu'][j] * steps),
        }
        for name, prediction in predictions.items():
            row = {'asset': asset, 'model': name, **metrics(actual, prediction)}
            row['coverage_10_90'] = (float(np.mean((actual >= lower[result.split:, j])
                                                   & (actual <= upper[result.split:, j])))
                                        if name == 'memory ensemble' else np.nan)
            rows.append(row)
    return pd.DataFrame(rows).set_index(['asset', 'model'])


def buy_and_hold(prices, weights, start_level=100.):
    """Initial capital weights applied to rebased prices, invariant to share units."""
    validate_prices(prices)
    if set(prices.columns) != set(weights):
        raise ValueError('Weights must match the assets exactly.')
    w = pd.Series(weights, dtype=float).reindex(prices.columns)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0 or not np.isfinite(start_level) or start_level <= 0:
        raise ValueError('Require finite nonnegative weights and a positive initial level.')
    return (prices / prices.iloc[0]) @ (w / w.sum()) * start_level
