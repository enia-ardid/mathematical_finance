"""Observed-history forecasts using the unchanged temporal-memory recurrence."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from temporal_memory import fit_parameters, grunwald_weights, laplacian, metrics, validate_prices

HORIZONS = (1, 5, 21)
KAPPA_SCALES = (0., .5, 1.)


def conditional_moments(history, parameters, kappa, horizon, sector_model):
    """Mean and marginal variance of future cumulative log-returns, conditional on history.

    All returns in history are observed. Future returns follow the original
    linear recurrence with independent Gaussian innovations and fixed parameters.
    """
    history = np.asarray(history, dtype=float)
    alpha = np.asarray(parameters['alpha'])
    mu, sigma = np.asarray(parameters['mu']), np.asarray(parameters['sigma'])
    assets = len(alpha)
    if (history.ndim != 2 or history.shape[1] != assets or len(history) < 1
            or not np.isfinite(history).all() or horizon < 1):
        raise ValueError('Require a finite observed return matrix and a positive horizon.')
    steps = len(history)
    weights = np.column_stack([grunwald_weights(a, steps + horizon) for a in alpha])
    returns = np.vstack([history, np.zeros((horizon, assets))])
    center = mu if sector_model else np.zeros(assets)
    for t in range(steps, steps + horizon):
        memory = np.sum(weights[:t] * (returns[:t] - center)[::-1], axis=0)
        returns[t] = mu + kappa * laplacian(returns[t - 1]) + memory
    cumulative_mean = returns[steps:].cumsum(axis=0)

    # Response of each future return to a unit innovation at the first future step.
    identity = np.eye(assets)
    lattice = np.column_stack([laplacian(identity[:, j]) for j in range(assets)])
    impulse = np.zeros((horizon, assets, assets))
    impulse[0] = identity
    for t in range(1, horizon):
        for lag in range(1, t + 1):
            coefficient = np.diag(weights[lag - 1])
            if lag == 1:
                coefficient = coefficient + kappa * lattice
            impulse[t] += coefficient @ impulse[t - lag]
    cumulative_impulse = impulse.cumsum(axis=0)
    variance = np.sum((cumulative_impulse * sigma[None, None, :])**2, axis=2).cumsum(axis=0)
    if not np.isfinite(cumulative_mean).all() or not np.isfinite(variance).all():
        raise FloatingPointError('Nonfinite forecast moments.')
    return cumulative_mean, variance


def forecast_from_history(observed_prices, parameters, kappa, horizon, sector_model):
    """Forecast without accepting future prices or calibrating to a future endpoint."""
    validate_prices(observed_prices)
    ordered = observed_prices[parameters['order']]
    history = np.log(ordered).diff().iloc[1:].to_numpy()
    mean, variance = conditional_moments(history, parameters, kappa, horizon, sector_model)
    anchor = ordered.iloc[-1].to_numpy()
    z = norm.ppf(.9)
    with np.errstate(over='raise', invalid='raise', under='raise'):
        point = anchor * np.exp(mean + .5 * variance)
        lower = anchor * np.exp(mean - z * np.sqrt(variance))
        upper = anchor * np.exp(mean + z * np.sqrt(variance))
        drift = anchor * np.exp(np.arange(1, horizon + 1)[:, None] * parameters['mu'])
    return {'memory': point, 'drift': drift, 'lower': lower, 'upper': upper,
            'log_mean': mean, 'log_variance': variance}


@dataclass
class ForecastCalibration:
    choices: pd.DataFrame
    scores: pd.DataFrame
    validation_origins: tuple
    development_end: pd.Timestamp
    horizons: tuple


def select_calibration(development_prices, sector_model=False, horizons=HORIZONS,
                       validation_fraction=.4, origin_step=21):
    """Select coupling and forecast blend using rolling origins inside development only."""
    validate_prices(development_prices)
    horizons = tuple(sorted(set(horizons)))
    if (not horizons or any(not isinstance(h, int) or h < 1 for h in horizons)
            or not 0 < validation_fraction < 1 or origin_step < 1):
        raise ValueError('Require positive integer horizons/stride and a validation fraction in (0, 1).')
    last_origin = len(development_prices) - 1 - max(horizons)
    first_origin = max(100, int((1 - validation_fraction) * (len(development_prices) - 1)))
    origins = list(range(first_origin, last_origin + 1, origin_step))
    if len(origins) < 3:
        raise ValueError('Need at least three complete internal validation origins; provide more development data.')
    scales = KAPPA_SCALES if sector_model else (0.,)
    samples = {(h, scale): [] for h in horizons for scale in scales}
    for origin in origins:
        observed = development_prices.iloc[:origin + 1]
        params = fit_parameters(observed, sector_model)
        order = params['order']
        anchor = observed[order].iloc[-1].to_numpy()
        for scale in scales:
            forecast = forecast_from_history(observed, params, scale * params['kappa_base'], max(horizons), sector_model)
            for h in horizons:
                actual = development_prices[order].iloc[origin + h].to_numpy() / anchor
                memory = forecast['memory'][h - 1] / anchor
                drift = forecast['drift'][h - 1] / anchor
                samples[h, scale].append((actual, memory, drift))
    rows = []
    for (h, scale), observations in samples.items():
        actual, memory, drift = np.stack(observations).transpose(1, 0, 2).reshape(3, -1)
        difference = memory - drift
        denominator = difference @ difference
        weight = float(np.clip(difference @ (actual - drift) / denominator, 0., 1.)) if denominator > 0 else 0.
        prediction = drift + weight * difference
        rows.append({'horizon': h, 'kappa_scale': scale, 'memory_weight': weight,
                     'validation_mse': float(np.mean((prediction - actual)**2)),
                     'memory_mse': float(np.mean((memory - actual)**2)),
                     'drift_mse': float(np.mean((drift - actual)**2)),
                     'validation_targets': len(actual)})
    scores = pd.DataFrame(rows)
    # Stable candidate order breaks ties in favor of smaller coupling.
    choices = scores.loc[scores.groupby('horizon', sort=True)['validation_mse'].idxmin()].set_index('horizon')
    return ForecastCalibration(choices, scores, tuple(development_prices.index[origins]),
                               development_prices.index[-1], horizons)


def rolling_forecast(prices, sector_model=False, test_fraction=.3, test_start=None,
                     horizons=HORIZONS, origin_step=5, refit_every=21):
    """Rolling forecasts: past test observations become available at subsequent origins.

    Model hyperparameters remain frozen after development. Distribution parameters
    are refitted on expanding observed history at least refit_every sessions apart.
    """
    validate_prices(prices)
    if origin_step < 1 or refit_every < 1 or not 0 < test_fraction < 1:
        raise ValueError('Require positive strides and a test fraction in (0, 1).')
    if test_start is None:
        split = int((1 - test_fraction) * (len(prices) - 1))
    else:
        split = int(prices.index.searchsorted(pd.Timestamp(test_start), side='left')) - 1
    if split < 1 or len(prices) - 1 - split < max(horizons):
        raise ValueError('Test period must include the longest forecast horizon.')
    calibration = select_calibration(prices.iloc[:split + 1], sector_model, horizons)
    rows, fitted = [], []
    last_fit = -refit_every
    # All horizons share origins and have complete targets; no partial end blocks.
    for origin in range(split, len(prices) - max(horizons), origin_step):
        observed = prices.iloc[:origin + 1]
        if origin - last_fit >= refit_every:
            params = fit_parameters(observed, sector_model)
            last_fit = origin
            fitted.append({'origin': str(prices.index[origin].date()),
                           'parameters': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in params.items()}})
        order = params['order']
        anchor = observed[order].iloc[-1].to_numpy()
        cache = {}
        for h in calibration.horizons:
            choice = calibration.choices.loc[h]
            scale, weight = choice['kappa_scale'], choice['memory_weight']
            if scale not in cache:
                cache[scale] = forecast_from_history(observed, params, scale * params['kappa_base'],
                                                     max(horizons), sector_model)
            forecast = cache[scale]
            actual = prices[order].iloc[origin + h].to_numpy()
            for j, asset in enumerate(order):
                memory, drift = forecast['memory'][h - 1, j], forecast['drift'][h - 1, j]
                rows.append({'origin': prices.index[origin], 'target': prices.index[origin + h],
                             'last_fit': prices.index[last_fit], 'horizon': h, 'asset': asset,
                             'anchor': anchor[j], 'actual': actual[j], 'memory': memory,
                             'calibrated': drift + weight * (memory - drift), 'drift': drift,
                             'constant': anchor[j], 'memory_lower': forecast['lower'][h - 1, j],
                             'memory_upper': forecast['upper'][h - 1, j],
                             'memory_weight': weight, 'kappa': scale * params['kappa_base']})
    return pd.DataFrame(rows), calibration, fitted


def forecast_metrics(forecasts):
    """Compare all methods at identical origins, horizons and asset scales."""
    rows = []
    for (asset, horizon), block in forecasts.groupby(['asset', 'horizon'], sort=True):
        actual, anchor = block['actual'].to_numpy(), block['anchor'].to_numpy()
        for method in ['memory', 'calibrated', 'drift', 'constant']:
            prediction = block[method].to_numpy()
            error = (prediction - actual) / anchor
            naive_error = (anchor - actual) / anchor
            naive_mse = float(np.mean(naive_error**2))
            rows.append({'asset': asset, 'horizon': horizon, 'method': method,
                         'origins': len(block), **metrics(actual, prediction),
                         'relative_RMSE_percent': float(100 * np.sqrt(np.mean(error**2))),
                         'skill_vs_constant': 1 - float(np.mean(error**2)) / naive_mse if naive_mse > 0 else np.nan,
                         'memory_weight': float(block['memory_weight'].iloc[0]) if method == 'calibrated' else np.nan,
                         'raw_interval_coverage': float(np.mean((actual >= block['memory_lower']) &
                                                                (actual <= block['memory_upper']))) if method == 'memory' else np.nan})
    return pd.DataFrame(rows).set_index(['asset', 'horizon', 'method'])
