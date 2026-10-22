"""Chronological Network GARCH benchmarks; the fractional models stay frozen.

The recursion follows Zhou et al. (2020), equation (2.2), on standardized
returns. Weighted correlation edges and unit initialization are explicit
adaptations; a binary graph and a pooled no-network control are also fitted.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import softmax

from fractional_diffusion import _base, HORIZONS
from temporal_memory import validate_prices

METHODS = ('network_garch', 'network_garch_binary', 'pooled_garch')
STARTS = ((.05, .05, .85), (.1, .05, .7), (.02, .03, .93))


def network_matrix(correlation, binary=False):
    correlation = np.asarray(correlation, dtype=float)
    if (correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]
            or not np.isfinite(correlation).all() or not np.allclose(correlation, correlation.T)):
        raise ValueError('Need a finite symmetric correlation matrix.')
    weights = ((correlation >= .5).astype(float) if binary
               else np.maximum(correlation, 0.)**2)
    np.fill_diagonal(weights, 0.)
    degrees = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, degrees, out=np.zeros_like(weights), where=degrees > 0)


def _validate(q, network):
    q, network = np.asarray(q, dtype=float), np.asarray(network, dtype=float)
    if (q.ndim != 2 or not len(q) or not q.shape[1] or not np.isfinite(q).all()
            or (q < 0).any() or network.shape != (q.shape[1], q.shape[1])
            or not np.isfinite(network).all() or (network < 0).any()
            or not np.allclose(np.diag(network), 0.)
            or not np.all(np.isclose(network.sum(axis=1), 0.) | np.isclose(network.sum(axis=1), 1.))):
        raise ValueError('Need nonnegative squared returns and a row-normalized, zero-diagonal network.')
    return q, network


def network_after(q, network, parameters):
    """h[t+1] after each observed q[t], with h[0] = 1."""
    q, network = _validate(q, network)
    omega, alpha, spillover, beta = np.asarray(parameters, dtype=float)
    if (not np.isfinite(parameters).all() or omega <= 0
            or min(alpha, spillover, beta) < 0 or alpha + spillover + beta >= 1):
        raise ValueError('Require positive intercept and nonnegative feedback summing below one.')
    forcing = omega + alpha*q + spillover*(q @ network.T)
    return lfilter([1.], [1., -beta], forcing, axis=0,
                   zi=np.full((1, q.shape[1]), beta))[0]


def _parameters(theta, active):
    weights = .999 * softmax(np.r_[theta[1:], 0.])[:-1]
    if active:
        alpha, spillover, beta = weights
    else:
        alpha, beta = weights
        spillover = 0.
    return np.array([np.exp(theta[0]), alpha, spillover, beta])


def objective_gradient(theta, q, network, active=True):
    """Mean marginal Gaussian QLIKE and its analytic transformed gradient."""
    parameters = _parameters(theta, active)
    omega, alpha, spillover, beta = parameters
    after = network_after(q, network, parameters)
    variance = np.vstack([np.ones(q.shape[1]), after[:-1]])
    forcing = np.stack([np.ones_like(q), q, q @ network.T, variance], axis=2)
    derivative_after = lfilter([1.], [1., -beta], forcing, axis=0)
    derivative = np.concatenate([np.zeros((1, q.shape[1], 4)), derivative_after[:-1]])
    raw = np.mean((1/variance - q/variance**2)[..., None]*derivative, axis=(0, 1))
    chosen = [1, 2, 3] if active else [1, 3]
    weights, gradient = parameters[chosen], raw[chosen]
    jacobian = np.r_[omega*raw[0], weights*(gradient - np.dot(weights, gradient)/.999)]
    return float(np.mean(np.log(variance) + q/variance)), jacobian


def fit_network(q, network):
    q, network = _validate(q, network)
    active = bool(network.any())
    attempts = []
    for alpha, spillover, beta in STARTS:
        weights = np.array([alpha, spillover, beta] if active else [alpha, beta])
        initial = np.r_[np.log(1-weights.sum()), np.log(weights/(.999-weights.sum()))]
        result = minimize(objective_gradient, initial, args=(q, network, active), jac=True,
                          method='L-BFGS-B', bounds=[(-12., 4.)] + [(-12., 12.)]*len(weights),
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7})
        attempts.append(dict(parameters=_parameters(result.x, active).tolist(),
                             objective=float(result.fun), converged=bool(result.success),
                             message=str(result.message), iterations=int(result.nit)))
    finite = [a for a in attempts if np.isfinite(a['objective']) and np.isfinite(a['parameters']).all()]
    if not finite:
        raise FloatingPointError('All Network GARCH starts failed; no replacement is allowed.')
    best = min(finite, key=lambda a: a['objective']).copy()
    best['attempts'] = attempts
    return best


def forecast_network(q, network, parameters, horizon=21):
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Require a positive integer horizon.')
    first = network_after(q, network, parameters)[-1]
    omega, alpha, spillover, beta = parameters
    result = [first]
    for _ in range(1, horizon):
        result.append(omega + (alpha+beta)*result[-1] + spillover*(network @ result[-1]))
    result = np.asarray(result)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError('Nonfinite or nonpositive Network GARCH forecast.')
    return result


def append_network_benchmarks(prices, forecasts, audits):
    """Reuse the frozen origins and verify common training scales and targets."""
    validate_prices(prices)
    returns = np.log(prices).diff().iloc[1:].to_numpy()
    templates = forecasts[forecasts.method == 'persistent_asymmetric']
    if templates.empty or templates.duplicated(['origin', 'asset', 'horizon']).any():
        raise ValueError('Need unique fractional templates for one panel.')
    by_date = {a['origin']: a for a in audits}
    rows, records, active_date = [], [], None
    for date, block in templates.groupby('origin', sort=True):
        origin = int(prices.index.get_loc(date))
        if block.last_fit.nunique() != 1 or len(block) != len(prices.columns)*len(HORIZONS):
            raise ValueError('Incomplete targets or inconsistent refits.')
        refit = pd.Timestamp(block.last_fit.iloc[0])
        if refit > date:
            raise ValueError('Future refit date.')
        if refit != active_date:
            fit_origin = int(prices.index.get_loc(refit))
            offset = max(0, fit_origin-1008)
            base = _base(returns[offset:fit_origin])
            recorded = by_date[str(refit.date())]
            for key in ['mean', 'scale', 'correlation', 'graph']:
                np.testing.assert_allclose(base[key], recorded[key], rtol=1e-10, atol=1e-13)
            mean, scale = base['mean'], base['scale']
            q = (returns[offset:fit_origin]-mean)**2/scale
            networks = {'network_garch': network_matrix(base['correlation']),
                        'network_garch_binary': network_matrix(base['correlation'], binary=True),
                        'pooled_garch': np.zeros_like(base['correlation'])}
            fits = {method: fit_network(q, matrix) for method, matrix in networks.items()}
            records.append(dict(origin=str(refit.date()), first_training_return=str(prices.index[offset+1].date()),
                                last_training_return=str(refit.date()), observations=len(q),
                                assets=list(prices.columns), fits=fits,
                                networks={k:v.tolist() for k,v in networks.items()}))
            active_date = refit
        observed = (returns[offset:origin]-mean)**2/scale
        daily = {method: forecast_network(observed, networks[method], fits[method]['parameters'])*scale
                 for method in METHODS}
        keyed = block.set_index(['horizon', 'asset'])
        for horizon in HORIZONS:
            actual = ((returns[origin:origin+horizon]-mean)**2).sum(axis=0)
            for j, asset in enumerate(prices.columns):
                template = keyed.loc[(horizon, asset)].to_dict()
                if pd.Timestamp(template['target']) != prices.index[origin+horizon]:
                    raise ValueError('Target dates do not match the retained trading observations.')
                np.testing.assert_allclose([template['realized_proxy'], template['reference_scale']],
                                           [actual[j], horizon*scale[j]], rtol=1e-10, atol=1e-13)
                for method in METHODS:
                    prediction = float(daily[method][:horizon,j].sum())
                    row = dict(template, asset=asset, horizon=horizon, method=method, variance=prediction)
                    row['qlike'] = np.log(prediction/row['reference_scale'])+row['realized_proxy']/prediction
                    row['normalized_square_error'] = ((row['realized_proxy']-prediction)/row['reference_scale'])**2
                    rows.append(row)
    return pd.concat([forecasts, pd.DataFrame(rows)], ignore_index=True), records
