"""Persistent, asymmetric fractional diffusion with past-only horizon-aware selection.

The earlier model stays in fractional_diffusion.py. Conditional symmetry is
required for the asymmetric multi-step moment recursion, not just unit variance.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import softmax

from fractional_diffusion import (
    MEMORY, ORDERS, FEEDBACK, DIFFUSION, DECAYS, HORIZONS, _base,
    fractional_weights, diffusion_operator, fit_variance_models, conditional_variances,
)
from temporal_memory import validate_prices

SMOOTHING = (0., .5, .9)
ASYMMETRY = (0., 1.)


def kernel(choice):
    if choice['family'] == 'fractional':
        return fractional_weights(choice['kernel_parameter'])
    weights = choice['kernel_parameter'] ** np.arange(MEMORY)
    return weights / weights.sum()


def transformed_squares(returns, mean, scale, asymmetry):
    centered = np.asarray(returns) - mean
    return centered**2 / scale * (1 + asymmetry * (centered < 0)) / (1 + asymmetry / 2)


def lagged_memory(signals, weights):
    """At row t, use only signals before t, with unit presample initialization."""
    padded = np.vstack([np.ones((len(weights), signals.shape[1])), signals])
    return lfilter(np.r_[0., weights], [1.], padded, axis=0)[len(weights):]


def variance_filter(memory, operator, feedback, smoothing):
    intercept = (1 - smoothing) * (1 - feedback)
    forcing = intercept + (1 - smoothing) * feedback * (memory @ operator.T)
    return lfilter([1.], [1., -smoothing], forcing, axis=0,
                   zi=np.full((1, memory.shape[1]), smoothing))[0]


def expected_paths(known, previous, weights, operator, feedback, smoothing):
    """Vectorized exact moments for several origins; known excludes future signals.

    known[step, origin, asset] contains the contribution of already observed
    signals to that step's memory. previous is h at each origin's last return.
    """
    horizon = len(known)
    result = np.empty_like(known)
    a, c = (1 - smoothing) * feedback, (1 - smoothing) * (1 - feedback)
    for step in range(horizon):
        memory = known[step].copy()
        if step:
            memory += np.einsum('k,koj->oj', weights[:step], result[:step][::-1])
        result[step] = c + smoothing * previous + a * (memory @ operator.T)
        previous = result[step]
    return result


def select_refinements(returns, validation=252, window=1008):
    """All validation labels finish by the refit date, at every scored horizon."""
    values = np.asarray(returns, dtype=float)
    n, maximum = len(values), max(HORIZONS)
    if n < MEMORY + validation + 1:
        raise ValueError('Need at least 505 observed returns.')
    base = _base(values[max(0, n - validation - window):n - validation])
    q = (values - base['mean'])**2 / base['scale']
    targets = np.arange(n - validation, n - maximum + 1)
    totals = np.vstack([np.zeros(q.shape[1]), q.cumsum(axis=0)])
    actual = {h: totals[targets + h] - totals[targets] for h in HORIZONS}
    operators = {t: diffusion_operator(base['graph'], t)
                 for t in (DIFFUSION if q.shape[1] > 1 else (0.,))}
    rows = []
    for family, parameters in [('fractional', ORDERS), ('exponential', DECAYS)]:
        for parameter in parameters:
            weights = kernel({'family': family, 'kernel_parameter': parameter})
            for asymmetry in ASYMMETRY:
                signals = transformed_squares(values, base['mean'], base['scale'], asymmetry)
                memory = lagged_memory(signals, weights)
                known = np.stack([lagged_memory(signals, weights[step:])[targets]
                                  for step in range(maximum)])
                for strength, operator in operators.items():
                    for feedback in FEEDBACK:
                        for smoothing in SMOOTHING:
                            filtered = variance_filter(memory, operator, feedback, smoothing)
                            expected = expected_paths(known, filtered[targets - 1], weights,
                                                      operator, feedback, smoothing)
                            scores = {}
                            for h in HORIZONS:
                                prediction = expected[:h].sum(axis=0)
                                scores[f'qlike_{h}'] = float(np.mean(np.log(prediction / h) + actual[h] / prediction))
                            rows.append(dict(family=family, kernel_parameter=parameter,
                                             asymmetry=asymmetry, diffusion=strength,
                                             feedback=feedback, smoothing=smoothing,
                                             validation_qlike=float(np.mean(list(scores.values()))), **scores))
    scores = pd.DataFrame(rows)
    fractional = scores.family == 'fractional'
    masks = {'persistent_symmetric': fractional & (scores.asymmetry == 0),
             'persistent_asymmetric': fractional,
             'asymmetric_no_graph': fractional & (scores.diffusion == 0),
             'asymmetric_exponential': scores.family == 'exponential'}
    choices = {name: scores.loc[scores[mask].validation_qlike.idxmin()].to_dict()
               for name, mask in masks.items()}
    return choices, scores


def _gjr_parameters(theta):
    alpha, beta, half_gamma = .999 * softmax([theta[1], theta[2], theta[3], 0.])[:3]
    return float(np.exp(theta[0])), float(alpha), float(beta), float(2 * half_gamma)


def gjr_after(squares, negative, parameters):
    omega, alpha, beta, gamma = parameters
    return lfilter([1.], [1., -beta], omega + (alpha + gamma * negative) * squares, zi=[beta])[0]


def fit_gjr(centered, scale):
    squares, negative = centered**2 / scale, centered < 0
    def objective(theta):
        variance = np.r_[1., gjr_after(squares, negative, _gjr_parameters(theta))[:-1]]
        return float(np.mean(np.log(variance) + squares / variance))
    fits = []
    for alpha, beta, gamma in [(.05, .85, .1), (.05, .9, .05)]:
        persistence = alpha + beta + gamma / 2
        remainder = .999 - persistence
        initial = [np.log(1 - persistence), np.log(alpha / remainder),
                   np.log(beta / remainder), np.log(gamma / 2 / remainder)]
        fits.append(minimize(objective, initial, method='L-BFGS-B',
                             bounds=[(-10, 2), (-10, 10), (-10, 10), (-10, 10)],
                             options={'maxiter': 300, 'ftol': 1e-10}))
    eligible = [f for f in fits if np.isfinite(f.fun)]
    if not eligible:
        raise FloatingPointError('Both GJR fits failed.')
    best = min(eligible, key=lambda f: f.fun)
    return dict(parameters=_gjr_parameters(best.x), converged=bool(best.success),
                objective=float(best.fun), message=str(best.message))


def forecast_choice(returns, fitted, choice, horizon=21):
    if not isinstance(horizon, int) or not 1 <= horizon <= MEMORY:
        raise ValueError('Require an integer horizon from 1 to the memory length.')
    signals = transformed_squares(returns, fitted['mean'], fitted['scale'], choice['asymmetry'])
    weights = kernel(choice)
    operator = diffusion_operator(fitted['graph'], choice['diffusion'])
    filtered = variance_filter(lagged_memory(signals, weights), operator,
                               choice['feedback'], choice['smoothing'])
    history = np.vstack([np.ones((MEMORY, signals.shape[1])), signals])[-MEMORY:][::-1]
    known = np.stack([weights[step:] @ history[:MEMORY - step] for step in range(horizon)])[:, None, :]
    return expected_paths(known, filtered[-1:], weights, operator,
                          choice['feedback'], choice['smoothing'])[:, 0] * fitted['scale']


def refined_variances(returns, fitted, horizon=21):
    result = {name: forecast_choice(returns, fitted, choice, horizon)
              for name, choice in fitted['refined'].items()}
    centered = np.asarray(returns)[fitted['start']:] - fitted['mean']
    prediction = []
    for j, fit in enumerate(fitted['gjr']):
        omega, alpha, beta, gamma = fit['parameters']
        first = gjr_after(centered[:, j]**2 / fitted['scale'][j], centered[:, j] < 0, fit['parameters'])[-1]
        persistence = alpha + beta + gamma / 2
        equilibrium = omega / (1 - persistence)
        prediction.append((equilibrium + persistence**np.arange(horizon) * (first - equilibrium)) * fitted['scale'][j])
    result['gjr_garch'] = np.column_stack(prediction)
    return result


def simulate_refined(returns, fitted, choice, horizon=21, paths=8000, seed=1729):
    weights, b, eta = kernel(choice), choice['smoothing'], choice['feedback']
    operator = diffusion_operator(fitted['graph'], choice['diffusion'])
    signals = transformed_squares(returns, fitted['mean'], fitted['scale'], choice['asymmetry'])
    filtered = variance_filter(lagged_memory(signals, weights), operator, eta, b)
    history = np.zeros((paths, MEMORY + horizon, signals.shape[1]))
    history[:, :MEMORY] = np.vstack([np.ones((MEMORY, signals.shape[1])), signals])[-MEMORY:]
    noise = np.random.default_rng(seed).multivariate_normal(np.zeros(signals.shape[1]),
                                                          fitted['correlation'], size=(paths, horizon))
    previous, result, variance = filtered[-1], [], []
    for step in range(horizon):
        memory = np.einsum('k,pkj->pj', weights, history[:, step:step + MEMORY][:, ::-1])
        current = (1 - b) * (1 - eta) + b * previous + (1 - b) * eta * (memory @ operator.T)
        shock = np.sqrt(current * fitted['scale']) * noise[:, step]
        history[:, MEMORY + step] = transformed_squares(shock, 0., fitted['scale'], choice['asymmetry'])
        result.append(shock + fitted['mean'])
        variance.append(current * fitted['scale'])
        previous = current
    return np.stack(result, axis=1), np.stack(variance, axis=1)


def rolling_refinement(prices, start='2023-01-01', origin_step=5):
    validate_prices(prices)
    returns = np.log(prices).diff().iloc[1:].to_numpy()
    positions = [o for o in range(252, len(prices) - max(HORIZONS), origin_step)
                 if prices.index[o] >= pd.Timestamp(start)]
    if not positions or positions[0] < 505:
        raise ValueError('Evaluation needs at least 505 earlier returns.')
    rows, audits, last_fit = [], [], -21
    for origin in positions:
        observed = returns[:origin]
        if origin - last_fit >= 21:
            fitted = fit_variance_models(observed)
            choices, scores = select_refinements(observed)
            fitted['refined'] = choices
            centered = observed[fitted['start']:] - fitted['mean']
            fitted['gjr'] = [fit_gjr(centered[:, j], fitted['scale'][j]) for j in range(centered.shape[1])]
            last_fit = origin
            audits.append(dict(origin=str(prices.index[origin].date()),
                               validation_first_target=str(prices.index[origin - 251].date()),
                               validation_last_target=str(prices.index[origin].date()),
                               choices=choices, candidate_scores=scores.to_dict(orient='records'),
                               gjr=fitted['gjr'], garch=fitted['garch'],
                               mean=fitted['mean'].tolist(), scale=fitted['scale'].tolist(),
                               graph=fitted['graph'].tolist(), correlation=fitted['correlation'].tolist()))
        forecast = conditional_variances(observed, fitted)
        forecast.update(refined_variances(observed, fitted))
        for h in HORIZONS:
            actual = ((returns[origin:origin + h] - fitted['mean'])**2).sum(axis=0)
            reference = h * fitted['scale']
            for method, variance in forecast.items():
                prediction = variance[:h].sum(axis=0)
                if not (np.isfinite(prediction).all() and (prediction > 0).all()):
                    raise FloatingPointError('Invalid forecast.')
                for j, asset in enumerate(prices.columns):
                    rows.append(dict(origin=prices.index[origin], target=prices.index[origin + h],
                                     last_fit=prices.index[last_fit], asset=asset, horizon=h, method=method,
                                     realized_proxy=actual[j], variance=prediction[j], reference_scale=reference[j],
                                     qlike=np.log(prediction[j] / reference[j]) + actual[j] / prediction[j],
                                     normalized_square_error=((actual[j] - prediction[j]) / reference[j])**2))
    return pd.DataFrame(rows), audits
