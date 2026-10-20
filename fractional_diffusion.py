"""Positive fractional-memory variance models with implicit graph diffusion.

This is a new, explicitly specified model of conditional variance. The legacy
return recurrence is retained elsewhere and is not relabeled as this model.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import softmax
from sklearn.covariance import LedoitWolf

from temporal_memory import validate_prices

MEMORY = 252
ORDERS = (.2, .4, .6, .8)
FEEDBACK = (.8, .95, .99)
DIFFUSION = (0., .25, 1., 4.)
DECAYS = (.9, .94, .97, .99)
HORIZONS = (1, 5, 21)


def fractional_weights(order, length=MEMORY):
    """Normalized coefficients of the truncated fractional integral (1-B)^(-d)."""
    if not np.isfinite(order) or not 0 < order <= 1 or not isinstance(length, int) or length < 1:
        raise ValueError('Require 0 < order <= 1 and positive integer memory length.')
    weights = np.ones(length)
    for k in range(1, length):
        weights[k] = weights[k - 1] * (k - 1 + order) / k
    return weights / weights.sum()


def graph_laplacian(returns):
    """Symmetric graph from positive squared shrinkage correlations, mean degree one."""
    values = np.asarray(returns, dtype=float)
    scale = values.std(axis=0, ddof=1)
    if values.ndim != 2 or not np.isfinite(values).all() or (scale <= 1e-12).any():
        raise ValueError('Need a finite nonconstant return panel.')
    if values.shape[1] == 1:
        return np.zeros((1, 1)), np.ones((1, 1))
    covariance = LedoitWolf().fit(values / scale).covariance_
    diagonal = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(diagonal, diagonal)
    adjacency = np.maximum(correlation, 0.)**2
    np.fill_diagonal(adjacency, 0.)
    degree = adjacency.sum(axis=1)
    graph = np.diag(degree) - adjacency
    graph /= max(float(degree.mean()), 1e-12)
    return graph, correlation


def diffusion_operator(graph, strength):
    graph = np.asarray(graph, dtype=float)
    if (graph.ndim != 2 or graph.shape[0] != graph.shape[1] or strength < 0
            or not np.isfinite(strength) or not np.isfinite(graph).all()
            or not np.allclose(graph, graph.T) or not np.allclose(graph.sum(axis=1), 0)):
        raise ValueError('Require a finite symmetric Laplacian and nonnegative diffusion.')
    off_diagonal = graph - np.diag(np.diag(graph))
    if (off_diagonal > 1e-12).any():
        raise ValueError('Graph Laplacian must have nonpositive off-diagonals.')
    return np.linalg.solve(np.eye(len(graph)) + strength * graph, np.eye(len(graph)))


def ewma_after(squares, decay):
    """Variance state after observing each square, initialized at unit variance."""
    squares = np.asarray(squares)
    return lfilter([1 - decay], [1, -decay], squares, axis=0,
                   zi=np.full((1, squares.shape[1]), decay))[0]


def _base(returns):
    mean = returns.mean(axis=0)
    scale = returns.var(axis=0, ddof=1)
    if (scale <= 1e-12).any() or not np.isfinite(scale).all():
        raise ValueError('Insufficient nonconstant returns for variance estimation.')
    graph, correlation = graph_laplacian(returns)
    return {'mean': mean, 'scale': scale, 'graph': graph, 'correlation': correlation}


def select_parameters(returns, validation=252, window=1008):
    """Select kernels on trailing validation with nuisance estimates fixed before it."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < MEMORY + validation + 1:
        raise ValueError('Need at least 505 returns before selecting the variance model.')
    base = _base(returns[max(0, n - validation - window):n - validation])
    q = (returns - base['mean'])**2 / base['scale']
    strengths = DIFFUSION if q.shape[1] > 1 else (0.,)
    operators = {t: diffusion_operator(base['graph'], t) for t in strengths}
    rows = []
    for family, values in [('fractional', ORDERS), ('exponential', DECAYS)]:
        for value in values:
            if family == 'fractional':
                memory = lfilter(np.r_[0., fractional_weights(value)], [1.], q, axis=0)
            else:
                memory = np.vstack([np.ones(q.shape[1]), ewma_after(q, value)[:-1]])
            for strength, operator in operators.items():
                diffused = memory[-validation:] @ operator.T
                for feedback in FEEDBACK:
                    v = 1 - feedback + feedback * diffused
                    loss = float(np.mean(np.log(v) + q[-validation:] / v))
                    rows.append({'family': family, 'kernel_parameter': value, 'feedback': feedback,
                                 'diffusion': strength, 'validation_qlike': loss})
    scores = pd.DataFrame(rows)
    choices = {}
    for name, mask in [
        ('fractional_diffusion', scores.family == 'fractional'),
        ('fractional_only', (scores.family == 'fractional') & (scores.diffusion == 0)),
        ('exponential_diffusion', scores.family == 'exponential'),
    ]:
        choices[name] = scores.loc[scores[mask].validation_qlike.idxmin()].to_dict()
    ewma_scores = {}
    for decay in DECAYS:
        v = np.vstack([np.ones(q.shape[1]), ewma_after(q, decay)[:-1]])[-validation:]
        ewma_scores[decay] = float(np.mean(np.log(v) + q[-validation:] / v))
    choices['ewma'] = {'decay': min(ewma_scores, key=ewma_scores.get),
                       'validation_scores': ewma_scores}
    return choices, scores


def _garch_parameters(theta):
    omega = np.exp(theta[0])
    alpha, beta = .999 * softmax([theta[1], theta[2], 0.])[:2]
    return float(omega), float(alpha), float(beta)


def garch_after(q, parameters):
    omega, alpha, beta = parameters
    return lfilter([1.], [1., -beta], omega + alpha * np.asarray(q), zi=[beta])[0]


def fit_garch(q):
    """Independent Gaussian-QMLE GARCH(1,1), positivity and persistence < .999."""
    def objective(theta):
        parameters = _garch_parameters(theta)
        variance = np.r_[1., garch_after(q, parameters)[:-1]]
        return float(np.mean(np.log(variance) + q / variance))
    fits = []
    for alpha, beta in [(.05, .9), (.1, .8)]:
        remainder = .999 - alpha - beta
        initial = [np.log(1 - alpha - beta), np.log(alpha / remainder), np.log(beta / remainder)]
        fits.append(minimize(objective, initial, method='L-BFGS-B',
                             bounds=[(-10, 2), (-10, 10), (-10, 10)],
                             options={'maxiter': 300, 'ftol': 1e-10}))
    eligible = [f for f in fits if np.isfinite(f.fun)]
    if not eligible:
        raise FloatingPointError('Both GARCH fits failed; do not report an invalid benchmark.')
    best = min(eligible, key=lambda f: f.fun)
    return {'parameters': _garch_parameters(best.x), 'converged': bool(best.success),
            'objective': float(best.fun), 'message': str(best.message)}


def fit_variance_models(returns, window=1008):
    returns = np.asarray(returns, dtype=float)
    choices, scores = select_parameters(returns, window=window)
    start = max(0, len(returns) - window)
    base = _base(returns[start:])
    q = (returns[start:] - base['mean'])**2 / base['scale']
    base.update(choices=choices, scores=scores, start=start,
                garch=[fit_garch(q[:, j]) for j in range(q.shape[1])])
    return base


def conditional_variances(returns, fitted, horizon=21):
    """Exact marginal future return variances under each fixed fitted variance model."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Horizon must be a positive integer.')
    q = (np.asarray(returns) - fitted['mean'])**2 / fitted['scale']
    if len(q) < MEMORY:
        raise ValueError('Insufficient observed memory.')
    scale, choices = fitted['scale'], fitted['choices']
    output = {'historical': np.tile(scale, (horizon, 1))}
    state = ewma_after(q[fitted['start']:], choices['ewma']['decay'])[-1]
    output['ewma'] = np.tile(state * scale, (horizon, 1))
    garch = []
    for j, entry in enumerate(fitted['garch']):
        omega, alpha, beta = entry['parameters']
        first = garch_after(q[fitted['start']:, j], entry['parameters'])[-1]
        equilibrium = omega / (1 - alpha - beta)
        garch.append((equilibrium + (alpha + beta)**np.arange(horizon) * (first - equilibrium)) * scale[j])
    output['garch'] = np.column_stack(garch)
    for name in ('fractional_diffusion', 'fractional_only', 'exponential_diffusion'):
        selected = choices[name]
        operator = diffusion_operator(fitted['graph'], selected['diffusion'])
        feedback, parameter = selected['feedback'], selected['kernel_parameter']
        future = []
        if selected['family'] == 'fractional':
            weights = fractional_weights(parameter)
            history = np.asarray(q[-MEMORY:]).copy()
            for _ in range(horizon):
                memory = weights @ history[-MEMORY:][::-1]
                variance = 1 - feedback + feedback * operator @ memory
                future.append(variance)
                history = np.vstack([history, variance])
        else:
            state = ewma_after(q[fitted['start']:], parameter)[-1]
            for _ in range(horizon):
                variance = 1 - feedback + feedback * operator @ state
                future.append(variance)
                state = parameter * state + (1 - parameter) * variance
        output[name] = np.asarray(future) * scale
    if not all(np.isfinite(v).all() and (v > 0).all() for v in output.values()):
        raise FloatingPointError('Invalid forecast variance.')
    return output


def simulate_fractional(returns, fitted, horizon=21, paths=2000, seed=1729):
    """Gaussian-innovation simulations of the new fractional variance model."""
    selected = fitted['choices']['fractional_diffusion']
    weights = fractional_weights(selected['kernel_parameter'])
    operator = diffusion_operator(fitted['graph'], selected['diffusion'])
    q = (np.asarray(returns) - fitted['mean'])**2 / fitted['scale']
    history = np.zeros((paths, MEMORY + horizon, q.shape[1]))
    history[:, :MEMORY] = q[-MEMORY:]
    innovations = np.random.default_rng(seed).multivariate_normal(np.zeros(q.shape[1]),
                                                               fitted['correlation'], size=(paths, horizon))
    result, variances = [], []
    for step in range(horizon):
        memory = np.einsum('k,pkj->pj', weights, history[:, step:step + MEMORY][:, ::-1])
        normalized = 1 - selected['feedback'] + selected['feedback'] * memory @ operator.T
        variance = normalized * fitted['scale']
        shock = np.sqrt(variance) * innovations[:, step]
        result.append(fitted['mean'] + shock)
        variances.append(variance)
        history[:, MEMORY + step] = shock**2 / fitted['scale']
    return np.stack(result, axis=1), np.stack(variances, axis=1)


def rolling_variance_forecasts(prices, start='2023-01-01', origin_step=5):
    validate_prices(prices)
    returns = np.log(prices).diff().iloc[1:].to_numpy()
    positions = [o for o in range(252, len(prices) - max(HORIZONS), origin_step)
                 if prices.index[o] >= pd.Timestamp(start)]
    if not positions or positions[0] < 505:
        raise ValueError('Evaluation needs enough prehistory for the 252-day validation.')
    rows, audits = [], []
    last_fit = -21
    fitted = None
    for origin in positions:
        observed = returns[:origin]
        if origin - last_fit >= 21:
            fitted = fit_variance_models(observed)
            last_fit = origin
            audits.append({'origin': str(prices.index[origin].date()),
                           'validation_first_target': str(prices.index[origin - 251].date()),
                           'validation_last_target': str(prices.index[origin].date()),
                           'choices': fitted['choices'], 'garch': fitted['garch'],
                           'mean': fitted['mean'].tolist(), 'scale': fitted['scale'].tolist(),
                           'graph': fitted['graph'].tolist(), 'correlation': fitted['correlation'].tolist(),
                           'candidate_scores': fitted['scores'].to_dict(orient='records')})
        forecast = conditional_variances(observed, fitted)
        for horizon in HORIZONS:
            realized = ((returns[origin:origin + horizon] - fitted['mean'])**2).sum(axis=0)
            reference_scale = horizon * fitted['scale']
            for name, variance in forecast.items():
                prediction = variance[:horizon].sum(axis=0)
                for j, asset in enumerate(prices.columns):
                    rows.append({'origin': prices.index[origin], 'target': prices.index[origin + horizon],
                                 'last_fit': prices.index[last_fit], 'asset': asset, 'horizon': horizon,
                                 'method': name, 'realized_proxy': realized[j], 'variance': prediction[j],
                                 'reference_scale': reference_scale[j],
                                 'qlike': np.log(prediction[j] / reference_scale[j]) + realized[j] / prediction[j],
                                 'normalized_square_error': ((realized[j] - prediction[j]) / reference_scale[j])**2})
    return pd.DataFrame(rows), audits, fitted
