"""Causal, regularized forecast heads around the unchanged memory simulator.

The simulator supplies features, not a promised predictive advantage. Candidate
selection and uncertainty calibration use only outcomes already observed.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from forecasting import HORIZONS, forecast_from_history
from temporal_memory import fit_parameters, validate_prices

CONTROLS = ('lag1', 'mean5', 'mean21', 'mean63', 'vol_ratio')
PENALTIES = (1., 10.)
WINDOWS = (504, 1008)
MIN_TRAIN_ORIGINS = 40
MIN_SELECTION_ORIGINS = 20


def build_features(prices, sector_model=False, origin_step=5, first_origin=252):
    """Features at fixed historical origins plus the latest observed price.

    Targets may be missing at the end; they are never used before target_position.
    Parameter refits have at least 21 observations between them. No historical
    feature is recomputed with parameters fitted at a later origin.
    """
    validate_prices(prices)
    if not isinstance(origin_step, int) or origin_step < 1 or first_origin < 252:
        raise ValueError('Require a positive integer stride and at least 252 initial returns.')
    if len(prices) <= first_origin:
        raise ValueError('Insufficient history for adaptive features.')
    returns = np.log(prices).diff().iloc[1:]
    ewma = returns.pow(2).ewm(alpha=.06, adjust=False).mean().pow(.5)
    origins = list(range(first_origin, len(prices), origin_step))
    if origins[-1] != len(prices) - 1:
        origins.append(len(prices) - 1)
    frames = []
    last_fit = -21
    for origin in origins:
        observed = prices.iloc[:origin + 1]
        if origin - last_fit >= 21:
            parameters = fit_parameters(observed, sector_model)
            last_fit = origin
        order = parameters['order']
        past = returns[order].iloc[:origin]
        anchor = observed[order].iloc[-1].to_numpy()
        raw = {scale: forecast_from_history(observed, parameters, scale * parameters['kappa_base'],
                                           max(HORIZONS), sector_model)
               for scale in ((0., .5, 1.) if sector_model else (0.,))}
        for horizon in HORIZONS:
            target_pos = origin + horizon
            known = target_pos < len(prices)
            drift = raw[0.]['drift'][horizon - 1] / anchor - 1
            block = pd.DataFrame({
                'origin': prices.index[origin], 'origin_position': origin,
                'target': prices.index[target_pos] if known else pd.NaT,
                'target_position': target_pos, 'last_fit': prices.index[last_fit],
                'horizon': horizon, 'asset': order, 'anchor': anchor,
                'actual_return': prices[order].iloc[target_pos].to_numpy() / anchor - 1 if known else np.nan,
                'drift': drift, 'lag1': past.iloc[-1].to_numpy(),
                'mean5': past.iloc[-5:].mean().to_numpy(),
                'mean21': past.iloc[-21:].mean().to_numpy(),
                'mean63': past.iloc[-63:].mean().to_numpy(),
                'vol_ratio': np.log(np.maximum(past.iloc[-20:].std().to_numpy(), 1e-10) /
                                    np.maximum(past.iloc[-252:].std().to_numpy(), 1e-10)),
                'ewma_scale': np.maximum(ewma[order].iloc[origin - 1].to_numpy() * np.sqrt(horizon), 1e-10),
                'fixed_scale': parameters['sigma'] * np.sqrt(horizon),
            })
            for scale, forecast in raw.items():
                block[f'memory_{scale:g}'] = forecast['memory'][horizon - 1] / anchor - 1 - drift
            reference = raw[.5 if sector_model else 0.]
            block['raw_memory'] = reference['memory'][horizon - 1] / anchor - 1
            block['raw_lower'] = reference['lower'][horizon - 1] / anchor - 1
            block['raw_upper'] = reference['upper'][horizon - 1] / anchor - 1
            frames.append(block.sort_values('asset'))
    return pd.concat(frames, ignore_index=True)


def ridge_prediction(training, current, columns, penalty):
    """Mean-loss ridge on standardized features, with an unpenalized intercept."""
    x = training[list(columns)].to_numpy()
    z = current[list(columns)].to_numpy()
    y = (training.actual_return - training.drift).to_numpy()
    if penalty <= 0 or not all(np.isfinite(a).all() for a in (x, z, y)):
        raise ValueError('Ridge requires positive penalty and finite training/predictor values.')
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    x, z = (x - center) / scale, (z - center) / scale
    intercept = y.mean()
    beta = np.linalg.solve(x.T @ x / len(x) + penalty * np.eye(x.shape[1]),
                           x.T @ (y - intercept) / len(x))
    predicted = current.drift.to_numpy() + intercept + z @ beta
    # A direct return head must still describe positive prices. Extreme invalid
    # predictions fall back to the current drift reference, before any scoring.
    invalid = ~np.isfinite(predicted) | (predicted <= -1)
    predicted[invalid] = current.drift.to_numpy()[invalid]
    return predicted, {'columns': list(columns), 'center': center.tolist(), 'scale': scale.tolist(),
                       'intercept': float(intercept), 'coefficients': beta.tolist(),
                       'fallback_count': int(invalid.sum()), 'training_rows': len(training)}


def _choose(history, origin, candidates):
    """Score already-issued candidate predictions after their targets mature."""
    eligible = [r for r in history if r['position'] >= origin - 252 and r['target_position'] <= origin
                and all(np.isfinite(r['predictions'][name]).all() for name in candidates)]
    if len(eligible) < MIN_SELECTION_ORIGINS:
        return 'drift', len(eligible), None
    scores = {name: float(np.mean([np.mean((r['predictions'][name] - r['actual'])**2)
                                  for r in eligible])) for name in candidates}
    return min(scores, key=scores.get), len(eligible), scores


def _interval_width(history, origin, current_scale):
    eligible = [r for r in history if r['position'] >= origin - 252 and r['target_position'] <= origin]
    if len(eligible) < MIN_SELECTION_ORIGINS:
        return norm.ppf(.9) * current_scale, len(eligible), float(norm.ppf(.9))
    errors = np.concatenate([np.abs(r['actual'] - r['adaptive']) / r['scale'] for r in eligible])
    # Finite-sample upper empirical quantile; temporal/panel exchangeability is
    # not assumed, so this does not promise distribution-free coverage.
    rank = min(len(errors), int(np.ceil((len(errors) + 1) * .8)))
    q = float(np.partition(errors, rank - 1)[rank - 1])
    return q * current_scale, len(eligible), q


def adaptive_forecast(features):
    """Prequential fitting, selection, and interval calibration for each horizon.

    Known target rows satisfy target_position <= current origin. With overlapping
    horizons this boundary purges labels which would otherwise leak future data.
    Candidate results are retained, including the no-memory ablation.
    """
    memory_columns = tuple(c for c in features if c.startswith('memory_'))
    if not memory_columns:
        raise ValueError('Missing memory features.')
    candidates = [(f'{family}_w{window}_r{penalty:g}', window, penalty,
                   CONTROLS + memory_columns if family == 'memory' else CONTROLS)
                  for family in ('memory', 'controls') for window in WINDOWS for penalty in PENALTIES]
    full_names = ['constant', 'drift'] + [c[0] for c in candidates if c[0].startswith('memory_')]
    control_names = ['constant', 'drift'] + [c[0] for c in candidates if c[0].startswith('controls_')]
    output, audits = [], []
    for horizon, panel in features.groupby('horizon', sort=True):
        history = []
        for position, current in panel.groupby('origin_position', sort=True):
            current = current.sort_values('asset')
            preds = {'constant': np.zeros(len(current)), 'drift': current.drift.to_numpy()}
            fitted = {}
            for name, window, penalty, columns in candidates:
                training = panel[(panel.target_position <= position) &
                                 (panel.origin_position >= position - window)]
                if training.origin_position.nunique() < MIN_TRAIN_ORIGINS:
                    preds[name] = np.full(len(current), np.nan)
                    continue
                preds[name], fitted[name] = ridge_prediction(training, current, columns, penalty)
                fitted[name]['last_training_target'] = str(training.target.max().date())
                fitted[name]['last_training_origin'] = str(training.origin.max().date())
            selected, n_validation, scores = _choose(history, position, full_names)
            control_selected, _, control_scores = _choose(history, position, control_names)
            prediction = preds[selected]
            scale = current.ewma_scale.to_numpy()
            width, n_intervals, q = _interval_width(history, position, scale)
            block = current[['origin', 'target', 'origin_position', 'target_position', 'last_fit',
                             'horizon', 'asset', 'anchor', 'actual_return', 'raw_memory',
                             'raw_lower', 'raw_upper']].copy()
            block['adaptive'] = prediction
            block['without_memory'] = preds[control_selected]
            block['constant'], block['drift'] = preds['constant'], preds['drift']
            block['lower'] = np.maximum(-1., prediction - width)
            block['upper'] = prediction + width
            fixed_width = norm.ppf(.9) * current.fixed_scale.to_numpy()
            block['fixed_lower'] = np.maximum(-1., prediction - fixed_width)
            block['fixed_upper'] = prediction + fixed_width
            block['selected'] = selected
            block['selected_without_memory'] = control_selected
            block['validation_origins'] = n_validation
            block['interval_origins'] = n_intervals
            block['interval_multiplier'] = q
            output.append(block)
            audits.append({'origin': str(current.origin.iloc[0].date()), 'horizon': int(horizon),
                           'selected': selected, 'selected_without_memory': control_selected,
                           'selection_mse': scores, 'ablation_selection_mse': control_scores,
                           'fits': fitted})
            history.append({'position': position, 'target_position': position + horizon,
                            'actual': current.actual_return.to_numpy(), 'predictions': preds,
                            'adaptive': prediction, 'scale': scale})
    return pd.concat(output, ignore_index=True), audits


def evaluate_adaptive(forecasts, start='2023-01-01'):
    """Identical origins across methods/horizons; exclude incomplete end blocks."""
    counts = forecasts.groupby(['origin', 'asset']).actual_return.count()
    complete = counts[counts == len(HORIZONS)].index
    keep = pd.MultiIndex.from_frame(forecasts[['origin', 'asset']]).isin(complete)
    scored = forecasts[keep & (forecasts.origin >= pd.Timestamp(start))].copy()
    if scored.empty:
        raise ValueError('No complete forecasts in the requested evaluation period.')
    scored['year'] = scored.origin.dt.year.astype(str)
    samples = pd.concat([scored, scored.assign(year='all')], ignore_index=True)
    rows, intervals = [], []
    for (asset, horizon, year), block in samples.groupby(['asset', 'horizon', 'year']):
        actual = block.actual_return.to_numpy()
        naive_mse = np.mean(actual**2)
        for method in ('adaptive', 'without_memory', 'raw_memory', 'drift', 'constant'):
            error = block[method].to_numpy() - actual
            rows.append({'asset': asset, 'horizon': horizon, 'year': year, 'method': method,
                         'origins': len(block), 'relative_RMSE_percent': 100 * np.sqrt(np.mean(error**2)),
                         'relative_MAE_percent': 100 * np.mean(np.abs(error)),
                         'bias_percent': 100 * np.mean(error),
                         'skill_vs_constant': 1 - np.mean(error**2) / naive_mse if naive_mse else np.nan})
        for name, lower_col, upper_col in [('adaptive', 'lower', 'upper'),
                                           ('fixed_gaussian', 'fixed_lower', 'fixed_upper'),
                                           ('raw_memory', 'raw_lower', 'raw_upper')]:
            lower, upper = block[lower_col].to_numpy(), block[upper_col].to_numpy()
            score = upper - lower + 10 * np.maximum(lower - actual, 0) + 10 * np.maximum(actual - upper, 0)
            intervals.append({'asset': asset, 'horizon': horizon, 'year': year, 'method': name,
                              'origins': len(block), 'coverage': np.mean((actual >= lower) & (actual <= upper)),
                              'mean_width_percent': 100 * np.mean(upper - lower),
                              'interval_score_percent': 100 * np.mean(score)})
    return pd.DataFrame(rows), pd.DataFrame(intervals), scored


def paired_comparisons(scored, seed=1729, repetitions=2000):
    """Descriptive paired block bootstrap; sectors resampled together by origin."""
    rng = np.random.default_rng(seed)
    rows = []
    for (panel, horizon), block in scored.groupby(['panel', 'horizon']):
        for reference in ('constant', 'drift', 'without_memory'):
            differences = block.assign(loss_difference=(block.adaptive - block.actual_return)**2 -
                                       (block[reference] - block.actual_return)**2).groupby('origin').loss_difference.mean().to_numpy()
            n, length = len(differences), max(1, int(np.ceil(horizon / 5)))
            # Circular moving blocks preserve local target-overlap dependence.
            starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n / length))))
            indices = ((starts[..., None] + np.arange(length)) % n).reshape(repetitions, -1)[:, :n]
            lower, upper = np.quantile(differences[indices].mean(axis=1), [.025, .975])
            rows.append({'panel': panel, 'horizon': horizon, 'reference': reference,
                         'origins': n, 'block_length': length, 'mean_loss_difference': differences.mean(),
                         'descriptive_lower_95': lower, 'descriptive_upper_95': upper})
    return pd.DataFrame(rows)
