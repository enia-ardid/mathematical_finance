"""Frozen-universe replication: FIGARCH reference and a fixed forecast combination.

The fractional model is imported unchanged. The equal-weight combination is a
forecast ensemble, not a newly estimated fractional stochastic recurrence.
"""

import numpy as np
import pandas as pd
from arch.univariate import FIGARCH, Normal, ZeroMean

from fractional_diffusion import HORIZONS
from temporal_memory import validate_prices


def fit_figarch(standardized):
    standardized = np.ascontiguousarray(standardized, dtype=float)
    model = ZeroMean(standardized, rescale=False,
                     volatility=FIGARCH(p=1, q=1, power=2., truncation=1000),
                     distribution=Normal())
    fit = model.fit(disp='off', show_warning=True,
                    options={'maxiter': 1000, 'ftol': 1e-9})
    if not np.isfinite(fit.params).all() or not np.isfinite(fit.loglikelihood):
        raise FloatingPointError('FIGARCH returned a nonfinite fit; no substitution is permitted.')
    return dict(parameters=fit.params.to_list(),
                backcast=float(model.volatility.backcast(np.asarray(standardized))),
                convergence_flag=int(fit.convergence_flag),
                message=str(fit.optimization_result.message),
                loglikelihood=float(fit.loglikelihood))


def forecast_figarch(standardized, fit, horizon=21):
    """Update with observed residuals only, retaining the fitting-time backcast."""
    values = np.ascontiguousarray(standardized, dtype=float)
    process = FIGARCH(p=1, q=1, power=2., truncation=1000)
    prediction = process.forecast(np.asarray(fit['parameters']), values,
                                 fit['backcast'], process.variance_bounds(values),
                                 start=len(values) - 1, horizon=horizon,
                                 method='analytic').forecasts[-1]
    if not (np.isfinite(prediction).all() and (prediction > 0).all()):
        raise FloatingPointError('FIGARCH forecast is not finite and positive.')
    return prediction


def append_equal_combination(forecasts):
    """Combine matched variances, then recompute losses; never average losses."""
    keys = ['panel', 'origin', 'asset', 'horizon']
    left = forecasts[forecasts.method == 'persistent_asymmetric'].copy()
    right = forecasts[forecasts.method == 'gjr_garch'].copy()
    if left.empty or len(left) != len(right):
        raise ValueError('Both component forecasts must cover the same targets.')
    aligned = left.merge(right, on=keys, suffixes=('', '_gjr'), how='outer',
                         validate='one_to_one', indicator=True)
    if not aligned['_merge'].eq('both').all():
        raise ValueError('Component forecast keys differ.')
    for col in ['target', 'last_fit', 'realized_proxy', 'reference_scale']:
        if not aligned[col].eq(aligned[col + '_gjr']).all():
            raise ValueError(f'Component {col} conventions differ.')
    output = aligned[list(forecasts.columns)].copy()
    output['method'] = 'equal_combination'
    output['variance'] = .5 * (aligned.variance + aligned.variance_gjr)
    output['qlike'] = np.log(output.variance / output.reference_scale) + output.realized_proxy / output.variance
    output['normalized_square_error'] = ((output.realized_proxy - output.variance) / output.reference_scale)**2
    return pd.concat([forecasts, output], ignore_index=True)


def append_figarch(prices, forecasts, audits):
    """Use the original study's origins, means, scales and refit schedule exactly."""
    validate_prices(prices)
    returns = np.log(prices).diff().iloc[1:].to_numpy()
    templates = forecasts[forecasts.method == 'persistent_asymmetric']
    rows, fit_audits, active_date = [], [], None
    audit_by_date = {a['origin']: a for a in audits}
    for date, block in templates.groupby('origin', sort=True):
        origin = int(prices.index.get_loc(date))
        refit_date = str(pd.Timestamp(block.last_fit.iloc[0]).date())
        if refit_date != active_date:
            audit = audit_by_date[refit_date]
            fit_origin = int(prices.index.get_loc(pd.Timestamp(refit_date)))
            offset = max(0, fit_origin - 1008)
            mean, scale = np.asarray(audit['mean']), np.asarray(audit['scale'])
            fit_values = (returns[offset:fit_origin] - mean) / np.sqrt(scale)
            fits = [fit_figarch(fit_values[:, j]) for j in range(len(prices.columns))]
            fit_audits.append(dict(origin=refit_date, fits=dict(zip(prices.columns, fits))))
            active_date = refit_date
        observed = (returns[offset:origin] - mean) / np.sqrt(scale)
        daily = np.column_stack([forecast_figarch(observed[:, j], f) * scale[j]
                                 for j, f in enumerate(fits)])
        for h in HORIZONS:
            for j, asset in enumerate(prices.columns):
                row = block[(block.horizon == h) & (block.asset == asset)].iloc[0].to_dict()
                prediction = float(daily[:h, j].sum())
                row.update(method='figarch', variance=prediction,
                           qlike=float(np.log(prediction / row['reference_scale']) + row['realized_proxy'] / prediction),
                           normalized_square_error=float(((row['realized_proxy'] - prediction) / row['reference_scale'])**2))
                rows.append(row)
    return pd.concat([forecasts, pd.DataFrame(rows)], ignore_index=True), fit_audits
