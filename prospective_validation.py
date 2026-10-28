"""Frozen forecasting rules and alternative daily volatility measurements.

The core fractional model is imported unchanged. OHLC targets are evaluation
measurements only: they never enter parameter selection in this protocol.
"""

from copy import deepcopy
import numpy as np
import pandas as pd
from fractional_diffusion import fit_variance_models, conditional_variances
from refined_fractional import select_refinements, fit_gjr, refined_variances, forecast_choice

METHODS = ('persistent_asymmetric', 'matched_no_graph', 'matched_graph_025', 'matched_graph_1', 'matched_graph_4', 'asymmetric_no_graph',
           'asymmetric_exponential', 'gjr_garch', 'equal_combination', 'ewma')
FIELDS = ('Open', 'High', 'Low', 'Close')


def validate_ohlc(frame, assets):
    """Long table with adjusted OHLC; no filling, duplicates, or silent exclusions."""
    required = {'Date', 'asset', *FIELDS}
    if not required <= set(frame):
        raise ValueError('Missing OHLC columns.')
    data = frame.copy()
    data['Date'] = pd.to_datetime(data.Date, errors='raise')
    if data.Date.dt.tz is not None or not data.Date.eq(data.Date.dt.normalize()).all():
        raise ValueError('Session dates must be timezone-naive calendar dates.')
    if set(data.asset) != set(assets) or data.duplicated(['Date','asset']).any():
        raise ValueError('Missing/extra assets or duplicate OHLC observations.')
    values = data[list(FIELDS)].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('OHLC must be finite and positive.')
    if ((data.High < data[['Open','Close','Low']].max(axis=1)) |
            (data.Low > data[['Open','Close','High']].min(axis=1))).any():
        raise ValueError('Inconsistent high/low bounds; no clipping or repair.')
    if not data.groupby('Date').size().eq(len(assets)).all():
        raise ValueError('Incomplete session panel.')
    return data.sort_values(['Date','asset']).reset_index(drop=True)


def prices_from_ohlc(frame, assets):
    data = validate_ohlc(frame, assets)
    return data.pivot(index='Date',columns='asset',values='Close').loc[:,assets]


def daily_measurements(frame, assets):
    """RS + overnight component, with the gap centering deferred to the origin.

    RS is an intraday diffusion-variance proxy. The overnight-plus-RS target
    requires additional mean/covariance assumptions to estimate whole-day
    conditional variance; it is not labeled high-frequency realized variance.
    """
    data = validate_ohlc(frame, assets)
    rows=[]
    for asset, block in data.groupby('asset',sort=False):
        block=block.sort_values('Date')
        previous=block.Close.shift()
        gap=np.log(block.Open/previous)
        cc=np.log(block.Close/previous)
        rs=np.log(block.High/block.Open)*np.log(block.High/block.Close) + np.log(block.Low/block.Open)*np.log(block.Low/block.Close)
        for date, r, g, v in zip(block.Date.iloc[1:],cc.iloc[1:],gap.iloc[1:],rs.iloc[1:],strict=True):
            rows.append(dict(Date=date,asset=asset,close_return=float(r),overnight_return=float(g),rs=float(v)))
    return pd.DataFrame(rows)


def fit_frozen_model(ohlc, assets):
    prices=prices_from_ohlc(ohlc,assets)
    returns=np.log(prices).diff().iloc[1:].to_numpy()
    fitted=fit_variance_models(returns)
    fitted['refined'],_=select_refinements(returns)
    centered=returns[fitted['start']:] - fitted['mean']
    fitted['gjr']=[fit_gjr(centered[:,j],fitted['scale'][j]) for j in range(len(assets))]
    if not all(item['converged'] for item in fitted['gjr'] + fitted['garch']):
        raise FloatingPointError('Nonconverged benchmark fit; retain a failed origin, no substitution.')
    gaps=daily_measurements(ohlc,assets).pivot(index='Date',columns='asset',values='overnight_return').loc[:,assets]
    fitted['overnight_mean']=gaps.iloc[fitted['start']:].mean().to_numpy()
    fitted['fit_origin']=str(prices.index[-1].date())
    fitted['assets']=list(assets)
    # DataFrame is a selection audit, not needed to restore the forecasting state.
    fitted.pop('scores')
    return fitted


def forecasts_at_origin(ohlc, fitted, horizons=(1,5,21), lead_sessions=1):
    if not isinstance(lead_sessions,int) or lead_sessions<0:raise ValueError('Nonnegative integer lead required.')
    maximum=max(horizons)+lead_sessions
    assets=fitted['assets'];prices=prices_from_ohlc(ohlc,assets)
    returns=np.log(prices).diff().iloc[1:].to_numpy()
    daily=refined_variances(returns,fitted,maximum)
    matched=deepcopy(fitted['refined']['persistent_asymmetric'])
    matched['diffusion']=0.
    daily['matched_no_graph']=forecast_choice(returns,fitted,matched,maximum)
    for name,strength in [('matched_graph_025',.25),('matched_graph_1',1.),('matched_graph_4',4.)]:
        matched['diffusion']=strength
        daily[name]=forecast_choice(returns,fitted,matched,maximum)
    daily['ewma']=conditional_variances(returns,fitted,maximum)['ewma']
    daily['equal_combination']=.5*(daily['persistent_asymmetric']+daily['gjr_garch'])
    rows=[]
    for method in METHODS:
        if not np.isfinite(daily[method]).all() or (daily[method]<=0).any():
            raise FloatingPointError('Invalid forecast; no substitution.')
        for horizon in horizons:
            for j,asset in enumerate(assets):
                rows.append(dict(asset=asset,method=method,horizon=horizon,
                                 variance=float(daily[method][lead_sessions:lead_sessions+horizon,j].sum()),
                                 reference_scale=float(horizon*fitted['scale'][j]),
                                 mean=float(fitted['mean'][j]),overnight_mean=float(fitted['overnight_mean'][j])))
    return pd.DataFrame(rows)


def score_forecasts(forecasts, measurements, target_dates):
    """Exact scheduled dates required: missing sessions never shift a horizon."""
    expected=pd.DatetimeIndex(target_dates)
    if not expected.is_unique or not expected.is_monotonic_increasing:
        raise ValueError('Target dates must be unique and increasing.')
    data=measurements.copy();data['Date']=pd.to_datetime(data.Date)
    if data.duplicated(['Date','asset']).any():
        raise ValueError('Duplicate target records.')
    data=data.set_index(['Date','asset'])
    rows=[]
    for row in forecasts.to_dict('records'):
        horizon=int(row['horizon'])
        if len(expected)<horizon:
            raise ValueError('Incomplete target horizon.')
        keys=pd.MultiIndex.from_product([expected[:horizon],[row['asset']]],names=['Date','asset'])
        if not keys.isin(data.index).all():
            raise ValueError('Missing scheduled target; do not bridge missing sessions.')
        block=data.loc[keys]
        if not np.isfinite(block[['close_return','overnight_return','rs']]).all().all() or (block.rs<0).any():
            raise ValueError('Invalid target measurement.')
        targets={'close_squared':float(((block.close_return-row['mean'])**2).sum()),
                 'overnight_rs':float(((block.overnight_return-row['overnight_mean'])**2+block.rs).sum())}
        variance,reference=row['variance'],row['reference_scale']
        if not np.isfinite([variance,reference]).all() or min(variance,reference)<=0:
            raise ValueError('Invalid forecast or reference scale.')
        for proxy,value in targets.items():
            rows.append(row | dict(proxy=proxy,realized_proxy=value,
                                  qlike=float(np.log(variance/reference)+value/variance),
                                  normalized_square_error=float(((value-variance)/reference)**2)))
    return pd.DataFrame(rows)


def restore_fit(record):
    result=deepcopy(record)
    for key in ['mean','scale','graph','correlation','overnight_mean']:
        result[key]=np.asarray(result[key],float)
    return result
