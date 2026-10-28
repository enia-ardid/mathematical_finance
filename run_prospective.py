"""Freeze, collect and score a future evaluation without rewriting earlier records.

Commands: init, tick, status, report. Run tick after 17:00 America/New_York.
Collection is local; hashes alone are not an independent timestamp service.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from prospective_validation import (validate_ohlc, daily_measurements, fit_frozen_model,
                                    forecasts_at_origin, score_forecasts, restore_fit, METHODS)
from forecast_diagnostics import dm_test, holm

ROOT=Path(__file__).resolve().parent
NY=ZoneInfo('America/New_York')
CODE=('run_prospective.py','prospective_validation.py','fractional_diffusion.py',
      'refined_fractional.py','temporal_memory.py','forecast_diagnostics.py','requirements.txt')
PACKAGES=('numpy','pandas','scipy','scikit-learn','arch','yfinance','nolds','setuptools','curl_cffi','peewee')
PANELS={'sectors':['XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','XLC'],
        'cross_asset':['AGG','IEF','TLT','TIP','GLD','SLV','DBC','VNQ']}
# Published NYSE holidays; no inference from data availability or measured losses.
HOLIDAYS=('2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25','2026-06-19',
          '2026-07-03','2026-09-07','2026-11-26','2026-12-25',
          '2027-01-01','2027-01-18','2027-02-15','2027-03-26','2027-05-31','2027-06-18',
          '2027-07-05','2027-09-06','2027-11-25','2027-12-24')
EARLY=('2026-11-27','2026-12-24','2027-11-26')


def now():
    return datetime.now(timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(datetime,pd.Timestamp)):return value.isoformat()
    raise TypeError(type(value).__name__)


def write_once(path,value):
    """Exclusive creation. Existing records, including failures, cannot be replaced."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,indent=2,default=serial,allow_nan=False)+'\n'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)  # Atomic publication, refusing an existing destination.
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def calendar(first):
    first=pd.Timestamp(first)
    if first.year!=2026 or first<pd.Timestamp('2026-10-27') or first>pd.Timestamp('2026-11-10'):
        raise ValueError('This calendar supports study launches from 27 October to 10 November 2026.')
    dates=pd.bdate_range(first,'2027-12-31').difference(pd.to_datetime(HOLIDAYS))
    # 52 origins, five sessions apart, and a one-session embargo plus 21 target sessions after the last.
    needed=5*51+1+21+1
    if len(dates)<needed:raise ValueError('Insufficient frozen calendar.')
    dates=dates[:needed]
    return [dict(date=str(d.date()),open_utc=datetime(d.year,d.month,d.day,9,30,tzinfo=NY).astimezone(timezone.utc).isoformat(),
                 close_utc=datetime(d.year,d.month,d.day,13 if str(d.date()) in EARLY else 16,tzinfo=NY).astimezone(timezone.utc).isoformat())
            for d in dates]


def init_study(directory,first):
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):raise FileExistsError('Use an empty study directory.')
    sessions=calendar(first)
    if datetime.fromisoformat(sessions[0]['open_utc'])<=now():raise ValueError('The first study session must still be in the future.')
    protocol=dict(version=1,created_at_utc=now().isoformat(),
                  study='Forward evaluation of frozen fractional-diffusive rules',
                  status='No prospective outcomes at registration; local record, not independent preregistration.',
                  parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  panels=PANELS,sessions=sessions,origin_dates=[s['date'] for s in sessions[:256:5]],
                  horizons=[1,5,21],lead_sessions=1,methods=list(METHODS),history_start='2019-01-01',
                  refit_every_origins=5,estimation_window=1008,validation_length=252,
                  frozen_rules='Unchanged select_refinements and GJR; monthly (25-session) refits, fixed grids and tie rules. Coefficients may update under these rules.',
                  matched_ablation='Set only the selected refined diffusion to 0, 0.25, 1 or 4; retain kernel, feedback, smoothing, asymmetry, mean and scale. The primary mechanism contrast is fixed tau=1 versus tau=0, even if the selected model chooses zero diffusion.',
                  proxies=['overnight_rs','close_squared'],primary_proxy='overnight_rs',
                  primary_comparisons=[dict(model='matched_graph_1',reference='matched_no_graph',horizon=5),
                                       dict(model='persistent_asymmetric',reference='gjr_garch',horizon=21)],
                  primary_family='Four tests: two named comparisons in each of two panels; Holm 5%, two-sided DM, Bartlett HAC lag 4; lag 9 sensitivity.',
                  inference='Equal-weight assets within each origin, no cross-panel pooling; report raw and Holm p-values. All 52 scheduled origins must be complete for formal primary tests; no claim of adequate power.',
                  missing_rule='Retain all 52 scheduled origins per panel; no replacements or bridging missing target dates. Matched complete-panel scores only, with explicit coverage; any missing origin makes primary inference inconclusive; descriptive matched scores remain available.',
                  nonconvergence='Fail the whole panel at that origin; no fallback. A missed scheduled refit invalidates dependent origins until the next scheduled refit.',
                  capture_window='17:00 through 23:59:59 America/New_York on each frozen session; first successful complete snapshot; no backfill after the window.',
                  forecast_deadline='23:59:59 America/New_York at the origin, checked after fitting; late computation is retained but ineligible.',
                  target_embargo='Skip the next session: an h-session target sums conditional daily variances at steps 2 through h+1. Its first close-to-close interval starts at the next session close, after the forecast deadline.',
                  target_release='First valid same-session evening capture, including previous close from the same vintage. Later corrections cannot replace it.',
                  adjustment='Yahoo raw OHLC and Adj Close retained; each OHLC multiplied by same-date Adj Close / Close. No repairs, filling or ticker substitutions.',
                  overnight_mean='Trailing 1008 observed close-to-open log returns, estimated at the common refit date; held fixed until the next scheduled refit.',
                  range_scope='Overnight centered square + Rogers-Satchell. Whole-day interpretation requires adequate centering, intraday diffusion assumptions and no material overnight/intraday covariance. Not high-frequency realized variance.',
                  calendar_source='https://www.nyse.com/markets/hours-calendars',
                  calendar_exceptions='Unexpected exchange closure requires a dated prospective amendment; never infer calendar changes from missing prices.',
                  report_after_utc=datetime.fromisoformat(sessions[-1]['date']+'T23:59:59').replace(tzinfo=NY).astimezone(timezone.utc).isoformat(),early_reporting='Coverage only; no interim performance ranking or stopping on significance.',
                  code_sha256={name:sha(ROOT/name) for name in CODE},
                  environment=dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in PACKAGES}))
    write_once(directory/'protocol.json',protocol)
    write_once(directory/'freeze.json',dict(protocol_sha256=sha(directory/'protocol.json')))
    print(json.dumps(dict(protocol=str(directory/'protocol.json'),sha256=sha(directory/'protocol.json'),
                          first_origin=protocol['origin_dates'][0],last_origin=protocol['origin_dates'][-1],
                          final_target=sessions[-1]['date']),indent=2))


def read_protocol(directory):
    path=Path(directory)/'protocol.json'
    frozen=json.loads((Path(directory)/'freeze.json').read_text())
    if sha(path)!=frozen['protocol_sha256']:raise ValueError('Frozen protocol changed.')
    p=json.loads(path.read_text())
    for name,expected in p['code_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError(f'Frozen source changed: {name}; use a separate version/amendment.')
    if platform.python_version()!=p['environment']['python']:raise ValueError('Python version changed.')
    for name,expected in p['environment']['packages'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError(f'Frozen package changed: {name}')
    return p


def eligible_session(p,instant):
    if instant.tzinfo is None:raise ValueError('Timezone-aware UTC clock required.')
    local=instant.astimezone(NY);date=local.date().isoformat()
    if date not in {s['date'] for s in p['sessions']} or local.hour<17:return None
    return date


def download_snapshot(assets,start,date,directory):
    import yfinance as yf
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    yf.set_tz_cache_location(str(directory/'provider-cache'))
    requested=now()
    raw=yf.download(assets,start=start,end=str((pd.Timestamp(date)+pd.Timedelta(days=1)).date()),
                    auto_adjust=False,actions=True,repair=False,keepna=True,threads=False,progress=False)
    received=now()
    raw.rename_axis('Date').to_csv(directory/'provider.csv')
    rows=[]
    for asset in assets:
        block=raw.xs(asset,axis=1,level='Ticker')
        if 'Adj Close' not in block:raise ValueError('Adjusted close absent.')
        factor=block['Adj Close']/block['Close']
        if not np.isfinite(factor).all() or (factor<=0).any():raise ValueError('Invalid corporate-action adjustment.')
        for t,row in block.iterrows():
            rows.append(dict(Date=pd.Timestamp(t).tz_localize(None).normalize(),asset=asset,
                             **{field:float(row[field]*factor.loc[t]) for field in ['Open','High','Low','Close']}))
    frame=validate_ohlc(pd.DataFrame(rows),assets)
    if str(frame.Date.max().date())!=date:raise ValueError('Snapshot does not end at the scheduled session.')
    frame.to_csv(directory/'ohlc.csv',index=False)
    write_once(directory/'source.json',dict(provider='Yahoo Finance via yfinance',version=yf.__version__,
               requested_at_utc=requested.isoformat(),received_at_utc=received.isoformat(),
               requested_start=start,requested_end_exclusive=str((pd.Timestamp(date)+pd.Timedelta(days=1)).date()),
               raw_sha256=sha(directory/'provider.csv'),adjusted_sha256=sha(directory/'ohlc.csv')))
    return frame,received


def load_cached_fit(directory, refit_date, panel):
    prior=Path(directory)/'forecasts'/refit_date/panel
    status=json.loads((prior/'status.json').read_text())
    if status['status']!='eligible':raise ValueError('Required scheduled refit failed.')
    if sha(prior/'fit.json')!=status['fit_sha256']:raise ValueError('Archived fitted state changed.')
    if status['protocol_sha256']!=sha(Path(directory)/'protocol.json'):raise ValueError('Fit belongs to a different protocol.')
    return restore_fit(json.loads((prior/'fit.json').read_text()))


def tick(directory):
    """One bounded collection pass; no sleep loop and no automatic remote writes."""
    directory=Path(directory);p=read_protocol(directory);date=eligible_session(p,now())
    if date is None:
        print('Outside the fixed collection window; nothing downloaded.');return
    dates=[s['date'] for s in p['sessions']]
    assets=[a for panel in p['panels'].values() for a in panel]
    session=directory/'sessions'/date;session.mkdir(parents=True,exist_ok=True)
    capture=session/'capture.json'
    if not capture.exists():
        stamp=now().strftime('%Y%m%dT%H%M%S%fZ')
        attempt=session/'attempts'/stamp
        start=p['history_start'] if date in p['origin_dates'] else str((pd.Timestamp(date)-pd.Timedelta(days=10)).date())
        try:
            frame,received=download_snapshot(assets,start,date,attempt)
            if eligible_session(p,received)!=date:raise ValueError('Snapshot received after the capture window.')
            measurements=daily_measurements(frame,assets)
            last=measurements[measurements.Date==pd.Timestamp(date)]
            if len(last)!=len(assets):raise ValueError('Missing previous close.')
            position=dates.index(date)
            if position and str(pd.Timestamp(frame.Date.unique()[-2]).date())!=dates[position-1]:
                raise ValueError('Previous close is not the previous scheduled session.')
            write_once(attempt/'measurements.json',last.to_dict('records'))
            write_once(capture,dict(attempt=str(attempt.relative_to(directory)),received_at_utc=received.isoformat(),
                                   ohlc_sha256=sha(attempt/'ohlc.csv'),measurements_sha256=sha(attempt/'measurements.json'),
                                   protocol_sha256=sha(directory/'protocol.json')))
        except Exception as exc:
            write_once(session/'failures'/f'{stamp}.json',dict(at_utc=now().isoformat(),error=f'{type(exc).__name__}: {exc}'))
            raise
    info=json.loads(capture.read_text());snapshot=directory/info['attempt']/'ohlc.csv'
    if sha(snapshot)!=info['ohlc_sha256']:raise ValueError('Captured vintage changed.')
    if info['protocol_sha256']!=sha(directory/'protocol.json'):raise ValueError('Capture belongs to a different protocol.')
    if date not in p['origin_dates']:
        print(f'Archived first target vintage for {date}.');return
    frame=pd.read_csv(snapshot);index=p['origin_dates'].index(date)
    for panel,subset in p['panels'].items():
        output=directory/'forecasts'/date/panel
        if (output/'status.json').exists():continue
        output.mkdir(parents=True,exist_ok=True)
        try:
            sample=validate_ohlc(frame[frame.asset.isin(subset)],subset)
            refit_date=p['origin_dates'][index//p['refit_every_origins']*p['refit_every_origins']]
            if date==refit_date:
                fitted=fit_frozen_model(sample,subset)
                fitted['history_dates']=[str(pd.Timestamp(t).date()) for t in sorted(sample.Date.unique())]
                write_once(output/'fit.json',fitted)
            else:
                fitted=load_cached_fit(directory,refit_date,panel)
                history=[str(pd.Timestamp(t).date()) for t in sorted(sample.Date.unique())]
                if history[:len(fitted['history_dates'])]!=fitted['history_dates']:raise ValueError('Historical session index changed between refits.')
            forecasts=forecasts_at_origin(sample,fitted,p['horizons'],p['lead_sessions'])
            forecasts.insert(0,'panel',panel);forecasts.insert(1,'origin',date)
            forecasts['last_fit']=refit_date
            forecasts['target']=forecasts.horizon.map(lambda h:dates[dates.index(date)+p['lead_sessions']+h])
            forecasts.to_csv(output/'forecasts.csv',index=False)
            completed=now();eligible=eligible_session(p,completed)==date
            write_once(output/'status.json',dict(status='eligible' if eligible else 'late',completed_at_utc=completed.isoformat(),
                       forecast_sha256=sha(output/'forecasts.csv'),capture_sha256=sha(capture),
                       fit_sha256=sha(directory/'forecasts'/refit_date/panel/'fit.json'),
                       protocol_sha256=sha(directory/'protocol.json')))
        except Exception as exc:
            write_once(output/'status.json',dict(status='failed',completed_at_utc=now().isoformat(),error=f'{type(exc).__name__}: {exc}'))
    print(json.dumps(coverage(directory,p),indent=2))


def coverage(directory,p):
    rows=[];current=now()
    for date in p['origin_dates']:
        for panel in p['panels']:
            path=Path(directory)/'forecasts'/date/panel/'status.json'
            deadline=datetime.fromisoformat(date+'T23:59:59').replace(tzinfo=NY)
            state=json.loads(path.read_text())['status'] if path.exists() else ('missed' if current>deadline else 'pending')
            rows.append(dict(origin=date,panel=panel,status=state))
    return dict(protocol_sha256=sha(Path(directory)/'protocol.json'),origins=rows,
                final_target=p['sessions'][-1]['date'],performance='Withheld until the fixed final reporting date.')


def report(directory,output):
    directory=Path(directory);p=read_protocol(directory)
    final=datetime.fromisoformat(p['report_after_utc'])
    if now()<=final:raise ValueError('Final reporting window has not opened; only coverage may be inspected.')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use an empty report directory.')
    dates=[s['date'] for s in p['sessions']];measurements=[];scored=[];missing=[]
    for date in dates:
        base=directory/'sessions'/date
        if not (base/'capture.json').exists():continue
        capture=json.loads((base/'capture.json').read_text())
        if capture['protocol_sha256']!=sha(directory/'protocol.json'):raise ValueError('Target belongs to a different protocol.')
        if eligible_session(p,datetime.fromisoformat(capture['received_at_utc']))!=date:raise ValueError('Target capture missed its declared window.')
        target_path=directory/capture['attempt']/'measurements.json'
        if sha(target_path)!=capture['measurements_sha256']:raise ValueError('Target record changed.')
        measurements.extend(json.loads(target_path.read_text()))
    data=pd.DataFrame(measurements,columns=['Date','asset','close_return','overnight_return','rs'])
    for date in p['origin_dates']:
        for panel in p['panels']:
            base=directory/'forecasts'/date/panel;status_path=base/'status.json'
            if not status_path.exists() or json.loads(status_path.read_text())['status']!='eligible':continue
            status=json.loads(status_path.read_text())
            if status['protocol_sha256']!=sha(directory/'protocol.json'):raise ValueError('Forecast belongs to a different protocol.')
            if eligible_session(p,datetime.fromisoformat(status['completed_at_utc']))!=date:raise ValueError('Ineligible forecast timestamp.')
            if sha(base/'forecasts.csv')!=status['forecast_sha256']:raise ValueError('Forecast record changed.')
            forecasts=pd.read_csv(base/'forecasts.csv')
            for h in p['horizons']:
                try:scored.append(score_forecasts(forecasts[forecasts.horizon==h],data,dates[dates.index(date)+1+p['lead_sessions']:dates.index(date)+h+1+p['lead_sessions']]))
                except ValueError as exc:missing.append(dict(origin=date,panel=panel,horizon=h,reason=str(exc)))
    output.mkdir(parents=True,exist_ok=True)
    result=pd.concat(scored,ignore_index=True) if scored else pd.DataFrame()
    result.to_csv(output/'scores.csv',index=False)
    if len(result):
        result.groupby(['panel','horizon','proxy','method']).agg(qlike=('qlike','mean'),normalized_square_error=('normalized_square_error','mean'),origins=('origin','nunique'),rows=('origin','size')).reset_index().to_csv(output/'metrics.csv',index=False)
    tests=[]
    for panel in p['panels']:
        for comparison in p['primary_comparisons']:
            block=result[(result.panel==panel)&(result.horizon==comparison['horizon'])&(result.proxy==p['primary_proxy'])] if len(result) else result
            if len(block):
                pivot=block.pivot(index=['origin','asset'],columns='method',values='qlike')
                difference=(pivot[comparison['model']]-pivot[comparison['reference']]).groupby('origin').mean().sort_index()
            else:difference=pd.Series(dtype=float)
            for lag in [4,9]:
                test=dm_test(difference.to_numpy(),lag) if list(difference.index)==p['origin_dates'] else dict(status='insufficient_coverage',p_value=np.nan)
                tests.append(dict(panel=panel,**comparison,hac_lags=lag,origins=len(difference),**test))
    tests=pd.DataFrame(tests)
    for lag in [4,9]:
        mask=tests.hac_lags==lag;tests.loc[mask,'holm_p_value']=holm(tests.loc[mask,'p_value'].to_numpy())
    tests.to_csv(output/'primary_tests.csv',index=False)
    write_once(output/'coverage.json',coverage(directory,p)|dict(missing_target_windows=missing))
    print(f'Wrote {len(result)} scored rows; all scheduled origins retained in coverage.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','tick','status','report'])
    parser.add_argument('--study',type=Path,default=ROOT/'artifacts/prospective-v1')
    parser.add_argument('--first-origin',default='2026-10-27')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/prospective-report')
    args=parser.parse_args()
    if args.command=='init':init_study(args.study,args.first_origin)
    elif args.command=='tick':tick(args.study)
    elif args.command=='status':print(json.dumps(coverage(args.study,read_protocol(args.study)),indent=2))
    else:report(args.study,args.output)


if __name__=='__main__':main()
