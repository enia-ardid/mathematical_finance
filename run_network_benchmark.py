"""Add a separately recorded Network GARCH comparison to frozen ETF forecasts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from network_benchmark import append_network_benchmarks, STARTS
from run_paper import summarize, paired_losses
from temporal_memory import load_prices

LABELS = {'persistent_asymmetric': 'Fractional', 'equal_combination': 'Fixed 50:50 ensemble',
          'gjr_garch': 'GJR-GARCH', 'figarch': 'FIGARCH', 'network_garch': 'Network GARCH (weighted)',
          'network_garch_binary': 'Network GARCH (binary)', 'pooled_garch': 'Pooled GARCH'}
SOURCES = ['network_benchmark.py', 'run_network_benchmark.py', 'tests/test_network_benchmark.py',
           'fractional_diffusion.py', 'refined_fractional.py', 'temporal_memory.py',
           'run_paper.py', 'requirements.txt']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(args):
    result = {'forecasts': args.base_study/'forecasts.csv',
              'us_fit_audit': args.us_study/'fit_audit.json',
              'international_fit_audit': args.base_study/'international_fit_audit.json',
              'international_prices': args.external_dir/'prices.csv',
              'international_source': args.external_dir/'source.json',
              'external_summary': args.base_study/'summary.json'}
    result.update({f'us_raw/{p.name}':p for p in sorted((args.us_data_dir/'raw').glob('*.csv'))})
    if not any(k.startswith('us_raw/') for k in result):
        raise ValueError('US raw inputs are missing.')
    return result


def record_protocol(args):
    if args.protocol.exists():
        raise FileExistsError('The protocol already exists; do not rewrite a recorded study.')
    protocol = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(), review_date='2026-10-22',
        status='Follow-up comparison specified after inspecting earlier ETF results, before fitting these network benchmarks. Exploratory; not an independent test or registered preregistration.',
        source='Zhou, Li, Pan and Wang (2020), Network GARCH Model, equation (2.2), DOI 10.5705/ss.202018.0234.',
        model='h[t] = omega + alpha*q[t-1] + lambda*W*q[t-1] + beta*h[t-1]. Shared coefficients in standardized asset units; marginal Gaussian QMLE.',
        graphs={'network_garch':'Row-normalized positive squared shrinkage correlations from the common trailing 1008-return fit; diagonal zero; isolated rows zero.',
                'network_garch_binary':'Same fitted correlations thresholded at >= 0.5, diagonal zero, then row-normalized; fixed sensitivity analysis.',
                'pooled_garch':'Identical shared-parameter fit without a network term; lambda fixed to zero.'},
        adaptations='Weighted estimated graph in the primary comparison instead of binary shareholder links; common fitted mean/scale; unit initial variance. Binary sensitivity also uses estimated correlations. No transfer of fixed-exogenous-network asymptotic guarantees to this experiment.',
        fitting=dict(window=1008, initialization='h[0]=1 in normalized units, fixed while filtering observed returns between refits.',
                     starts=STARTS, transform='omega=exp(theta0); feedback weights=.999*softmax(logits plus zero slack).',
                     optimizer='L-BFGS-B, analytic gradient, maxiter500, ftol1e-10, gtol1e-7, log-intercept bounds[-12,4], logit bounds[-12,12].',
                     selection='Lowest finite training objective across all three starts; all attempts and convergence flags retained. Abort if none is finite; no substitution.'),
        evaluation='US sectors and eight country ETFs only; retain original 186 origins per panel, refit dates, scales, targets and 1/5/21-session horizons. No new market download, model tuning or change to existing forecasts.',
        scores='Mean normalized QLIKE, normalized RMSE, annual and per-asset summaries; paired whole-panel circular bootstrap, 2000 draws, seed1729, blocks5/10 origins, nominal 95% ranges.',
        stopping_rule='Run the recorded design once and report every benchmark regardless of ranking. Any required computational correction must be recorded separately.',
        input_sha256={k:digest(p) for k,p in inputs(args).items()},
        code_sha256={p:digest(p) for p in SOURCES})
    args.protocol.write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
    print(f'Recorded protocol: {args.protocol}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-study',type=Path,default=Path('artifacts/external-study-complete'))
    parser.add_argument('--us-study',type=Path,default=Path('artifacts/refinement-study'))
    parser.add_argument('--us-data-dir',type=Path,default=Path('artifacts/yahoo-through-2026-10-16'))
    parser.add_argument('--external-dir',type=Path,default=Path('artifacts/yahoo-international-through-2026-10-16'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/network-study'))
    parser.add_argument('--protocol',type=Path,default=Path('paper/network_protocol.json'))
    parser.add_argument('--record-protocol',action='store_true')
    parser.add_argument('--publish-assets',action='store_true')
    args=parser.parse_args()
    if args.record_protocol:
        record_protocol(args)
        return
    protocol=json.loads(args.protocol.read_text())
    if {k:digest(p) for k,p in inputs(args).items()} != protocol['input_sha256']:
        raise ValueError('Inputs differ from the recorded snapshot.')
    for path,expected in protocol['code_sha256'].items():
        if digest(path)!=expected:
            raise ValueError(f'Source differs from the recorded design: {path}')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; preserve earlier runs.')
    args.output.mkdir(parents=True)
    original=pd.read_csv(args.base_study/'forecasts.csv',parse_dates=['origin','target','last_fit'])
    us_audit=json.loads((args.us_study/'fit_audit.json').read_text())
    international_audit=json.loads((args.base_study/'international_fit_audit.json').read_text())
    _,sectors=load_prices(args.us_data_dir)
    countries=pd.read_csv(args.external_dir/'prices.csv',index_col='Date',parse_dates=['Date'])
    frames,records=[],{}
    for panel,prices,audit in [('sectors',sectors,us_audit['sectors']),('international',countries,international_audit)]:
        print(f'Fitting {panel}: weighted network, binary network and pooled control...',flush=True)
        reference=original[(original.panel==panel)&original.method.isin(LABELS)].copy()
        frame,records[panel]=append_network_benchmarks(prices,reference,audit)
        pd.testing.assert_frame_equal(frame.iloc[:len(reference)].reset_index(drop=True),reference.reset_index(drop=True))
        frame.to_csv(args.output/f'{panel}_forecasts.csv',index=False)
        frames.append(frame)
        print(f'{panel}: {frame.origin.nunique()} origins, {len(records[panel])} refits.',flush=True)
    forecasts=pd.concat(frames,ignore_index=True)
    metrics=summarize(forecasts)
    comparisons=[]
    for model in ['persistent_asymmetric','equal_combination','network_garch','network_garch_binary']:
        renamed=forecasts.assign(method=forecasts.method.replace({model:'fractional_diffusion'}))
        comparisons.append(paired_losses(renamed).assign(model=model))
    comparison=pd.concat(comparisons,ignore_index=True)
    assets=forecasts.groupby(['panel','asset','horizon','method']).agg(qlike=('qlike','mean'),origins=('origin','nunique')).reset_index()
    selected=[fit for panel in records.values() for record in panel for fit in record['fits'].values()]
    summary=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest(args.protocol),
                 input_sha256=protocol['input_sha256'],code_sha256=protocol['code_sha256'],
                 fitted_models=len(selected),nonconverged=sum(not x['converged'] for x in selected),
                 failed_starts=sum(not a['converged'] for x in selected for a in x['attempts']),
                 dates={p:dict(origins=int(b.origin.nunique()),first_origin=str(b.origin.min().date()),
                               last_origin=str(b.origin.max().date()),last_target=str(b.target.max().date()))
                        for p,b in forecasts.groupby('panel')},
                 prior_forecasts='All reused forecast rows are unchanged; the earlier study files are read-only inputs.')
    forecasts.to_csv(args.output/'forecasts.csv',index=False)
    metrics.to_csv(args.output/'metrics.csv',index=False)
    comparison.to_csv(args.output/'comparisons.csv',index=False)
    assets.to_csv(args.output/'asset_metrics.csv',index=False)
    (args.output/'fit_audit.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if args.publish_assets:
        for name in ['metrics.csv','comparisons.csv','asset_metrics.csv','summary.json']:
            shutil.copyfile(args.output/name,Path('paper')/('network_'+name))
        table=metrics[metrics.year=='all'].pivot(index='method',columns=['panel','horizon'],values='qlike')
        table=table.reindex(index=list(LABELS),columns=pd.MultiIndex.from_product([['sectors','international'],[1,5,21]]))
        table.index=[LABELS[m] for m in table.index]
        table.to_latex('paper/tables/network_scores.tex',float_format='%.4f',multicolumn_format='c')
    print(metrics[metrics.year=='all'].pivot(index=['panel','horizon'],columns='method',values='qlike').round(5).to_string(),flush=True)
    print(f"Selected fits: {summary['fitted_models']}; nonconverged: {summary['nonconverged']}; failed starts: {summary['failed_starts']}.",flush=True)


if __name__=='__main__':
    main()
