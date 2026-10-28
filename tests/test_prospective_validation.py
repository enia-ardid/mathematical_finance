import json
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from diffusion_risk import diffusion_risk,spectral_risk
from fractional_diffusion import diffusion_operator
from prospective_validation import daily_measurements,validate_ohlc,score_forecasts,forecasts_at_origin
from run_prospective import calendar,eligible_session,write_once,report,read_protocol,tick,sha,load_cached_fit


class DiffusionRiskTests(unittest.TestCase):
    def test_spectral_identity_with_noncommuting_noise(self):
        graph=np.array([[1.,-1,0],[-1,2,-1],[0,-1,1]])
        covariance=np.array([[.2,.05,0],[.05,.4,.1],[0,.1,.3]])
        signal=np.array([1.,.1,2.])
        for strength in [0.,.2,1.,8.]:
            analytic=diffusion_risk(signal,covariance,graph,strength)
            spectral=spectral_risk(signal,covariance,graph,strength)
            self.assertAlmostEqual(analytic['risk_difference'],spectral['risk_difference'].sum())

    def test_smoothing_can_help_or_hurt(self):
        graph=np.array([[1.,-1.],[-1.,1.]])
        noise=.01*np.eye(2)
        self.assertLess(diffusion_risk([1,1],noise,graph,1)['risk_difference'],0)
        self.assertGreater(diffusion_risk([.1,1.9],noise,graph,1)['risk_difference'],0)
        self.assertAlmostEqual(diffusion_risk([.1,1.9],noise,graph,0)['risk_difference'],0)
        with self.assertRaises(ValueError):diffusion_risk([1,1],-noise,graph,1)

    def test_correlated_monte_carlo_agrees_with_exact_risk(self):
        graph=np.array([[1.,-1.],[-1.,1.]])
        mean=np.array([.8,1.2]);covariance=np.array([[.1,.05],[.05,.2]])
        x=np.random.default_rng(19).multivariate_normal(mean,covariance,size=80000)
        difference=((x@diffusion_operator(graph,.5).T-mean)**2).sum(axis=1)-((x-mean)**2).sum(axis=1)
        expected=diffusion_risk(mean,covariance,graph,.5)['risk_difference']
        self.assertLess(abs(difference.mean()-expected),6*difference.std(ddof=1)/np.sqrt(len(x)))


class ProspectiveTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(Date=d,asset='A',Open=o,High=h,Low=l,Close=c)
                             for d,o,h,l,c in [('2026-10-26',100,103,99,101),
                                              ('2026-10-27',102,106,100,104),
                                              ('2026-10-28',104,108,102,106)]])

    def test_ohlc_measurement_and_centered_gap(self):
        data=daily_measurements(self.fixture(),['A'])
        rs=np.log(106/102)*np.log(106/104)+np.log(100/102)*np.log(100/104)
        self.assertAlmostEqual(data.rs.iloc[0],rs)
        forecasts=pd.DataFrame([dict(asset='A',horizon=1,variance=.01,reference_scale=.02,mean=.001,overnight_mean=.002)])
        scores=score_forecasts(forecasts,data,['2026-10-27']).set_index('proxy')
        self.assertAlmostEqual(scores.loc['overnight_rs','realized_proxy'],(np.log(102/101)-.002)**2+rs)
        self.assertAlmostEqual(scores.loc['close_squared','realized_proxy'],(np.log(104/101)-.001)**2)
        scaled=self.fixture();scaled[['Open','High','Low','Close']]*=2
        pd.testing.assert_frame_equal(data,daily_measurements(scaled,['A']))

    def test_invalid_data_and_missing_session_are_not_repaired(self):
        invalid=self.fixture();invalid.loc[1,'High']=90
        with self.assertRaises(ValueError):validate_ohlc(invalid,['A'])
        with self.assertRaises(ValueError):validate_ohlc(pd.concat([self.fixture(),self.fixture().iloc[:1]]),['A'])
        data=daily_measurements(self.fixture(),['A']).iloc[1:]
        forecasts=pd.DataFrame([dict(asset='A',horizon=1,variance=.01,reference_scale=.02,mean=0.,overnight_mean=0.)])
        with self.assertRaises(ValueError):score_forecasts(forecasts,data,['2026-10-27'])

    def test_calendar_deadlines_dst_and_holidays(self):
        sessions=calendar('2026-10-27');p={'sessions':sessions}
        dates=[s['date'] for s in sessions]
        self.assertEqual(len(sessions),278)
        self.assertNotIn('2026-11-26',dates)
        self.assertNotIn('2027-03-26',dates)
        bydate={s['date']:s for s in sessions}
        self.assertEqual(bydate['2026-11-27']['close_utc'],'2026-11-27T18:00:00+00:00')
        self.assertEqual(bydate['2026-10-27']['open_utc'],'2026-10-27T13:30:00+00:00')
        self.assertEqual(bydate['2026-11-02']['open_utc'],'2026-11-02T14:30:00+00:00')
        self.assertIsNone(eligible_session(p,datetime(2026,10,27,20,59,tzinfo=timezone.utc)))
        self.assertEqual(eligible_session(p,datetime(2026,10,27,21,1,tzinfo=timezone.utc)),'2026-10-27')
        self.assertIsNone(eligible_session(p,datetime(2026,10,28,4,1,tzinfo=timezone.utc)))

    def test_records_are_immutable_and_early_ranking_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.json';write_once(path,{'value':1})
            with self.assertRaises(FileExistsError):write_once(path,{'value':2})
            self.assertEqual(json.loads(path.read_text()),{'value':1})
            with patch('run_prospective.read_protocol',return_value={'report_after_utc':'2099-01-01T00:00:00+00:00'}):
                with self.assertRaises(ValueError):report(folder,Path(folder)/'results')

    def test_complete_collection_is_idempotent_and_keeps_first_vintage(self):
        p=dict(sessions=calendar('2026-10-27'),origin_dates=['2026-10-27'],
               panels={'test':['A']},history_start='2019-01-01',refit_every_origins=5,horizons=[1],lead_sessions=1)
        sample=self.fixture().iloc[:2].copy()
        current=datetime(2026,10,27,21,10,tzinfo=timezone.utc)
        forecast=pd.DataFrame([dict(asset='A',method='persistent_asymmetric',horizon=1,
                                   variance=.01,reference_scale=.02,mean=0.,overnight_mean=0.)])
        def download(assets,start,date,directory):
            directory.mkdir(parents=True)
            sample.to_csv(directory/'ohlc.csv',index=False)
            return validate_ohlc(sample,assets),current
        with tempfile.TemporaryDirectory() as folder:
            study=Path(folder);write_once(study/'protocol.json',p)
            with patch('run_prospective.read_protocol',return_value=p), patch('run_prospective.now',return_value=current), \
                 patch('run_prospective.download_snapshot',side_effect=download) as downloader, \
                 patch('run_prospective.fit_frozen_model',return_value={'assets':['A']}) as fitter, \
                 patch('run_prospective.forecasts_at_origin',return_value=forecast),patch('builtins.print'):
                tick(study);before=(study/'sessions/2026-10-27/capture.json').read_bytes();tick(study)
                self.assertEqual(downloader.call_count,1);self.assertEqual(fitter.call_count,1)
                self.assertEqual(before,(study/'sessions/2026-10-27/capture.json').read_bytes())
            status=json.loads((study/'forecasts/2026-10-27/test/status.json').read_text())
            self.assertEqual(status['status'],'eligible')
            stored=pd.read_csv(study/'forecasts/2026-10-27/test/forecasts.csv')
            self.assertEqual(stored.target.iloc[0],'2026-10-29')

    def test_frozen_protocol_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'protocol.json';write_once(path,{'original':True})
            write_once(Path(folder)/'freeze.json',{'protocol_sha256':sha(path)})
            path.write_text('{"original":false}')
            with self.assertRaisesRegex(ValueError,'Frozen protocol changed'):read_protocol(folder)

    def test_cached_fit_cannot_be_changed_between_origins(self):
        with tempfile.TemporaryDirectory() as folder:
            study=Path(folder);write_once(study/'protocol.json',{'study':'fixed'})
            base=study/'forecasts/2026-10-27/test'
            fit={key:[1.] for key in ['mean','scale','graph','correlation','overnight_mean']}
            write_once(base/'fit.json',fit)
            write_once(base/'status.json',dict(status='eligible',fit_sha256=sha(base/'fit.json'),protocol_sha256=sha(study/'protocol.json')))
            self.assertEqual(load_cached_fit(study,'2026-10-27','test')['mean'][0],1.)
            (base/'fit.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Archived fitted state changed'):
                load_cached_fit(study,'2026-10-27','test')

    def test_embargo_uses_steps_two_through_h_plus_one(self):
        fit=dict(assets=['A'],mean=np.zeros(1),scale=np.ones(1),overnight_mean=np.zeros(1),
                 refined={'persistent_asymmetric':{'diffusion':0.}})
        path=np.arange(1.,23.)[:,None]
        daily={name:path for name in ['persistent_asymmetric','asymmetric_no_graph','asymmetric_exponential','gjr_garch']}
        with patch('prospective_validation.refined_variances',return_value=daily), \
             patch('prospective_validation.forecast_choice',return_value=path), \
             patch('prospective_validation.conditional_variances',return_value={'ewma':path}):
            forecasts=forecasts_at_origin(self.fixture(),fit)
        gjr=forecasts[forecasts.method=='gjr_garch'].set_index('horizon')
        self.assertEqual(gjr.loc[1,'variance'],2.)
        self.assertEqual(gjr.loc[5,'variance'],20.)
        self.assertEqual(gjr.loc[21,'variance'],252.)

    def test_late_fit_is_archived_but_ineligible(self):
        p=dict(sessions=calendar('2026-10-27'),origin_dates=['2026-10-27'],
               panels={'test':['A']},history_start='2019-01-01',refit_every_origins=5,horizons=[1],lead_sessions=1)
        sample=self.fixture().iloc[:2].copy()
        clock=[datetime(2026,10,27,21,10,tzinfo=timezone.utc)]
        def download(assets,start,date,directory):
            directory.mkdir(parents=True);sample.to_csv(directory/'ohlc.csv',index=False)
            return validate_ohlc(sample,assets),clock[0]
        def fit(*args):
            clock[0]=datetime(2026,10,28,4,1,tzinfo=timezone.utc)
            return {'assets':['A']}
        prediction=pd.DataFrame([dict(asset='A',method='persistent_asymmetric',horizon=1,
                                      variance=.01,reference_scale=.02,mean=0.,overnight_mean=0.)])
        with tempfile.TemporaryDirectory() as folder:
            study=Path(folder);write_once(study/'protocol.json',p)
            with patch('run_prospective.read_protocol',return_value=p),patch('run_prospective.now',side_effect=lambda:clock[0]), \
                 patch('run_prospective.download_snapshot',side_effect=download),patch('run_prospective.fit_frozen_model',side_effect=fit), \
                 patch('run_prospective.forecasts_at_origin',return_value=prediction),patch('builtins.print'):
                tick(study)
            status=json.loads((study/'forecasts/2026-10-27/test/status.json').read_text())
            self.assertEqual(status['status'],'late')
            self.assertTrue((study/'forecasts/2026-10-27/test/forecasts.csv').exists())

    def test_paired_ablation_changes_only_diffusion(self):
        rng=np.random.default_rng(81)
        closes=100*np.exp(np.cumsum(rng.normal(0,.01,(530,2)),axis=0))
        rows=[]
        for i,date in enumerate(pd.bdate_range('2020-01-01',periods=len(closes))):
            for j,asset in enumerate(['A','B']):
                rows.append(dict(Date=date,asset=asset,Open=closes[i,j],Close=closes[i,j],High=closes[i,j]*1.01,Low=closes[i,j]*.99))
        choice=dict(family='fractional',kernel_parameter=.4,feedback=.95,smoothing=.5,diffusion=0.,asymmetry=1.)
        fitted=dict(assets=['A','B'],mean=np.zeros(2),scale=np.full(2,.0001),graph=np.array([[1.,-1],[-1,1]]),
                    start=0,overnight_mean=np.zeros(2),refined={'persistent_asymmetric':choice,'asymmetric_no_graph':choice,
                    'asymmetric_exponential':dict(choice,family='exponential',kernel_parameter=.94)},
                    gjr=[{'parameters':(.05,.05,.85,.1)}]*2)
        # EWMA is irrelevant to the equality under test; isolate that benchmark call.
        with patch('prospective_validation.conditional_variances',return_value={'ewma':np.full((22,2),.0001)}):
            frame=forecasts_at_origin(pd.DataFrame(rows),fitted)
        selected=frame[frame.method=='persistent_asymmetric'].variance.to_numpy()
        ablation=frame[frame.method=='matched_no_graph'].variance.to_numpy()
        np.testing.assert_array_equal(selected,ablation)
        self.assertEqual(choice['diffusion'],0.)


if __name__=='__main__':unittest.main()
