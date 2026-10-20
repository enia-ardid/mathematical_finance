import unittest

import numpy as np
import pandas as pd

from adaptive_forecasting import (
    CONTROLS, _choose, _interval_width, adaptive_forecast, build_features,
    evaluate_adaptive, ridge_prediction,
)
from temporal_memory import synthetic_prices


class AdaptiveForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices, _ = synthetic_prices(650)
        cls.features = build_features(cls.prices)
        cls.forecasts, cls.audits = adaptive_forecast(cls.features)

    def test_future_values_cannot_change_past_features_fits_or_choices(self):
        altered = self.prices.copy()
        cutoff = altered.index[610]
        altered.loc[cutoff:] *= np.linspace(1.2, 2., len(altered.loc[cutoff:]))[:, None]
        features = build_features(altered)
        stable = [c for c in features if c != 'actual_return']
        pd.testing.assert_frame_equal(self.features.loc[self.features.origin < cutoff, stable],
                                      features.loc[features.origin < cutoff, stable])
        forecasts, audits = adaptive_forecast(features)
        stable = [c for c in forecasts if c != 'actual_return']
        pd.testing.assert_frame_equal(self.forecasts.loc[self.forecasts.origin < cutoff, stable],
                                      forecasts.loc[forecasts.origin < cutoff, stable])
        before = [a for a in self.audits if a['origin'] < str(cutoff.date())]
        after = [a for a in audits if a['origin'] < str(cutoff.date())]
        self.assertEqual(before, after)

    def test_every_fit_uses_only_matured_labels(self):
        fitted = 0
        for audit in self.audits:
            for fit in audit['fits'].values():
                fitted += 1
                self.assertLessEqual(fit['last_training_target'], audit['origin'])
                self.assertLess(fit['last_training_origin'], audit['origin'])
        self.assertGreater(fitted, 0)

    def test_sector_features_do_not_use_future_cross_asset_information(self):
        _, sectors = synthetic_prices(320)
        before = build_features(sectors, sector_model=True)
        altered = sectors.copy()
        cutoff = altered.index[300]
        altered.loc[cutoff:] *= np.linspace(1, 3, len(altered.columns))
        after = build_features(altered, sector_model=True)
        columns = [c for c in before if c != 'actual_return']
        pd.testing.assert_frame_equal(before.loc[before.origin < cutoff, columns],
                                      after.loc[after.origin < cutoff, columns])

    def test_recent_constant_prices_produce_finite_features(self):
        prices = self.prices.iloc[:300].copy()
        prices.iloc[-30:] = prices.iloc[-31].to_numpy()
        features = build_features(prices)
        self.assertTrue(np.isfinite(features[list(CONTROLS)]).all().all())

    def test_ridge_learns_a_signed_memory_correction(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=500)
        train = pd.DataFrame({'memory_0': x, 'drift': .001, 'actual_return': .001 - .02 * x})
        current = pd.DataFrame({'memory_0': [1., -1.], 'drift': [.001, .001]})
        prediction, fitted = ridge_prediction(train, current, ['memory_0'], .01)
        np.testing.assert_allclose(prediction, [-.019, .021], atol=.0003)
        self.assertLess(fitted['coefficients'][0], 0)

    def test_selector_excludes_overlapping_unobserved_targets(self):
        history = [{'position': i, 'target_position': i + 21, 'actual': np.array([.01]),
                    'predictions': {'constant': np.array([0.]), 'drift': np.array([.01])}}
                   for i in range(30)]
        # This very bad drift outcome has not happened yet at origin 50.
        history.append({'position': 49, 'target_position': 70, 'actual': np.array([0.]),
                        'predictions': {'constant': np.array([0.]), 'drift': np.array([100.])}})
        selected, count, scores = _choose(history, 50, ['constant', 'drift'])
        self.assertEqual(selected, 'drift')
        self.assertEqual(count, 30)
        self.assertEqual(scores['drift'], 0)

    def test_intervals_ignore_unobserved_errors(self):
        history = [{'position': i, 'target_position': i + 21, 'actual': np.array([.02]),
                    'adaptive': np.array([0.]), 'scale': np.array([.01])} for i in range(30)]
        history.append({'position': 49, 'target_position': 70, 'actual': np.array([100.]),
                        'adaptive': np.array([0.]), 'scale': np.array([.01])})
        width, count, q = _interval_width(history, 50, np.array([.03]))
        self.assertEqual(count, 30)
        self.assertEqual(q, 2.)
        np.testing.assert_allclose(width, [.06])

    def test_latest_is_unscored_and_horizons_share_evaluation_origins(self):
        latest = self.forecasts[self.forecasts.origin == self.prices.index[-1]]
        self.assertEqual(set(latest.horizon), {1, 5, 21})
        self.assertTrue(latest.actual_return.isna().all())
        self.assertTrue(latest.target.isna().all())
        metrics, intervals, scored = evaluate_adaptive(self.forecasts, self.prices.index[580])
        origins = [set(b.origin) for _, b in scored.groupby('horizon')]
        self.assertTrue(all(o == origins[0] for o in origins))
        self.assertFalse(scored.actual_return.isna().any())
        self.assertTrue(np.isfinite(metrics.relative_RMSE_percent).all())
        self.assertTrue(intervals.coverage.between(0, 1).all())
        self.assertTrue((intervals.interval_score_percent >= intervals.mean_width_percent).all())

    def test_interval_score_penalizes_missed_targets(self):
        frame = self.forecasts.copy()
        frame['actual_return'] = .1
        frame['lower'], frame['upper'] = -.02, .02
        _, intervals, _ = evaluate_adaptive(frame, frame.origin.min())
        score = intervals[intervals.method == 'adaptive'].interval_score_percent
        np.testing.assert_allclose(score, 84.)  # 100 * (.04 + 2/.2 * .08)

    def test_forecast_head_has_share_unit_invariance(self):
        scaled = self.features.copy()
        scaled['anchor'] *= 100
        after, audits = adaptive_forecast(scaled)
        columns = [c for c in after if c != 'anchor']
        pd.testing.assert_frame_equal(after[columns], self.forecasts[columns])
        self.assertEqual(audits, self.audits)


if __name__ == '__main__':
    unittest.main()
