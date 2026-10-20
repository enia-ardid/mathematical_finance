import unittest

import numpy as np
import pandas as pd

from forecasting import conditional_moments, forecast_from_history, forecast_metrics, rolling_forecast, select_calibration
from temporal_memory import fit_parameters, simulate_returns, synthetic_prices


class ConditionalForecastTests(unittest.TestCase):
    def test_conditional_mean_continues_the_unchanged_recurrence(self):
        alpha = np.array([.4, .6])
        mu = np.array([.001, -.0005])
        history = simulate_returns([.01, -.02], np.tile(mu, (8, 1)), np.zeros((8, 2)), alpha, .12, 3, True)
        full = simulate_returns([.01, -.02], np.tile(mu, (13, 1)), np.zeros((13, 2)), alpha, .12, 3, True)
        mean, variance = conditional_moments(history, {'alpha': alpha, 'mu': mu, 'sigma': np.zeros(2)}, .12, 5, True)
        np.testing.assert_allclose(mean, full[8:].cumsum(axis=0), atol=1e-15)
        np.testing.assert_array_equal(variance, np.zeros_like(variance))

    def test_two_step_variance_has_cross_sector_innovation_effects(self):
        sigma = np.array([.02, .03])
        params = {'alpha': [.5, .7], 'mu': [0, 0], 'sigma': sigma}
        _, variance = conditional_moments(np.zeros((5, 2)), params, .1, 2, True)
        # With two reflected nodes, B1 = [[.8,.2],[.2,.8]]. The cumulative
        # second-step innovation response is I+B1, plus a fresh innovation.
        cumulative_response = np.array([[1.8, .2], [.2, 1.8]])
        expected = sigma**2 + (cumulative_response**2) @ (sigma**2)
        np.testing.assert_allclose(variance[0], sigma**2)
        np.testing.assert_allclose(variance[1], expected)

    def test_one_step_price_mean_includes_lognormal_correction(self):
        dates = pd.date_range('2020-01-01', periods=3)
        prices = pd.DataFrame({'A': [100., 101., 102.]}, index=dates)
        params = {'order': ['A'], 'alpha': [.5], 'mu': [.001], 'sigma': [.02]}
        r = np.diff(np.log(prices['A']))
        expected_log_return = .001 + r[-1] - .5 * r[-2]
        forecast = forecast_from_history(prices, params, 0., 1, False)
        self.assertAlmostEqual(forecast['memory'][0, 0], 102 * np.exp(expected_log_return + .5 * .02**2))

    def test_forecast_prefix_does_not_depend_on_requested_horizon(self):
        prices, _ = synthetic_prices(360)
        params = fit_parameters(prices.iloc[:200], False)
        short = forecast_from_history(prices.iloc[:240], params, 0., 5, False)
        long = forecast_from_history(prices.iloc[:240], params, 0., 21, False)
        for key in short:
            np.testing.assert_array_equal(short[key], long[key][:5])


class ForecastEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp, cls.sectors = synthetic_prices(360)

    def test_future_changes_cannot_change_earlier_forecasts_or_calibration(self):
        for sector, prices in [(False, self.sp), (True, self.sectors)]:
            with self.subTest(sector=sector):
                before, calibration, _ = rolling_forecast(prices, sector)
                change_date = prices.index[280]
                altered = prices.copy()
                altered.loc[change_date:] *= np.linspace(2, 5, len(altered.loc[change_date:]))[:, None]
                after, changed_calibration, _ = rolling_forecast(altered, sector)
                pd.testing.assert_frame_equal(calibration.choices, changed_calibration.choices)
                pd.testing.assert_frame_equal(calibration.scores, changed_calibration.scores)
                columns = ['origin', 'target', 'asset', 'horizon', 'memory', 'calibrated', 'drift', 'constant', 'kappa']
                pd.testing.assert_frame_equal(before.loc[before.origin < change_date, columns],
                                              after.loc[after.origin < change_date, columns])
                self.assertTrue((before.last_fit <= before.origin).all())
                self.assertTrue((before.origin < before.target).all())
                self.assertTrue((before.target > calibration.development_end).all())

    def test_every_horizon_has_the_same_origins_and_baseline_anchor(self):
        forecasts, _, _ = rolling_forecast(self.sp)
        origins = [tuple(frame.origin) for _, frame in forecasts.groupby('horizon')]
        self.assertTrue(all(value == origins[0] for value in origins))
        for row in forecasts.itertuples():
            self.assertEqual(row.constant, self.sp.loc[row.origin, 'SP500'])
            self.assertEqual(row.actual, self.sp.loc[row.target, 'SP500'])
            self.assertEqual(self.sp.index.get_loc(row.target) - self.sp.index.get_loc(row.origin), row.horizon)

    def test_calibration_bounds_and_validation_cutoff(self):
        development = self.sectors.iloc[:260]
        calibration = select_calibration(development, True)
        self.assertTrue(calibration.scores.memory_weight.between(0, 1).all())
        self.assertTrue((calibration.scores.validation_mse <= calibration.scores.drift_mse + 1e-15).all())
        for origin in calibration.validation_origins:
            self.assertLessEqual(development.index.get_loc(origin) + 21, len(development) - 1)

    def test_normalized_error_and_baseline_skill_are_scale_invariant(self):
        forecasts, _, _ = rolling_forecast(self.sp)
        before = forecast_metrics(forecasts)
        scaled = forecasts.copy()
        columns = ['anchor', 'actual', 'memory', 'calibrated', 'drift', 'constant', 'memory_lower', 'memory_upper']
        scaled[columns] *= 1000
        after = forecast_metrics(scaled)
        np.testing.assert_allclose(before.relative_RMSE_percent, after.relative_RMSE_percent)
        np.testing.assert_allclose(before.skill_vs_constant, after.skill_vs_constant, atol=1e-13)
        self.assertTrue((before.xs('constant', level='method').skill_vs_constant == 0).all())

    def test_insufficient_validation_history_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'three complete'):
            select_calibration(self.sp.iloc[:130])


if __name__ == '__main__':
    unittest.main()
