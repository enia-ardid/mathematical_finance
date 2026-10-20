import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_memory import (
    SECTORS, buy_and_hold, calibrated_path, estimate_hurst, evaluate,
    fit_parameters, grunwald_weights, laplacian, load_prices, metrics,
    run_model, simulate_returns, synthetic_prices, validate_prices,
)


class NumericalTests(unittest.TestCase):
    def test_binomial_coefficients(self):
        np.testing.assert_allclose(grunwald_weights(.5, 4), [1, -.5, -.125, -.0625])
        np.testing.assert_array_equal(grunwald_weights(1, 4), [1, -1, 0, 0])

    def test_recurrence_is_not_a_gl_differential_equation(self):
        # At alpha=1, D r = 1 with r0=0 gives [0, 1, 2]. The project's
        # shifted plus-sign recurrence instead gives [0, 1, 2, 2].
        observed = simulate_returns([0], np.ones((4, 1)), np.zeros((4, 1)), [1], 0, 0, False)
        np.testing.assert_allclose(observed.ravel(), [0, 1, 2, 2])
        self.assertNotEqual(observed[-1, 0], 3.)

    def test_reflected_boundaries_and_single_asset(self):
        np.testing.assert_allclose(laplacian([1, 2, 4]), [2, 1, -4])
        np.testing.assert_allclose(laplacian([3, 3, 3]), [0, 0, 0])
        np.testing.assert_allclose(laplacian([7]), [0])

    def test_standard_smape_factor(self):
        self.assertAlmostEqual(metrics([1., 1.], [3., 3.])['sMAPE_percent'], 100.)
        self.assertEqual(metrics([0., 0.], [0., 0.])['sMAPE_percent'], 0.)

    def test_metrics_reject_broadcasting(self):
        with self.assertRaises(ValueError):
            metrics(np.ones(3), np.ones((3, 1)))

    def test_weights_reject_invalid_order(self):
        for alpha in [0., -1., 1.1, np.nan]:
            with self.assertRaises(ValueError):
                grunwald_weights(alpha, 10)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp, cls.sectors = synthetic_prices(240)

    def test_held_out_values_cannot_change_predictions_or_fit(self):
        for sector_model, prices in [(False, self.sp), (True, self.sectors)]:
            with self.subTest(sector_model=sector_model):
                original = run_model(prices, sector_model, n_sim=2, n_calibration=2)
                changed = prices.copy()
                # Includes the very first test price: catches the old endpoint leak.
                changed.iloc[original.split + 1:] *= np.linspace(2, 20, len(prices) - original.split - 1)[:, None]
                rerun = run_model(changed, sector_model, n_sim=2, n_calibration=2)
                self.assertEqual(original.parameters['order'], rerun.parameters['order'])
                self.assertEqual(original.kappa_scores, rerun.kappa_scores)
                self.assertEqual(original.kappa, rerun.kappa)
                for name in ['alpha', 'hurst', 'mu', 'sigma']:
                    np.testing.assert_array_equal(original.parameters[name], rerun.parameters[name])
                np.testing.assert_array_equal(original.paths, rerun.paths)
                self.assertFalse(evaluate(original).equals(evaluate(rerun)))

    def test_prediction_prefix_does_not_depend_on_horizon(self):
        train = self.sectors.iloc[:180]
        params = fit_parameters(train, True)
        a = calibrated_path(train, 5, params, .1, 3, True)
        b = calibrated_path(train, 30, params, .1, 3, True)
        np.testing.assert_array_equal(a, b[:len(a)])

    def test_all_prices_and_returns_are_date_aligned(self):
        result = run_model(self.sp, n_sim=2)
        pd.testing.assert_index_equal(result.mean.index, self.sp.index[1:])
        pd.testing.assert_frame_equal(result.actual, self.sp.iloc[1:])
        self.assertEqual(result.paths.shape, (2, len(self.sp) - 1, 1))

    def test_hurst_is_deterministic_and_restores_rng(self):
        values = np.log(self.sp).diff().iloc[1:, 0].to_numpy()
        np.random.seed(91)
        state = np.random.get_state()
        first = estimate_hurst(values)
        after = np.random.random(3)
        np.random.set_state(state)
        np.testing.assert_array_equal(after, np.random.random(3))
        np.random.seed(87)
        self.assertEqual(first, estimate_hurst(values))

    def test_constant_price_baseline_uses_last_training_observation(self):
        result = run_model(self.sp, n_sim=2)
        actual = self.sp.iloc[result.split + 1:, 0].to_numpy()
        baseline = np.full(len(actual), self.sp.iloc[result.split, 0])
        expected = metrics(actual, baseline)
        actual_metrics = evaluate(result).loc[('SP500', 'constant price')]
        for key, value in expected.items():
            self.assertAlmostEqual(actual_metrics[key], value)

    def test_bad_data_fails_instead_of_being_filled(self):
        for value in [np.nan, 0, -1, np.inf]:
            bad = self.sp.copy()
            bad.iloc[10, 0] = value
            with self.assertRaises(ValueError):
                validate_prices(bad)
        with self.assertRaises(ValueError):
            validate_prices(self.sp.iloc[::-1])
        with self.assertRaises(ValueError):
            validate_prices(pd.concat([self.sp, self.sp]))

    def test_insufficient_training_data_is_rejected(self):
        with self.assertRaises(ValueError):
            run_model(self.sp.iloc[:50])

    def test_csv_input_and_inception_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / 'raw'
            raw.mkdir()
            self.sp.rename_axis('Date').to_csv(raw / 'sp500_close.csv')
            sectors = self.sectors.copy()
            sectors.iloc[:3, -1] = np.nan
            sectors.rename_axis('Date').to_csv(raw / 'sector_etfs_close.csv')
            sp, actual = load_prices(temp)
            pd.testing.assert_index_equal(sp.index, self.sp.index[3:], check_names=False)
            np.testing.assert_allclose(actual, self.sectors.iloc[3:], rtol=1e-14)
            sectors.iloc[50, -1] = np.nan
            sectors.rename_axis('Date').to_csv(raw / 'sector_etfs_close.csv')
            with self.assertRaises(ValueError):
                load_prices(temp)
            sectors.iloc[50, -1] = self.sectors.iloc[50, -1]
            reordered = sectors.iloc[[1, 0, *range(2, len(sectors))]]
            reordered.rename_axis('Date').to_csv(raw / 'sector_etfs_close.csv')
            with self.assertRaises(ValueError):
                load_prices(temp)


class PortfolioTests(unittest.TestCase):
    def test_initial_capital_weights_and_share_unit_invariance(self):
        dates = pd.date_range('2020-01-01', periods=3)
        prices = pd.DataFrame({'A': [100., 110., 120.], 'B': [10., 10., 10.]}, index=dates)
        expected = [100., 105., 110.]
        np.testing.assert_allclose(buy_and_hold(prices, {'A': .5, 'B': .5}), expected)
        prices['B'] *= 1000
        np.testing.assert_allclose(buy_and_hold(prices, {'A': .5, 'B': .5}), expected)

    def test_weights_follow_column_names(self):
        _, prices = synthetic_prices(240)
        weights = {symbol: i + 1 for i, symbol in enumerate(SECTORS)}
        pd.testing.assert_series_equal(buy_and_hold(prices, weights), buy_and_hold(prices.iloc[:, ::-1], weights))


if __name__ == '__main__':
    unittest.main()
