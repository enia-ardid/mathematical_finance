import unittest

import numpy as np
import pandas as pd

from fractional_diffusion import MEMORY, diffusion_operator
from refined_fractional import (
    kernel, lagged_memory, variance_filter, expected_paths, transformed_squares,
    forecast_choice, simulate_refined, fit_gjr, gjr_after, rolling_refinement,
)
from temporal_memory import synthetic_prices


class RefinedVarianceTests(unittest.TestCase):
    def fixture(self):
        values = np.random.default_rng(6).normal(size=(570, 2)) * [.01, .015]
        fitted = dict(mean=np.array([.001, -.001]), scale=np.array([.0001, .000225]),
                      graph=np.array([[1., -1.], [-1., 1.]]),
                      correlation=np.array([[1., .3], [.3, 1.]]))
        choice = dict(family='fractional', kernel_parameter=.4, feedback=.95,
                      smoothing=.5, diffusion=.25, asymmetry=1.)
        return values, fitted, choice

    def test_lagged_memory_uses_only_previous_observations(self):
        signals = np.array([[2., 3], [5., 7], [11., 13]])
        result = lagged_memory(signals, np.array([.6, .3, .1]))
        np.testing.assert_allclose(result, [[1., 1.], [1.6, 2.2], [3.7, 5.2]])

    def test_vectorized_validation_matches_separate_origin_forecasts(self):
        values, fitted, choice = self.fixture()
        weights = kernel(choice)
        signals = transformed_squares(values, fitted['mean'], fitted['scale'], 1.)
        operator = diffusion_operator(fitted['graph'], choice['diffusion'])
        filtered = variance_filter(lagged_memory(signals, weights), operator, .95, .5)
        origins = np.array([510, 535, 548])
        known = np.stack([lagged_memory(signals, weights[step:])[origins] for step in range(21)])
        result = expected_paths(known, filtered[origins - 1], weights, operator, .95, .5)
        for i, origin in enumerate(origins):
            expected = forecast_choice(values[:origin], fitted, choice)
            np.testing.assert_allclose(result[:, i] * fitted['scale'], expected, rtol=1e-12)

    def test_asymmetry_is_normalized_for_symmetric_innovations(self):
        values = np.array([-2., -1., 1., 2.])[:, None]
        for delta in [0., 1., 3.]:
            self.assertAlmostEqual(transformed_squares(values, 0., 1., delta).mean(), (values**2).mean())
        result = transformed_squares(values, 0., 1., 1.)
        self.assertAlmostEqual(result[0, 0] / result[-1, 0], 2.)

    def test_exact_moments_and_variance_floor_against_simulation(self):
        values, fitted, choice = self.fixture()
        expected = forecast_choice(values, fitted, choice, 8)
        simulated, variance = simulate_refined(values, fitted, choice, horizon=8)
        error = variance.std(axis=0) / np.sqrt(len(variance))
        self.assertTrue((np.abs(variance.mean(axis=0) - expected) <= 6 * error + 1e-12).all())
        self.assertTrue((variance >= (1 - choice['feedback']) * fitted['scale'] - 1e-14).all())
        self.assertTrue(np.isfinite(simulated).all())
        np.testing.assert_array_equal(forecast_choice(values, fitted, choice, 3), expected[:3])

    def test_gjr_recursion_and_stationary_fit(self):
        values = np.array([.1, -3., .4, -1., 2.])
        parameters = (.05, .05, .85, .1)
        state, expected = 1., []
        for value in values:
            state = .05 + (.05 + .1 * (value < 0)) * value**2 + .85 * state
            expected.append(state)
        np.testing.assert_allclose(gjr_after(values**2, values < 0, parameters), expected)
        fit = fit_gjr(np.random.default_rng(10).normal(size=600), 1.)
        omega, alpha, beta, gamma = fit['parameters']
        self.assertTrue(np.isfinite(fit['objective']))
        self.assertGreater(omega, 0)
        self.assertGreaterEqual(min(alpha, beta, gamma), 0)
        self.assertLess(alpha + beta + gamma / 2, .999)

    def test_future_observations_cannot_change_selection_or_forecasts(self):
        _, panel = synthetic_prices(610)
        prices = panel.iloc[:, :2]
        first, audits = rolling_refinement(prices, start=prices.index[552])
        cutoff = prices.index[575]
        changed = prices.copy()
        changed.loc[cutoff:] *= np.linspace(1.5, 2.5, len(changed.loc[cutoff:]))[:, None]
        second, changed_audits = rolling_refinement(changed, start=prices.index[552])
        columns = ['origin', 'target', 'last_fit', 'asset', 'method', 'horizon', 'variance', 'reference_scale']
        pd.testing.assert_frame_equal(first.loc[first.origin < cutoff, columns], second.loc[second.origin < cutoff, columns])
        self.assertEqual([a for a in audits if a['origin'] < str(cutoff.date())],
                         [a for a in changed_audits if a['origin'] < str(cutoff.date())])
        self.assertTrue(all(a['validation_last_target'] == a['origin'] for a in audits))


if __name__ == '__main__':
    unittest.main()
