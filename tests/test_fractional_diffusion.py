import unittest

import numpy as np
import pandas as pd
from scipy.special import gammaln

from fractional_diffusion import (
    MEMORY, conditional_variances, diffusion_operator, fit_garch, fractional_weights,
    garch_after, graph_laplacian, rolling_variance_forecasts, simulate_fractional,
)
from temporal_memory import synthetic_prices


class FractionalVarianceTests(unittest.TestCase):
    def test_weights_match_fractional_binomial_coefficients(self):
        k = np.arange(80)
        for order in (.2, .6, 1.):
            exact = np.exp(gammaln(k + order) - gammaln(order) - gammaln(k + 1))
            exact /= exact.sum()
            weights = fractional_weights(order, 80)
            np.testing.assert_allclose(weights, exact, rtol=1e-12)
            self.assertTrue((weights > 0).all())
            self.assertAlmostEqual(weights.sum(), 1.)
        np.testing.assert_allclose(fractional_weights(1., 80), np.ones(80) / 80)

    def test_diffusion_preserves_constants_positivity_and_reduces_energy(self):
        graph = np.array([[1., -1, 0], [-1, 3, -2], [0, -2, 2]])
        for strength in [0, .25, 1, 100]:
            operator = diffusion_operator(graph, strength)
            np.testing.assert_allclose(operator.sum(axis=1), 1.)
            np.testing.assert_allclose(operator.sum(axis=0), 1.)
            self.assertTrue((operator >= -1e-15).all())
            original = np.array([.1, 2., .8])
            result = operator @ original
            self.assertGreaterEqual(result.min(), original.min() - 1e-13)
            self.assertLessEqual(result.max(), original.max() + 1e-13)
            self.assertLessEqual(result @ graph @ result, original @ graph @ original + 1e-13)

    def test_invalid_graph_is_rejected(self):
        with self.assertRaises(ValueError):
            diffusion_operator(np.array([[-1., 1.], [1., -1.]]), 1.)

    def test_graph_has_asset_permutation_equivariance(self):
        _, prices = synthetic_prices(550)
        returns = np.log(prices).diff().iloc[1:].to_numpy()
        permutation = np.array([4, 0, 2, 1, 3, 10, 7, 8, 5, 9, 6])
        graph, corr = graph_laplacian(returns)
        changed, changed_corr = graph_laplacian(returns[:, permutation])
        np.testing.assert_allclose(changed, graph[np.ix_(permutation, permutation)], atol=1e-14)
        np.testing.assert_allclose(changed_corr, corr[np.ix_(permutation, permutation)], atol=1e-14)

    def test_garch_filter_matches_scalar_recursion(self):
        squares = np.array([.1, 3., .4, 1., 2.])
        parameters = (.05, .1, .85)
        state, expected = 1., []
        for value in squares:
            state = .05 + .1 * value + .85 * state
            expected.append(state)
        np.testing.assert_allclose(garch_after(squares, parameters), expected)

    def test_garch_fit_has_positive_stationary_parameters(self):
        squares = np.random.default_rng(13).normal(size=600)**2
        result = fit_garch(squares)
        omega, alpha, beta = result['parameters']
        self.assertGreater(omega, 0)
        self.assertGreaterEqual(alpha, 0)
        self.assertGreaterEqual(beta, 0)
        self.assertLess(alpha + beta, 1.)
        self.assertTrue(np.isfinite(result['objective']))

    def test_multistep_moments_agree_with_simulated_conditional_variance(self):
        returns = np.random.default_rng(21).normal(size=(MEMORY + 10, 2)) * [.01, .015]
        choice = {'family': 'fractional', 'kernel_parameter': .6, 'feedback': .95, 'diffusion': .5}
        fitted = {'mean': np.zeros(2), 'scale': np.array([.0001, .000225]), 'start': 0,
                  'graph': np.array([[1., -1.], [-1., 1.]]), 'correlation': np.array([[1., .3], [.3, 1.]]),
                  'choices': {'fractional_diffusion': choice, 'fractional_only': dict(choice, diffusion=0.),
                              'exponential_diffusion': dict(choice, family='exponential', kernel_parameter=.94),
                              'ewma': {'decay': .94}},
                  'garch': [{'parameters': (.05, .05, .9)}] * 2}
        expected = conditional_variances(returns, fitted, 8)['fractional_diffusion']
        simulated, variance = simulate_fractional(returns, fitted, horizon=8, paths=8000)
        standard_error = variance.std(axis=0) / np.sqrt(len(variance))
        self.assertTrue((np.abs(variance.mean(axis=0) - expected) <= 6 * standard_error + 1e-12).all())
        self.assertTrue((variance >= .05 * fitted['scale'] - 1e-15).all())
        self.assertTrue(np.isfinite(simulated).all())
        prefix = conditional_variances(returns, fitted, 3)['fractional_diffusion']
        np.testing.assert_array_equal(prefix, expected[:3])

    def test_future_prices_cannot_change_past_forecasts_or_fits(self):
        _, sectors = synthetic_prices(660)
        prices = sectors.iloc[:, :2]
        first, audit, _ = rolling_variance_forecasts(prices, start=prices.index[552])
        altered = prices.copy()
        cutoff = prices.index[615]
        altered.loc[cutoff:] *= np.linspace(2., 3., len(altered.loc[cutoff:]))[:, None]
        second, changed, _ = rolling_variance_forecasts(altered, start=prices.index[552])
        columns = ['origin', 'target', 'last_fit', 'asset', 'horizon', 'method', 'variance', 'reference_scale']
        pd.testing.assert_frame_equal(first.loc[first.origin < cutoff, columns], second.loc[second.origin < cutoff, columns])
        self.assertEqual([a for a in audit if a['origin'] < str(cutoff.date())],
                         [a for a in changed if a['origin'] < str(cutoff.date())])
        self.assertTrue((first.last_fit <= first.origin).all())
        self.assertTrue((first.origin < first.target).all())
        self.assertTrue(np.isfinite(first.qlike).all())


if __name__ == '__main__':
    unittest.main()
