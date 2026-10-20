import unittest

import numpy as np
import pandas as pd
from arch.univariate import FIGARCH, Normal, ZeroMean

from external_validation import append_equal_combination, fit_figarch, forecast_figarch


class ExternalValidationTests(unittest.TestCase):
    def components(self):
        shared = dict(panel='example', origin=pd.Timestamp('2025-01-01'),
                      target=pd.Timestamp('2025-01-02'), last_fit=pd.Timestamp('2025-01-01'),
                      asset='A', horizon=1, realized_proxy=4., reference_scale=1.)
        return pd.DataFrame([dict(shared, method=method, variance=v, qlike=np.log(v) + 4/v,
                                  normalized_square_error=(4-v)**2)
                             for method, v in [('persistent_asymmetric', 1.), ('gjr_garch', 9.)]])

    def test_combination_recomputes_loss_of_matched_variances(self):
        inputs = self.components()
        before = inputs.copy(deep=True)
        combined = append_equal_combination(inputs).iloc[-1]
        self.assertEqual(combined.variance, 5.)
        self.assertAlmostEqual(combined.qlike, np.log(5.) + 4/5)
        self.assertNotAlmostEqual(combined.qlike, inputs.qlike.mean())
        pd.testing.assert_frame_equal(inputs, before)

    def test_combination_rejects_unmatched_keys_or_targets(self):
        for col, value in [('asset', 'B'), ('realized_proxy', 7.), ('reference_scale', 2.)]:
            inputs = self.components()
            inputs.loc[1, col] = value
            with self.assertRaises(ValueError):
                append_equal_combination(inputs)
        with self.assertRaises(ValueError):
            append_equal_combination(self.components().iloc[:1])

    def test_figarch_matches_library_at_fit_and_updated_origin(self):
        values = np.random.default_rng(29).standard_normal((650, 2))[:, 0]
        self.assertFalse(values.flags.c_contiguous)
        fitted = fit_figarch(values[:600])
        self.assertEqual(fitted['convergence_flag'], 0)
        model = ZeroMean(values[:600], rescale=False, volatility=FIGARCH(truncation=1000), distribution=Normal())
        reference = model.fix(fitted['parameters']).forecast(horizon=21, reindex=False).variance.to_numpy()[-1]
        np.testing.assert_allclose(forecast_figarch(values[:600], fitted), reference, rtol=1e-12)
        # The fixed forecast wrapper's initialization is explicitly retained.
        process = FIGARCH(truncation=1000)
        contiguous = np.ascontiguousarray(values[:610])
        reference = process.forecast(np.asarray(fitted['parameters']), contiguous, fitted['backcast'],
                                     process.variance_bounds(contiguous), start=609, horizon=21).forecasts[-1]
        np.testing.assert_array_equal(forecast_figarch(values[:610], fitted), reference)
        changed = values.copy()
        changed[610:] *= 100
        np.testing.assert_array_equal(forecast_figarch(changed[:610], fitted), reference)
        np.testing.assert_array_equal(forecast_figarch(values[:610], fitted, 5), reference[:5])

    def test_persistence_equals_geometrically_smoothed_fractional_kernel(self):
        from refined_fractional import variance_filter, lagged_memory
        from fractional_diffusion import fractional_weights
        weights = fractional_weights(.4)
        # An isolated excess innovation reveals the deterministic filter response.
        signals = np.ones((600, 1))
        signals[0] = 2.
        b, eta = .9, .95
        result = variance_filter(lagged_memory(signals, weights), np.eye(1), eta, b)[:, 0] - 1
        effective = np.convolve(weights, (1-b) * b**np.arange(600))[:599]
        np.testing.assert_allclose(result[1:], eta * effective, atol=1e-14)
        self.assertAlmostEqual(effective.sum(), 1., places=12)


if __name__ == '__main__':
    unittest.main()
