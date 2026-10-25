import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac

from forecast_diagnostics import aligned_pair, dm_test, holm, mz_test, win_counts


class DiagnosticTests(unittest.TestCase):
    def test_dm_matches_independent_hac_regression(self):
        rng = np.random.default_rng(51)
        d = rng.normal(size=250)
        d[1:] += .6 * d[:-1]
        for lag in (0, 4, 9):
            fit = sm.OLS(d, np.ones((len(d), 1))).fit()
            expected = np.sqrt(cov_hac(fit, nlags=lag, use_correction=False)[0, 0])
            actual = dm_test(d, lag)
            self.assertAlmostEqual(actual["standard_error"], expected, places=12)
            self.assertAlmostEqual(actual["dm"], d.mean() / expected, places=12)
            self.assertAlmostEqual(dm_test(-d, lag)["p_value"], actual["p_value"])
            self.assertAlmostEqual(dm_test(7*d, lag)["dm"], actual["dm"])

    def test_calibration_matches_independent_regression(self):
        rng = np.random.default_rng(12)
        x = np.exp(rng.normal(size=300) / 2)
        y = .2 + .8*x + rng.normal(size=300)*x*.3
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        for lag in (4, 9):
            actual = mz_test(x, y, lag)
            expected = cov_hac(fit, nlags=lag, use_correction=True)
            np.testing.assert_allclose([actual["alpha"], actual["beta"]], fit.params, atol=1e-12)
            np.testing.assert_allclose([actual["se_alpha"], actual["se_beta"]],
                                       np.sqrt(np.diag(expected)), atol=1e-12)
            delta = fit.params - [0, 1]
            self.assertAlmostEqual(actual["wald"], delta @ np.linalg.solve(expected, delta), places=9)

    def test_holm_matches_statsmodels(self):
        p = np.array([.7, .012, .003, .09, .02])
        np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])
        self.assertTrue(np.isnan(holm([np.nan, .01])[0]))
        self.assertEqual(holm([np.nan, .01])[1], .02)

    def test_pair_alignment_and_win_count_conservation(self):
        rows = []
        for asset in ("A", "B"):
            for i, loss in enumerate([-1., 0., 1.]):
                for method in ("m", "r"):
                    rows.append(dict(panel="p", horizon=1, origin=i, target=i+1,
                                     asset=asset, method=method, realized_proxy=2.,
                                     reference_scale=1., qlike=loss if method=="m" else 0.))
        f = pd.DataFrame(rows)
        pair = aligned_pair(f.sample(frac=1, random_state=1), "m", "r")
        counts = win_counts(pair)
        self.assertTrue((counts[["wins", "ties", "losses"]].sum(axis=1)==counts.origins).all())
        self.assertTrue((counts[["wins", "ties", "losses"]]==1).all().all())
        with self.assertRaises(ValueError):
            aligned_pair(f.iloc[1:], "m", "r")
        f.loc[0, "realized_proxy"] = 3
        with self.assertRaises(ValueError):
            aligned_pair(f, "m", "r")

    def test_degenerate_and_invalid_inputs_are_explicit(self):
        self.assertEqual(dm_test(np.zeros(30), 4)["status"], "degenerate_loss_variance")
        x = np.arange(30.)+1
        self.assertEqual(mz_test(x, x, 4)["status"], "singular_hac")
        with self.assertRaises(ValueError):
            mz_test(np.ones(30), x, 4)
        with self.assertRaises(ValueError):
            dm_test([np.nan]*30, 4)
        with self.assertRaises(ValueError):
            dm_test(x, 30)


if __name__ == "__main__":
    unittest.main()
