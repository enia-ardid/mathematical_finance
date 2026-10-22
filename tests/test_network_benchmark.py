import unittest

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from fractional_diffusion import _base, garch_after, HORIZONS
from network_benchmark import (network_matrix, network_after, objective_gradient,
                               fit_network, forecast_network, append_network_benchmarks)


class NetworkBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.q = np.random.default_rng(29).standard_normal((400, 3))**2
        self.network = np.array([[0., 1., 0.], [.4, 0., .6], [0., 0., 0.]])
        self.parameters = [.03, .08, .05, .83]

    def test_filter_matches_published_recursion_and_isolated_node(self):
        expected, state = [], np.ones(3)
        for q in self.q:
            state = .03+.08*q+.05*(self.network@q)+.83*state
            expected.append(state)
        np.testing.assert_allclose(network_after(self.q, self.network, self.parameters), expected, rtol=1e-13)
        np.testing.assert_allclose(np.asarray(expected)[:,2], garch_after(self.q[:,2], [.03,.08,.83]))

    def test_gradient_matches_finite_differences_in_both_models(self):
        for active, theta in [(True, np.array([-3.,.1,-.5,2.])), (False,np.array([-3.,.1,2.]))]:
            matrix = self.network if active else np.zeros((3,3))
            _, gradient = objective_gradient(theta,self.q,matrix,active)
            numeric = approx_derivative(lambda t: objective_gradient(t,self.q,matrix,active)[0],theta)
            np.testing.assert_allclose(gradient,numeric.ravel(),rtol=2e-5,atol=2e-7)

    def test_no_network_forecast_reduces_to_analytic_garch(self):
        prediction = forecast_network(self.q,np.zeros((3,3)),[.03,.08,0.,.83])
        first = np.array([garch_after(self.q[:,j],[.03,.08,.83])[-1] for j in range(3)])
        expected = .03/(1-.91)+.91**np.arange(21)[:,None]*(first-.03/(1-.91))
        np.testing.assert_allclose(prediction,expected,rtol=1e-13)

    def test_asset_relabeling_preserves_forecasts_and_graphs(self):
        permutation = [2,0,1]
        original = forecast_network(self.q,self.network,self.parameters)
        permuted = forecast_network(self.q[:,permutation],self.network[np.ix_(permutation,permutation)],self.parameters)
        np.testing.assert_allclose(permuted,original[:,permutation],rtol=1e-14)
        correlation = np.corrcoef(self.q.T)
        for binary in [False,True]:
            matrix = network_matrix(correlation,binary)
            permuted = network_matrix(correlation[np.ix_(permutation,permutation)],binary)
            np.testing.assert_allclose(permuted,matrix[np.ix_(permutation,permutation)])

    def test_exact_moments_agree_with_conditional_simulation(self):
        prediction = forecast_network(self.q,self.network,self.parameters,horizon=5)
        rng = np.random.default_rng(947)
        current = np.broadcast_to(prediction[0],(100000,3)).copy()
        for step in range(5):
            np.testing.assert_allclose(current.mean(axis=0),prediction[step],rtol=.008,atol=.002)
            squares = current*rng.standard_normal(current.shape)**2
            current = .03+.08*squares+.05*(squares@self.network.T)+.83*current

    def test_fit_is_finite_convergent_and_respects_feedback_constraint(self):
        fit = fit_network(self.q,self.network)
        self.assertTrue(fit['converged'],fit['message'])
        self.assertEqual(len(fit['attempts']),3)
        omega,alpha,spillover,beta = fit['parameters']
        self.assertGreater(omega,0.)
        self.assertLess(alpha+spillover+beta,.999)
        self.assertLessEqual(fit['objective'],np.mean(self.q)+1e-5)

    def test_invalid_network_or_parameters_are_rejected(self):
        for bad in [np.eye(3),self.network*2, self.network*np.nan]:
            with self.assertRaises(ValueError):
                network_after(self.q,bad,self.parameters)
        with self.assertRaises(ValueError):
            network_after(self.q,self.network,[.01,.1,.1,.9])

    def test_appended_forecasts_preserve_targets_and_ignore_future_prices(self):
        values = np.random.default_rng(112).normal(0.,.01,(650,2))
        prices = pd.DataFrame(100*np.exp(np.vstack([np.zeros(2),values.cumsum(axis=0)])),
                              index=pd.bdate_range('2020-01-01',periods=651),columns=['A','B'])
        base = _base(np.log(prices).diff().iloc[1:551].to_numpy())
        origin = prices.index[550]
        audits = [dict(origin=str(origin.date()),**{k:v.tolist() for k,v in base.items()})]
        returns = np.log(prices).diff().iloc[1:].to_numpy()
        rows=[]
        for h in HORIZONS:
            for j,asset in enumerate(prices.columns):
                actual = ((returns[550:550+h,j]-base['mean'][j])**2).sum()
                ref = h*base['scale'][j]
                rows.append(dict(panel='test',origin=origin,last_fit=origin,target=prices.index[550+h],
                                 asset=asset,horizon=h,method='persistent_asymmetric',realized_proxy=actual,
                                 reference_scale=ref,variance=ref,qlike=actual/ref,
                                 normalized_square_error=((actual-ref)/ref)**2))
        template = pd.DataFrame(rows)
        output,audit = append_network_benchmarks(prices,template,audits)
        changed = prices.copy()
        changed.iloc[600:] *= np.linspace(1.,3.,51)[:,None]
        later,_ = append_network_benchmarks(changed,template,audits)
        pd.testing.assert_frame_equal(output,later)
        pd.testing.assert_frame_equal(output.iloc[:len(template)].reset_index(drop=True),template)
        self.assertEqual(len(output),4*len(template))
        self.assertEqual(audit[0]['last_training_return'],str(origin.date()))
        self.assertTrue((output.variance>0).all())


if __name__ == '__main__':
    unittest.main()
