# Relation to existing volatility models

The model combines a finite fractional temporal kernel with a common positive graph resolvent. The following comparisons distinguish that restriction from existing fractional and network volatility models.

| Reference | Existing contribution | Relationship to this project |
| --- | --- | --- |
| [Baillie, Bollerslev and Mikkelsen (1996), FIGARCH](https://doi.org/10.1016/S0304-4076(95)01749-6) | Fractionally integrated conditional heteroskedasticity. | Our normalized finite fractional-integral kernel has subunit total feedback. It is not the FIGARCH fractional-difference recursion. FIGARCH is an estimated reference in the external study. |
| [Zhou, Li, Pan and Wang (2020), Network GARCH](https://www3.stat.sinica.edu.tw/statistica/j30n4/j30n403/j30n403.html) | Network information reduces the parameter count in multivariate GARCH; stationarity and QMLE theory are developed. | Network parsimony and stability arguments predate this project. Our lag matrices share a normalized kernel and a graph resolvent. A [direct follow-up benchmark](network_benchmark.md) fits their first-order recursion with explicit correlation-network and initialization adaptations. |
| [Mattera and Otto (2024), Network log-ARCH](https://eprints.gla.ac.uk/316957/1/316957.pdf) | Network log-variance modeling and forecasting with alternative network constructions. | We filter lagged squared innovations in variance units; their log-scale specification is a different model and has not been fitted here. |
| [Pan and Pan (2024), Threshold Network GARCH](https://doi.org/10.1111/jtsa.12743) | Asymmetric responses within a network GARCH model, with stationarity and estimation theory. | Our normalized asymmetric transformation preserves the stated conditional-moment recursion under symmetric innovations. |
| [Peiyi Zhou (2026), Generalised Network GARCH](https://doi.org/10.1002/for.70191) | Network-augmented GARCH with own and neighboring shock and variance terms, and covariance forecasting. | Our common resolvent constrains the spatial response at every lag. We do not estimate a comparable dynamic covariance model. |

The shared kernel and resolvent yield exact conditional second-moment forecasts and tractable impulse responses. The stationarity proofs apply standard positive-filter arguments to this restricted family. The empirical comparison fits the first-order Network GARCH recursion on weighted and binary graphs. Network log-ARCH, threshold Network GARCH and generalized dynamic-covariance models remain outside the fitted comparison.

## Effective memory under persistence

For fixed parameters, write the persistent model as

$$
h_t=(1-b)(1-\eta)\mathbf1+b h_{t-1}
 +(1-b)\eta A_\tau\sum_{k=0}^{M-1}p_kx_{t-1-k}.
$$

Unrolling the scalar persistence filter, after removing its initialization transient, gives

$$
h_t=(1-\eta)\mathbf1+\eta A_\tau
 \sum_{j=1}^{\infty}\widetilde p_j x_{t-j},\qquad
\widetilde p_j=(1-b)\sum_{k=0}^{\min(M-1,j-1)}p_k b^{j-1-k}.
$$

The effective weights are nonnegative and sum to one. For $0<b<1$ and $j\ge M$, $\widetilde p_{j+1}=b\widetilde p_j$. For $b=0$, they reduce to the original finite kernel. Thus persistence smooths the fractional kernel with a geometric filter; it does not create an asymptotically hyperbolic tail.

If $L=U\Lambda U^\top$, each graph mode has multiplier $(1+\tau\lambda_\ell)^{-1}$, identical at every temporal lag. The model imposes a separable temporal/spatial response: it cannot represent a different diffusion geometry for each lag. The impulse-response identity is checked against the implemented recursion in the test suite.
