# Forecasting from observed history

This page specifies the original conditional-forecast and fixed-blend workflow. The later [adaptive extension](adaptive_forecasting.md) reuses these exact conditional moments as features, adds signed regularized calibration and evaluates empirical uncertainty intervals separately.

The forecasting workflow uses the same return recurrence as the original simulation. The weights, Hurst mapping, Gaussian innovations, sector ordering and reflected Laplacian are unchanged. The forecasting layer changes the **conditioning information and calibration**: it starts from observed history, anchors prices at the forecast origin and chooses calibration on internal validation dates.

`temporal_memory.py` defines the return recurrence. `forecasting.py` supplies conditional inference and evaluation; `run_experiment.py --mode simulation` still runs the earlier experiment. The [model specification](methodology.md) describes that core and its interpretation as a discrete heuristic.

## Prices, returns and information

Let $P_{t,j}>0$ be the observed price of asset $j$ at retained trading observation $t$. Its log-return is

$$
r_{t,j}=\log\left(\frac{P_{t,j}}{P_{t-1,j}}\right).
$$

At forecast origin $o$, the available information is $\mathcal F_o=\sigma(P_0,\ldots,P_o)$. An $h$-session forecast targets $P_{o+h}$ and may use only $\mathcal F_o$. Horizons are **1, 5 and 21 trading observations**. A 21-session forecast is not a prediction made one day before its target, even when all forecasts are drawn on the same price chart.

The forecast begins at the observed price $P_o$ and retains all observed returns through $o$. Future returns use the model recursively. There is no fitted endpoint adjustment or price-level scaling in this workflow. The historical COVID scenario is not extrapolated into future dates; its realized effects are already present in the observed history.

## The unchanged recurrence

For each asset, the history coefficients are

$$
w_{0,j}=1,\qquad
w_{k,j}=\left(1-\frac{\alpha_j+1}{k}\right)w_{k-1,j}.
$$

The discrete model at unit time and lattice spacing is

$$
\mathbf r_t=\boldsymbol\mu+\boldsymbol\varepsilon_t+\kappa L\mathbf r_{t-1}
 +\sum_{k=0}^{t-1}W_k(\mathbf r_{t-1-k}-\mathbf c),
\qquad W_k=\operatorname{diag}(w_{k,1},\ldots,w_{k,J}),
$$

$$
\boldsymbol\varepsilon_t\sim\mathcal N(\mathbf0,D),\qquad
D=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_J^2).
$$

Here $\mathbf c=\mathbf0$ for the index and $\mathbf c=\boldsymbol\mu$ for sectors. The index has $\kappa=0$. The sector Laplacian is

$$
(L\mathbf r)_j=
\begin{cases}
2(r_2-r_1),&j=1,\\
r_{j-1}-2r_j+r_{j+1},&1<j<J,\\
2(r_{J-1}-r_J),&j=J.
\end{cases}
$$

For a single asset, $L=0$. Parameters come from expanding observed history. The order parameter remains

$$
\alpha_j=\operatorname{clip}\!\left(\frac12+s\left(H_j-\frac12\right),\,0.05,\,0.95\right),
\qquad
s=\begin{cases}1,&\text{index},\\0.6,&\text{sectors}.\end{cases}
$$

This Hurst mapping is a modeling assumption. The recurrence is not obtained by isolating a Grünwald–Letnikov derivative; the [original-model notes](legacy_model_notes.md) explain the distinction. The forecasts are derived from the discrete recurrence.

## Exact conditional means, without Monte Carlo noise

With fixed parameters, the recurrence is linear with Gaussian innovations. Its conditional future log-prices are Gaussian. Compute future conditional return means by setting future innovations to zero while retaining observed past returns. Write their cumulative mean as $m_{h,j}$.

The impulse-response matrices account for uncertainty propagating through both temporal memory and neighboring sectors. Define

$$
B_\ell=W_{\ell-1}+\mathbf1_{\{\ell=1\}}\kappa L,\qquad
A_0=I,\qquad
A_q=\sum_{\ell=1}^{q}B_\ell A_{q-\ell},\qquad
C_q=\sum_{v=0}^{q}A_v.
$$

Then the conditional covariance of cumulative future log-returns is

$$
\Sigma_h=\sum_{q=0}^{h-1}C_qDC_q^{\mathsf T},\qquad v_{h,j}=(\Sigma_h)_{jj}.
$$

The code computes these marginal variances without storing unnecessary full covariance matrices. The raw model's expected price and median are different:

$$
\widehat P^{\mathrm{memory}}_{o+h,j}
=P_{o,j}\exp\!\left(m_{h,j}+\frac12v_{h,j}\right),
\qquad
\operatorname{median}(P_{o+h,j}\mid\mathcal F_o)
=P_{o,j}\exp(m_{h,j}).
$$

This is the exact mean under the **specified fitted Gaussian model**, not an assertion about the true market distribution. It removes finite-ensemble noise without changing the recurrence. The raw 10th–90th percentile interval is

$$
\left[
P_{o,j}e^{m_{h,j}-z_{0.9}\sqrt{v_{h,j}}},\;
P_{o,j}e^{m_{h,j}+z_{0.9}\sqrt{v_{h,j}}}
\right],\qquad z_{0.9}=\Phi^{-1}(0.9).
$$

Coverage is measured for the raw model only. These intervals omit parameter uncertainty and are not empirically calibrated intervals for the blended forecast.

## External forecast calibration

The reference with training log-drift is

$$
\widehat P^{\mathrm{drift}}_{o+h,j}=P_{o,j}e^{h\widehat\mu_j}.
$$

The calibrated point forecast is a convex combination in **price levels**:

$$
\widehat P^{\mathrm{cal}}_{o+h,j}
=\lambda_h\widehat P^{\mathrm{memory}}_{o+h,j}
 +(1-\lambda_h)\widehat P^{\mathrm{drift}}_{o+h,j},
\qquad 0\le\lambda_h\le1.
$$

This is an explicit forecast-calibration layer, not a modification of the simulator or a proof that the memory mechanism works. The index and sector panel receive separate calibration; within sectors, one weight per horizon is shared across all eleven assets. When $\lambda_h=0$, the method has selected the drift baseline and no improvement can be attributed to memory.

At internal validation origins, normalize every price by its own origin price. Let $y_i$ be the normalized observed target, $a_i$ the normalized memory forecast and $b_i$ the normalized drift forecast. The least-squares blend has a closed form:

$$
\widehat\lambda_h
=\operatorname{clip}\!\left(
\frac{\sum_i(a_i-b_i)(y_i-b_i)}{\sum_i(a_i-b_i)^2},\,0,\,1\right).
$$

Use zero if the denominator is zero. Normalization gives assets comparable weight regardless of their share price. Sector coupling still uses the original training-correlation scale,

$$
\kappa_{\mathrm{base}}
=\frac12\max\!\left(0,\tanh\left[\frac{1}{M}\sum_{i<j}\operatorname{atanh}(\rho_{ij})\right]\right),
\qquad M=\frac{J(J-1)}2.
$$

Correlations are clipped just inside $\pm1$. For each horizon, choose among $\kappa=s_\kappa\kappa_{\mathrm{base}}$, with $s_\kappa\in\{0,0.5,1\}$, by the normalized validation error after blending. Ties favor smaller coupling. The reported raw-memory forecast uses that same selected coupling, before applying the blend. Neither coupling scale nor blend weight is selected using outer test targets.

## Chronological selection and evaluation

The default outer test is the last 30% of returns. `--test-start` instead defines the first permissible target date. All earlier prices form development data.

Internal validation origins occupy the last 40% of development, spaced 21 observations apart. At each origin, estimate parameters using only the prefix available there, then predict the next 1, 5 and 21 sessions. At least three complete origins are required. All horizons use the same origins. Fit the blend and choose coupling from these validation errors, then freeze both choices before the outer test.

Outer forecast origins advance every five observations. Distribution parameters and sector order are refitted on expanding observed history at the first origin and whenever at least 21 observations have elapsed since the last refit. With five-observation origins, this means a refit every 25 observations. Earlier test observations can be used once they have occurred; later observations cannot affect an earlier forecast. The tests perturb future prices and verify this property. The blend and coupling **scale** remain fixed, although the refitted correlation-based coupling magnitude can change.

Every reported method has identical origins and targets. Forecasts near the end of the sample are omitted unless all three horizons have complete targets. Five-session forecasts have non-overlapping target intervals; 21-session targets overlap. Reported errors are descriptive, with no independent-error significance claim. This rolling task has a different information set and horizon from the old multi-year continuation; their R² values should not be compared as if they were the same benchmark.

This design follows the [rolling-origin evaluation principle](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation). The current 2026 experiment was specified before inspecting that newly downloaded period, but it is still a retrospective backtest on a revised data snapshot, not a prospective live trial or formal preregistration.

## Reading the scores

For forecast error $e_i=\widehat P_i-P_i$ and origin price $P_{o_i}$,

$$
\operatorname{RMSE}=\sqrt{\frac1N\sum_i e_i^2},\qquad
\operatorname{MAE}=\frac1N\sum_i|e_i|,\qquad
R^2=1-\frac{\sum_i e_i^2}{\sum_i(P_i-\overline P)^2}.
$$

The main cross-asset score is

$$
\operatorname{relative\ RMSE}(\%)
=100\sqrt{\frac1N\sum_i\left(\frac{e_i}{P_{o_i}}\right)^2}.
$$

With the same-origin constant forecast $\widehat P^{\mathrm{constant}}_{o+h}=P_o$, define

$$
\operatorname{skill}
=1-\frac{\sum_i[(\widehat P_i-P_i)/P_{o_i}]^2}
          {\sum_i[(P_{o_i}-P_i)/P_{o_i}]^2}.
$$

Positive skill beats that baseline; negative skill does not. A high price-level R² is insufficient evidence of predictive skill because recent prices already explain much of the next price level. The results report the raw model, calibrated forecast, drift baseline and constant baseline together, including zero calibration weights and losses to either reference.
