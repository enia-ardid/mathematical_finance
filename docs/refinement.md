# Persistent asymmetric fractional diffusion

The follow-up model preserves the fractional kernel and graph operator from the [first variance specification](fractional_diffusion.md). It adds a persistent variance state, a normalized response to negative innovations, and a validation objective covering all three forecast horizons. The earlier model and its recorded results remain unchanged.

## Model and assumptions

Let $u_t=r_t-\mu$, $q_{t,j}=u_{t,j}^2/s_j^2$, and define

$$x_{t,j}=q_{t,j}\frac{1+\delta\mathbf1\{u_{t,j}<0\}}{1+\delta/2},\qquad
m_t=\sum_{k=0}^{M-1}p_k(d)x_{t-1-k}.$$

The positive fractional-integral weights $p_k(d)$ sum to one, $M=252$, and $A_\tau=(I+\tau L)^{-1}$ is the graph diffusion operator. The normalized conditional variance is

$$h_t=(1-b)(1-\eta)\mathbf1+b h_{t-1}+(1-b)\eta A_\tau m_t,
\qquad v_t=s^2\odot h_t.$$

Here $0\le b<1$, $0\le\eta<1$, and $\delta\ge0$. The parameter $b$ controls persistence in the variance state. For $\delta=1$, a negative innovation contributes twice as much as a positive innovation of equal magnitude. The normalization preserves $E[x_t\mid\mathcal F_{t-1}]=h_t$ **under conditionally symmetric, unit-variance innovations**. Zero means and unit variances alone are insufficient for this identity. Gaussian innovations satisfy the assumption and are used in simulations.

The initialization uses unit normalized variance and unit transformed squared innovations before the observed history. Nonnegative weights and diffusion then give $v_{t,j}\ge(1-\eta)s_j^2$. With fixed parameters and iid symmetric innovations, a shared-noise coupling satisfies

$$e_t\le b e_{t-1}+(1-b)\eta\sum_{k=0}^{M-1}p_k e_{t-1-k},
\qquad b+(1-b)\eta<1.$$

The contraction gives a unique causal stationary solution with finite first moments for normalized variance, $E[h_t]=\mathbf1$, and return variance $s_j^2$. These are model properties under the stated assumptions, not guarantees for rolling estimates or observed markets. The finite fractional window and recursive exponential persistence do not establish asymptotic long-range dependence.

Multi-step expected variances use the same recursion with future $x$ replaced by its conditional expectation. The implementation computes these moments exactly under the fitted model. Monte Carlo tests check this calculation against pathwise asymmetric simulations.

## Chronological selection and comparisons

The [follow-up protocol](../paper/refinement_protocol.json) was recorded before running the new variants, after the earlier historical results had been inspected. This is explicitly exploratory research on reused data.

At each refit, the final 252 observed returns form a validation block. Means, scales and graph estimates used for validation are fitted strictly before that block. Every validation forecast must have its full 21-session target observed by the refit date. The objective is the equally weighted mean of normalized QLIKE at horizons 1, 5 and 21, pooled over assets and eligible daily validation origins. Parameters are then refitted on observed history, and all three outer horizons use the same selected specification.

The original grids for fractional order, feedback and diffusion are retained. New grids are $b\in\{0,0.5,0.9\}$ and $\delta\in\{0,1\}$. There are 288 candidates for sectors and 72 for the scalar index per kernel family. Exact ties follow the recorded iteration order. A larger candidate family may overfit validation; chronological ordering does not eliminate that risk.

The full model is compared with a symmetric variant, a variant without graph diffusion, and an exponential-kernel variant with the same candidate count. Each ablation has its own validation selection. The exponential kernel is normalized and truncated to the same 252 observations. The earlier EWMA, GARCH and fractional references are recomputed on identical targets. Independent GJR-GARCH models provide an additional asymmetric benchmark; the asymmetric variance mechanism follows [Glosten, Jagannathan and Runkle](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x).

GJR-GARCH uses $v_t=\omega+(\alpha+\gamma\mathbf1\{u_{t-1}<0\})u_{t-1}^2+\beta v_{t-1}$ in normalized units, Gaussian QMLE, two fixed starts, and $\alpha+\beta+\gamma/2<0.999$. Optimizer convergence is recorded. Its multi-step moments also assume conditional symmetry.

## Recorded results

The follow-up uses the same 186 origins, assets, target convention and snapshot as the first study. All 40,176 earlier forecast records and loss values reproduce exactly. Lower normalized QLIKE is better:

| Panel | Horizon | Earlier fractional | Refined fractional | GARCH | GJR-GARCH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Index | 1 | 0.65138 | 0.58187 | 0.60036 | 0.52512 |
| Index | 5 | 0.42225 | 0.35738 | 0.36402 | 0.31285 |
| Index | 21 | 0.54672 | 0.49770 | 0.47141 | 0.43604 |
| Sectors | 1 | 0.64053 | 0.62417 | 0.62969 | 0.60528 |
| Sectors | 5 | 0.56274 | 0.54779 | 0.54987 | 0.54007 |
| Sectors | 21 | 0.61963 | 0.60432 | 0.60211 | 0.59305 |

The refinement improves on the earlier fractional model in all six pooled comparisons and on symmetric GARCH in four. GJR-GARCH has the lowest loss of these four models in all six cases.

With ten-origin blocks, nominal 95% ranges for the improvement relative to the earlier fractional model exclude zero in five of six comparisons. The exception is one-session sector forecasting. All ranges comparing the refined model with GARCH include zero. These ranges are descriptive: the same history informed the follow-up and there is no correction for the sequence of research decisions.

Asymmetry is selected at all 38 index refits and 35 of 38 sector refits. A nonzero persistence parameter is selected at every refit. Nonzero diffusion is selected at 18 of 38 sector refits, and the graph ablation remains close: graph diffusion modestly helps the one-session sector average, slightly hurts the five-session average and has little effect at 21 sessions. Improvements are not uniform by year: the index deteriorates at 21 sessions in 2024, while sectors deteriorate at 5/21 sessions in 2024 and one session in 2025 relative to the earlier fractional model.

All 456 selected GJR fits report convergence. The offline synthetic run also completes. The tests include validation forecasts checked against separate truncated-history forecasts, asymmetric moment normalization, simulation agreement and future-price perturbations.

## Reproduce

```bash
python run_refinement.py --synthetic --output artifacts/refinement-synthetic
python run_refinement.py --data-dir artifacts/yahoo-through-2026-10-16 \
  --output artifacts/refinement-study --publish-assets
python -m unittest discover -s tests -v
```

The [implementation](../refined_fractional.py) and [runner](../run_refinement.py) retain origin-level forecasts and complete candidate scores in ignored local artifacts. Published aggregate outputs are [metrics](../paper/refinement_metrics.csv), [paired loss comparisons](../paper/refinement_comparisons.csv) and [provenance](../paper/refinement_summary.json). Paired circular block-bootstrap ranges use the same 2,000 replicates and five-/ten-origin blocks as the earlier study. They are nominal descriptive ranges, without adjustment for repeated research on this snapshot.
