# Direct Network GARCH comparison

This follow-up fits the recursion in [Zhou, Li, Pan and Wang (2020), equation (2.2)](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A30n403.pdf) on the recorded sector and country panels. It adds benchmarks without changing the fractional model, the 50:50 combination, or any earlier forecast. The [protocol](../paper/network_protocol.json) was recorded before fitting these benchmarks, after the previous results were known. The comparison is exploratory.

## Specification

For the common standardized squared innovations $z_{t,j}=(r_{t,j}-\mu_j)^2/s_j^2$,

$$
h^N_t=\omega\mathbf1+\alpha z_{t-1}
 +\lambda Bz_{t-1}+\beta h^N_{t-1},
\qquad v^N_t=s^2\odot h^N_t.
$$

The intercept is positive and the nonnegative feedback coefficients sum to less than $0.999$. Coefficients are shared across assets in standardized units. Three specifications are reported:

- **Weighted network:** row-normalized positive squared shrinkage correlations, estimated on the same fitting window as the fractional model.
- **Binary network:** the same correlations thresholded at $R_{ij}\ge0.5$, then row-normalized. The threshold is fixed.
- **Pooled GARCH:** the same estimation procedure with $\lambda=0$.

Networks have zero diagonals; isolated rows contribute zero. These are applications of the published first-order recursion with documented adaptations: correlation networks instead of shareholder links, common standardization, and unit initial variance. The weighted case extends binary adjacency to nonnegative weights. We do not transfer the original paper's exogenous-network asymptotic results to rolling estimated graphs, or claim to reproduce its empirical tables.

Marginal Gaussian QMLE minimizes

$$
\frac1{TJ}\sum_{t,j}\left(\log h^N_{t,j}+\frac{z_{t,j}}{h^N_{t,j}}\right).
$$

Three fixed starts, an analytic gradient and constrained parameter transforms determine each fit. Training uses the preceding 1,008 returns; parameters and initialization are retained while new observed returns update the state. Beyond the first forecast,

$$
\mathbb E_o[h^N_{t+1}]
=\omega\mathbf1+\bigl[(\alpha+\beta)I+\lambda B\bigr]\mathbb E_o[h^N_t].
$$

These daily moments are summed over 1, 5 and 21 sessions. Every method shares the original targets, means, scales and 186 origins per panel. The index is omitted because it has no cross-asset network.

## Results

All **228 selected fits and 684 optimization starts converge**. Mean normalized QLIKE is:

| Model | Sectors 1 | Sectors 5 | Sectors 21 | Countries 1 | Countries 5 | Countries 21 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fractional | 0.62417 | 0.54779 | 0.60432 | 0.52898 | 0.66359 | 0.67147 |
| Fixed 50:50 ensemble | 0.60600 | 0.53714 | 0.59255 | 0.51308 | 0.65094 | 0.66134 |
| GJR-GARCH | 0.60528 | 0.54007 | 0.59305 | 0.50833 | 0.64958 | 0.66226 |
| FIGARCH | 0.65230 | 0.54224 | 0.59664 | 0.54380 | 0.66713 | 0.67299 |
| Network GARCH, weighted | 0.61486 | 0.54653 | 0.60044 | 0.53968 | 0.66425 | 0.67411 |
| Network GARCH, binary | 0.61723 | 0.54901 | 0.60287 | 0.54007 | 0.66480 | 0.67497 |
| Pooled GARCH | 0.63211 | 0.55126 | 0.59969 | 0.52962 | 0.66139 | 0.67254 |

Lower is better. The weighted network beats the standalone fractional model at all sector horizons and loses at all country horizons. All six paired 95% ranges include zero with both five- and ten-origin blocks. The fixed ensemble beats both network versions and the pooled control in all six aggregate comparisons. Against the weighted network, only the one-session country improvement excludes zero at both block lengths; the five-session country finding depends on block length.

Network feedback itself improves the pooled GARCH control at sector horizons 1/5, worsens it at sector horizon 21 and all country horizons, and has intervals including zero throughout. The result remains mixed; adding a direct network reference does not establish universal predictive superiority.

## Verification and reproduction

Eight additional tests check the literal recursion, analytic gradients against numerical derivatives, reduction to ordinary GARCH, conditional forecasts against simulations, asset relabeling, valid estimation, invalid-input rejection and future-price independence. The complete suite passes **61 tests**.

Using the recorded local snapshots and study files:

~~~bash
python run_network_benchmark.py --output artifacts/network-reproduction --publish-assets
~~~

The runner checks source and input hashes and refuses an existing output directory. To specify a separate study on different inputs, supply their paths and a new protocol path with --record-protocol before fitting; preserve the protocol used for each run. A fresh price download is a different snapshot.

The implementation is [network_benchmark.py](../network_benchmark.py). Published outputs include [aggregate and annual metrics](../paper/network_metrics.csv), [asset metrics](../paper/network_asset_metrics.csv), [paired comparisons](../paper/network_comparisons.csv) and [metadata](../paper/network_summary.json). Detailed forecasts and fitting attempts remain in ignored local artifacts.
