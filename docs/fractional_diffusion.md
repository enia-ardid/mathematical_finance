# Fractional memory and graph diffusion for variance

This page describes the first variance specification and its preserved results. The later [persistent asymmetric refinement](refinement.md) improves its pooled scores at all six evaluated panel–horizon combinations and is included in the updated manuscript.

The revised [manuscript](../paper/main.tex) preserves the project's two central ideas while correcting their mathematical role. It models conditional variance directly. The earlier simulator and forecast heads are retained in their own modules; the new implementation is [fractional_diffusion.py](../fractional_diffusion.py).

For log-returns $r_t$, a fitted mean $\mu$ and positive fitted asset scales $s_j^2$, define $z_{t,j}=(r_{t,j}-\mu_j)^2/s_j^2$. The normalized finite fractional-integral kernel is

$$g_0(d)=1,\qquad g_k(d)=\frac{k-1+d}{k}g_{k-1}(d),\qquad
p_k(d)=\frac{g_k(d)}{\sum_{v=0}^{M-1}g_v(d)},\quad 0<d\le1.$$

These positive coefficients come from $(1-B)^{-d}$, with $M=252$ in the experiment. They are fractional-integral weights; the code does not reinterpret the old signed return convolution as a valid fractional derivative. Finite truncation does not establish infinite-range dependence.

Let $L$ be a symmetric graph Laplacian fitted from past shrinkage correlations. The implicit diffusion operator and conditional variance are

$$A_\tau=(I+\tau L)^{-1},\qquad
v_t=s^2\odot\left[(1-\eta)\mathbf1+
\eta A_\tau\sum_{k=0}^{M-1}p_k(d)z_{t-1-k}\right],
\qquad 0\le\eta<1,\quad\tau\ge0.$$

The graph operator is nonnegative and doubly stochastic, preserves constants and contracts graph energy. Consequently, $v_{t,j}\ge(1-\eta)s_j^2$. For fixed parameters and temporally iid unit-second-moment innovations, the squared-return recursion admits a causal stationary solution with finite first moments, hence finite second moments for returns. The paper supplies the coupling proof and distinguishes it from claims about fourth moments, rolling-fit stationarity or predictive accuracy.

Multi-step expected variances follow by replacing future squared innovations with their recursively computed conditional expectations. Gaussian-innovation simulation is also available. The empirical target is a sum of subsequent daily squared returns centered at the common origin-fitted mean, not intraday realized variance or directional price prediction.

The [paper bundle](../paper/README.md) documents the evaluation. There are 186 common forecast origins during 2023–2026. All models use past-only estimation and compare on identical targets. The fractional graph model has lower pooled QLIKE than tuned EWMA at 1, 5 and 21 observations in both panels, but GARCH has lower pooled scores in all six comparisons. The diffusion and exponential-kernel ablations, annual results and descriptive uncertainty ranges remain visible.

```bash
python run_paper.py --synthetic --output artifacts/paper-synthetic
python run_paper.py --data-dir artifacts/yahoo-through-2026-10-16 --output artifacts/paper-study --publish-assets
```

All historical windows were already available in the project. The new protocol prevents per-origin look-ahead and records the finite candidate set; it does not make this a prospective test or establish independent confirmation. Publication claims must follow the measured results and the stated mathematical assumptions.
