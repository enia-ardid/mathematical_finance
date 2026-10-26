# Notes on the original return model

The original return recurrence and its numerical outputs are documented in *ETFs Simulation with Temporal Memory: A Fractional-Diffusive Approach* (14 October 2026) and the notebooks at commit `9194bfbbb354dc9ecc331b932c5968005c4cccc8`. The report's SHA-256 is `d78a5d707833d831b5c3e45b691083c58471e183cc120fcb8e6af0c864758b21`; the file is not redistributed here.

These notes describe the mathematical discrepancies, implementation corrections and interpretation of the historical results. The conditional-variance model in the current manuscript is defined separately in [fractional_diffusion.md](fractional_diffusion.md) and [refinement.md](refinement.md).

## Discrete recurrence and continuous-time interpretation

**Equations (9) and (17) do not describe the same discretization.** From the report's finite GL sum,

$$h^{-\alpha}\sum_{k=0}^{n}w_k r_{n-k}=F_n,$$

isolating the current state gives

$$r_n=h^\alpha F_n-\sum_{k=1}^{n}w_k r_{n-k}.$$

Equation (17) instead **adds** a sum beginning with `w0` applied to `r[n-1]`. Both its sign and lag index differ. At α = 1, unit forcing and r₀ = 0, a first-order GL equation gives `0, 1, 2, 3`; the implemented recurrence gives `0, 1, 2, 2`. The test `test_recurrence_is_not_a_gl_differential_equation` makes this discrepancy explicit. The current implementation preserves and names the discrete model; changing its dynamics to solve the differential equation would require new results and a specified initial-history convention. For a Caputo interpretation with 0 < α < 1, the corresponding GL approximation is applied to `r-r₀`; selecting that interpretation would be an additional modeling decision.

**The log transformation in §5.1 is not derived.** Even for an ordinary spatial derivative,

$$\frac{\partial_{xx}P}{P}=\partial_{xx}\log P+(\partial_x\log P)^2.$$

For geometric Brownian motion, Itô's formula gives

$$d\log P=\left(\mu-\frac{\sigma^2}{2}\right)dt+\sigma\,dW.$$

A fractional derivative does not inherit the ordinary logarithmic chain rule. Passing from a log-price equation to a log-return equation also changes the forcing and initial conditions. Equations (15)–(16) therefore cannot be asserted as consequences of (7). The maintained implementation therefore starts from the discrete return recurrence. See the standard [Black–Scholes derivation](https://www.columbia.edu/~yao/bsnotes04b.pdf) for the classical case.

**The Black–Scholes PDE in (5) uses the wrong squared variable.** With option value $V$, underlying price $S$ and risk-free rate $r_f$, the classical equation is

$$\partial_tV+\frac12\sigma^2S^2\partial_{SS}V+r_fS\partial_SV-r_fV=0.$$

The diffusion coefficient contains $S^2$, not $V^2$. Its pricing measure and terminal payoff also differ from a physical-measure return simulation. Sector-neighbor diffusion is not the spatial derivative of the option-pricing PDE. The code does not price options.

**α = H is an assumption, not a general theorem.** Hurst estimation on returns, a fractional derivative order, fractional Gaussian noise and ARFIMA differencing parameters are not interchangeable. The original preliminary index cell also passed log-prices to a `kind='price'` estimator and mixed estimates on incompatible inputs; it did not supply the final simulator's parameter. The maintained pipeline estimates H consistently on training returns and preserves the declared mapping without claiming identification. [Hosking's fractional differencing model](https://doi.org/10.1093/biomet/68.1.165) is a different stochastic specification, not a justification for substituting H into this recurrence.

**The stated convergence and stability results are unsupported.** The claimed truncation expression and the addition of deterministic and stochastic orders in §4.3 do not establish convergence of the actual coupled recurrence. White noise is not an ordinary pointwise `N(0,1)` function; a Wiener increment has variance proportional to the time step. Because the code fixes the step at one day, its noise scaling does not establish a continuous-time limit. Moderate κ or correlated sectors do not remove stability restrictions. [Yuste and Acedo](https://arxiv.org/abs/cs/0311011) give an example of scheme-specific stability analysis, not a result that can be transferred to this implementation.

**Boundary conditions differ between text and code.** Equation (14) repeats the boundary value in a ghost cell, while both notebooks reflect the adjacent interior value, doubling the boundary second difference. The maintained code documents the reflected convention. The lattice is an imposed ordering of categories; refinements in Δx are not observed refinements of an economic spatial field.

**The computational-complexity claim is incorrect.** Computing weights recursively is O(T), but every simulated step still sums its history. The implemented full-memory algorithm is quadratic in the number of dates, not free of convolution cost.

**Several qualitative statements exceed the model.** Gaussian forcing with fixed coefficients does not establish heavy tails or volatility clustering. A change of parameters over a date interval is not a Dirac impulse. Correlation is not causal contagion. ARIMA models can encode temporal dependence; describing them as having no memory is too broad. The code simulates returns and prices, not a separately estimated volatility process.

## Implementation corrections

| Original issue | Resolution and effect |
| --- | --- |
| Sector drift correction targets `prices_win_df.iloc[split_idx]`, the first test price | All calibration functions now accept training prices only; last training price is the endpoint. The recorded sector experiments use this correction. |
| Sector ordering and coupling-grid scale use full-sample correlations | Both are estimated inside the training-only fit. Changing test prices cannot alter either choice. |
| Implicit `pct_change()` filling can introduce false zero returns | Returns are recomputed from validated positive prices with log differences. Interior missing values fail explicitly. |
| Unseeded Hurst RANSAC can change α between runs | Fixed estimation seed, with caller RNG restored; simulation seeds remain separate. |
| Summing weighted raw ETF prices misinterprets capital weights | Apply weights to individually rebased prices; test invariance to share denomination. |
| sMAPE omits a factor of two relative to its usual 0–200% definition | Formula and name are explicit; values under the revised definition are twice the old convention. |
| Reconstruction metrics include training dates and compare a single selected sector path | Use the full predefined ensemble and evaluate baskets and index on the same held-out dates. |
| Notebook globals, reused function names, fallback variables and stale outputs | One tested implementation supplies the notebook and CLI; current outputs are regenerated. |

The beta notebook additionally omits XLP, hard-codes 1,644 dates and arrays of length ten, mixes dates when integrating prices, uses full-sample fits, and contains outputs from different execution states. It is retained as a labeled historical artifact, not a supported runnable analysis.

## Interpretation of historical scores

The report's index `R² ≈ 0.82` is the score of an averaged price path, not an average of per-seed R² scores with uncertainty. Table 1 labels approximately `0.85` as training R², but the final notebook's stored table reports `0.6755` for training and `0.8507` over the full sample. Table 1 includes κ even though the direct index simulator has no sector coupling. The reconstruction score near `0.7481` compares real and simulated ETF baskets over aligned **all-sample** dates, not held-out predictions of the actual index.

The report says SPY, while the code downloads `^GSPC`; these are different instruments. It also refers to ten sectors and Ward linkage, while the final notebook uses eleven sectors and average linkage on correlation distances. The stated COVID multipliers differ from the final index notebook, and the final sector simulation disables the COVID scenario. The beta and final notebooks therefore represent different specifications.

The local CSVs cover 2021–2026, not the report's period. Their provider and price-adjustment convention could not be recovered. New runs on those files check execution but cannot reproduce or independently substantiate the historical claims. A separate new Yahoo Finance download approximately recovers the index score (R² 0.8164), while the corrected sector reconstruction performs poorly; see the [execution record](results.md). This is a new experiment with recorded provenance, not an exact reproduction of the original snapshot or a validation of the derivation.

## Relation to the variance model

The price-forecast experiments use the discrete recurrence directly, with observed history, conditional moments and chronological calibration. The [variance model](fractional_diffusion.md) instead applies positive fractional-integral weights to squared innovations and smooths them over an estimated correlation graph. Its equations and moment arguments are independent of the original pricing-PDE derivation.

The original report's reference [6], attributed to Klimov and Herrmann, could not be verified by exact-title and author searches. It is not used in the current manuscript. An unsuccessful search does not establish that the reference does not exist.
