# Frozen forward evaluation

This study evaluates future outcomes under fixed rules. It does not reuse the 2023–2026 results as a new test set. The [public protocol](../paper/prospective_protocol.json) fixes the code hashes, dependency versions, calendar, asset panels, comparisons and reporting date. No prospective performance result is available at launch.

## Design

| Item | Fixed rule |
|---|---|
| First origin | October 27, 2026 |
| Last origin | November 2, 2027 |
| Last target session | December 3, 2027 |
| Origin schedule | 52 origins, five exchange sessions apart |
| Refit schedule | Every fifth origin: 25 sessions |
| Panels | Eleven sector ETFs; AGG, IEF, TLT, TIP, GLD, SLV, DBC, VNQ |
| Core | Existing positive fractional-diffusive model; unchanged grids and selection functions |
| Estimation / validation | 1008 / 252 observed returns |
| Target lengths | 1, 5, 21 sessions, following a one-session embargo |
| Capture | First complete snapshot received 17:00–23:59:59 New York time on each scheduled session |
| Forecast deadline | 23:59:59 New York time at the origin |
| Reporting | After the last target's full capture window; no interim performance rankings |

Freezing the model means fixing its specification and update rules. Coefficients may change at scheduled refits. The model is not retuned after inspecting forward evaluation losses. The new cross-asset universe broadens asset coverage but does not create independent market shocks.

Forecasts issued after an origin's close must not be described as preceding the overnight interval that already started at that close. The study skips the next session: an $h$-session prediction sums the existing model's daily forecasts at steps $2,\ldots,h+1$. The first target interval starts at the following session's close, after the issuance deadline. For an October 27 origin, the first scored daily return ends October 29, not October 28. This embargo makes these scores distinct from the earlier historical experiment.

## Matched diffusion comparison

The selected fractional specification retains its chosen memory order, feedback, persistence, asymmetry, mean and scale. Counterfactuals change only diffusion to $0$, $0.25$, $1$ or $4$. This avoids attributing gains from reselection of other parameters to diffusion itself. The main model still selects from its original grid, including zero diffusion.

Ten methods are saved at every eligible origin: the selected fractional model, four matched diffusion strengths, separately selected no-graph and exponential variants, GJR-GARCH, the fixed 50:50 combination and EWMA. The mechanism comparison uses fixed strength $1$ versus $0$, so it remains defined even when the selected model chooses no diffusion.

## OHLC-based variance measurement

The downloader retains raw OHLC, adjusted closes and corporate-action fields returned by Yahoo. Each day's OHLC is multiplied by that day's `Adj Close / Close`; the same adjustment factor applies to all four prices. Missing values, inconsistent extrema and incomplete panels are rejected. There is no filling, ticker substitution, automatic price repair or silent exclusion.

For adjusted prices, define the close-to-open return and intraday Rogers–Satchell term:

$$
g_t=\log(O_t/C_{t-1}),\qquad
RS_t=\log(H_t/O_t)\log(H_t/C_t)+\log(L_t/O_t)\log(L_t/C_t).
$$

The primary measurement is

$$
Y^{\mathrm{range}}_{o,h,j}=\sum_{k=2}^{h+1}
\left[(g_{o+k,j}-\widehat\mu^{\mathrm{gap}}_{o,j})^2+RS_{o+k,j}\right].
$$

The overnight mean is estimated from the trailing fitting window and held fixed until refit. RS estimates intraday variance under a log-Brownian model with drift; discrete extrema introduce measurement error. Combining it with the overnight component requires suitable centering and negligible overnight/intraday covariance to represent whole-day variance. It is **not high-frequency realized variance**, and we do not claim universal unbiasedness for ETF returns. See [Rogers and Satchell (1991)](https://doi.org/10.1214/aoap/1177005835).

The earlier centered squared-return target is retained as a secondary measurement, with the same embargo, origins, fitted means and forecasts. Range targets do not enter model selection. Both targets use normalized QLIKE and normalized squared error. A richer measurement adds information; it does not guarantee a better forecast ranking.

## Statistical decisions and missing observations

The primary family contains four tests: in each of two panels, fixed matched diffusion $1$ versus $0$ at five sessions, and selected fractional versus GJR at 21 sessions, using the range target. Asset losses are averaged within each origin before inference. Two-sided DM tests use Bartlett HAC with four origin lags, with nine lags as a fixed sensitivity; Holm correction covers the four primary tests separately for each lag choice.

Formal primary inference requires all 52 scheduled matched origins. Missing observations therefore cannot silently compress the time axis in HAC calculations. Incomplete coverage produces descriptive scores and an inconclusive primary test. All other strengths, horizons and close-based comparisons are secondary. Fifty-two observations do not guarantee sufficient power: a nonsignificant result does not establish equivalence.

A provider response missing `Close` or `Adj Close` cannot supply an adjustment factor and is rejected; its raw response and error remain in the local attempt log. Retrying within the same capture window is allowed, but filling a missing close or backdating a forecast is not.

The first valid daily snapshot is immutable. Later corrections do not replace targets. Missing captures, late forecasts and failed fits remain visible. Failed scheduled refits invalidate their dependent origins until the next scheduled refit; the code does not silently fit a replacement outside the schedule. No target horizon bridges a missing session. Unexpected exchange closures require a prospective calendar amendment rather than an outcome-dependent data repair.

## Running the study

The operational mode is manual. No background collector is installed. Run one collection pass on each scheduled exchange session within the evening window; the code chooses whether that session also requires a forecast.

The configured study lives in `artifacts/prospective-v1/`. Collection never writes to the existing experiment directories or changes the main model. Raw data and fitted states remain local.

```bash
# Once, before the first origin, in a NEW empty directory:
python run_prospective.py init --study artifacts/prospective-v1 --first-origin 2026-10-27

# Prepare the frozen runtime once (does not install a background service):
python prepare_prospective_runtime.py

# One pass. Outside the frozen evening window this performs no download:
python artifacts/prospective-runtime-v1/run_prospective.py tick --study artifacts/prospective-v1

# Coverage only, without inspecting performance:
python artifacts/prospective-runtime-v1/run_prospective.py status --study artifacts/prospective-v1

# Rejected before the final reporting date:
python artifacts/prospective-runtime-v1/run_prospective.py report --study artifacts/prospective-v1 --output artifacts/prospective-report
```

`tick` is idempotent. It can be invoked hourly by a local scheduler. The machine must be awake, connected and able to obtain the provider's data within the capture window. A missed window is recorded as missing and cannot be backfilled. Provider or machine downtime can make the final primary comparison inconclusive.

Run `python prepare_prospective_runtime.py` to prepare an isolated runtime and a macOS launchd configuration; this command does not install a background service. The frozen runtime is copied separately from the working repository so later development cannot silently change the active model. Python and relevant dependency versions are checked at each invocation. Changes require a separate study version or a dated amendment, not rewriting the original protocol.

Protocol publication before targets provides a public record of the design. The collector itself does not upload data or forecasts. Local clocks and hashes alone are not independent timestamp evidence; externally timestamped forecast commitments would strengthen the audit. No submission, acceptance, prospective accuracy gain or independent preregistration is claimed.

## Operational checks

An October 23 historical OHLC snapshot was used only to test download, model fitting and positive forecasts before launch. Its files are kept under `artifacts/prospective-preflight/`; they are excluded from forward evaluation. Synthetic tests exercise calendar boundaries, daylight saving, immutable captures, the one-session embargo, late forecasts, missing targets and graph-risk identities.
