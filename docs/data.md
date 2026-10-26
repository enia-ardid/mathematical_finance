# Data inputs and provenance

The public demo generates synthetic correlated Gaussian returns with seed 17. These are explicitly synthetic inputs for exercising the code, not replacements for historical prices.

## Local CSV format

Pass a directory containing:

```text
data_finance/
└── raw/
    ├── sp500_close.csv
    └── sector_etfs_close.csv
```

The first CSV has columns `Date,SP500`. The second has `Date,XLB,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY,XLC`; column order may vary. Dates must be unique and increasing, and observed prices must be finite and positive. Use prices with a documented, consistent adjustment convention. The loader trims leading sector inception gaps, aligns observed dates and recomputes log-returns from raw prices. It does not use existing `processed/` files.

The original local data files contain 1,255 prices per asset from **2021-01-05 to 2026-01-02**, with all eleven sectors and no missing values or duplicate dates. Their 1,254 stored log-returns agree with log differences of the raw prices to approximately `2e-15`. Their provider, adjustment convention and download date are undocumented. The return check establishes arithmetic consistency only; the market observations could not be independently verified. The files remain outside the published repository; checksums accompany the [local execution record](results.md).

## Obtain a new documented snapshot

```bash
python download_data.py --output data_finance --start 2018-01-01 --end 2026-10-17
python run_experiment.py --mode forecast --data-dir data_finance --test-start 2026-01-05 --max-prices 10000 --output artifacts/market
```

The downloader requests `^GSPC` and eleven sector ETFs from Yahoo Finance via `yfinance`, with `auto_adjust=True`. The end date is exclusive. It saves raw CSVs and `source.json` containing the provider, request dates, library version, download timestamp and SHA-256 hashes. It refuses to overwrite a populated raw-data directory. See the [download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) for adjustment and date semantics.

A fresh download may differ from historical notebook inputs because upstream data and adjustments can change. Network access and provider availability are required. Keeping a snapshot and its manifest is necessary for comparing runs; this repository does not distribute market-data CSVs or assert permission to redistribute them. `^GSPC` is the index, not SPY, and adjusted ETF returns need not match a price-index return series.

The recorded 2026 forecast experiment uses a snapshot downloaded on 17 October 2026, with common history from 19 June 2018 through 16 October 2026. See the [execution record](results.md) for the development cutoff, evaluation dates and hashes. The data were downloaded retrospectively: chronological code prevents use of future rows, but the snapshot does not establish which adjusted prices were available at each historical date. This is not a prospective or point-in-time data study.

## External country-ETF snapshot

The [external protocol](../paper/external_protocol.json) fixes EWA, EWC, EWG, EWH, EWJ, EWL, EWQ and EWU before their first download for this project. These represent Australia, Canada, Germany, Hong Kong, Japan, Switzerland, France and the United Kingdom. The [issuer's product list](https://www.ishares.com/us/literature/brochure/ishares-product-list-en-us.pdf) documents the funds; their selection reflects geographical spread, not measured forecasting performance.

```bash
python download_external.py --output artifacts/yahoo-international-new
```

The separate downloader reads the frozen universe and dates, requires every asset, validates finite positive prices, performs no filling or ticker substitution, and refuses to overwrite a nonempty directory. It saves `prices.csv` and a manifest with the protocol hash. The recorded snapshot was downloaded on 18 October 2026 at 19:52:00 UTC, after the protocol was written at 19:51:50 UTC. It contains 2,210 prices per asset from 2 January 2018 through 16 October 2026. Its SHA-256 is `eb0262113b33dc8b4b00184a3a98253c520da5cb5b3c0350789425a0253fbcad`.

This is a second universe, not an independent prospective dataset. These US-listed dollar-denominated funds share trading hours and global risk factors with the US development panel. Adjustment revisions, a deliberately selected universe and possible survivorship effects remain limitations. The protocol timestamp is a local audit record rather than independent proof of preregistration.

## Data vintages and forecast availability

A row's trading date and the date on which that version became available are different pieces of information. The rolling code uses rows dated through the forecast origin from a fixed downloaded snapshot. It does not reconstruct historical versions of those rows.

For a single asset, let $P_t^{(T)}$ and $P_t^{(T')}$ denote the price for trading date $t$ in two download vintages. Define $a_t=P_t^{(T')}/P_t^{(T)}$. Then

$$
r_t^{(T')}-r_t^{(T)}
=\log a_t-\log a_{t-1}.
$$

If the same multiplier applies to both prices in a return, it cancels. Uniform back-adjustment over a span therefore leaves returns inside that span unchanged. A change in the multiplier between adjacent dates can change the corresponding return. The [yfinance repair documentation](https://ranaroussi.github.io/yfinance/advanced/price_repair.html) describes missing dividend or split adjustments and corrections to historical prices. Those are possible mechanisms, not evidence that these errors occurred in the study's ETFs.

Changed returns can affect the fitting means and scales, estimated graph, selected parameters and forecasts. A change inside a target window also affects its squared-return proxy and score. The direction of the effect is not known in advance. A single saved vintage cannot measure this sensitivity or establish that revisions favor a particular method. Downloading another historical snapshot now would measure a difference between two snapshots, not recover what was observable at every 2023–2026 origin.

### Requirements for a prospective extension

This is a description of a future study, not a registered protocol or an active collection service.

1. Fix the code version, assets, horizons, refit and validation rules, score, collection period and reporting date before observing the new targets. Keep the existing studies intact.
2. At each scheduled origin, save the complete received price vintage, request and reception times, adjustment convention, corporate-action records where available, and hashes. Archive the fitted state and all benchmark forecasts before the first target session, with an externally verifiable timestamp.
3. Specify the target convention and capture delay before launch. Keep the first eligible target release immutable and retain later corrected versions as a separate sensitivity analysis. Score a given origin against its archived mean and scale.
4. Preserve scheduled origins with missing data, failed fits or late forecasts in the audit log. Define their treatment in advance and report coverage; do not drop difficult periods after seeing their errors.
5. Report after the fixed collection period, regardless of performance, allowing the longest horizon to mature. Any model or rule change creates a separately documented amendment and subsequent evaluation segment.

The existing code and results have not been relabeled as prospective. Such a study needs forecasts recorded before outcomes occur and subsequent elapsed market time.
