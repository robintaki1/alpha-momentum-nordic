# 01 Phase 1 Infrastructure and Data

This document is the first execution spec under the PRD. `AlphaMomentum_PRD_v4.md` is still the master document, but this file translates Phase 1 into a decision-complete implementation spec so the next agent or developer can build the pipeline without guessing.

## 1. Target State and Done Definition

Phase 1 is complete only if all four scripts below work together and `validate.py` returns exit code `0`.

Scripts in scope:

- `python/download_eodhd.py`
- `python/download_riksbank_fx.py`
- `python/build_universe.py`
- `python/validate.py`

Phase 1 gives the green light to Phase 2 only if all of the following are true:

- all required artifacts exist in `data/`
- `security_status_pti_monthly.parquet` exists and `universe_pti.parquet` contains `Full Nordics`, `SE-only`, and liquid-subset eligibility plus the tradability inputs required by later-phase capacity screens
- `prices_adjusted_daily.parquet` exists as a locally reconstructed point-in-time-safe adjusted series and validates against raw prices and corporate actions
- delistings, corporate actions, and FX series are loaded and validated, including vendor active/delisted roster reconciliation with no unresolved in-scope price histories ending more than `30` calendar days before `end_date`
- the Phase 1 artifacts are sufficient to rebuild the current-month eligible universe without manual `live_tickers.csv`
- benchmark prices exist for both the Nordic and global benchmark
- the C++ module and the Python logic return the same answers on a deterministic fixture
- `validate.py` writes a clear final line: `PASS: Phase 1 green`

## 2. Run Order

The run order is fixed and must not be changed in v1:

1. build the C++ module according to the PRD
2. run `download_eodhd.py`
3. run `download_riksbank_fx.py`
4. run `build_universe.py`
5. run `validate.py`

If a step fails, the next step must not be run as if everything were complete.

## 3. Shared Rules and Defaults

The following defaults apply throughout Phase 1 unless the CLI explicitly overrides them:

- `start_date = 1999-01-01`
- `end_date = auto`, which means the most recently completed trading day in `Europe/Stockholm`
- `base_currency = SEK`
- `universe_mode = full_nordics_main_markets`
- `run_se_only_diagnostic = true`
- `min_listing_months = 18`
- `min_price_sek = 20`
- `sim_capital_sek = 250000`
- `max_order_fraction_of_60d_median_daily_value = 0.01`
- `include_delisted = true`
- `primary_passive_benchmark = XACT Norden`
- `secondary_opportunity_cost_benchmark = Vanguard FTSE All-World UCITS ETF`

Precedence for config is fixed:

1. CLI arguments
2. `config.py`
3. hardcoded default in the script

All scripts must use the same precedence rule.

Capacity screening is intentionally deferred. In Phase 1, `build_universe.py` must store tradability inputs such as `median_daily_value_60d_sek`, but it must not apply a `TOP_N`-dependent liquidity exclusion because `TOP_N` is selected later in Phase 2.

## 4. Required Artifacts

Phase 1 must produce these files in `data/`:

- `security_master.parquet`
- `security_status_pti_monthly.parquet`
- `prices_raw_daily.parquet`
- `prices_adjusted_daily.parquet`
- `corporate_actions.parquet`
- `delisted_metadata.parquet`
- `benchmark_prices.parquet`
- `riksbank_fx_daily.parquet`
- `universe_pti.parquet`

`security_master.parquet` is the static identity/reference table for securities. It must not be used as the source of historical eligibility, historical primary-share status, or historical market-cap state.

`security_status_pti_monthly.parquet` is the mandatory point-in-time artifact for listing status, primary-share status, main-universe eligibility, and monthly market-cap state. `build_universe.py` must derive and write it before constructing `universe_pti.parquet`.

`prices_adjusted_daily.parquet` is the locally reconstructed point-in-time-safe adjusted series. Vendor-adjusted prices, if downloaded, are QA/reference-only and are not the production adjusted-price artifact.

All output files must be written atomically:

- write first to `data/staging/`
- validate the file locally
- then move it to the final name
- if the script fails, a previously working final file must not be overwritten with half-finished content

## 5. Script Contracts

### 5.1 `download_eodhd.py`

Purpose:

- fetch Nordic security metadata
- fetch daily raw OHLCV
- fetch corporate actions
- fetch metadata for delisted companies
- fetch benchmark prices
- optionally fetch vendor-adjusted history for QA/reference only

CLI:

```bash
python python/download_eodhd.py \
  --start 1999-01-01 \
  --end auto \
  --universe-mode full_nordics_main_markets \
  --include-delisted true \
  --out-dir data
```

Inputs:

- `EODHD_API_KEY` via environment or `config.py`
- normalized exchange and instrument configuration in the script
- benchmark configuration in the script or `config.py`, including configured benchmark IDs and benchmark types

Outputs:

- `data/security_master.parquet`
- `data/prices_raw_daily.parquet`
- `data/corporate_actions.parquet`
- `data/delisted_metadata.parquet`
- `data/benchmark_prices.parquet`

Decisions that must be hardcoded in v1:

- the main universe is only `Nasdaq Stockholm`, `Nasdaq Helsinki`, `Nasdaq Copenhagen`, `Oslo Bors`
- only common shares with a primary Nordic listing may be included
- `First North`, `NGM`, `Spotlight`, `Euronext Growth`, ETFs, certificates, preferred shares, subscription rights, ADR/GDR, SPAC-like vehicles, and other non-ordinary equity instruments must not be loaded into the main universe
- `security_id` must be vendor-stable and in v1 must be set to `eodhd_symbol`
- active and delisted names must not be mixed together; status must be explicit
- vendor roster membership must be preserved per security as `active_only`, `delisted_only`, or `both`

Normalization rules:

- keep the vendor symbol as `eodhd_symbol`
- add normalized columns `country_code`, `exchange_group`, `security_type_normalized`, `share_class_normalized`, `issuer_group_id`
- `country_code` is issuer domicile/reference metadata only; listing-venue filters such as `SE-only` must use point-in-time primary listing venue via `exchange_group`
- preserve the metadata required to determine the most liquid ordinary primary share later in point-in-time monthly status derivation
- other share classes may exist in `security_master.parquet`, but point-in-time primary-share and main-universe eligibility are written later to `security_status_pti_monthly.parquet`
- benchmark rows must be tagged with the configured benchmark ID and an explicit `benchmark_type` of `total_return` or `proxy`

Error handling and stop conditions:

- missing API key => exit code `1`
- if one of the four Nordic main exchanges cannot be loaded => exit code `1`
- if benchmark series cannot be loaded => exit code `1`
- if a benchmark series cannot be mapped to the configured benchmark ID and `benchmark_type` => exit code `1`
- if vendor active/delisted roster membership cannot be reconciled for an in-scope security => exit code `1`
- if more than `1%` of symbol downloads fail => exit code `1`
- if symbol failures are `<= 1%`, the script may still stop with `1`; v1 must be strict and not allow silent partial success
- if output contains duplicates on `(security_id, date)` => exit code `1`

### 5.2 `download_riksbank_fx.py`

Purpose:

- fetch FX series for `NOK/SEK`, `DKK/SEK`, `EUR/SEK`
- store them as a pure observation series without forward fill

CLI:

```bash
python python/download_riksbank_fx.py \
  --start 1999-01-01 \
  --end auto \
  --base-currency SEK \
  --pairs NOK,DKK,EUR \
  --out data/riksbank_fx_daily.parquet
```

Inputs:

- The Riksbank SWEA API
- Required v1 series IDs: `SEKNOKPMI`, `SEKDKKPMI`, `SEKEURPMI`

Outputs:

- `data/riksbank_fx_daily.parquet`

Rules:

- store only actual observation dates
- no forward fill in the stored file
- `build_universe.py` must later perform an `asof` join to the latest available rate on or before the current trading day
- observations after `end_date` are allowed in the raw response but ignored for freshness checks

Error handling and stop conditions:

- if any of the currencies `NOK`, `DKK`, `EUR` is missing entirely => exit code `1`
- if the latest observation on or before `end_date` for any required currency is more than `7` calendar days older than `end_date` => exit code `1`
- if `sek_per_ccy <= 0` occurs => exit code `1`

### 5.3 `build_universe.py`

Purpose:

- construct locally reconstructed point-in-time-safe adjusted prices
- derive monthly point-in-time security status
- create a point-in-time universe per month
- create `Full Nordics`, `SE-only`, and a liquidity-near subset
- mark exact variant-specific exclusion reasons per security and month

CLI:

```bash
python python/build_universe.py \
  --input-dir data \
  --universe-mode full_nordics_main_markets \
  --emit-se-only true \
  --emit-liquid-subset true \
  --out data/universe_pti.parquet
```

Inputs:

- `data/security_master.parquet`
- `data/prices_raw_daily.parquet`
- `data/corporate_actions.parquet`
- `data/delisted_metadata.parquet`
- `data/riksbank_fx_daily.parquet`

Output:

- `data/security_status_pti_monthly.parquet`
- `data/prices_adjusted_daily.parquet`
- `data/universe_pti.parquet`

Fixed rules:

- the rebalance anchor is exchange-specific: `anchor_trade_date` is the last trading day of the month for each `exchange_group`
- `next_execution_date` is the first trading day of the following month for that same `exchange_group`
- each symbol may carry its latest observed trading day on or before `anchor_trade_date` into the PTI artifacts for diagnostics
- a security is rankable only if `asof_trade_date = anchor_trade_date`
- if `asof_trade_date < anchor_trade_date`, the row may remain in the PTI artifacts for reporting/diagnostics only and must be ineligible in any variant that would otherwise be eligible, using `stale_last_trade` as the exclusion reason for that variant
- if the latest observed trading day is more than `5` calendar days older than `anchor_trade_date`, the security remains stale and ineligible
- no forward fill of price or volume is allowed
- `MIN_LISTING_MONTHS = 18`
- `MIN_PRICE_SEK = 20` on raw close, converted to SEK using the latest available Riksbank rate on or before the symbol's as-of day
- `build_universe.py` must compute and persist `close_raw_sek`, `trade_value_sek`, and `median_daily_value_60d_sek` as tradability inputs for later phases
- `trade_value_sek` and `close_raw_sek` must use the latest available Riksbank rate on or before the same trading date and must never use a future FX observation
- `median_daily_value_60d_sek` must be calculated over the trailing `60` trading days ending at `asof_trade_date`
- the runtime capacity rule using `SIM_CAPITAL_SEK / TOP_N` is not applied in Phase 1; later phases apply it using the run's current `TOP_N`
- delisted before `next_execution_date` => not eligible
- `build_universe.py` must first derive `security_status_pti_monthly.parquet` and `prices_adjusted_daily.parquet`, then derive `universe_pti.parquet`
- for each `rebalance_month` and `issuer_group_id`, the selected primary share is the Nordic ordinary-share listing with the highest trailing `60`-trading-day median `trade_value_sek` ending at `asof_trade_date`
- fixed tie-breakers for primary-share selection are: higher `market_cap_sek_asof_anchor`, then older `listing_date`, then lexical `security_id`

Universe variants:

- `Full Nordics`: all eligible securities on Nordic main lists
- `SE-only`: same rules but only if the point-in-time primary listing venue is `Nasdaq Stockholm` main market; issuer domicile must not be used for this filter
- `largest-third-by-market-cap`: the top third of already eligible securities in each month, sorted by point-in-time `market_cap_sek_asof_anchor` from `security_status_pti_monthly.parquet`

Market-cap rule:

- `build_universe.py` must use point-in-time monthly market cap from `security_status_pti_monthly.parquet`
- if `market_cap` is missing for more than `5%` of otherwise eligible securities in a rebalance month, the script must exit with code `1`
- v1 must not silently switch to daily value as a fallback for this subset; if market cap is missing, that is a data problem that must stop the run

Variant-specific exclusion reasons that must be explicit in the output:

- `non_primary_listing`
- `non_common_share`
- `excluded_venue`
- `outside_se_only_scope`
- `too_short_listing_history`
- `price_below_min_sek`
- `missing_price_or_volume`
- `stale_last_trade`
- `delisted_before_execution`
- `bad_corporate_action_data`
- `bad_outlier_data`
- `outside_liquid_subset_cut`

Error handling and stop conditions:

- missing input file => exit code `1`
- if an otherwise eligible common-share candidate is missing `issuer_group_id` => exit code `1`
- if a variant marks the same security eligible and also assigns that variant a non-empty exclusion reason in the same month => exit code `1`
- if `SE-only` is empty in any month after `2005-01` => exit code `1`
- if `Full Nordics` is empty in any month after `2005-01` => exit code `1`

### 5.4 `validate.py`

Purpose:

- validate that Phase 1 is logically and technically complete
- check data artifacts, schemas, data quality, FX, universe, and C++/Python parity

CLI:

```bash
python python/validate.py \
  --input-dir data \
  --require-cpp true \
  --require-benchmarks true
```

Inputs:

- all required artifacts in `data/`
- built C++ module in the project root or a defined import path
- reference logic in `engine.py`

Output:

- terminal report with sections
- exit code `0` on pass
- exit code `1` on fail

Sections that must be printed:

- `Artifacts`
- `Schema`
- `Raw prices`
- `Adjusted prices`
- `Corporate actions`
- `Delistings`
- `FX`
- `Universe`
- `Benchmarks`
- `CPP parity`
- `Final verdict`

`validate.py` may write warnings, but warnings must not replace fail in blocking checks.

## 6. Data Model

### 6.1 `security_master.parquet`

One row per `security_id`.

This is the static identity/reference table. It must not be used as the source of historical listing eligibility, historical primary-share status, or historical market-cap state.

`country_code` in this file is issuer domicile/reference metadata only. Listing-venue filters such as `SE-only` must use point-in-time primary listing venue via `exchange_group`.

Required columns:

- `security_id`
- `eodhd_symbol`
- `ticker_local`
- `isin` nullable
- `company_name`
- `country_code`
- `exchange_name`
- `exchange_group`
- `issuer_group_id`
- `vendor_roster_status`
- `currency`
- `security_type_raw`
- `security_type_normalized`
- `share_class_raw`
- `share_class_normalized`
- `listing_date`
- `delisting_date` nullable
- `is_delisted`
- `shares_outstanding` nullable
- `last_metadata_refresh_ts`

Allowed `vendor_roster_status` values in v1:

- `active_only`
- `delisted_only`
- `both`

### 6.2 `security_status_pti_monthly.parquet`

Unique key: `(rebalance_month, security_id)`

This is the point-in-time monthly status table used to construct historical universe membership and the monthly market-cap subset. It is also where the deterministic monthly primary-share selection result is persisted.

`SE-only` must be derived from point-in-time primary listing venue in `exchange_group`, not from issuer domicile metadata.

Required columns:

- `rebalance_month`
- `anchor_trade_date`
- `next_execution_date`
- `security_id`
- `country_code`
- `exchange_group`
- `is_listed_asof_anchor`
- `delisted_before_next_execution`
- `is_primary_common_share_asof_anchor`
- `eligible_for_main_universe_asof_anchor`
- `listing_age_months`
- `market_cap_local_asof_anchor` nullable
- `market_cap_sek_asof_anchor` nullable
- `status_source`

### 6.3 `prices_raw_daily.parquet`

Unique key: `(security_id, date)`

Required columns:

- `security_id`
- `date`
- `open_raw`
- `high_raw`
- `low_raw`
- `close_raw`
- `volume`
- `currency`
- `trade_value_local` nullable
- `trade_value_sek` nullable
- `source`

`trade_value_local` must in v1 be calculated as `close_raw * volume` if the vendor does not provide daily traded value.

`trade_value_sek` must use the latest available Riksbank rate on or before that same trading date and must never use a future FX observation.

### 6.4 `prices_adjusted_daily.parquet`

Unique key: `(security_id, date)`

This is the locally reconstructed point-in-time-safe adjusted series. It must be built from `prices_raw_daily.parquet` and `corporate_actions.parquet` using only corporate actions effective on or before each date. Vendor-adjusted prices, if downloaded, are QA/reference-only and are not this artifact.

Required columns:

- `security_id`
- `date`
- `adj_close`
- `adj_factor`
- `currency`
- `source`

The signal model in later phases may only read `adj_close` from this file, not attempt to reconstruct adjustments ad hoc and not read vendor `adj_close` as a production source.

### 6.5 `corporate_actions.parquet`

One row per event.

Required columns:

- `security_id`
- `event_date`
- `action_type`
- `action_value`
- `action_ratio` nullable
- `currency` nullable
- `source`

Allowed `action_type` values in v1:

- `split`
- `reverse_split`
- `cash_dividend`
- `special_dividend`
- `cash_merger`
- `delisting_cashout`

### 6.6 `delisted_metadata.parquet`

One row per delisted security.

Every security whose `vendor_roster_status` is `delisted_only` or `both` must appear in this file.

Required columns:

- `security_id`
- `eodhd_symbol`
- `company_name`
- `country_code`
- `exchange_name`
- `currency`
- `listing_date`
- `delisting_date`
- `delisting_reason` nullable
- `last_trade_date` nullable
- `cashout_value` nullable
- `source`

### 6.7 `benchmark_prices.parquet`

Unique key: `(benchmark_id, date)`

Required columns:

- `benchmark_id`
- `benchmark_name`
- `benchmark_type`
- `date`
- `currency`
- `close_raw`
- `adj_close`
- `close_sek`
- `source`

`benchmark_id` must at minimum exist for:

- primary Nordic benchmark
- secondary global benchmark

Allowed `benchmark_type` values in v1:

- `total_return`
- `proxy`

### 6.8 `riksbank_fx_daily.parquet`

Unique key: `(currency, date)`

Required columns:

- `currency`
- `date`
- `sek_per_ccy`
- `source`

### 6.9 `universe_pti.parquet`

One row per `(rebalance_month, security_id)`

Required columns:

- `rebalance_month`
- `anchor_trade_date`
- `next_execution_date`
- `security_id`
- `country_code`
- `exchange_group`
- `currency`
- `asof_trade_date`
- `listing_age_months`
- `close_raw_local`
- `close_raw_sek`
- `trade_value_sek`
- `median_daily_value_60d_sek`
- `market_cap_sek`
- `is_eligible_full_nordics`
- `is_eligible_se_only`
- `is_eligible_liquid_subset`
- `exclusion_reason_full_nordics`
- `exclusion_reason_se_only`
- `exclusion_reason_liquid_subset`

`exclusion_reason_full_nordics` must be empty if and only if `is_eligible_full_nordics = true`.

`exclusion_reason_se_only` must be empty if and only if `is_eligible_se_only = true`.

`exclusion_reason_liquid_subset` must be empty if and only if `is_eligible_liquid_subset = true`.

The same security may be eligible in one variant and excluded in another in the same rebalance month.

`market_cap_sek` in this file is the point-in-time as-of-anchor market cap copied from `security_status_pti_monthly.parquet`.

`anchor_trade_date` and `next_execution_date` must follow the row's `exchange_group` trading calendar.

`close_raw_sek` and `trade_value_sek` must use the latest available Riksbank rate on or before `asof_trade_date`, and `median_daily_value_60d_sek` must use the trailing `60` trading days ending at `asof_trade_date`.

If `asof_trade_date < anchor_trade_date`, the row may remain in this file for diagnostics/reporting only, but it must be ineligible in any variant that would otherwise be eligible and must not be ranked or traded.

`SE-only` eligibility must be derived from point-in-time primary listing venue in `exchange_group`, not from `country_code`.

## 7. Validation Rules

### 7.1 Data and schema

`validate.py` must fail if:

- any required artifact is missing
- any file is empty
- required columns are missing
- primary keys are not unique
- dates are not ascending within each instrument
- a row in `universe_pti.parquet` has no matching `(rebalance_month, security_id)` row in `security_status_pti_monthly.parquet`

### 7.2 Raw and adjusted prices

`validate.py` must fail if:

- `open_raw`, `high_raw`, `low_raw`, `close_raw` are `<= 0`
- `volume < 0`
- `trade_value_local` is inconsistent with `close_raw * volume`
- `trade_value_sek` does not match the correct `asof` FX rate on or before the same trading date
- `trade_value_sek` uses a future FX observation
- `high_raw < max(open_raw, close_raw, low_raw)`
- `low_raw > min(open_raw, close_raw, high_raw)`
- `adj_close <= 0`
- `adj_factor <= 0`
- locally reconstructed adjusted prices are inconsistent with raw prices and the corporate actions effective on or before each date
- an adjustment change appears before the corresponding corporate-action effective date
- price is outside the reasonable interval `1 <= close_raw <= 100000` after already known bad rows have been filtered out
- an extreme price move over `70%` between two observation days lacks a matching corporate action

### 7.3 Corporate actions and delistings

`validate.py` must fail if:

- split/reverse split exists but is not reflected in the locally reconstructed `adj_factor` from the event date onward
- a corporate action that should affect the adjusted series is missing from the local adjusted-price reconstruction on or after its effective date
- a delisted security is missing `delisting_date`
- a security has price data long after `delisting_date` without explanation in metadata
- a cashout/delisting action exists but the corresponding security does not exist in delisted metadata
- an in-scope security with historical prices is missing `vendor_roster_status`
- a security whose `vendor_roster_status` is `delisted_only` or `both` is missing from `delisted_metadata.parquet`
- a security with in-scope historical prices ends more than `30` calendar days before `end_date` and has neither delisting evidence nor explicit vendor active-roster evidence

### 7.4 FX

`validate.py` must fail if:

- `NOK`, `DKK`, or `EUR` is missing from the FX file
- `sek_per_ccy <= 0`
- the latest observation on or before `end_date` for any required currency is more than `7` calendar days older than `end_date`
- gaps in the observation series prevent an `asof` join from providing a rate on or before the relevant trading day
- a required currency is treated as fresh only because of observations after `end_date`

### 7.5 Universe

`validate.py` must fail if:

- a security from an excluded venue appears as eligible
- a security appears as `SE-only` eligible even though its point-in-time primary listing venue is not `Nasdaq Stockholm` main market
- a non-common share appears as eligible
- an otherwise eligible common-share candidate is missing `issuer_group_id`
- primary-share selection within an `issuer_group_id` does not match the documented trailing `60`-trading-day median `trade_value_sek` rule and fixed tie-breakers
- a security with `listing_age_months < 18` appears as eligible
- a security with `close_raw_sek < 20` appears as eligible
- a security with `asof_trade_date < anchor_trade_date` appears as eligible in any variant
- a row with `asof_trade_date < anchor_trade_date` would otherwise satisfy a variant's non-stale filters but is not marked with `stale_last_trade` for that variant
- `close_raw_sek` does not match the correct `asof` FX rate on or before `asof_trade_date`
- `median_daily_value_60d_sek` does not match the documented trailing `60`-trading-day calculation ending at `asof_trade_date`
- an eligible security in any variant has null or nonpositive `close_raw_sek`, `trade_value_sek`, or `median_daily_value_60d_sek`
- `anchor_trade_date` is not the last trading day of the month for that row's `exchange_group`
- `next_execution_date` is not the first trading day of the following month for that row's `exchange_group`
- `is_eligible_full_nordics = true` and `exclusion_reason_full_nordics` is not empty, or `is_eligible_full_nordics = false` and `exclusion_reason_full_nordics` is empty
- `is_eligible_se_only = true` and `exclusion_reason_se_only` is not empty, or `is_eligible_se_only = false` and `exclusion_reason_se_only` is empty
- `is_eligible_liquid_subset = true` and `exclusion_reason_liquid_subset` is not empty, or `is_eligible_liquid_subset = false` and `exclusion_reason_liquid_subset` is empty
- `SE-only` or `Full Nordics` is missing monthly rows after the backtest history has started

### 7.6 Benchmarks

`validate.py` must fail if:

- configured benchmark IDs are missing or duplicated
- the configured Nordic benchmark is missing from `benchmark_prices.parquet`
- the configured global benchmark is missing from `benchmark_prices.parquet`
- a configured benchmark is missing explicit `benchmark_type`
- a configured benchmark uses a `benchmark_type` other than `total_return` or `proxy`
- benchmark series contain nonpositive prices
- benchmark series cannot be converted to SEK
- benchmark series are too gappy to support the strategy's monthly sampling convention
- benchmark series do not cover the full reporting and evaluation window without explicit justification
- configured benchmark monthly sampling does not align with the strategy's monthly convention

### 7.7 C++ parity

`validate.py` must build or read a small deterministic fixture from real Phase 1 data and verify:

- the same signal values in Python and C++ within tolerance `1e-10`
- the same sort order for top names in the fixture month
- the same number of eligible securities after filtering

If the C++ module is missing or cannot be imported in Phase 1, `validate.py` must fail.

## 8. Green Light to Phase 2

Phase 2 may start only if everything below is true:

- `download_eodhd.py` succeeded without partial success
- `download_riksbank_fx.py` succeeded without missing currencies
- the latest FX observation on or before `end_date` for each required currency is not stale by more than `7` calendar days
- `build_universe.py` created `security_status_pti_monthly.parquet`, `prices_adjusted_daily.parquet`, and `universe_pti.parquet`
- `universe_pti.parquet` contains `Full Nordics`, `SE-only`, and liquid-subset eligibility plus the tradability inputs required by later-phase capacity screens
- the Phase 1 artifacts are sufficient to rebuild the current-month eligible universe without manual `live_tickers.csv`
- delisting roster reconciliation is complete and there are no unresolved in-scope price histories ending more than `30` calendar days before `end_date`
- `validate.py` returned exit code `0`
- benchmark prices exist for the configured Nordic and global benchmark IDs, with explicit `benchmark_type`
- `validate.py` wrote `PASS: Phase 1 green`

If any point fails, the status must be `Phase 1 not complete`.

## 9. Subsequent Docs

When this spec is implemented and stable, the later markdown specs should be organized like this:

1. `02_phase2_research_validation.md` — current combined Phase 2 and Phase 3 spec
2. Optional future split docs for Phase 4 and Phase 5 operations if those procedures later need their own markdown specs

The later documents must not be written as if Phase 1 already works unless `validate.py` is actually green.

The later documents must use the Phase 1 artifacts as the source of truth for rebuilding the live universe and must not reintroduce `live_tickers.csv` as a curated live-universe source.
