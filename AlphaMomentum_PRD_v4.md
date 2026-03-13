# PRODUCT REQUIREMENTS DOCUMENT

**Alpha-Momentum Nordic**

**Trading System**

*C++ Core · pybind11 · Gated Deployment · Nordic exchanges*

| Field | Value |
| --- | --- |
| Version | 4.0 — Gated Deployment |
| Date | March 2026 |
| Critical rule | The live engine is NOT built before the backtest is validated |
| Quick Run | ~2–5 min — daily iteration |
| Mega Run | ~15–30 min — robust evening run when needed |
| Paper trading | 6 months mandatory before real money |
| Simulation capital | 250 000 SEK default in backtest and capacity control |
| Execution | Signal at month-end, execute at the next trading day's open |
| Live data | EODHD — same provider for backtest and live signal |

## ⚡ The Most Important Rule in This Document

🛑 STOP: The live engine is NOT built and is NOT run before backtest and untouched final holdout validation are fully complete.

There is a logical order that must not be broken. The live engine is meaningless — and potentially harmful to your capital — if you do not know that the strategy actually holds out-of-sample. Building a polished Streamlit dashboard and starting to follow the signals before you have validated the strategy is putting the cart before the horse.

> **Why the order is critical**
>
> - The parameters in the live engine must come from untouched final holdout validation — not from gut feel.
> - If untouched final holdout validation fails (OOS Sharpe < 0.3), you must NOT proceed to paper trading or live.
> - Paper trading reveals whether the signals are practically executable — spreads, liquidity, turnover.
> - You need 6 months of data to know whether you can psychologically stick to the system.
> - Most people abandon their strategy exactly when it starts recovering after a drawdown.

### Run Types

| Run type | Primary phase | Purpose | Must NOT do |
| --- | --- | --- | --- |
| `Signal Run` | Phase 4-6 | Monthly operational signal generation with the already locked parameter tuple | No grid search, bootstrap, negative controls, or parameter selection |
| `Validation Run` | Phase 2 | Lightweight research iteration, mapped to `Quick Run` | No holdout usage and no live/paper-trading execution |
| `Mega Validation Run` | Phase 2 | Heavier research certification, mapped to `Mega Run`, and the only profile allowed to lock parameters | No holdout usage and no auto-promotion if consistency checks warn |
| `Untouched Final Holdout` | Phase 3 | One-time post-selection gate on unseen data | No parameter reselection or tuning after the run |

## 0. Literature & Motivation

The strongest external evidence in this document supports Nordic medium-term price momentum as the baseline — not a fully proven claim of unique retail alpha. The project name comes from the alpha-momentum/QMOM literature, but PRD v4 uses a simpler production core because it is currently better supported by both the literature and the project's goal: personal, monthly investing with realistic simulation.

### 0.1 Evidence Hierarchy

- **Production baseline:** pure Nordic price momentum with realistic data, liquidity, and execution assumptions.
- **Research hypothesis:** alpha/residual momentum may exist as a research track, but must NOT become the live default unless it first beats the price baseline out-of-sample, after costs and under paper trading.
- **Practical decision criterion:** if the strategy does not beat inexpensive passive benchmark alternatives after modeled costs, the default decision should be passive index investing.

### 0.2 Core References and How They Are Used

| Source | Type | What it actually contributes | PRD consequence |
| --- | --- | --- | --- |
| Grobys, Fatmy & Rajalin (2024), *Applied Economics* | Peer-reviewed article | Strongest direct support that momentum still exists in the Nordics. Also shows that combinations do not necessarily beat pure momentum after broader risk adjustment. | Supports keeping the main track simple and momentum-driven. |
| Nilsson & Picone (2021), GU | Master's thesis | Direct Nordic support for alpha/idiosyncratic momentum and lower crash risk relative to price momentum in their sample. | Motivates alpha as a research track, but not as the production default. |
| Jacobsen & Nyhegn (2021), CBS | Master's thesis | Support for momentum in the Nordics 2007–2021, but the effect is clearer in small companies. | Motivates liquidity filters, capacity control, and caution around small-cap-driven alpha. |
| Annerstedt & Schönström (2006), Lund | Master's thesis | Historical support for momentum in the Nordics, especially on the 3–12 month horizon. | Good historical motivation, but too old to justify today's net expectations on its own. |
| Kobelyats'kyy & Fulgentiusson (2011), Lund | Student thesis | Swedish momentum appears to exist and is not well explained by CAPM/FF3. | Supports the baseline hypothesis, but not a Nordic live implementation by itself. |
| Bergsten (2019), Uppsala | Master's thesis | Price momentum appears to beat intermediate past returns in Sweden; crash risk is real. | Supports a simple baseline and explicit crash-risk reporting. |
| Savolainen (2024), LUT | Master's thesis | Value+momentum combinations can improve the risk profile, but standalone momentum is not unambiguously strong in all variants. | Motivates baselines and passive benchmark comparisons, not greater model complexity now. |
| Alpha Architect QMOM investment case | Marketing material | Good process inspiration around screening, liquidity, and rebalance. Not independent evidence. | May be used as design inspiration, not as validation of alpha. |
| Björkman & Durling (2018), Linköping | Master's thesis | HFT affects Swedish market quality, liquidity, and volatility. Not a momentum study. | Motivates execution realism, but does not validate the signal. |

### 0.3 Conclusion from the Literature

- There is good support that Nordic price momentum is a reasonable hypothesis to backtest and potentially use.
- There is some support for alpha/residual momentum in the Nordics, but the support is thinner and methodologically more fragile than for pure price momentum.
- Several studies indirectly suggest that much of the premium may be stronger in smaller and less liquid companies. Therefore capacity filters, price filters, and a point-in-time universe are not "extra features" but central realism requirements.
- Many student studies ignore or simplify trading costs. The PRD should therefore be stricter than the literature in cost modeling and benchmark comparisons.
- The literature does not prove that a concentrated long-only retail portfolio should automatically inherit the alpha from academic long-short portfolios. Long-short results must therefore never be treated as direct live evidence for this project.
- Several Nordic studies suggest that equal weighting is more realistic than value weighting in a concentrated Nordic strategy, because single large names can otherwise dominate the outcome disproportionately.
- Swedish evidence also suggests that simple price momentum appears more robust than `intermediate past returns`, reinforcing the decision not to broaden the signal in this revision.

### 0.4 Non-Negotiable Implications for This Project

- EODHD with delisted history is mandatory in universe construction.
- `equal-weight`, simple `12-1` price momentum, and passive index alternatives must always be reported side by side with the main strategy.
- External FF3/FF5 residualization does not belong in the main track before region, currency, publication timing, and factor construction have been validated point-in-time.
- The project is built for personal, monthly systematic investing through a regular broker — not for HFT or intraday execution.
- Long-short academic results may be used as context, but not as a decision basis for live without separate validation in long-only format.
- The production universe must be kept explicitly Nordic and the main track must be limited to common shares on Nordic main lists, not all smaller lists that happen to exist at the broker.

## 1. Phase Overview — Gated Pipeline

The system is built in six phases with explicit gates between each phase. Whoever implements the system must NOT start the next phase until the current phase is complete and approved.

> **PHASE 1 — Infrastructure & Data** `[ACTIVE]`
>
> - Build the C++ core with pybind11.
> - Run `validate.py` — C++ == Python `engine.py`.
> - Obtain backtest data via EOD Historical Data - All World (primary source).
> - Download Riksbank FX data.
> - Build a clean point-in-time universe plus the tradability inputs used by later-phase capacity screens.

> **PHASE 2 — Backtest & In-Sample Rolling-Origin Validation** `[LOCKED]`
>
> - Quick Run — 36 combinations, ~2–5 min, lightweight validation profile for research iteration.
> - Iterate parameters until you are satisfied.
> - Mega Run — 168 combinations, ~15–30 min, heavier validation profile for final in-sample selection.
> - Choose the best parameters based on in-sample rolling-origin validation, Sharpe, and plateau stability.
> - Run the Mega consistency check on the locked candidate versus the lighter validation profile.
> - Run baselines and execution sensitivity without new parameter selection.
> - 🔒 Unlocks when: Phase 1 is fully complete and `validate.py` gives the green light.

> **PHASE 3 — Untouched Final Holdout Validation** `[LOCKED]`
>
> - Run the untouched final holdout ONCE with Mega Run's best parameters.
> - Accept the results regardless of outcome — no tweaks afterward.
> - If OOS Sharpe < 0.3: go back to Phase 2 and investigate why. No paper trading or live before new validation.
> - Document: in-sample vs out-of-sample, baselines, and execution sensitivity.
> - 🔒 Unlocks when: Phase 2 is complete and Mega Run has produced the best parameters.

> **PHASE 4 — Paper Trading (6 months)** `[LOCKED]`
>
> - Build `live_signal.py` with the untouched holdout's validated parameters.
> - Run the monthly Signal Run via EODHD — the same provider as in backtest.
> - Document each month's signal, reference price, actual fill, and what actually happened.
> - Review: is turnover reasonable? Are the companies actually buyable?
> - Review: can you psychologically follow the system?
> - 🔒 Unlocks when: untouched final holdout validation shows OOS Sharpe >= 0.3.

> **PHASE 5 — Live Deployment (optional)** `[LOCKED]`
>
> - Build the full Streamlit dashboard.
> - Start with a small capital allocation — max 20% of the intended amount.
> - Evaluate monthly against the paper-trading benchmark.
> - Increase allocation gradually if live matches paper trading.
> - 🔒 Unlocks when: 6 months of paper trading are complete and documented.

> **PHASE 6 — Continuous Management** `[LOCKED]`
>
> - Monthly rebalancing using exchange-specific `anchor_trade_date` and `next_execution_date`.
> - Quarterly review of data quality, benchmark gap, and real trading costs.
> - Annual re-validation — run a fresh untouched final holdout validation with newly appended unseen data.
> - If annual re-validation raises a consistency warning, freeze parameter changes until a manual review explains it.
> - Document deviations between signal and actual execution.
> - 🔒 Unlocks when: Phase 5 is active and functioning.

## 2. Phase 1 — Infrastructure & Data

### 2.1 C++ Build Environment

#### Installation per platform

```bash
# Mac
brew install cmake eigen pybind11
xcode-select --install

# Ubuntu/Debian
sudo apt install cmake libeigen3-dev python3-dev g++
pip install pybind11

# Windows
# 1. Visual Studio Build Tools 2022 (free)
#    Choose: C++ build tools + Windows 11 SDK
# 2. vcpkg
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg install eigen3 pybind11
```

#### Build and verify

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
cp alpha_momentum_cpp*.so ../   # Linux/Mac
cp alpha_momentum_cpp*.pyd ../  # Windows
python -c "import alpha_momentum_cpp; print('C++ OK')"
python python/validate.py  # Should print: ✓ Validation OK
```

🛑 STOP: Do NOT proceed to Phase 2 if `validate.py` fails. All backtest results are unreliable if C++ and Python give different answers.

### 2.2 Data Sources

#### Backtest data

> **Borsdata vs EODHD — decision guide**
>
> **EOD Historical Data - All World (20 USD/month) — primary source:**
>
> - Active subscription used for backtest data in this project.
> - Confirmed to have delisted companies.
> - Known decimal errors in Nordic data — filter with: `price < 1 OR price > 100000` → remove.
> - The goal is to use EODHD for both historical equity data and the monthly live signal, so vendor mismatch is minimized.
>
> **Borsdata Pro (249 kr/month) — alternative source:**
>
> - Better Nordic data quality, API available at the Pro level.
> - May require a 6-month subscription — verify at `borsdata.se/priser`.
> - Check that delisted companies exist in the API before you pay.
> - Terms: may not be used commercially or distributed to others.
>
> **Decision:** EODHD is the default for backtest data. Borsdata is used only if you later want to switch to a more expensive but more Nordic-specific source.

Backtest data must contain daily raw OHLCV, corporate actions (splits/dividends), metadata for delisted companies, vendor active/delisted roster reconciliation, and the inputs required to locally reconstruct point-in-time-safe adjusted price series.

Signal calculation must use locally reconstructed point-in-time adjusted prices. Vendor-adjusted prices may be downloaded only for QA/reference and must never be the production signal source. Liquidity filters, capacity control, and execution assumptions must use raw price and volume data.

#### FX data — Sveriges Riksbank (always free)

```text
# NOK/SEK
https://api.riksbank.se/swea/v1/Observations/SEKNOKPMI/1999-01-01/2026-03-31
# DKK/SEK
https://api.riksbank.se/swea/v1/Observations/SEKDKKPMI/1999-01-01/2026-03-31
# EUR/SEK
https://api.riksbank.se/swea/v1/Observations/SEKEURPMI/1999-01-01/2026-03-31
```

FX freshness is defined by staleness before the requested `end_date`, not by observations after it. The latest observation on or before `end_date` for each required currency must not be more than `7` calendar days older than `end_date`, and later observations may be ignored.

#### Live signal data — EODHD (same provider as backtest)

EODHD is also used for the live signal in Phases 4–6. That reduces differences in symbol mapping, corporate actions, and price series between research and real usage.

- Fetches the same type of data family that the backtest is built on.
- Runs once per month after all covered exchanges have closed and vendor EOD data is available for all of them, taking seconds to minutes.
- No extra data source is required beyond the EODHD subscription.
- The goal is for the live signal to be as close as possible to the backtest's data logic.

#### Passive benchmark data — must exist

- Primary passive benchmark for the main track: inexpensive Nordic index ETF or Nordic total-return proxy, default `XACT Norden`.
- Secondary opportunity-cost benchmark: inexpensive global index ETF, default `Vanguard FTSE All-World UCITS ETF`.
- Configure explicit benchmark IDs for both the primary Nordic benchmark and the secondary global benchmark.
- Each configured benchmark must be labeled explicitly as `total_return` or `proxy`; a proxy must never be silent.
- Benchmark series should be adjusted/total return in SEK or the closest practical proxy, and they must be aligned to the same monthly sampling convention used by the strategy.
- If the strategy after costs is not clearly better than the Nordic passive benchmark, passive Nordic index investing should be the default choice.
- If the strategy beats the Nordic benchmark but not the global benchmark, that must be documented explicitly as an active regional choice, not as self-evident superiority.

### 2.3 Data Quality, Point-in-Time Universe & Tradability

- The production universe in the main track is `Full Nordics`: common shares with a primary Nordic listing on `Nasdaq Stockholm`, `Nasdaq Helsinki`, `Nasdaq Copenhagen`, and `Oslo Børs`.
- Exclude from the main track: `First North`, `NGM`, `Spotlight`, `Euronext Growth`, ETFs, certificates, preferred shares, subscription rights, SPAC-like vehicles, ADR/GDR, and other non-ordinary equity instruments.
- `First North` may only be used as a separate diagnostic after the main track on main markets has already been validated. `NGM` and `Spotlight` are out of scope in this revision.
- `SE-only` means the subset whose point-in-time primary listing venue is `Nasdaq Stockholm` main market. It must not be defined by issuer domicile.
- If a company has multiple share classes or a dual listing, only the most liquid primary Nordic ordinary share may be used.
- Primary-share selection must be monthly, point-in-time, and deterministic. Listings of the same underlying issuer must be grouped with `issuer_group_id`, and the selected listing is the Nordic ordinary share with the highest trailing `60`-trading-day median `trade_value_sek` ending at that row's point-in-time `asof_trade_date`.
- Fixed tie-breakers for primary-share selection are: higher `market_cap_sek_asof_anchor`, then older `listing_date`, then lexical `security_id`.
- Missing `issuer_group_id` for an otherwise eligible common-share candidate is a blocking data failure.
- Static security metadata must not be the source of historical eligibility or historical market-cap state. Point-in-time listing status, primary-share status, main-universe eligibility, and market-cap state must come from a monthly point-in-time security-status artifact.
- No forward fills for price or volume. If data is missing at rebalance, the stock is not eligible that month.
- A security is rankable only if it has price and volume on its own `anchor_trade_date`. If `asof_trade_date < anchor_trade_date`, the row may be retained for diagnostics/reporting only, but it must be excluded from ranking, capacity checks, and execution.
- A stock may only be included if it was listed at the signal date and was not delisted before execution.
- `MIN_LISTING_MONTHS = 18` before a stock may become eligible in the signal model.
- `MIN_PRICE_SEK = 20` on raw closing price at the signal date, converted to SEK.
- Default runtime simulation capital is `SIM_CAPITAL_SEK = 250_000`.
- Phase 1 must store and validate `close_raw_sek`, `trade_value_sek`, and `median_daily_value_60d_sek` as tradability inputs for later phases.
- The capacity rule `SIM_CAPITAL_SEK / TOP_N <= 1%` of 60-day median daily traded value in SEK is applied later in Phase 2–6 using the run's current `TOP_N`; it is not a Phase 1 universe-construction gate.
- FX conversion to SEK must use the latest available Riksbank rate on or before the relevant trading day.
- Extreme price moves without a matching split or dividend event are marked as bad data and excluded until the series is clean again.
- Robustness reports must always be run both on the full eligible universe and on a more capacity-near subset, default `largest-third-by-market-cap`, derived from point-in-time monthly market-cap state, to test small-cap dependence.

## 3. Phase 2 — Backtest & In-Sample Rolling-Origin Validation

🔒 GATE: Phase 2 does NOT start until `validate.py` is green and backtest data has been downloaded.

### 3.1 Quick Run (~2–5 minutes)

Used for daily iteration during development. Run freely — a few minutes per tweak is acceptable.

This is the `Validation Run` profile. It is a Phase 2 research loop only and must never be confused with the monthly `Signal Run` used later in paper trading or live operation.

The grid is kept small in this revision. Robustness should come from better simulation, not from more alpha parameters or optimization of cost assumptions.

| Parameter | Values | Count |  |
| --- | --- | --- | --- |
| L (lookback) | 6, 9, 12, 18 | 4 |  |
| skip | 0, 1, 2 | 3 |  |
| top_n | 10, 15, 20 | 3 |  |
| TOTAL |  | 36 | ~2–5 min |

```bash
python python/grid_search.py --mode quick
```

Quick Run is for iterative research only. It does not lock the final parameter tuple for Phase 3.

### 3.2 Mega Run (~15–30 minutes, run when needed)

🌙 Mega Run is run ONCE when you are satisfied with the Quick Run results. It should be small enough to run the same evening without turning into an all-night job.

This is the `Mega Validation Run` profile. It is the only Phase 2 profile allowed to lock the winning parameter tuple, and this revision does NOT introduce a separate `deep` or `overnight` profile beyond it.

| Parameter | Values (Mega) | Count | vs Quick |
| --- | --- | --- | --- |
| L | 3, 6, 9, 12, 15, 18, 24 | 7 | 4→7 |
| skip | 0, 1, 2, 3 | 4 | 3→4 |
| top_n | 5, 8, 10, 12, 15, 20 | 6 | 3→6 |
| TOTAL | 7×4×6 | 168 | 4.7× more |

```bash
python python/grid_search.py --mode mega
```

Mega Run locks the winner under the Mega profile only, then writes `results/consistency_report.json` by comparing that same locked tuple under the lighter validation profile.

### 3.3 Parameter Selection after Mega Run

Rank combinations in this order — NEVER on CAGR alone:

- All parameter selection in Phase 2 must use only the in-sample window `2000-01` → `2017-12`.
- The untouched holdout window `2018-01` → `2026-01` must never be used by any Phase 2 chart, metric, selector, bootstrap, or negative-control pass/fail decision.
- Phase 2 uses these fixed rolling-origin folds:
  - `fold_1`: train `2000-01` → `2007-12`, validate `2008-01` → `2009-12`
  - `fold_2`: train `2000-01` → `2009-12`, validate `2010-01` → `2011-12`
  - `fold_3`: train `2000-01` → `2011-12`, validate `2012-01` → `2013-12`
  - `fold_4`: train `2000-01` → `2013-12`, validate `2014-01` → `2015-12`
  - `fold_5`: train `2000-01` → `2015-12`, validate `2016-01` → `2017-12`
- Parameter selection must be done under a fixed default cost, not by optimizing across multiple cost levels. Cost sensitivity is reported separately after the parameters are chosen.
- Robustness: net Sharpe > 0.4 in at least 4 of 5 fixed in-sample validation folds.
- Validation Sharpe: high risk-adjusted return across the in-sample validation folds.
- Deflated Sharpe Ratio: positive after correcting for the full Mega Run grid.
- Stationary bootstrap CI: 95% confidence interval lower bound > 0, using monthly net returns, mean block length `6`, and `2000` resamples.
- Universe sensitivity: low std of Sharpe across universe subsets.
- Plateau stability: one-step neighbors in the grid should have median validation Sharpe `>= 0`.
- Plateau stability: median validation Sharpe for the neighbors should be at least `70%` of the selected combination's median validation Sharpe.
- Negative controls: cross-sectional score shuffles and block-shuffled return-path null runs must pass the full selection gates in at most `5%` of runs.
- MaxDD: when Sharpe is equal, choose lower drawdown.
- The monthly `Signal Run` is excluded from Phase 2 statistical comparison because it is an operational path with locked parameters only, not a research selector.

⚠ DO NOT choose parameters based on CAGR. That leads to overfitting.

### 3.4 Mega Run Consistency Check

After Mega Run locks the winning parameter tuple, Phase 2 must immediately run a post-selection consistency check on that same locked tuple under the lighter `Validation Run` profile.

Hard rules:

- Mega Run selects the winner using the Mega profile only.
- The consistency step compares the locked candidate under `quick` versus `mega` validation settings on the same in-sample folds and comparison window.
- The consistency step must NOT rerun full parameter selection under the lighter profile.
- The consistency step must NOT use the monthly `Signal Run`, because that path is operational rather than statistical.
- The system must never average the two profiles or auto-prefer Mega merely because it is heavier.

`results/consistency_report.json` must contain:

- `locked_candidate`
- `comparison_window`
- `quick_profile`
- `mega_profile`
- `differences`
- `consistency_warning`
- `warning_reasons`

The comparable metrics in `quick_profile` and `mega_profile` must include:

- net Sharpe
- max drawdown
- total return
- primary benchmark edge
- fold pass count
- bootstrap CI lower bound
- negative-control pass rate summary

Any of the following must set `consistency_warning = true`:

- absolute net Sharpe difference `> 0.15`
- absolute max drawdown difference `> 5 percentage points`
- absolute total return difference `> 10 percentage points`
- primary benchmark edge flips sign
- candidate-level gate status differs between the two profiles

If `consistency_warning = true` during Phase 2:

- no new parameter tuple may be promoted to Phase 3 until the inconsistency is explained
- the report must include a short diagnosis section covering likely causes: sample fragility, cost sensitivity, universe dependence, or implementation mismatch

### 3.5 Reports & Baselines

- Every Quick Run, Mega Run, and untouched holdout report must be compared against `equal-weight` of the full eligible universe.
- Every report must also be compared against simple `12-1` price momentum with the same `TOP_N`.
- Every report must be compared against the configured primary Nordic benchmark ID and configured secondary global opportunity-cost benchmark ID.
- Benchmark comparisons must report whether each configured benchmark is `total_return` or `proxy`; proxies must remain explicitly labeled in all reporting.
- Mandatory report sections in Phase 2: turnover, average capacity usage, max capacity usage, monthly hit rate, worst 10 months, and fold-by-fold summaries for `fold_1` through `fold_5`.
- Mandatory report sections in Phase 3: turnover, average capacity usage, max capacity usage, monthly hit rate, worst 10 months, and holdout subperiods (`2018–2019`, `2020–2021`, `2022–2023`, `2024–end date`).
- Always report both `Full Nordics` and `SE-only` under the same methodology. `Full Nordics` is the main track; `SE-only` means point-in-time primary listings on `Nasdaq Stockholm` main market and is a mandatory robustness and fallback variant.
- Always report results for both the full universe and `largest-third-by-market-cap` or an equivalent more liquid subset.
- Add a calendar diagnostic: at least `January vs non-January`, but use it only as analysis, never as an active timing rule in this revision.
- Add cost sensitivity across several reasonable levels, default `10/25/50/75/100 bps` for SEK names, plus separate extra FX friction for non-SEK names. Clearly flag if the edge disappears in `Full Nordics` but survives in `SE-only`.
- The primary Phase 2 selection model must use the tiered historical cost model from 7.3; the flat bps grid remains secondary sensitivity reporting and must never replace the primary selection model.
- Clearly separate long-only results from any long-short comparisons in reporting; long-short must never be presented as the main evidence for the live format.
- The same chosen parameters must be run with `next_open` (primary) and `next_close` (sensitivity), but `next_close` must NEVER be used for parameter selection.
- If the strategy after all modeled costs does not beat the Nordic passive benchmark, the default decision should be passive index investing, not the strategy.
- If `Full Nordics` only works under unrealistically low assumptions for non-SEK friction but `SE-only` holds, `SE-only` should be the live candidate before abandoning the project entirely.

## 4. Phase 3 — Untouched Final Holdout Validation

🔒 GATE: Phase 3 does NOT start until Mega Run is complete and the best parameters have been selected.

### 4.1 Specification

- In-sample — `2000-01` → `2017-12` — parameters were selected here.
- Out-of-sample — `2018-01` → `2026-01` — never touched during grid search.
- Phase 3 consumes only the Mega-selected winning parameter tuple after any Phase 2 consistency warning has been resolved.
- The holdout runner must read only the locked parameter tuple from `results/selection_summary.json` and must reject ad-hoc CLI parameter overrides.
- Signal is generated on the close of `anchor_trade_date`, defined as the last trading day of month `m` for each instrument's `exchange_group`.
- Primary execution takes place on the `open` of `next_execution_date`, defined as the first trading day of month `m+1` for that same `exchange_group`.
- The same selected parameters are also run on the `close` of each instrument's `next_execution_date` as a sensitivity test.
- The sensitivity test is documented but never used for parameter selection.
- Runs — EXACTLY ONE — no tweaks afterward regardless of result.

```bash
python python/walk_forward.py
```

### 4.2 Interpretation Guide — What Happens Now?

| OOS Sharpe | Assessment | Next step |
| --- | --- | --- |
| > 0.5 | ★ Strong | Proceed to Phase 4 — paper trading |
| 0.3 – 0.5 | OK | Proceed to Phase 4 with increased caution |
| 0.1 – 0.3 | ⚠ Weak | Go back to Phase 2 — investigate why, no paper trading or live |
| < 0.1 | 🛑 Fail | Do NOT run live — likely overfit |

🛑 STOP: If OOS Sharpe < 0.1: stop here. Do NOT build the live engine. The strategy does not hold out-of-sample.

Below `0.3`, the system does not proceed to Phase 4. The interval `0.1–0.3` means going back to Phase 2 for a redo; below `0.1` is a hard stop.

## 5. Phase 4 — Paper Trading (6 months mandatory)

🔒 GATE: Phase 4 does NOT start until untouched final holdout validation shows OOS Sharpe >= 0.3.

### 5.1 Why Paper Trading Is Mandatory

Untouched final holdout validation shows whether the strategy holds historically. Paper trading shows whether it holds in reality and whether you hold up.

> **What paper trading reveals that backtest cannot**
>
> - Practical executability: are the selected companies actually buyable at reasonable spreads and reasonable slippage?
> - Turnover in practice: does the estimated turnover cost match reality?
> - Psychological endurance: can you follow the system when it underperforms for 3 months in a row?
> - Signal quality: do the signals seem reasonable when you see them in real time?
> - Data quality: are any EODHD or corporate-action errors revealed in live data?
> - Most people abandon their strategy exactly when it starts recovering.

### 5.2 live_signal.py — Built in Phase 4

⚠ `live_signal.py` is NOT built in Phase 1 or 2. It is built only when untouched final holdout validation is approved.

`live_signal.py` must do exactly this and nothing more:

- Read `config.py` for parameters (the untouched holdout's validated values).
- Rebuild the current-month eligible universe from the same Phase 1 artifacts and the same rule set used by backtest and untouched final holdout validation; `live_tickers.csv` must not be the source of truth.
- Fetch the latest 24 months of data via EODHD for the research-defined live universe.
- Build the signal with the same locally reconstructed point-in-time adjusted-price logic used by the backtest; never use vendor `adj_close` as the production signal source.
- Use the same point-in-time share-selection logic, variant exclusions, exchange-specific `anchor_trade_date`, exchange-specific `next_execution_date`, and runtime capacity screen as the research pipeline.
- Run only after all covered exchanges have closed and vendor EOD data is available for all of them; otherwise defer to the next pre-open window.
- If an operational emergency override/blocklist is ever introduced, it must be empty by default, logged explicitly, and must never replace the research-defined universe.
- Call the C++ module: `cpp.compute_signals(...)`.
- Compare with the previous month's portfolio.
- Generate `signals/YYYY-MM_trades.csv` and `signals/YYYY-MM_report.txt`, including the planned exchange-specific `next_execution_date` for each trade.

`live_signal.py` is a signal generator, not an execution engine.

It is the operational `Signal Run`, not a research mode. It must not perform grid search, bootstrap resampling, negative controls, or parameter selection.

### 5.3 Monthly Paper Trading Routine

Do this once per month for 6 months, after all covered exchanges have closed and vendor EOD data is available for all of them; otherwise defer to the next pre-open window:

- Run: `python python/live_signal.py`
- Open `signals/YYYY-MM_report.txt` — review the signals.
- Note: what would you have bought/sold if it were real money?
- On each trade's exchange-specific `next_execution_date`: note the model's reference price and your actual fill price.
- Calculate slippage in bps relative to the model's primary execution assumption (`next_open`).
- The following month: compare the outcome with the signal.
- Document deviations and observations.

This monthly routine is operational only. It must not run research selection logic, grid search, bootstrap, or negative-control testing.

### 5.4 Paper Trading Documentation

Create a simple log (Excel or CSV) with these columns per month:

| Column | Example | Type | Purpose |
| --- | --- | --- | --- |
| Month | 2026-04 | str |  |
| Signal_ticker | SAND.ST | str | What the system said |
| Signal_action | BUY | str | BUY/SELL/HOLD |
| Planned_execution_day | 2026-05-02 | date | First allowed trading day |
| Trade_currency | NOK | str | Separate SEK from non-SEK fills |
| Reference_price_model | 245.10 | float | Price according to the primary model |
| Actual_fill_price | 246.40 | float | Your real price |
| Slippage_bps | 53 | float | Difference versus the model |
| Model_cost_bps | 50 | float | Expected one-way cost according to the model |
| Observed_total_cost_bps | 68 | float | Brokerage + FX + spread/slippage estimate |
| Actual_return | +4.2% | float | The month after |
| Portfolio_return | +2.1% | float | The portfolio's full return |
| Non_execution_reason | Insufficient liquidity | str | If the trade did not go through |
| Comment | Wide spread | str | Observations |

### 5.5 Approved or Failed Paper Trading

Phase 5 may start only if all of the following requirements are met after 6 months of paper trading:

- At least `90%` of planned trades were practically executable.
- Average observed total one-way cost must not exceed the model's corresponding cost assumption by more than `25 bps`.
- 6-month cumulative paper-trading return must not be more than `5 percentage points` below the model's corresponding signal portfolio.
- Paper trading must not turn a previous advantage versus the Nordic benchmark into a clear disadvantage after observed costs.
- Recurring symbol-mapping, corporate-action, or data-source errors must be understood and fixed before live.
- If `Full Nordics` fails mainly because of non-SEK friction but `SE-only` passes the gates, `SE-only` may proceed as a separate live candidate.

## 6. Phase 5 — Live Deployment (optional)

🔒 GATE: Phase 5 does NOT start until 6 months of paper trading are documented and approved according to the criteria in 5.5.

### 6.1 Capital Allocation — Gradual

⚠ NEVER start with full capital allocation immediately. Build up gradually.

| Month | Allocation | Condition for increase |
| --- | --- | --- |
| Month 1–3 | 20% of intended amount | Always start small |
| Month 4–6 | 50% of intended amount | Live matches paper trading within ±3pp and real costs stay close to the model |
| Month 7+ | 100% of intended amount | 6 months of live track record without benchmark gap or cost gap clearly worsening |

### 6.2 Streamlit Dashboard — Built in Phase 5

⚠ The dashboard is NOT built until Phase 5. The focus in Phases 1–4 is backtest and validation, not UI.

#### Pages in the dashboard

- 📊 Live Signal — Current portfolio, trades, latest signal
- 📈 Backtest — Equity curve, annual table, drawdown, key metrics
- 🔬 Holdout Validation — In-sample vs OOS equity curve and key metrics
- ⚡ Quick Run — Run grid search directly from the UI with a progress bar
- 🌙 Mega Run — Start mega run, robustness matrix, consistency report, Monte Carlo CI
- ⚙️ Configuration — Edit `config.py`, system status

## 7. Signal Construction

### 7.1 Default Parameters (Starting Values)

These are default values for Quick Run and initial testing. They are updated after Mega Run and untouched final holdout validation.

| Parameter | Default | Grid values | Academic rationale |
| --- | --- | --- | --- |
| L (lookback) | 12 months | 3–24 | Jegadeesh-Titman (1993): 12m optimal |
| SKIP | 1 month | 0–3 | Avoids short-term reversal |
| TOP_N | 10 stocks | 5–20 | Balance of concentration/diversification |
| BASE_COST_BPS_SEK | 25 bps | fixed default | One-way all-in for SEK names: brokerage + spread + slippage |
| EXTRA_FX_COST_BPS_NON_SEK | 25 bps | separate sensitivity assumption | Extra one-way FX friction for NOK/DKK/EUR names until real Nordea cost is verified |

This revision uses pure price momentum in the main track. FF3 residualization is removed from the production logic because the wrong region, wrong currency, or wrong publication timing can create false alpha in backtest.

No new alpha parameters are added in this revision. Robustness should come from better simulation, not from a larger grid or greater model complexity.

Alpha/residual momentum may only be reintroduced as a separate research track if it first shows clear outperformance versus the price baseline out-of-sample, after costs, and under paper trading.

The portfolio must in the main track be weighted `equal-weight` across the selected `TOP_N`. Value weighting or another weighting scheme requires a separate research track because several Nordic studies warn that concentration in individual large companies can otherwise dominate the outcome.

### 7.2 Look-Ahead-Free Protocol

> **At month-end m — calculate signal for m+1**
>
> - Price momentum: `P[m-SKIP-1] / P[m-L-SKIP-1] - 1`
> - The main track uses pure price momentum, not external FF3 residualization.
> - External factor models must not be used in the main track without separate validation of region, currency, and publication timing.
> - Signal calculation uses locally reconstructed point-in-time adjusted prices built from raw prices and corporate actions effective on or before each date; vendor-adjusted series are QA-only.
> - If an issuer has multiple ordinary Nordic listings, the live and backtest pipelines must apply the same monthly point-in-time primary-share selection based on trailing `60`-trading-day median `trade_value_sek`, with the documented tie-breakers.
> - Only securities with price and volume on their own `anchor_trade_date` are rankable. If `asof_trade_date < anchor_trade_date`, the row may be retained for diagnostics/reporting only and must not enter ranking or execution.
> - Liquidity and execution use raw price and volume data.
> - Rebalancing uses each instrument's own `exchange_group` calendar: signal on `anchor_trade_date` in month `m`, execution in the market of `m+1` on `next_execution_date`.
> - The first allowed fill is the instrument's exchange-specific `next_execution_date`; never the same day's close on which the signal was created.
> - Live operation must rebuild the same eligible universe as research; manual ticker curation is not allowed as the source of truth.

### 7.3 Execution and Return Model

- Primary backtest execution takes place on each instrument's exchange-specific `next_execution_date` `open` in month `m+1`.
- The sensitivity test runs the same chosen parameters on each instrument's exchange-specific `next_execution_date` `close`.
- The primary historical selection model uses a tiered one-way cost:
  - brokerage: `max(39 SEK, 10 bps * order_notional_sek)`
  - spread/slippage buckets on `median_daily_value_60d_sek`: `15/25/40/60/100 bps` for `>= 50m`, `20–50m`, `10–20m`, `5–10m`, `< 5m`
  - low-price add-on: `+10 bps` when `close_raw_sek < 50`
  - participation add-on from `order_notional_sek / median_daily_value_60d_sek`: `0/10/20/40 bps` for `<= 0.25%`, `0.25–0.50%`, `0.50–0.75%`, `0.75–1.00%`
  - `> 1.00%` participation fails the capacity gate
  - `next_open` multiplies spread/slippage plus participation by `1.25x`
  - `next_close` multiplies spread/slippage plus participation by `1.00x`
- `BASE_COST_BPS_SEK` remains a secondary sensitivity-reporting grid and is charged on both buys and sells only in the flat-bps comparison runs.
- `EXTRA_FX_COST_BPS_NON_SEK` remains a secondary sensitivity-reporting grid. The primary historical model instead uses explicit low/base/high FX friction scenarios of `10/25/50 bps` for NOK, DKK, and EUR names in `Full Nordics`.
- Nordea-like brokerage for Nordic online trades must be treated separately from the currency-friction assumption; the goal is to avoid hiding non-SEK costs in a single average that is too forgiving.
- The runtime capacity screen uses `SIM_CAPITAL_SEK`, the run's current `TOP_N`, and validated tradability inputs including `close_raw_sek`, `trade_value_sek`, and `median_daily_value_60d_sek`; it is not precomputed as a Phase 1 universe exclusion.
- Live and research must use the same eligible-universe construction and runtime capacity logic; manual live ticker curation is not allowed as a substitute.
- Runtime ranking, capacity screening, and execution apply only to securities whose `asof_trade_date = anchor_trade_date`; stale rows may be retained for diagnostics/reporting only.
- Holding-period return must be calculated with a one-day delay from signal to position and use locally reconstructed point-in-time adjusted total return prices during the holding period.
- Delisted companies must be included in the history.
- Delisting coverage completeness must be validated by reconciling vendor active and delisted rosters, and unresolved in-scope price histories ending more than `30` calendar days before `end_date` must fail Phase 1 validation.
- If a company disappears during the holding period and no clear cash-out or corporate action exists, force-exit at the last available price with `DELIST_FALLBACK_HAIRCUT = -0.30`.

### 7.4 Momentum Crash Risk

Momentum can underperform heavily in rapid regime shifts and after sharp market reversals. The PRD should therefore report subperiods, worst months, execution sensitivity, and comparison with passive benchmark alternatives, but should NOT try to add any crash-timing model in this revision.

Low-vol combinations, volatility scaling, and value+momentum combinations are treated in this revision as backlog/research tracks. The literature makes them interesting, but not compelling enough to increase model complexity in the production core right now.

## 8. File Structure

```text
alpha_momentum_nordic/
├── 02_phase2_research_validation.md
├── CMakeLists.txt
├── config.py
├── requirements.txt
├── .env                          ← DO NOT COMMIT TO GIT
├── .gitignore
│
├── src/                          ← C++ core (Phase 1)
│   ├── data.hpp
│   ├── stats.hpp
│   ├── signal.hpp / signal.cpp
│   ├── portfolio.hpp / portfolio.cpp
│   ├── engine.hpp / engine.cpp
│   ├── grid.hpp / grid.cpp
│   ├── walk_forward.hpp / walk_forward.cpp
│   ├── monte_carlo.hpp / monte_carlo.cpp
│   ├── bindings.cpp
│   └── main.cpp
│
├── python/
│   ├── download_borsdata.py      ← Phase 1 (alternative source)
│   ├── download_eodhd.py         ← Phase 1 (static master + raw + actions + delisted)
│   ├── download_riksbank_fx.py   ← Phase 1
│   ├── build_universe.py         ← Phase 1 (PTI status + local adjusted prices + universe)
│   ├── validate.py               ← Phase 1 (gate)
│   ├── grid_search.py            ← Phase 2
│   ├── validation_protocol.py    ← Phase 2–3 (folds, DSR, bootstrap, costs, controls)
│   ├── walk_forward.py           ← Phase 3
│   ├── live_signal.py            ← Phase 4 (not built until then)
│   └── dashboard.py              ← Phase 5 (not built until then)
│
├── engine.py                     ← ORIGINAL do not touch
├── run.py                        ← ORIGINAL do not touch
│
├── data/
│   ├── security_master.parquet
│   ├── security_status_pti_monthly.parquet
│   ├── prices_raw_daily.parquet
│   ├── prices_adjusted_daily.parquet
│   ├── corporate_actions.parquet
│   ├── delisted_metadata.parquet
│   ├── benchmark_prices.parquet
│   ├── universe_pti.parquet
│   └── riksbank_fx_daily.parquet
│
├── signals/                      ← Created in Phase 4
│   └── YYYY-MM_{portfolio,trades,report}.*
│
├── results/
│   ├── grid_results_quick.csv
│   ├── selection_summary.json
│   ├── consistency_report.json
│   ├── walk_forward_results.json
│   └── mega_run/
│       └── ...
│
└── papertrading/                 ← Created in Phase 4
    └── papertrading_log.csv
```

## 9. Configuration (config.py)

Updated automatically after Mega Run and untouched final holdout validation. Parameters must never be hardcoded in other files.

```python
# ── Signal parameters (updated after untouched final holdout validation) ──
L            = 12   # Default value — replaced by untouched final holdout validation
SKIP         = 1
TOP_N        = 10
POSITION_WEIGHTING = 'equal'
VOL_TARGET   = 0.22
VOL_SCALE    = False
UNIVERSE_MODE = 'full_nordics_main_markets'
RUN_SE_ONLY_DIAGNOSTIC = True

# ── Capital & tradability ──
SIM_CAPITAL_SEK = 250_000   # Runtime capacity-screen input, not a Phase 1 universe gate
MIN_LISTING_MONTHS = 18
MIN_PRICE_SEK = 20
MAX_ORDER_FRACTION_OF_60D_MEDIAN_DAILY_VALUE = 0.01   # Applied later with the run's current TOP_N
RUN_LIQUID_SUBSET_DIAGNOSTIC = True
LIQUID_SUBSET_MODE = 'largest_third_by_market_cap'

# ── Execution & data hygiene ──
EXECUTION_MODEL = 'next_open'
RUN_EXECUTION_SENSITIVITY = True   # Also run the same parameters on the next day's close
USE_ADJUSTED_CLOSE_FOR_SIGNAL = True   # Uses locally reconstructed point-in-time adjusted prices
BASE_COST_BPS_SEK = 25
EXTRA_FX_COST_BPS_NON_SEK = 25
COST_SENSITIVITY_BPS_SEK = [10, 25, 50, 75, 100]
FX_COST_SENSITIVITY_BPS_NON_SEK = [0, 10, 25, 50]
INCLUDE_DELISTED = True
DELIST_FALLBACK_HAIRCUT = -0.30
RUN_CALENDAR_DIAGNOSTIC = True

# ── Phase control ──
CURRENT_PHASE        = 1   # Updated manually
HOLDOUT_VALIDATION_DONE = False
OOS_SHARPE              = None  # Filled in after untouched final holdout validation
PAPER_TRADING_MONTHS = 0    # Incremented monthly

# ── In-sample validation & holdout ──
ROLLING_ORIGIN_FOLDS = [
    ('fold_1', '2000-01', '2007-12', '2008-01', '2009-12'),
    ('fold_2', '2000-01', '2009-12', '2010-01', '2011-12'),
    ('fold_3', '2000-01', '2011-12', '2012-01', '2013-12'),
    ('fold_4', '2000-01', '2013-12', '2014-01', '2015-12'),
    ('fold_5', '2000-01', '2015-12', '2016-01', '2017-12'),
]
INSAMPLE_END   = '2017-12'
OOS_START      = '2018-01'
OOS_END        = '2026-01'
OOS_SHARPE_MIN = 0.3   # Below this: do not run live

# ── Run profiles ──
SIGNAL_PROFILE = {
    'selection_enabled': False,
    'bootstrap_resamples': 0,
    'cross_sectional_shuffle_runs': 0,
    'block_shuffled_null_runs': 0,
}
VALIDATION_PROFILE_QUICK = {
    'selection_enabled': True,
    'bootstrap_resamples': 500,
    'cross_sectional_shuffle_runs': 100,
    'block_shuffled_null_runs': 50,
}
VALIDATION_PROFILE_MEGA = {
    'selection_enabled': True,
    'bootstrap_resamples': 2000,
    'cross_sectional_shuffle_runs': 500,
    'block_shuffled_null_runs': 200,
}

# ── Mega Run and consistency policy ──
MEGA_WF_PASSES_REQUIRED = 4
CONSISTENCY_WARNING_THRESHOLDS = {
    'net_sharpe_abs_diff': 0.15,
    'max_drawdown_abs_diff': 0.05,
    'total_return_abs_diff': 0.10,
    'benchmark_edge_flip': True,
    'gate_status_mismatch': True,
}

# ── Anti-overfitting validation ──
BOOTSTRAP_METHOD = 'stationary'
BOOTSTRAP_BLOCK_LENGTH_MONTHS = 6
USE_DEFLATED_SHARPE = True
NEGATIVE_CONTROL_PASS_RATE_MAX = 0.05
PRIMARY_SELECTION_COST_MODEL = 'tiered_v1'
# Tiered historical costs stay the same across quick and mega profiles in this revision.

# ── Tiered historical cost model ──
BROKERAGE_MIN_SEK = 39
BROKERAGE_BPS = 10
SPREAD_BPS_BUCKETS = [
    (50_000_000, 15),
    (20_000_000, 25),
    (10_000_000, 40),
    (5_000_000, 60),
    (0, 100),
]
LOW_PRICE_THRESHOLD_SEK = 50
LOW_PRICE_ADDON_BPS = 10
PARTICIPATION_BPS_BUCKETS = [
    (0.0025, 0),
    (0.0050, 10),
    (0.0075, 20),
    (0.0100, 40),
]
NEXT_OPEN_IMPACT_MULTIPLIER = 1.25
NEXT_CLOSE_IMPACT_MULTIPLIER = 1.00
FX_FRICTION_SCENARIOS_BPS = {'low': 10, 'base': 25, 'high': 50}

# ── Data sources ──
BORSDATA_API_KEY = os.environ.get('BORSDATA_API_KEY')
EODHD_API_KEY    = os.environ.get('EODHD_API_KEY')
PRIMARY_SOURCE   = 'eodhd'   # EOD Historical Data - All World (20 USD/month)
BASE_CURRENCY    = 'SEK'
PRIMARY_PASSIVE_BENCHMARK = 'XACT Norden'
PRIMARY_PASSIVE_BENCHMARK_ID = 'XACT_NORDEN'
PRIMARY_PASSIVE_BENCHMARK_TYPE = 'proxy'   # 'total_return' or 'proxy'
SECONDARY_OPPORTUNITY_COST_BENCHMARK = 'Vanguard FTSE All-World UCITS ETF'
SECONDARY_OPPORTUNITY_COST_BENCHMARK_ID = 'VWRL_UK'
SECONDARY_OPPORTUNITY_COST_BENCHMARK_TYPE = 'proxy'   # 'total_return' or 'proxy'
```

## 10. Realistic Timeline

| Period | Phase | Activity |
| --- | --- | --- |
| Week 1–2 | Phase 1 | C++ build, validation, daily data download, point-in-time universe |
| Week 2–4 | Phase 2 | Quick Run iteration, Mega Run, baselines, and execution sensitivity |
| Week 4 | Phase 3 | Untouched final holdout validation — one run, accept the result |
| Month 2–7 | Phase 4 | Paper trading monthly, document everything |
| Month 8+ | Phase 5 | Live deployment if untouched final holdout validation and paper trading are approved |

> **Summary of costs**
>
> - Backtest data: EOD Historical Data - All World — 20 USD/month (primary source)
> - Alternative backtest source: Borsdata Pro — 249 kr/month if you later want to switch
> - Live signal data: included in the EODHD subscription
> - Nordea online brokerage for Nordic trades must be modeled separately from FX friction; non-SEK names require an explicit extra cost assumption until your account setup is verified in practice
> - Streamlit dashboard: runs locally — 0 kr
> - C++ tools: CMake, Eigen, pybind11 — all free and open source
> - Total recurring cost after setup: at least 20 USD/month as long as the EODHD subscription stays active, plus actual Nordea friction when trading

Alpha-Momentum Nordic · PRD v4.0 · Gated Deployment · March 2026

EODHD · Sveriges Riksbank · Vanguard FTSE All-World UCITS ETF · XACT Norden

## 11. Appendix A — Evidence Matrix & Cross-Study Conclusions

### 11.1 Evidence Matrix

| Source | Type | Region / sample | Main finding | Method weakness | PRD implication | Classification |
| --- | --- | --- | --- | --- | --- | --- |
| Grobys, Fatmy & Rajalin (2024), *Applied Economics* | Peer-reviewed article | Nordics, 1999–2022 | Momentum and low-vol remain; combinations do not beat pure momentum after broader risk adjustment. | Partly deals with combination strategies and not your exact retail implementation. | Support for a simple momentum core and for low-vol as backlog, not default. | Validates |
| Nilsson & Picone (2021), GU | Master's thesis | Nordics, several momentum variants | Alpha momentum looks best on a risk-adjusted basis and appears to mitigate crash risk. | Thesis, not peer-reviewed; costs are not fully modeled. | Motivates alpha as a research track but not as the live default. | Partially validates |
| Annerstedt & Schönström (2006), Lund | Master's thesis | Nordics, 1991–2006 | Momentum profits on the 3–12 month horizon; delistings are considered. | Old study with limited direct relevance to today's frictions. | Support for a Nordic baseline and for delisting requirements in the data. | Partially validates |
| Jacobsen & Nyhegn (2021), CBS | Master's thesis | Nordics, 2007–2021 | Momentum exists in both small and large companies, but is stronger in small companies; a January effect appears in the sample. | Thesis and a small implementation gap between academic portfolio and retail long-only. | Support for small-cap diagnostics and calendar reporting, not for seasonal timing. | Partially validates |
| Kobelyats'kyy & Fulgentiusson (2011), Lund | Student thesis | Sweden | Swedish momentum appears to exist and is not well explained by CAPM/FF3. | Sweden only and an older sample. | Local data point that supports the baseline hypothesis. | Partially validates |
| Bergsten (2019), Uppsala | Master's thesis | Sweden, 1999–2018 | Pure momentum beats intermediate past returns; volatility scaling can mitigate crash risk. | Costs are explicitly ignored. | Support for a simple signal in the core; vol scaling is backlog. | Partially validates |
| Savolainen (2024), LUT | Master's thesis | Nordics, 1999–2023 | Value+momentum can improve the risk profile; standalone momentum is not unambiguously strong in all variants. | No realistic trading costs; focus is multifactor portfolios. | Support for baselines, passive benchmarks, and combination tracks as backlog. | Weakens |
| Alpha Architect QMOM investment case | Marketing material | USA / ETF case | Good process ideas around screening, momentum windows, and rebalance. | Not independent research, not Nordic, not neutral evidence. | Design inspiration, not methodological proof. | Inspires but does not prove |
| Björkman & Durling (2018), Linköping | Master's thesis | Sweden / HFT market quality | HFT appears to improve liquidity and lower volatility in their sample. | Not a momentum study. | Motivates execution realism and caution around liquidity assumptions. | Inspires but does not prove |

### 11.2 Cross-Study Conclusions

- Price momentum in the Nordics is better supported than alpha/residual momentum in this literature bundle.
- The evidence is heavily thesis-dominated; only one clearly peer-reviewed Nordic source in the bundle.
- Small-cap/illiquidity likely carries part of the premium, making capacity filters and `largest-third-by-market-cap` diagnostics central.
- Equal weighting often appears more realistic than value weighting in a Nordic context because individual large companies can otherwise dominate the portfolio.
- Transaction costs and real-world implementability are repeatedly underestimated in the studies.
- Long-short results often look stronger than long-only, but they must not be mixed into the decision basis for this project.
- `January vs non-January` is interesting as a diagnostic, but not as a basis for adding calendar rules in this revision.
- Low-vol/vol scaling and value+momentum are meaningful backlog tracks, but the literature does not justify replacing the simpler production core with them now.
- Swedish evidence provides extra support for not adding `intermediate past returns` to the main track.
- The most important practical conclusion is still that the strategy must beat passive index alternatives after realistic costs and paper trading, otherwise passive index investing should be the default.

## 12. Appendix B — Codex Skills for This Project

The skills in this appendix are work-environment tools for research, documentation, and later implementation. They are not strategy validation and not product requirements in themselves.

| Skill | Current status | Trigger | Phase |
| --- | --- | --- | --- |
| `pdf` | Use now | Paper review, PDF extraction, future PDF output | Now / Phase 0–2 |
| `sora` | Installed but not core | Demo, pitch, or video later | Later / Phase 5+ |
| `jupyter-notebook` | Use later | Exploratory comparison of backtests, benchmark gaps, OOS windows | Phase 2–4 |
| `spreadsheet` | Use later | Paper trading logs, grid results, benchmark and cost tables | Phase 2–4 |
| `doc` | Use later | If `.docx` becomes an active working format again | As needed |
| `playwright` | Use later | When the Streamlit dashboard exists and needs UI testing | Phase 5 |
| `security-best-practices` | Use later | When real Python/app code, `.env`, and secrets handling exist | Phase 1+ codebase |

### 12.1 Skills That Are Not Worth Prioritizing Now

- `openai-docs`, `figma`, `figma-implement-design`, `cloudflare-deploy`, `netlify-deploy`, `render-deploy`, `vercel-deploy`, `notion-*`, `slides`, `speech`, `transcribe`, `chatgpt-apps`
- Reason: they do not solve the project's current bottlenecks, which are literature, methodology, data realism, and later backtest/paper-trading validation.

### 12.2 Conclusion from the Skills Audit

- `pdf` is the only skill that provides immediate core value in the current phase, because the project's main bottleneck right now is paper review and document-driven methodology.
- `sora` can be useful only if the project later needs demo, pitch, or training material; it should not affect research or implementation priorities now.
- The next most reasonable installations are `jupyter-notebook` and `spreadsheet`, but only once actual backtest results, benchmark gaps, and paper-trading logs exist to analyze.
- None of the other listed skills currently adds anything more important than better data, better cost modeling, or better validation.
