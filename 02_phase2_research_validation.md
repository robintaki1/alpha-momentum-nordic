# 02 Phase 2 Research Validation

This document supplements `AlphaMomentum_PRD_v4.md` and turns the Phase 2 and Phase 3 run hierarchy into an implementation-ready spec. It assumes Phase 1 will later produce the required point-in-time artifacts and that the research engine will later produce evaluation manifests consumed by the orchestration scripts.

## 1. Target State and Done Definition

Phase 2 is complete only if all of the following are true:

- candidate parameters are evaluated only on the fixed in-sample rolling-origin folds
- no Phase 2 metric, chart, bootstrap, or negative control touches the untouched holdout window
- `python/grid_search.py` writes `results/grid_results_{mode}.csv`
- `python/grid_search.py` writes `results/selection_summary.json`
- `python/grid_search.py --mode mega` also writes `results/consistency_report.json`
- at most one parameter tuple is locked as the winner
- if no candidate passes the hard gates, the summary states that explicitly and Phase 3 does not proceed
- if `consistency_warning = true`, Phase 3 stays blocked until the inconsistency is explained in the report

Phase 3 is complete only if all of the following are true:

- `python/walk_forward.py` reads only the locked Mega-selected parameter tuple from `results/selection_summary.json`
- ad-hoc parameter overrides are rejected
- holdout evaluation uses only the untouched window `2018-01` through `2026-01`
- `results/walk_forward_results.json` is written for the required universe, execution, and FX scenarios

## 2. Run Profiles and Boundaries

The run taxonomy is layered on top of the existing PRD names:

- `signal`: monthly operational path used in Phase 4-6; locked parameters only; no statistical selection logic
- `quick`: light validation profile used for iterative research in Phase 2; mapped to `Quick Run`
- `mega`: heavy validation profile used for final in-sample selection in Phase 2; mapped to `Mega Run`
- `untouched final holdout`: separate Phase 3 gate, not a validation profile

Hard boundaries:

- `signal` is never part of Phase 2 statistical comparison
- `quick` and `mega` may use only the fixed in-sample folds
- `untouched final holdout` may never be used by `quick` or `mega`
- no separate `deep` or `overnight` profile is introduced in this revision

Default profile settings:

- `signal`
  - `selection_enabled = false`
  - `bootstrap_resamples = 0`
  - `cross_sectional_shuffle_runs = 0`
  - `block_shuffled_null_runs = 0`
- `quick`
  - `selection_enabled = true`
  - `bootstrap_resamples = 500`
  - `cross_sectional_shuffle_runs = 100`
  - `block_shuffled_null_runs = 50`
- `mega`
  - `selection_enabled = true`
  - `bootstrap_resamples = 2000`
  - `cross_sectional_shuffle_runs = 500`
  - `block_shuffled_null_runs = 200`

The tiered historical cost model stays unchanged across `quick` and `mega` in this revision.

## 3. Fixed Validation Protocol

These folds are mandatory and must not be regenerated dynamically:

- `fold_1`: train `2000-01` -> `2007-12`, validate `2008-01` -> `2009-12`
- `fold_2`: train `2000-01` -> `2009-12`, validate `2010-01` -> `2011-12`
- `fold_3`: train `2000-01` -> `2011-12`, validate `2012-01` -> `2013-12`
- `fold_4`: train `2000-01` -> `2013-12`, validate `2014-01` -> `2015-12`
- `fold_5`: train `2000-01` -> `2015-12`, validate `2016-01` -> `2017-12`

The untouched initial holdout is:

- `2018-01` -> `2026-01`

Hard leakage rule:

- any Phase 2 evaluation row whose validation window starts at or after `2018-01` must fail immediately

## 4. Input Contracts

### 4.1 `python/grid_search.py`

This script does not rebuild Phase 1 data or compute raw backtests directly. It consumes a future engine-produced JSON evaluation manifest.

Required CLI:

```bash
python python/grid_search.py \
  --mode mega \
  --candidate-panel results/candidate_evaluations.json \
  --output-dir results
```

Required behavior:

- `--mode quick` evaluates the Quick Run grid on the fixed in-sample folds and writes `grid_results_quick.csv` plus `selection_summary.json`
- `--mode mega` evaluates the Mega Run grid on the fixed in-sample folds, locks the winner using Mega metrics only, then performs a metrics-only quick-profile comparison on that same locked candidate
- the quick-profile comparison is a single-candidate rerun under the `quick` profile; it must not rerun full lighter-profile parameter selection

Required input manifest shape:

```json
{
  "mode": "mega",
  "candidates": [
    {
      "params": {"l": 12, "skip": 1, "top_n": 10},
      "evaluations": [
        {
          "fold_id": "fold_1",
          "validate_start": "2008-01",
          "validate_end": "2009-12",
          "universe_variant": "Full Nordics",
          "execution_model": "next_open",
          "fx_scenario": "base",
          "cost_model_name": "tiered_v1",
          "monthly_returns": [0.01, -0.02]
        }
      ]
    }
  ],
  "negative_controls": {
    "cross_sectional_shuffle": {"run_count": 500, "pass_count": 12},
    "block_shuffled_null": {"run_count": 200, "pass_count": 4}
  },
  "consistency_reference": {
    "profile": "quick",
    "candidates": [
      {
        "params": {"l": 12, "skip": 1, "top_n": 10},
        "evaluations": [
          {
            "fold_id": "fold_1",
            "validate_start": "2008-01",
            "validate_end": "2009-12",
            "universe_variant": "Full Nordics",
            "execution_model": "next_open",
            "fx_scenario": "base",
            "cost_model_name": "tiered_v1",
            "monthly_returns": [0.01, -0.02]
          }
        ]
      }
    ],
    "negative_controls": {
      "cross_sectional_shuffle": {"run_count": 100, "pass_count": 3},
      "block_shuffled_null": {"run_count": 50, "pass_count": 1}
    }
  }
}
```

Rules for `consistency_reference`:

- it is required only when `mode = mega`
- it must contain the same parameter tuple space as needed to find the eventual locked candidate
- it exists only to compare the locked Mega winner under lighter validation settings

### 4.2 `python/walk_forward.py`

Required CLI:

```bash
python python/walk_forward.py \
  --selection results/selection_summary.json \
  --holdout-panel results/holdout_evaluations.json \
  --output-dir results
```

The holdout manifest must contain only the locked parameter tuple's evaluation rows for:

- `Full Nordics`
- `SE-only`
- `largest-third-by-market-cap`
- `next_open`
- `next_close`
- `low`, `base`, `high` FX scenarios

Phase 3 must consume only the Mega-selected winner after any `consistency_warning` from Phase 2 has been resolved.

## 5. Selection Logic

The main ranking track is:

- `Full Nordics`
- `next_open`
- `base` FX scenario
- `tiered_v1` cost model

Hard gates for a candidate:

- all five fixed folds must be present
- at least `4/5` folds with net Sharpe `> 0.4`
- deflated Sharpe score `> 0`
- stationary-bootstrap 95% Sharpe CI lower bound `> 0`
- negative-control pass rate `<= 5%` for both suites

Ranking order among candidates that pass the hard gates:

1. median validation Sharpe
2. lower universe-sensitivity standard deviation across `Full Nordics`, `SE-only`, `largest-third-by-market-cap`
3. plateau stability across one-step neighbors
4. lower max drawdown

If no candidate passes the hard gates:

- `selection_status` must be `no_candidate_passed_hard_gates`
- `locked_candidate` must be `null`
- Phase 3 must not run

## 6. Mega Run Consistency Policy

The consistency report compares the same locked candidate under `quick` and `mega` validation settings on the same in-sample window.

Required fields in both `quick_profile` and `mega_profile`:

- `net_sharpe`
- `max_drawdown`
- `total_return`
- `primary_benchmark_edge`
- `fold_pass_count`
- `bootstrap_ci_low`
- `negative_control_pass_rate_summary`
- `gate_status`

Required warning thresholds:

- absolute net Sharpe difference `> 0.15`
- absolute max drawdown difference `> 0.05`
- absolute total return difference `> 0.10`
- `primary_benchmark_edge` flips sign
- `gate_status` differs between profiles

Required behavior:

- any threshold breach sets `consistency_warning = true`
- the report must include `warning_reasons`
- the report must include a short diagnosis section covering likely causes: sample fragility, cost sensitivity, universe dependence, or implementation mismatch
- the system must never average `quick` and `mega`
- the system must never auto-prefer `mega` merely because it is heavier

## 7. Statistics and Historical Cost Model

### 7.1 Stationary bootstrap

- resample unit: monthly net returns
- mean block length: `6` months
- `quick` resamples: `500`
- `mega` resamples: `2000`
- output: `bootstrap_ci_low`, `bootstrap_ci_high`

### 7.2 Deflated Sharpe

- use the full Mega Run candidate count as the number of trials for the final selection correction
- report both:
  - `deflated_sharpe_score`
  - `deflated_sharpe_probability`

### 7.3 Negative controls

Cross-sectional score shuffle:

- shuffle scores within each rebalance month after all eligibility filters are applied
- select `TOP_N` from the shuffled ranking
- `quick` run count: `100`
- `mega` run count: `500`

Block-shuffled return-path null:

- resample the in-sample monthly return panel by contiguous blocks
- `quick` run count: `50`
- `mega` run count: `200`

Hard rule:

- each suite must pass the main candidate-selection gates in at most `5%` of runs

### 7.4 Historical cost model

The primary historical selection model is `tiered_v1`.

One-way cost components:

- brokerage: `max(39 SEK, 10 bps * order_notional_sek)`
- spread/slippage on `median_daily_value_60d_sek`
  - `>= 50m`: `15 bps`
  - `20m-50m`: `25 bps`
  - `10m-20m`: `40 bps`
  - `5m-10m`: `60 bps`
  - `< 5m`: `100 bps`
- low-price add-on: `+10 bps` if `close_raw_sek < 50`
- participation add-on from `order_notional_sek / median_daily_value_60d_sek`
  - `<= 0.25%`: `0 bps`
  - `0.25%-0.50%`: `10 bps`
  - `0.50%-0.75%`: `20 bps`
  - `0.75%-1.00%`: `40 bps`
  - `> 1.00%`: fail the capacity gate
- `next_open` multiplier: `1.25x` on spread/slippage plus participation
- `next_close` multiplier: `1.00x` on spread/slippage plus participation
- FX friction scenarios for non-SEK names:
  - `low`: `10 bps`
  - `base`: `25 bps`
  - `high`: `50 bps`

The flat bps grid remains secondary sensitivity reporting only.

## 8. Outputs

### 8.1 `results/grid_results_{mode}.csv`

Each row must include at minimum:

- `candidate_id`
- `l`
- `skip`
- `top_n`
- `fold_id`
- `universe_variant`
- `execution_model`
- `fx_scenario`
- `cost_model_name`
- `validation_sharpe`
- `validation_max_drawdown`
- `candidate_fold_pass_count`
- `candidate_median_validation_sharpe`
- `candidate_deflated_sharpe_score`
- `candidate_deflated_sharpe_probability`
- `candidate_bootstrap_ci_low`
- `candidate_bootstrap_ci_high`
- `candidate_universe_sensitivity_std`
- `candidate_plateau_neighbor_median_sharpe`
- `candidate_plateau_neighbor_ratio`
- `gate_fold_count`
- `gate_deflated_sharpe`
- `gate_bootstrap`
- `gate_negative_controls`

### 8.2 `results/selection_summary.json`

Must contain:

- `selection_status`
- `locked_candidate`
- `ranked_candidates`
- `negative_controls`
- `neighbor_diagnostics`

### 8.3 `results/consistency_report.json`

Must contain:

- `locked_candidate`
- `comparison_window`
- `quick_profile`
- `mega_profile`
- `differences`
- `consistency_warning`
- `warning_reasons`
- `diagnosis`

### 8.4 `results/walk_forward_results.json`

Must contain holdout metrics for:

- `Full Nordics`
- `SE-only`
- `largest-third-by-market-cap`
- `next_open`
- `next_close`
- `low`
- `base`
- `high`

It must also contain a Phase 4 gate summary that states:

- holdout base-scenario net Sharpe
- whether it is `>= 0.3`
- whether the primary benchmark edge survives modeled costs
- final `phase4_eligible` boolean
