# 04 Phase 4 Forward Monitoring and Governance

This document supplements `AlphaMomentum_PRD_v4.md` and defines the operating process for Phase 4 before any further monitoring automation or live-execution work is added.

The goal is simple:

- freeze the active monthly candidate and its shadow control clearly
- keep the governance trail readable after revalidation
- avoid retuning or stealth changes once paper trading starts

## 1. Current Frozen State (Reference Only)

As of `2026-03-18`, the repo preserves the following monthly candidate setup for
reference and audit only:

- Lead thesis: `ex_norway`
- Lead label: `Ex-Norway (Sweden and Denmark only)`
- Lead params: `L=12 / skip=1 / top_n=8 / strat=trend_filter / trend_ma=10`
- Lead source: `results/cadence_compare_rebuild/ex_norway/1m/selection_summary.json`

- Shadow control thesis: `baseline`
- Shadow label: `Baseline (Sweden, Denmark, Norway)`
- Shadow params: `L=12 / skip=1 / top_n=8 / strat=trend_filter / trend_ma=10`
- Shadow source: `results/cadence_compare_rebuild/baseline/1m/selection_summary.json`

The machine-readable preserved reference is:

- `results/forward_monitor/frozen_strategy_manifest.json`

The employer-facing package is:

- `portfolio/alpha_momentum_validation_package/`

## 2. Why This Reference Phase Exists

The authoritative cadence comparison summary now reports no validated winner under the
`legacy_entry_exit_costs` validation model. This file therefore describes a preserved
reference package and governance trail, not an active paper-trading playbook.

The turnover-aware equal-weight cadence model remains a stricter contrast branch, and the
preserved monthly package remains visible only as a historical/reference artifact.

The frozen monthly package is used to answer these practical questions:

- does the frozen lead still behave acceptably when new months arrive?
- does the lead continue to look better than the shadow control?
- do turnover, cost drag, and concentration stay tolerable?
- do the live books still look like something a disciplined investor could actually follow?

## 3. Allowed and Forbidden Actions

### Allowed

- rebuild the frozen monthly forward monitor
- refresh the portfolio package artifacts
- add audit-only diagnostics that can downgrade or disqualify the frozen strategy
- improve documentation and reporting
- improve data freshness checks, tradability checks, and logging

### Forbidden

- changing `L`, `skip`, or `top_n`
- changing the lead thesis or universe definition silently
- loosening validation gates after the fact
- expanding the grid again while claiming the same frozen strategy is still in force
- removing a bad month from the record
- replacing excluded names with discretionary hand-picked substitutions

Any change to the strategy definition must be treated as a new research cycle, not as a continuation of the same preserved monthly candidate.

## 4. Monthly Operating Procedure

Run this only when auditing the preserved candidate or comparing it against a future
revalidation cycle.

1. Run `run_monthly_rebalance.bat`
2. This rebuilds the forward monitor and applies the next paper-trade rebalance only if a new holding month is available
3. Open `results/forward_monitor/dashboard.html`
4. Open `portfolio/alpha_momentum_validation_package/research_audit_dossier.html`
5. Open `portfolio/alpha_momentum_validation_package/PROJECT_STATUS.html`
6. Review the frozen lead and shadow books side by side
7. Record the month in the monitoring log
8. Do not change the strategy definition

## 5. Monthly Review Rubric

Each monthly review should capture at least:

- lead realized return
- shadow realized return
- passive benchmark return
- lead versus shadow overlap
- estimated turnover and rebalance drag
- top-country concentration
- any stale-data or tradability issues
- whether the lead still looks meaningfully better than the shadow

If something looks wrong, document it. Do not fix it by retuning the model during the same frozen cycle.

## 6. Current Diagnostics That Must Stay Visible

The following checks should remain part of the forward-monitoring read:

- untouched holdout result
- certification PBO / POB warning level
- theory-aligned signal-null check
- path-order robustness watch
- pre-trade audit
- holdout breadth / top-name dependence
- lead versus shadow overlap
- authoritative cadence-comparison verdict

These are already surfaced in:

- `results/forward_monitor/dashboard.html`
- `results/forward_monitor/forward_monitor_summary.json`
- `results/forward_monitor/research_audit_dossier.html`
- `results/forward_monitor/holdout_name_dependence.csv`

## 7. Decision Calendar

The old monthly cycle originally used these review horizons:

- First serious review: after `6` monthly rebalances
- Stronger review: after `12` monthly rebalances

With the preserved legacy setup starting at `2026-03`, that means:

- first serious review around `2026-08`
- stronger go / no-go review around `2027-02`

Those dates are historical/reference checkpoints for the preserved monthly candidate,
not the current active go/no-go clock.

## 8. Suggested Monitoring Log Format

The monitoring log can be Markdown, CSV, or spreadsheet-based, but it should contain at least:

| Field | Example |
| --- | --- |
| review_month | `2026-03` |
| lead_return | `-30.5%` |
| shadow_return | `-30.4%` |
| benchmark_return | `...` |
| lead_turnover_name_fraction | `40.0%` |
| lead_estimated_rebalance_cost | `48.5 bps` |
| lead_top_country_weight | `SE 60%` |
| overlap_vs_shadow | `40.0%` |
| data_issues | `none` |
| tradability_issues | `none` |
| reviewer_note | `No parameter changes. Watch concentration.` |

## 9. Employer-Facing Record Keeping

The folder that should be kept tidy and shareable is:

- `portfolio/alpha_momentum_validation_package/`

Minimum files that should remain current:

- `README.html`
- `PROJECT_STATUS.html`
- `research_audit_dossier.html`
- `dashboard.html`
- `cadence_compare_summary/cadence_comparison_report.html`
- `cadence_compare_summary/dashboard.html`
- `cadence_compare_summary/cadence_comparison.json`
- `research_engine_rebuild/summary/dashboard.html`

## 10. Current Caveats That Must Be Disclosed Honestly

- `ex_norway` is a challenger thesis, not the cleanest ex-ante baseline
- the original baseline should remain visible as the shadow control
- the authoritative package uses the legacy entry/exit cost model; the turnover-aware model remains a stricter contrast
- certification PBO is in the caution zone, not the ideal zone
- path-order robustness remains weak as a diagnostic
- liquid-subset diagnostics remain suspended until PTI market-cap data exists

These caveats mean the preserved reference package must remain frozen and transparent.
They must be disclosed rather than hidden.

## 11. Done Definition For This Documentation Phase

This documentation step is complete when:

- the PRD reflects frozen Phase 4 governance
- this file exists as the operating spec
- the preserved manifest remains the source of truth for the paper-trading package
- the portfolio package remains the employer-facing bundle

Only after that should any additional Phase 4 tooling or logging automation be added.
