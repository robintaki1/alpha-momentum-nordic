# Alpha Momentum Research Dossier

Generated: `2026-03-17T01:42:05.668031+00:00`

## Executive Summary

- Active validated strategy under selected package model: `yes`
- Package model: `entry_exit_costs`
- Current verdict: `Selected package winner: Ex-Norway (Sweden and Denmark only) / Monthly with holdout Sharpe 1.153.`
- Cadence comparison summary available: `yes`
- Current package focus: `validated candidate branch`

This dossier is intentionally scoped to the package branch you chose to present. It highlights the candidate path that aligns with your preferred execution model and avoids promoting the discarded over-trading branch.

## What This Project Demonstrates

- Thesis-scoped research instead of unconstrained parameter fishing
- Fixed-fold validation with untouched holdout discipline
- Bootstrap, deflated-Sharpe, null-control, and CSCV PBO diagnostics
- Separate cadence-comparison research over weekly through semiannual schedules
- Packaging and reporting that make the repo readable for external reviewers

## Current Operating Reality

- The package currently presents a validated candidate under the selected entry/exit execution model.
- The discarded cadence transaction-cost model is not the package source of truth.
- The monitoring stack still exists, but the package story is anchored on the selected model rather than the over-trading branch.
- Current monitoring window: `2026-02 -> 2026-03`
- Thesis branches tracked in this package build: `2`

## Recommended Reading Order

1. `PROJECT_STATUS.html`
2. `cadence_compare_summary/dashboard.html`
3. `cadence_compare_summary/cadence_comparison_report.html`

## Remaining Gaps

- Liquid-subset diagnostics are still suspended because PTI market-cap history is missing.
- A more realistic banded rebalance model is still unimplemented.
- The next useful work should be a clearly scoped implementation branch, not silent parameter drift.

## Artifact Index

- `PROJECT_STATUS.html`: current plain-English repo status
- `cadence_compare_summary/dashboard.html`: authoritative current-state dashboard
- `cadence_compare_summary/cadence_comparison_report.html`: written verdict summary

## Employer-Facing Interpretation

This project demonstrates a serious validation workflow: scoped hypotheses, fixed folds, robust diagnostics, untouched holdout evaluation, cadence-sensitive re-testing, and a shareable reporting layer. Under the selected entry/exit execution model, the repo retains a validated candidate, while the discarded transaction-cost branch remains an internal contrast point.
