# 03 Broker Readiness and Live Execution

This document captures the current broker assumption for the project so implementation can start without repeatedly reopening the same question.

## Current Status Note

As of `2026-03-18`, the authoritative cadence summary reports no active validated winner
under the `legacy_entry_exit_costs` validation model. Broker readiness and live execution
therefore remain paused. This file remains the broker-scope assumption reference if a
future candidate is revalidated and approved for paper trading or live use.

## 1. Current Decision

As of `2026-03-13`, the intended live account type was `Nordea ISK`.

A manual broker-readiness check was completed in Nordea app/natbanken using the `Buy` flow up to the final confirmation screen for one sample stock per target market:

- `Volvo B` on `Nasdaq Stockholm`
- `Novo Nordisk B` on `Nasdaq Copenhagen`
- `Equinor` on `Oslo Bors`

The three currently in-scope corrected-baseline markets succeeded.

Decision for implementation:

- keep the corrected baseline universe as `Sweden + Denmark + Norway`
- do NOT collapse the project to `SE-only` for broker reasons
- continue using the same research data and validation path already defined in the PRD
- do not start broker/live implementation until a strategy validates again under the authoritative cadence model

## 2. What This Check Means

The passed broker-readiness check is a market-level validation only.

It means:

- the intended `ISK` account can trade the currently relevant corrected-baseline markets in principle
- broker access is not currently a blocker for Phase 1-3 implementation
- the corrected baseline remains a realistic target universe for research

It does NOT mean:

- every future stock in the backtest will always be tradable
- every month will have identical broker availability
- the project should build broker automation now

## 3. What To Implement Now

Implementation in Phase 1-3 should assume:

- universe construction remains `Nasdaq Stockholm`, `Nasdaq Copenhagen`, and `Oslo Bors`
- the same EODHD and Riksbank-based research pipeline remains valid
- non-SEK names remain in scope and must continue to use explicit FX-friction scenarios
- no broker API integration or order routing is required in this revision

## 4. What The Future Live Engine Must Do

When a future live-signal tool is built for a revalidated candidate, it must still apply a final broker tradability screen before producing the monthly trade plan.

Minimum behavior:

- evaluate tradability for the intended account type, currently `Nordea ISK`
- preserve the research-ranked candidates first, then filter only at the final broker gate
- exclude any stock that cannot actually be traded in the account at that time
- log a clear exclusion reason in the monthly report
- never silently replace excluded names with hand-picked discretionary substitutions

Suggested trade-plan fields:

- `broker_account_type`
- `broker_tradable`
- `broker_check_timestamp`
- `broker_block_reason`

## 5. Monthly Live Checklist Later

Before any future paper-trade restart or live-trade month after revalidation:

1. run `live_signal.py`
2. review the proposed trades
3. verify the shortlisted names can still be traded in `Nordea ISK`
4. document any broker blocks or special restrictions
5. trade only the names that pass the final broker screen

If broker blocks become common enough to materially change the portfolio composition, that is no longer just a live-execution detail. The restriction set must then be pulled back into research and paper-trading assumptions.
