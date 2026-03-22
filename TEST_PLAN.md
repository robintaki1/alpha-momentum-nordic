# Minimal Test Plan

This is a lightweight, high-value test outline that does not change the model or research scope.

## Proposed Test Areas

1. Momentum scoring and ranking
2. Strategy variant selection (banded, min_hold, weighting)
3. Cost model calculations and capacity gates
4. Validation metrics (Sharpe, deflated Sharpe, bootstrap CI)
5. Cadence schedule generation (weekly, biweekly, monthly, quarterly, semiannual)
6. Forward-monitor selection loading and frozen-manifest checks

## Suggested Test Locations

1. `C:\Users\robin\Desktop\alpha momentum\engine.py`
2. `C:\Users\robin\Desktop\alpha momentum\python\strategy_variants.py`
3. `C:\Users\robin\Desktop\alpha momentum\python\validation_protocol.py`
4. `C:\Users\robin\Desktop\alpha momentum\python\cadence_engine.py`
5. `C:\Users\robin\Desktop\alpha momentum\python\forward_monitor.py`

## Minimal Test Examples

1. Momentum scores are invariant to linear scaling of prices and reject non-positive inputs.
2. Banded rebalancing keeps names within the buffer and only fills to top_n when needed.
3. Min-hold strategy locks names for the requested holding period.
4. Deflated Sharpe returns 0 for insufficient data and reasonable values for stable series.
5. Cadence schedule produces deterministic period counts and correct offsets.
6. Frozen selection loader fails fast when the locked candidate is missing.
