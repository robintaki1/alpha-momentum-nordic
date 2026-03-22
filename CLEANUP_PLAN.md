# Stability Cleanup Plan (Non-Destructive)

Goal: clean up repo hygiene without removing any essential research or paper-trading components.

## P0: Keep the Gating Intact

1. Keep the PRD and phase docs as the source of truth and do not change strategy parameters in code without updating the docs.
2. Confirm that Phase 4 artifacts remain frozen and visible during paper trading.

## P1: Hygiene and Clarity

1. Verify encoding and fix any mojibake if present so Swedish text and punctuation render correctly.
2. Establish a clear artifact policy for `results`, `portfolio`, `papertrading`, and `_snapshots` (tracked vs generated).
3. Confirm that Python is the authoritative engine while C++ is dormant, or re-enable C++ with full parity checks.

## P2: Consistency and Reproducibility

1. Remove duplicated formatting helpers by centralizing shared utilities (dashboard markdown and param formatting).
2. Add a minimal test suite for core math, selection logic, and validation metrics.
3. Add a dependency lock (or pin versions) to reduce result drift across environments.

## P3: Live-Engine Readiness

1. Create a dedicated live-engine scaffold only after the paper-trading gate is satisfied.
2. Add a final broker tradability screen to the live signal stage and log exclusions.
