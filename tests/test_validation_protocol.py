from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_protocol import (
    candidate_aggregates,
    cross_sectional_score_shuffle_runs,
    deflated_sharpe_metrics,
    evaluate_holdout_candidate,
    fixed_folds,
    leakage_detected,
    negative_control_pass_rate,
    selection_summary,
    stationary_bootstrap_sharpe_ci,
    tiered_cost_breakdown,
    TieredCostInputs,
)


def build_candidate_manifest() -> dict:
    strong_pattern = [0.04, 0.01] * 12
    medium_pattern = [0.02, 0.005] * 12
    weak_pattern = [0.01, -0.01] * 12

    def rows_for_candidate(params: dict, pattern: list[float]) -> dict:
        evaluations = []
        for fold in fixed_folds():
            for variant in ("Full Nordics", "SE-only", "largest-third-by-market-cap"):
                scale = {
                    "Full Nordics": 1.0,
                    "SE-only": 0.9,
                    "largest-third-by-market-cap": 0.95,
                }[variant]
                evaluations.append(
                    {
                        "fold_id": fold.fold_id,
                        "validate_start": fold.validate_start,
                        "validate_end": fold.validate_end,
                        "universe_variant": variant,
                        "execution_model": "next_open",
                        "fx_scenario": "base",
                        "cost_model_name": "tiered_v1",
                        "monthly_returns": [value * scale for value in pattern],
                    }
                )
        return {"params": params, "evaluations": evaluations}

    return {
        "mode": "mega",
        "candidates": [
            rows_for_candidate({"l": 12, "skip": 1, "top_n": 10}, strong_pattern),
            rows_for_candidate({"l": 9, "skip": 1, "top_n": 10}, medium_pattern),
            rows_for_candidate({"l": 6, "skip": 1, "top_n": 10}, weak_pattern),
        ],
        "negative_controls": {
            "cross_sectional_shuffle": {"run_count": 500, "pass_count": 0},
            "block_shuffled_null": {"run_count": 200, "pass_count": 0},
        },
    }


def build_holdout_candidate(params: dict) -> dict:
    evaluations = []
    for variant in ("Full Nordics", "SE-only", "largest-third-by-market-cap"):
        for execution_model in ("next_open", "next_close"):
            for fx_scenario in ("low", "base", "high"):
                scale = {
                    "Full Nordics": 1.0,
                    "SE-only": 0.9,
                    "largest-third-by-market-cap": 0.95,
                }[variant]
                execution_scale = 1.0 if execution_model == "next_open" else 0.85
                fx_scale = {"low": 1.0, "base": 0.95, "high": 0.85}[fx_scenario]
                strategy_returns = [0.03 * scale * execution_scale * fx_scale, 0.01 * scale * execution_scale * fx_scale] * 48
                primary_benchmark_returns = [0.01, 0.005] * 48
                secondary_benchmark_returns = [0.008, 0.004] * 48
                evaluations.append(
                    {
                        "window_start": "2018-01",
                        "window_end": "2026-01",
                        "universe_variant": variant,
                        "execution_model": execution_model,
                        "fx_scenario": fx_scenario,
                        "cost_model_name": "tiered_v1",
                        "monthly_returns": strategy_returns,
                        "primary_benchmark_returns": primary_benchmark_returns,
                        "secondary_benchmark_returns": secondary_benchmark_returns,
                    }
                )
    return {"params": params, "evaluations": evaluations}


class ValidationProtocolTests(unittest.TestCase):
    def test_fixed_folds_never_touch_holdout(self) -> None:
        for fold in fixed_folds():
            self.assertFalse(leakage_detected(fold.validate_start))

    def test_stationary_bootstrap_is_reproducible(self) -> None:
        returns = [0.03, -0.01, 0.02, 0.01] * 12
        first = stationary_bootstrap_sharpe_ci(returns, seed=42)
        second = stationary_bootstrap_sharpe_ci(returns, seed=42)
        self.assertEqual(first, second)

    def test_deflated_sharpe_on_null_like_returns_is_non_positive(self) -> None:
        returns = [0.01, -0.01] * 24
        metrics = deflated_sharpe_metrics(returns, n_trials=20)
        self.assertLessEqual(metrics["score"], 0.0)

    def test_tiered_cost_model_applies_capacity_and_execution_differences(self) -> None:
        next_close = tiered_cost_breakdown(
            TieredCostInputs(
                order_notional_sek=20_000,
                median_daily_value_60d_sek=10_000_000,
                close_raw_sek=45,
                execution_model="next_close",
                is_non_sek_name=True,
                fx_scenario="base",
            )
        )
        next_open = tiered_cost_breakdown(
            TieredCostInputs(
                order_notional_sek=20_000,
                median_daily_value_60d_sek=10_000_000,
                close_raw_sek=45,
                execution_model="next_open",
                is_non_sek_name=True,
                fx_scenario="base",
            )
        )
        self.assertTrue(next_close.passes_capacity_gate)
        self.assertGreaterEqual(next_open.total_bps, next_close.total_bps)
        self.assertGreater(next_open.low_price_addon_bps, 0.0)
        self.assertGreater(next_open.fx_bps, 0.0)

    def test_negative_control_shuffle_stays_below_pass_rate_threshold(self) -> None:
        months = [
            {
                "positions": [
                    {"score": 0.9, "next_return": 0.0},
                    {"score": 0.8, "next_return": 0.0},
                    {"score": 0.7, "next_return": 0.0},
                ]
            }
            for _ in range(24)
        ]
        shuffled = cross_sectional_score_shuffle_runs(months, top_n=2, n_runs=25, seed=3)
        pass_rate = negative_control_pass_rate(shuffled, n_trials=10)
        self.assertLessEqual(pass_rate, 0.05)

    def test_candidate_selection_and_holdout_flow(self) -> None:
        manifest = build_candidate_manifest()
        aggregates, negative_controls = candidate_aggregates(manifest)
        summary = selection_summary(aggregates, negative_controls, mode="mega")
        self.assertEqual(summary["selection_status"], "selected")
        locked_candidate = summary["locked_candidate"]
        self.assertIsNotNone(locked_candidate)
        self.assertEqual(locked_candidate["params"], {"l": 12, "skip": 1, "top_n": 10})

        holdout_candidate = build_holdout_candidate(locked_candidate["params"])
        holdout_results = evaluate_holdout_candidate(holdout_candidate)
        self.assertTrue(holdout_results["phase4_gate"]["meets_sharpe_gate"])
        self.assertTrue(holdout_results["phase4_gate"]["beats_primary_benchmark"])
        self.assertTrue(holdout_results["phase4_gate"]["phase4_eligible"])


if __name__ == "__main__":
    unittest.main()
