from __future__ import annotations

import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

NORMAL_DIST = statistics.NormalDist()
REQUIRED_UNIVERSE_VARIANTS = (
    "Full Nordics",
    "SE-only",
    "largest-third-by-market-cap",
)
REQUIRED_EXECUTION_MODELS = ("next_open", "next_close")
REQUIRED_FX_SCENARIOS = ("low", "base", "high")
PRIMARY_TRACK = {
    "universe_variant": "Full Nordics",
    "execution_model": "next_open",
    "fx_scenario": "base",
    "cost_model_name": config.PRIMARY_SELECTION_COST_MODEL,
}


@dataclass(frozen=True)
class RollingFold:
    fold_id: str
    train_start: str
    train_end: str
    validate_start: str
    validate_end: str


@dataclass(frozen=True)
class TieredCostInputs:
    order_notional_sek: float
    median_daily_value_60d_sek: float
    close_raw_sek: float
    execution_model: str
    is_non_sek_name: bool = False
    fx_scenario: str = "base"


@dataclass(frozen=True)
class CostBreakdown:
    brokerage_sek: float
    brokerage_bps: float
    spread_slippage_bps: float
    low_price_addon_bps: float
    participation_bps: float
    fx_bps: float
    total_bps: float
    passes_capacity_gate: bool


def fixed_folds() -> list[RollingFold]:
    return [RollingFold(*fold) for fold in config.ROLLING_ORIGIN_FOLDS]


def month_to_ordinal(month: str) -> int:
    year, month_number = month.split("-")
    return int(year) * 12 + int(month_number) - 1


def candidate_id(params: dict[str, Any]) -> str:
    strategy_id = params.get("strategy_id", "baseline")
    return f"l{params['l']}_s{params['skip']}_n{params['top_n']}_strat={strategy_id}"


def annualized_sharpe(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    std = statistics.stdev(returns)
    if std == 0:
        return 0.0
    return statistics.fmean(returns) / std * math.sqrt(12.0)


def max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = 1.0 - (equity / peak if peak else 1.0)
        max_dd = max(max_dd, drawdown)
    return max_dd


def total_return(returns: Sequence[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def _sample_skewness(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean_value = statistics.fmean(values)
    std = statistics.stdev(values)
    if std == 0:
        return 0.0
    third_moment = statistics.fmean(((value - mean_value) / std) ** 3 for value in values)
    return third_moment


def _sample_kurtosis(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 3.0
    mean_value = statistics.fmean(values)
    std = statistics.stdev(values)
    if std == 0:
        return 3.0
    fourth_moment = statistics.fmean(((value - mean_value) / std) ** 4 for value in values)
    return fourth_moment


def deflated_sharpe_metrics(returns: Sequence[float], n_trials: int) -> dict[str, float]:
    if len(returns) < 3 or n_trials < 1:
        return {"score": 0.0, "probability": 0.0, "expected_max_noise_sharpe": 0.0}
    mean_value = statistics.fmean(returns)
    std = statistics.stdev(returns)
    if std == 0:
        return {"score": 0.0, "probability": 0.0, "expected_max_noise_sharpe": 0.0}

    monthly_sharpe = mean_value / std
    skewness = _sample_skewness(returns)
    kurtosis = _sample_kurtosis(returns)
    variance_term = 1.0 - skewness * monthly_sharpe + ((kurtosis - 1.0) / 4.0) * monthly_sharpe**2
    variance_term = max(variance_term, 1e-12)
    sharpe_std = math.sqrt(variance_term / max(len(returns) - 1, 1))

    if n_trials == 1:
        expected_max_noise_sharpe = 0.0
    else:
        gamma = 0.5772156649
        first = NORMAL_DIST.inv_cdf(max(1e-9, min(1 - 1e-9, 1.0 - 1.0 / n_trials)))
        second = NORMAL_DIST.inv_cdf(
            max(1e-9, min(1 - 1e-9, 1.0 - 1.0 / (n_trials * math.e)))
        )
        expected_max_noise_sharpe = sharpe_std * ((1.0 - gamma) * first + gamma * second)

    score = monthly_sharpe - expected_max_noise_sharpe
    probability = NORMAL_DIST.cdf(score / sharpe_std) if sharpe_std > 0 else 0.0
    return {
        "score": score,
        "probability": probability,
        "expected_max_noise_sharpe": expected_max_noise_sharpe,
    }


def _stationary_bootstrap_sample(values: Sequence[float], mean_block_length: int, rng: random.Random) -> list[float]:
    if not values:
        return []
    p = 1.0 / max(mean_block_length, 1)
    n_values = len(values)
    indices: list[int] = []
    index = rng.randrange(n_values)
    for position in range(n_values):
        if position == 0 or rng.random() < p:
            index = rng.randrange(n_values)
        else:
            index = (index + 1) % n_values
        indices.append(index)
    return [values[index] for index in indices]


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def stationary_bootstrap_sharpe_ci(
    returns: Sequence[float],
    mean_block_length: int = config.BOOTSTRAP_BLOCK_LENGTH_MONTHS,
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = 7,
) -> tuple[float, float]:
    if len(returns) < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    sharpe_values = [
        annualized_sharpe(_stationary_bootstrap_sample(returns, mean_block_length, rng))
        for _ in range(n_resamples)
    ]
    sharpe_values.sort()
    return (_quantile(sharpe_values, 0.025), _quantile(sharpe_values, 0.975))


def leakage_detected(validate_start: str, holdout_start: str | None = None) -> bool:
    holdout_start = holdout_start or config.OOS_START
    return month_to_ordinal(validate_start) >= month_to_ordinal(holdout_start)


def validate_phase2_evaluations(evaluations: Sequence[dict[str, Any]]) -> None:
    expected_folds = {fold.fold_id: fold for fold in fixed_folds()}
    for evaluation in evaluations:
        fold_id = evaluation["fold_id"]
        if fold_id not in expected_folds:
            raise ValueError(f"Unknown fold_id '{fold_id}' in Phase 2 candidate panel.")
        expected_fold = expected_folds[fold_id]
        if evaluation["validate_start"] != expected_fold.validate_start or evaluation["validate_end"] != expected_fold.validate_end:
            raise ValueError(f"Fold '{fold_id}' does not match the fixed validation protocol.")
        if leakage_detected(evaluation["validate_start"]):
            raise ValueError(f"Leakage detected: Phase 2 evaluation '{fold_id}' touches the untouched holdout.")


def spread_slippage_bps(median_daily_value_60d_sek: float) -> float:
    for threshold, bps in config.SPREAD_BPS_BUCKETS:
        if median_daily_value_60d_sek >= threshold:
            return bps
    return config.SPREAD_BPS_BUCKETS[-1][1]


def participation_bps(order_notional_sek: float, median_daily_value_60d_sek: float) -> tuple[float, bool]:
    if median_daily_value_60d_sek <= 0:
        return (float("inf"), False)
    ratio = order_notional_sek / median_daily_value_60d_sek
    if ratio > config.MAX_ORDER_FRACTION_OF_60D_MEDIAN_DAILY_VALUE:
        return (float("inf"), False)
    for threshold, bps in config.PARTICIPATION_BPS_BUCKETS:
        if ratio <= threshold:
            return (bps, True)
    return (float("inf"), False)


def tiered_cost_breakdown(inputs: TieredCostInputs) -> CostBreakdown:
    if inputs.order_notional_sek <= 0:
        raise ValueError("order_notional_sek must be positive.")
    brokerage_sek = max(
        config.BROKERAGE_MIN_SEK,
        inputs.order_notional_sek * (config.BROKERAGE_BPS / 10_000.0),
    )
    brokerage_bps = (brokerage_sek / inputs.order_notional_sek) * 10_000.0
    spread_bps = spread_slippage_bps(inputs.median_daily_value_60d_sek)
    participation_component_bps, passes_capacity_gate = participation_bps(
        inputs.order_notional_sek, inputs.median_daily_value_60d_sek
    )
    if not passes_capacity_gate:
        return CostBreakdown(
            brokerage_sek=brokerage_sek,
            brokerage_bps=brokerage_bps,
            spread_slippage_bps=spread_bps,
            low_price_addon_bps=0.0,
            participation_bps=participation_component_bps,
            fx_bps=0.0,
            total_bps=float("inf"),
            passes_capacity_gate=False,
        )

    multiplier = {
        "next_open": config.NEXT_OPEN_IMPACT_MULTIPLIER,
        "next_close": config.NEXT_CLOSE_IMPACT_MULTIPLIER,
    }.get(inputs.execution_model)
    if multiplier is None:
        raise ValueError(f"Unsupported execution_model '{inputs.execution_model}'.")
    impact_bps = (spread_bps + participation_component_bps) * multiplier
    low_price_addon_bps = config.LOW_PRICE_ADDON_BPS if inputs.close_raw_sek < config.LOW_PRICE_THRESHOLD_SEK else 0.0
    fx_bps = config.FX_FRICTION_SCENARIOS_BPS[inputs.fx_scenario] if inputs.is_non_sek_name else 0.0
    total_bps = brokerage_bps + impact_bps + low_price_addon_bps + fx_bps
    return CostBreakdown(
        brokerage_sek=brokerage_sek,
        brokerage_bps=brokerage_bps,
        spread_slippage_bps=impact_bps,
        low_price_addon_bps=low_price_addon_bps,
        participation_bps=participation_component_bps,
        fx_bps=fx_bps,
        total_bps=total_bps,
        passes_capacity_gate=True,
    )


def _weights_for_selection(selected: Sequence[dict[str, Any]], weighting: str, top_n: int) -> list[float]:
    if not selected:
        return []
    exposure = float(len(selected)) / float(max(1, top_n))
    if weighting in ("cap", "inv_vol"):
        raw_values = []
        for item in selected:
            value = item.get("weight_value")
            if value is None:
                raw_values.append(float("nan"))
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = float("nan")
            if not math.isfinite(numeric) or numeric <= 0.0:
                raw_values.append(float("nan"))
            else:
                raw_values.append(1.0 / numeric if weighting == "inv_vol" else numeric)
        if not any(math.isfinite(value) for value in raw_values):
            return [exposure / float(len(selected))] * len(selected)
        cleaned = [value if math.isfinite(value) else 0.0 for value in raw_values]
        total = sum(cleaned)
        if total <= 0.0:
            return [exposure / float(len(selected))] * len(selected)
        return [exposure * value / total for value in cleaned]
    return [exposure / float(len(selected))] * len(selected)


def cross_sectional_score_shuffle_runs(
    months: Sequence[dict[str, Any]],
    top_n: int,
    n_runs: int,
    seed: int = 11,
) -> list[list[float]]:

    rng = random.Random(seed)
    all_runs: list[list[float]] = []
    for _ in range(n_runs):
        run_returns: list[float] = []
        for month in months:
            positions = month["positions"]
            filter_on = month.get("filter_on", True)
            if not filter_on:
                run_returns.append(0.0)
                continue
            weighting = month.get("weighting", "equal")
            baseline_return = month.get("baseline_return")
            shuffled_scores = [position["score"] for position in positions]
            rng.shuffle(shuffled_scores)
            shuffled_positions = []
            for position, score in zip(positions, shuffled_scores):
                shuffled_positions.append(
                    {
                        "score": score,
                        "next_return": position["next_return"],
                        "weight_value": position.get("weight_value"),
                    }
                )
            shuffled_positions.sort(key=lambda item: item["score"], reverse=True)
            selected = shuffled_positions[:top_n]
            if not selected:
                run_returns.append(0.0)
                continue
            weights = _weights_for_selection(selected, weighting, top_n)
            if not weights:
                run_returns.append(0.0)
                continue
            selected_mean = sum(
                weight * float(item["next_return"]) for weight, item in zip(weights, selected, strict=True)
            )
            try:
                if baseline_return is not None:
                    run_returns.append(selected_mean - float(baseline_return))
                else:
                    run_returns.append(selected_mean)
            except (TypeError, ValueError):
                run_returns.append(selected_mean)
        all_runs.append(run_returns)
    return all_runs


def cross_sectional_score_actual_run(
    months: Sequence[dict[str, Any]],
    top_n: int,
) -> list[float]:
    run_returns: list[float] = []
    for month in months:
        positions = month["positions"]
        filter_on = month.get("filter_on", True)
        if not filter_on:
            run_returns.append(0.0)
            continue
        weighting = month.get("weighting", "equal")
        baseline_return = month.get("baseline_return")
        ordered = sorted(positions, key=lambda item: item["score"], reverse=True)
        selected = ordered[:top_n]
        if not selected:
            run_returns.append(0.0)
            continue
        weights = _weights_for_selection(selected, weighting, top_n)
        if not weights:
            run_returns.append(0.0)
            continue
        selected_mean = sum(
            weight * float(item["next_return"]) for weight, item in zip(weights, selected, strict=True)
        )
        try:
            if baseline_return is not None:
                run_returns.append(selected_mean - float(baseline_return))
            else:
                run_returns.append(selected_mean)
        except (TypeError, ValueError):
            run_returns.append(selected_mean)
    return run_returns


def block_shuffled_return_path_runs(
    returns: Sequence[float],
    block_length: int,
    n_runs: int,
    seed: int = 17,
) -> list[list[float]]:
    if block_length <= 0:
        raise ValueError("block_length must be positive.")
    rng = random.Random(seed)
    blocks = [list(returns[index : index + block_length]) for index in range(0, len(returns), block_length)]
    runs: list[list[float]] = []
    for _ in range(n_runs):
        shuffled_blocks = blocks[:]
        rng.shuffle(shuffled_blocks)
        run = [value for block in shuffled_blocks for value in block][: len(returns)]
        runs.append(run)
    return runs


def negative_control_pass_rate(
    return_runs: Sequence[Sequence[float]],
    n_trials: int,
    sharpe_threshold: float = 0.4,
) -> float:
    if not return_runs:
        return 1.0
    pass_count = 0
    for returns in return_runs:
        sharpe = annualized_sharpe(returns)
        bootstrap_low, _ = stationary_bootstrap_sharpe_ci(returns, n_resamples=250, seed=23)
        dsr_score = deflated_sharpe_metrics(returns, n_trials)["score"]
        if sharpe > sharpe_threshold and bootstrap_low > 0.0 and dsr_score > 0.0:
            pass_count += 1
    return pass_count / len(return_runs)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def required_negative_controls(negative_controls: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        cross_sectional = negative_controls["cross_sectional_shuffle"]
        block_shuffled = negative_controls["block_shuffled_null"]
    except KeyError as exc:
        raise ValueError("Missing required negative-control summary.") from exc
    return {
        "cross_sectional_shuffle": cross_sectional,
        "block_shuffled_null": block_shuffled,
    }


def negative_controls_pass(negative_controls: dict[str, Any]) -> bool:
    controls = required_negative_controls(negative_controls)
    gated_keys = getattr(config, "NEGATIVE_CONTROL_GATED_KEYS", tuple(controls.keys()))
    for key in gated_keys:
        payload = controls.get(key)
        if payload is None:
            return False
        run_count = payload["run_count"]
        pass_count = payload["pass_count"]
        if run_count <= 0:
            return False
        if (pass_count / run_count) > config.NEGATIVE_CONTROL_PASS_RATE_MAX:
            return False
    return True


def _filter_evaluations(
    evaluations: Iterable[dict[str, Any]],
    *,
    universe_variant: str | None = None,
    execution_model: str | None = None,
    fx_scenario: str | None = None,
    cost_model_name: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for evaluation in evaluations:
        if universe_variant is not None and evaluation["universe_variant"] != universe_variant:
            continue
        if execution_model is not None and evaluation["execution_model"] != execution_model:
            continue
        if fx_scenario is not None and evaluation["fx_scenario"] != fx_scenario:
            continue
        if cost_model_name is not None and evaluation["cost_model_name"] != cost_model_name:
            continue
        rows.append(evaluation)
    return rows


def _concatenate_returns(evaluations: Sequence[dict[str, Any]], ordered_fold_ids: Sequence[str] | None = None) -> list[float]:
    ordered = list(evaluations)
    if ordered_fold_ids is not None:
        fold_order = {fold_id: index for index, fold_id in enumerate(ordered_fold_ids)}
        ordered.sort(key=lambda item: fold_order[item["fold_id"]])
    returns: list[float] = []
    for evaluation in ordered:
        returns.extend(evaluation["monthly_returns"])
    return returns


def candidate_aggregates(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = manifest["candidates"]
    negative_controls = manifest.get("negative_controls", {})
    neg_controls_gate = negative_controls_pass(negative_controls)
    ordered_folds = [fold.fold_id for fold in fixed_folds()]
    n_trials = len(candidates)
    aggregates: list[dict[str, Any]] = []

    for candidate in candidates:
        params = candidate["params"]
        candidate_name = candidate_id(params)
        evaluations = candidate["evaluations"]
        validate_phase2_evaluations(evaluations)

        main_track_evaluations = _filter_evaluations(evaluations, **PRIMARY_TRACK)
        fold_map = {row["fold_id"]: row for row in main_track_evaluations}
        missing_folds = [fold_id for fold_id in ordered_folds if fold_id not in fold_map]
        if missing_folds:
            raise ValueError(f"Candidate {candidate_name} is missing required folds: {', '.join(missing_folds)}")

        per_fold_sharpes = {
            fold_id: annualized_sharpe(fold_map[fold_id]["monthly_returns"]) for fold_id in ordered_folds
        }
        per_fold_drawdowns = {
            fold_id: max_drawdown(fold_map[fold_id]["monthly_returns"]) for fold_id in ordered_folds
        }
        concatenated_returns = _concatenate_returns([fold_map[fold_id] for fold_id in ordered_folds])
        median_validation_sharpe = statistics.median(per_fold_sharpes.values())
        fold_pass_count = sum(value > 0.4 for value in per_fold_sharpes.values())
        bootstrap_ci_low, bootstrap_ci_high = stationary_bootstrap_sharpe_ci(concatenated_returns)
        dsr_metrics = deflated_sharpe_metrics(concatenated_returns, n_trials)

        universe_variant_sharpes: list[float] = []
        for variant in REQUIRED_UNIVERSE_VARIANTS:
            rows = _filter_evaluations(
                evaluations,
                universe_variant=variant,
                execution_model=PRIMARY_TRACK["execution_model"],
                fx_scenario=PRIMARY_TRACK["fx_scenario"],
                cost_model_name=PRIMARY_TRACK["cost_model_name"],
            )
            if len(rows) != len(ordered_folds):
                universe_variant_sharpes = []
                break
            universe_variant_sharpes.append(annualized_sharpe(_concatenate_returns(rows, ordered_folds)))
        universe_sensitivity_std = (
            statistics.pstdev(universe_variant_sharpes) if universe_variant_sharpes else float("inf")
        )
        aggregate = {
            "candidate_id": candidate_name,
            "params": params,
            "evaluations": evaluations,
            "per_fold_sharpes": per_fold_sharpes,
            "per_fold_drawdowns": per_fold_drawdowns,
            "fold_pass_count": fold_pass_count,
            "median_validation_sharpe": median_validation_sharpe,
            "bootstrap_ci_low": bootstrap_ci_low,
            "bootstrap_ci_high": bootstrap_ci_high,
            "deflated_sharpe_score": dsr_metrics["score"],
            "deflated_sharpe_probability": dsr_metrics["probability"],
            "max_drawdown": max_drawdown(concatenated_returns),
            "concatenated_returns": concatenated_returns,
            "universe_sensitivity_std": universe_sensitivity_std,
            "gate_fold_count": fold_pass_count >= config.MEGA_WF_PASSES_REQUIRED,
            "gate_deflated_sharpe": dsr_metrics["score"] > 0.0,
            "gate_bootstrap": bootstrap_ci_low > 0.0,
            "gate_negative_controls": neg_controls_gate,
        }
        aggregates.append(aggregate)

    _attach_plateau_diagnostics(aggregates)
    _attach_ranks(aggregates)
    return aggregates, negative_controls


def _attach_plateau_diagnostics(aggregates: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for aggregate in aggregates:
        params = aggregate["params"]
        strategy_id = params.get("strategy_id", "baseline")
        key = (strategy_id, int(params["l"]), int(params["skip"]))
        grouped.setdefault(key, []).append(aggregate)

    for group in grouped.values():
        top_n_values = sorted({item["params"]["top_n"] for item in group})
        by_top_n = {item["params"]["top_n"]: item for item in group}
        for aggregate in group:
            params = aggregate["params"]
            neighbors: list[dict[str, Any]] = []
            current_index = top_n_values.index(params["top_n"])
            for offset in (-1, 1):
                neighbor_index = current_index + offset
                if neighbor_index < 0 or neighbor_index >= len(top_n_values):
                    continue
                neighbor = by_top_n.get(top_n_values[neighbor_index])
                if neighbor is not None:
                    neighbors.append(neighbor)
            if neighbors:
                neighbor_sharpes = [neighbor["median_validation_sharpe"] for neighbor in neighbors]
                median_neighbor_sharpe = statistics.median(neighbor_sharpes)
                ratio = (
                    median_neighbor_sharpe / aggregate["median_validation_sharpe"]
                    if aggregate["median_validation_sharpe"] not in (0.0, float("inf"), float("-inf"))
                    else 0.0
                )
            else:
                median_neighbor_sharpe = float("-inf")
                ratio = 0.0
            aggregate["plateau_neighbor_median_sharpe"] = median_neighbor_sharpe
            aggregate["plateau_neighbor_ratio"] = ratio
            aggregate["gate_plateau_nonnegative"] = median_neighbor_sharpe >= 0.0
            aggregate["gate_plateau_ratio"] = ratio >= 0.7
            aggregate["neighbor_ids"] = [neighbor["candidate_id"] for neighbor in neighbors]


def _candidate_sort_key(aggregate: dict[str, Any]) -> tuple[Any, ...]:
    hard_gate = all(
        (
            aggregate["gate_fold_count"],
            aggregate["gate_deflated_sharpe"],
            aggregate["gate_bootstrap"],
            aggregate["gate_negative_controls"],
        )
    )
    strategy_id = aggregate.get("params", {}).get("strategy_id", "baseline")
    return (
        1 if hard_gate else 0,
        aggregate["median_validation_sharpe"],
        -aggregate["universe_sensitivity_std"],
        aggregate["plateau_neighbor_median_sharpe"],
        aggregate["plateau_neighbor_ratio"],
        -aggregate["max_drawdown"],
        -aggregate["params"]["l"],
        -aggregate["params"]["skip"],
        -aggregate["params"]["top_n"],
        strategy_id,
    )


def _attach_ranks(aggregates: list[dict[str, Any]]) -> None:
    ranked = sorted(aggregates, key=_candidate_sort_key, reverse=True)
    for rank, aggregate in enumerate(ranked, start=1):
        aggregate["rank"] = rank
    if ranked:
        selected = ranked[0]
        selected["selected"] = all(
            (
                selected["gate_fold_count"],
                selected["gate_deflated_sharpe"],
                selected["gate_bootstrap"],
                selected["gate_negative_controls"],
            )
        )
    for aggregate in aggregates:
        aggregate.setdefault("selected", False)


def selection_summary(aggregates: list[dict[str, Any]], negative_controls: dict[str, Any], mode: str) -> dict[str, Any]:
    ranked = sorted(aggregates, key=lambda item: item["rank"])
    selected = next((item for item in ranked if item["selected"]), None)
    summary = {
        "mode": mode,
        "selection_status": "selected" if selected is not None else "no_candidate_passed_hard_gates",
        "locked_candidate": None,
        "ranked_candidates": [],
        "negative_controls": negative_controls,
        "neighbor_diagnostics": [],
    }
    for aggregate in ranked:
        candidate_summary = {
            "candidate_id": aggregate["candidate_id"],
            "params": aggregate["params"],
            "rank": aggregate["rank"],
            "fold_pass_count": aggregate["fold_pass_count"],
            "median_validation_sharpe": aggregate["median_validation_sharpe"],
            "deflated_sharpe_score": aggregate["deflated_sharpe_score"],
            "deflated_sharpe_probability": aggregate["deflated_sharpe_probability"],
            "bootstrap_ci_low": aggregate["bootstrap_ci_low"],
            "bootstrap_ci_high": aggregate["bootstrap_ci_high"],
            "universe_sensitivity_std": aggregate["universe_sensitivity_std"],
            "plateau_neighbor_median_sharpe": aggregate["plateau_neighbor_median_sharpe"],
            "plateau_neighbor_ratio": aggregate["plateau_neighbor_ratio"],
            "max_drawdown": aggregate["max_drawdown"],
            "gate_fold_count": aggregate["gate_fold_count"],
            "gate_deflated_sharpe": aggregate["gate_deflated_sharpe"],
            "gate_bootstrap": aggregate["gate_bootstrap"],
            "gate_negative_controls": aggregate["gate_negative_controls"],
            "selected": aggregate["selected"],
        }
        summary["ranked_candidates"].append(candidate_summary)

    if selected is not None:
        summary["locked_candidate"] = summary["ranked_candidates"][0]
        for neighbor_id in selected["neighbor_ids"]:
            neighbor = next(item for item in ranked if item["candidate_id"] == neighbor_id)
            failure_reasons = []
            if not neighbor["gate_fold_count"]:
                failure_reasons.append("fewer than 4 of 5 folds cleared Sharpe > 0.4")
            if not neighbor["gate_deflated_sharpe"]:
                failure_reasons.append("deflated Sharpe score <= 0")
            if not neighbor["gate_bootstrap"]:
                failure_reasons.append("bootstrap CI lower bound <= 0")
            if not neighbor["gate_negative_controls"]:
                failure_reasons.append("negative-control pass rate exceeded threshold")
            if not failure_reasons:
                failure_reasons.append("lower rank than the locked candidate on median Sharpe, robustness, or drawdown")
            summary["neighbor_diagnostics"].append(
                {
                    "candidate_id": neighbor["candidate_id"],
                    "params": neighbor["params"],
                    "reasons": failure_reasons,
                }
            )
    return summary


def csv_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aggregate in aggregates:
        for evaluation in aggregate["evaluations"]:
            rows.append(
                {
                    "candidate_id": aggregate["candidate_id"],
                    "l": aggregate["params"]["l"],
                    "skip": aggregate["params"]["skip"],
                    "top_n": aggregate["params"]["top_n"],
                    "fold_id": evaluation["fold_id"],
                    "universe_variant": evaluation["universe_variant"],
                    "execution_model": evaluation["execution_model"],
                    "fx_scenario": evaluation["fx_scenario"],
                    "cost_model_name": evaluation["cost_model_name"],
                    "validation_sharpe": annualized_sharpe(evaluation["monthly_returns"]),
                    "validation_max_drawdown": max_drawdown(evaluation["monthly_returns"]),
                    "candidate_fold_pass_count": aggregate["fold_pass_count"],
                    "candidate_median_validation_sharpe": aggregate["median_validation_sharpe"],
                    "candidate_deflated_sharpe_score": aggregate["deflated_sharpe_score"],
                    "candidate_deflated_sharpe_probability": aggregate["deflated_sharpe_probability"],
                    "candidate_bootstrap_ci_low": aggregate["bootstrap_ci_low"],
                    "candidate_bootstrap_ci_high": aggregate["bootstrap_ci_high"],
                    "candidate_universe_sensitivity_std": aggregate["universe_sensitivity_std"],
                    "candidate_plateau_neighbor_median_sharpe": aggregate["plateau_neighbor_median_sharpe"],
                    "candidate_plateau_neighbor_ratio": aggregate["plateau_neighbor_ratio"],
                    "gate_fold_count": aggregate["gate_fold_count"],
                    "gate_deflated_sharpe": aggregate["gate_deflated_sharpe"],
                    "gate_bootstrap": aggregate["gate_bootstrap"],
                    "gate_negative_controls": aggregate["gate_negative_controls"],
                    "rank": aggregate["rank"],
                    "selected": aggregate["selected"],
                }
            )
    return rows


def load_locked_candidate(selection_summary_path: Path) -> dict[str, Any]:
    summary = load_json(selection_summary_path)
    if summary.get("selection_status") != "selected" or summary.get("locked_candidate") is None:
        raise ValueError("No locked candidate is available for the untouched holdout.")
    return summary["locked_candidate"]


def evaluate_holdout_candidate(
    candidate: dict[str, Any],
    holdout_start: str | None = None,
    holdout_end: str | None = None,
) -> dict[str, Any]:
    holdout_start = holdout_start or config.OOS_START
    holdout_end = holdout_end or config.OOS_END
    evaluations = candidate["evaluations"]
    results: dict[str, Any] = {}
    for variant in REQUIRED_UNIVERSE_VARIANTS:
        results[variant] = {}
        for execution_model in REQUIRED_EXECUTION_MODELS:
            results[variant][execution_model] = {}
            for fx_scenario in REQUIRED_FX_SCENARIOS:
                rows = _filter_evaluations(
                    evaluations,
                    universe_variant=variant,
                    execution_model=execution_model,
                    fx_scenario=fx_scenario,
                )
                if len(rows) != 1:
                    raise ValueError(
                        f"Holdout panel must contain exactly one row for {variant} / {execution_model} / {fx_scenario}."
                    )
                row = rows[0]
                if row["window_start"] != holdout_start or row["window_end"] != holdout_end:
                    raise ValueError("Holdout panel window does not match the untouched holdout.")
                strategy_returns = row["monthly_returns"]
                primary_benchmark_returns = row.get("primary_benchmark_returns")
                secondary_benchmark_returns = row.get("secondary_benchmark_returns")
                tertiary_benchmark_returns = row.get("tertiary_benchmark_returns")
                result_row = {
                    "net_sharpe": annualized_sharpe(strategy_returns),
                    "max_drawdown": max_drawdown(strategy_returns),
                    "total_return": total_return(strategy_returns),
                    "months": len(strategy_returns),
                }
                if primary_benchmark_returns is not None:
                    result_row["primary_benchmark_total_return"] = total_return(primary_benchmark_returns)
                    result_row["beats_primary_benchmark"] = result_row["total_return"] > result_row["primary_benchmark_total_return"]
                if secondary_benchmark_returns is not None:
                    result_row["secondary_benchmark_total_return"] = total_return(secondary_benchmark_returns)
                if tertiary_benchmark_returns is not None:
                    result_row["tertiary_benchmark_total_return"] = total_return(tertiary_benchmark_returns)
                results[variant][execution_model][fx_scenario] = result_row
    base_main = results["Full Nordics"]["next_open"]["base"]
    phase4_eligible = bool(
        base_main.get("beats_primary_benchmark")
        and base_main["net_sharpe"] >= config.OOS_SHARPE_MIN
    )
    return {
        "selected_params": candidate["params"],
        "holdout_window": {"start": holdout_start, "end": holdout_end},
        "results": results,
        "phase4_gate": {
            "base_main_net_sharpe": base_main["net_sharpe"],
            "meets_sharpe_gate": base_main["net_sharpe"] >= config.OOS_SHARPE_MIN,
            "beats_primary_benchmark": base_main.get("beats_primary_benchmark", False),
            "phase4_eligible": phase4_eligible,
        },
    }

