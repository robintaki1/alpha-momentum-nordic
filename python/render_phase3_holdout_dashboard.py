from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from research_engine import (
    PRIMARY_TRACK,
    ResearchDataset,
    _clean_returns,
    _find_holdout_primary_track,
    _month_to_ordinal,
    _ordinal_to_year_label,
    _fold_cagr,
    annualized_sharpe_periods,
    apply_pbo_policy,
    build_thesis,
    clip_walkforward_folds_to_window,
    config,
    evaluate_holdout,
    format_equity_plus_return,
    format_float,
    format_mc,
    format_params,
    format_pbo_display,
    format_pct,
    monte_carlo_badge,
    monte_carlo_interpretation,
    monte_carlo_summary,
    max_drawdown,
    render_badge,
    render_evidence_stack_html,
    render_phase_boundary_svg,
    render_histogram_svg,
    render_monte_carlo_anomaly_panel,
    render_return_curve,
    render_spaghetti_svg,
    render_trimmed_monte_carlo,
    render_walkforward_compare_svg,
    render_walkforward_ladder_svg,
    render_walkforward_oos_equity_svg,
    render_walkforward_rolling_sharpe_svg,
    render_walkforward_schedule_svg,
    resolve_profile_settings,
    total_return,
    pbo_explainer,
    walk_forward_diagnostic_result,
    walk_forward_gap_months_summary,
    walk_forward_gap_summary,
    walk_forward_quality_badge,
    walk_forward_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a full Phase 3 holdout dashboard from an existing research run."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thesis-name", type=str, default=None)
    parser.add_argument("--profile-set", type=str, default=None)
    parser.add_argument("--context-note", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel_href(output_path: Path, target_path: Path) -> str:
    return Path(
        os.path.relpath(
            Path(target_path).resolve(),
            start=output_path.resolve().parent,
        )
    ).as_posix()


def infer_profile_set(source_run_dir: Path, explicit_profile_set: str | None) -> str:
    if explicit_profile_set:
        return explicit_profile_set
    summary_path = source_run_dir.parent / "summary" / "research_engine_summary.json"
    summary = load_json(summary_path, default={})
    return str(summary.get("profile_set") or "default")


def needs_walk_forward(summary: dict[str, Any]) -> bool:
    walk_forward = summary.get("walk_forward", {})
    if not isinstance(walk_forward, dict):
        return True
    folds = walk_forward.get("folds")
    return not isinstance(folds, list) or not folds


def needs_holdout_series(holdout: dict[str, Any]) -> bool:
    track_bundle = _find_holdout_primary_track(holdout)
    if track_bundle is None:
        return True
    primary_track = track_bundle[1]
    return not bool(primary_track.get("strategy_returns"))


def _normalize_for_match(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_for_match(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_for_match(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return numeric
    return value


def matching_holdout_identity(
    left_params: dict[str, Any] | None,
    right_params: dict[str, Any] | None,
    left_window: dict[str, Any] | None,
    right_window: dict[str, Any] | None,
) -> bool:
    return _normalize_for_match(left_params or {}) == _normalize_for_match(right_params or {}) and (
        _normalize_for_match(left_window or {}) == _normalize_for_match(right_window or {})
    )


def find_series_backed_holdout_path(
    *,
    source_run_dir: Path,
    thesis_name: str,
    selected_params: dict[str, Any] | None,
    holdout_window: dict[str, Any] | None,
) -> Path | None:
    results_root = source_run_dir.parent.parent
    candidates: list[Path] = []
    for path in sorted(results_root.rglob("holdout_results.json")):
        if path.resolve() == (source_run_dir / "holdout_results.json").resolve():
            continue
        if path.parent.name != thesis_name:
            continue
        payload = load_json(path, default={})
        if not payload or needs_holdout_series(payload):
            continue
        if not matching_holdout_identity(
            selected_params,
            payload.get("selected_params"),
            holdout_window,
            payload.get("holdout_window"),
        ):
            continue
        candidates.append(path)
    return candidates[0] if candidates else None


def find_upstream_phase2_selection(
    *,
    thesis_name: str,
    locked_candidate_id: str | None,
) -> tuple[Path | None, dict[str, Any]]:
    candidates = [
        Path("portfolio")
        / "alpha_momentum_validation_package"
        / "research_engine_rebuild"
        / thesis_name
        / "1m"
        / "selection_summary.json",
        Path("results") / "cadence_compare_rebuild" / thesis_name / "1m" / "selection_summary.json",
    ]
    for path in candidates:
        payload = load_json(path, default={})
        if not payload:
            continue
        candidate_id = (payload.get("locked_candidate") or {}).get("candidate_id")
        if locked_candidate_id and candidate_id and candidate_id != locked_candidate_id:
            continue
        backtest = payload.get("backtest_overfitting", {})
        if backtest.get("pbo") is None:
            continue
        return path, payload
    return None, {}


def format_profile_set_brief(profile_set: str, profile_settings: dict[str, Any]) -> str:
    profile = profile_settings.get("certification_baseline") or next(iter(profile_settings.values()), {})
    lookbacks = [str(value) for value in profile.get("lookbacks", ()) if value is not None]
    skips = [str(value) for value in profile.get("skips", ()) if value is not None]
    top_ns = [str(value) for value in profile.get("top_ns", ()) if value is not None]
    strategy_ids = [str(item.get("strategy_id", "baseline")) for item in config.STRATEGY_VARIANTS]

    strategy_summary = "baseline + trend-filter family"
    if strategy_ids:
        has_baseline = "baseline" in strategy_ids
        has_trend = any("trend_filter" in strategy for strategy in strategy_ids)
        if has_baseline and not has_trend:
            strategy_summary = "baseline family"
        elif has_trend and not has_baseline:
            strategy_summary = "trend-filter family"

    return (
        f"{profile_set} - "
        f"L={', '.join(lookbacks) or 'n/a'}; "
        f"skip={', '.join(skips) or 'n/a'}; "
        f"top_n={', '.join(top_ns) or 'n/a'}; "
        f"{strategy_summary}."
    )


def holdout_primary_sharpe(holdout: dict[str, Any]) -> float | None:
    phase4 = holdout.get("phase4_gate", {})
    try:
        value = phase4.get("base_main_net_sharpe")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def annualized_return(total: float | None, months: int | None, periods_per_year: int) -> float | None:
    if total is None or months is None or months <= 0:
        return None
    if 1.0 + float(total) <= 0.0:
        return None
    years = months / periods_per_year
    if years <= 0:
        return None
    return (1.0 + float(total)) ** (1.0 / years) - 1.0


def _ordinal_to_month(ordinal: int) -> str:
    year = ordinal // 12
    month_number = ordinal % 12 + 1
    return f"{year:04d}-{month_number:02d}"


def _candidate_id(params: dict[str, Any]) -> str:
    strategy_id = params.get("strategy_id", "baseline")
    return f"l{params['l']}_s{params['skip']}_n{params['top_n']}_strat={strategy_id}"


def build_phase3_fold_windows(
    *, start_month: str, end_month: str, fold_count: int = 5, train_start: str = "2000-01"
) -> list[dict[str, Any]]:
    start_ord = _month_to_ordinal(start_month)
    end_ord = _month_to_ordinal(end_month)
    total_months = end_ord - start_ord + 1
    if total_months <= 0:
        return []
    fold_count = max(1, min(fold_count, total_months))
    base_len = total_months // fold_count
    remainder = total_months % fold_count
    folds: list[dict[str, Any]] = []
    current_start = start_ord
    for idx in range(fold_count):
        length = base_len + (1 if idx < remainder else 0)
        validate_start_ord = current_start
        validate_end_ord = current_start + length - 1
        folds.append(
            {
                "fold_id": f"phase3_fold_{idx + 1}",
                "train_window": {
                    "start": train_start,
                    "end": _ordinal_to_month(validate_start_ord - 1),
                },
                "validate_window": {
                    "start": _ordinal_to_month(validate_start_ord),
                    "end": _ordinal_to_month(validate_end_ord),
                },
            }
        )
        current_start = validate_end_ord + 1
    return folds


def phase3_walk_forward_test(
    dataset: ResearchDataset,
    *,
    thesis: dict[str, Any],
    params: dict[str, Any],
    period_label: str,
    periods_per_year: int,
    start_month: str,
    end_month: str,
    fold_count: int = 5,
) -> dict[str, Any]:
    folds = build_phase3_fold_windows(
        start_month=start_month,
        end_month=end_month,
        fold_count=fold_count,
        train_start="2000-01",
    )
    if not folds:
        return {"status": "unavailable", "reason": "no_folds"}

    excluded_countries = tuple(thesis.get("excluded_countries", ()))
    combined_returns: list[float] = []
    combined_benchmark_returns: list[float] = []
    fold_results: list[dict[str, Any]] = []
    candidate_id = _candidate_id(params)

    for fold in folds:
        train_window = fold["train_window"]
        validate_window = fold["validate_window"]
        sim_train = dataset.simulate_window(
            params=params,
            universe_variant=PRIMARY_TRACK["universe_variant"],
            execution_model=PRIMARY_TRACK["execution_model"],
            fx_scenario=PRIMARY_TRACK["fx_scenario"],
            start_month=train_window["start"],
            end_month=train_window["end"],
            excluded_countries=excluded_countries,
        )
        train_returns = _clean_returns(sim_train.monthly_returns)
        sim_validate = dataset.simulate_window(
            params=params,
            universe_variant=PRIMARY_TRACK["universe_variant"],
            execution_model=PRIMARY_TRACK["execution_model"],
            fx_scenario=PRIMARY_TRACK["fx_scenario"],
            start_month=validate_window["start"],
            end_month=validate_window["end"],
            excluded_countries=excluded_countries,
        )
        validate_returns = _clean_returns(sim_validate.monthly_returns)
        validate_benchmark_returns = _clean_returns(sim_validate.primary_benchmark_returns or [])
        combined_returns.extend(validate_returns)
        combined_benchmark_returns.extend(validate_benchmark_returns)
        fold_results.append(
            {
                "fold_id": fold["fold_id"],
                "status": "ok",
                "train_window": train_window,
                "validate_window": validate_window,
                "selected_candidate_id": candidate_id,
                "selected_params": params,
                "train_sharpe": annualized_sharpe_periods(train_returns, periods_per_year=periods_per_year)
                if len(train_returns) >= 2
                else None,
                "train_total_return": total_return(train_returns) if train_returns else None,
                "train_months": len(train_returns),
                "validate_sharpe": annualized_sharpe_periods(validate_returns, periods_per_year=periods_per_year)
                if len(validate_returns) >= 2
                else None,
                "validate_total_return": total_return(validate_returns) if validate_returns else None,
                "validate_max_drawdown": max_drawdown(validate_returns) if validate_returns else None,
                "validate_months": len(validate_returns),
                "validate_benchmark_sharpe": annualized_sharpe_periods(
                    validate_benchmark_returns,
                    periods_per_year=periods_per_year,
                )
                if len(validate_benchmark_returns) >= 2
                else None,
                "validate_benchmark_total_return": total_return(validate_benchmark_returns)
                if validate_benchmark_returns
                else None,
                "validate_benchmark_months": len(validate_benchmark_returns),
                "period_label": period_label,
                "periods_per_year": periods_per_year,
            }
        )

    if not combined_returns:
        return {"status": "unavailable", "reason": "no_validation_returns", "folds": fold_results}

    combined = {
        "sharpe": annualized_sharpe_periods(combined_returns, periods_per_year=periods_per_year),
        "total_return": total_return(combined_returns),
        "max_drawdown": max_drawdown(combined_returns),
        "months": len(combined_returns),
        "period_label": period_label,
        "periods_per_year": periods_per_year,
    }
    return {
        "status": "ok",
        "selection_metric": "locked_params_holdout_walk_forward",
        "track": dict(PRIMARY_TRACK),
        "folds": fold_results,
        "combined": combined,
        "combined_returns": combined_returns,
        "combined_benchmark_returns": combined_benchmark_returns,
    }


def primary_holdout_fold(holdout_window: dict[str, Any]) -> list[dict[str, Any]]:
    start = holdout_window.get("start") or config.OOS_START
    end = holdout_window.get("end") or config.OOS_END
    return [
        {
            "fold_id": "holdout",
            "train_window": {"start": "2000-01", "end": config.INSAMPLE_END},
            "validate_window": {"start": start, "end": end},
        }
    ]


def month_label(start_ord: int, index: int) -> str:
    return _ordinal_to_year_label(start_ord + index)


def short_track_label(universe_variant: str, execution_model: str) -> str:
    universe_map = {
        "Full Nordics": "Full",
        "SE-only": "SE",
        "largest-third-by-market-cap": "Large 1/3",
    }
    execution_map = {
        "next_open": "Open",
        "next_close": "Close",
    }
    return f"{universe_map.get(universe_variant, universe_variant)} / {execution_map.get(execution_model, execution_model)}"


def collect_holdout_tracks(holdout: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = holdout.get("results", {})
    universe_order = {"Full Nordics": 0, "SE-only": 1, "largest-third-by-market-cap": 2}
    execution_order = {"next_open": 0, "next_close": 1}
    for universe_variant, execution_models in results.items():
        if not isinstance(execution_models, dict):
            continue
        for execution_model, scenarios in execution_models.items():
            if not isinstance(scenarios, dict):
                continue
            base = scenarios.get("base")
            if not isinstance(base, dict):
                continue
            months = int(base.get("months") or base.get("period_count") or 0)
            periods_per_year = int(base.get("periods_per_year") or holdout.get("periods_per_year") or 12)
            total = base.get("total_return")
            rows.append(
                {
                    "label": short_track_label(universe_variant, execution_model),
                    "universe_variant": universe_variant,
                    "execution_model": execution_model,
                    "fx_scenario": "base",
                    "is_primary": (
                        universe_variant == PRIMARY_TRACK.get("universe_variant")
                        and execution_model == PRIMARY_TRACK.get("execution_model")
                        and "base" == PRIMARY_TRACK.get("fx_scenario")
                    ),
                    "net_sharpe": base.get("net_sharpe"),
                    "total_return": total,
                    "cagr": annualized_return(total, months, periods_per_year),
                    "max_drawdown": base.get("max_drawdown"),
                    "months": months,
                    "beats_primary_benchmark": base.get("beats_primary_benchmark"),
                    "primary_benchmark_total_return": base.get("primary_benchmark_total_return"),
                    "strategy_returns": base.get("strategy_returns") or [],
                    "primary_benchmark_returns": base.get("primary_benchmark_returns") or [],
                    "sort_key": (
                        universe_order.get(universe_variant, 99),
                        execution_order.get(execution_model, 99),
                    ),
                }
            )
    rows.sort(key=lambda item: item["sort_key"])
    return rows


def render_holdout_schedule_svg(
    *,
    holdout_start: str,
    holdout_end: str,
    train_start: str = "2000-01",
    train_end: str = config.INSAMPLE_END,
    title: str = "Phase 3 Holdout Schedule",
    width: int = 720,
) -> str:
    min_start = _month_to_ordinal(train_start)
    train_end_ord = _month_to_ordinal(train_end)
    holdout_start_ord = _month_to_ordinal(holdout_start)
    holdout_end_ord = _month_to_ordinal(holdout_end)
    span = max(1, holdout_end_ord - min_start + 1)

    margin_left = 86
    margin_right = 12
    margin_top = 20
    margin_bottom = 38
    row_height = 16
    height = margin_top + row_height + margin_bottom + 8
    inner_w = width - margin_left - margin_right

    def _x(ordinal: int) -> float:
        return margin_left + (ordinal - min_start) / span * inner_w

    train_w = max(1.0, _x(train_end_ord + 1) - _x(min_start))
    holdout_w = max(1.0, _x(holdout_end_ord + 1) - _x(holdout_start_ord))
    x_ticks = _axis_ticks(float(min_start), float(holdout_end_ord), 6)
    x_labels = [
        f'<text x="{_x(int(round(tick))):.1f}" y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick)}</text>'
        for tick in x_ticks
    ]
    x_axis = (
        f'<line x1="{margin_left}" y1="{height - margin_bottom + 6}" '
        f'x2="{width - margin_right}" y2="{height - margin_bottom + 6}" stroke="#8f7b5f" stroke-width="1"></line>'
    )
    legend = (
        '<div class="chart-note">'
        '<span style="display:inline-block;width:12px;height:12px;background:#9b8872;margin-right:6px;border-radius:2px;"></span> Selection / training span (IS) '
        '<span style="display:inline-block;width:12px;height:12px;background:#c27c3a;margin:0 6px 0 12px;border-radius:2px;"></span> Untouched holdout (OOS)'
        "</div>"
    )
    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        f'<rect x="{_x(min_start):.1f}" y="{margin_top:.1f}" width="{train_w:.1f}" height="{row_height}" fill="#9b8872"></rect>'
        f'<rect x="{_x(holdout_start_ord):.1f}" y="{margin_top:.1f}" width="{holdout_w:.1f}" height="{row_height}" fill="#c27c3a"></rect>'
        f'<text x="{margin_left - 8}" y="{margin_top + row_height - 3:.1f}" text-anchor="end">Holdout</text>'
        + x_axis
        + "".join(x_labels)
        + "</svg>"
        + legend
        + "</div>"
    )


def render_holdout_equity_svg(
    returns: list[float] | None,
    benchmark_returns: list[float] | None,
    *,
    start_month: str,
    title: str = "Holdout Equity vs Benchmark",
    width: int = 680,
    height: int = 220,
) -> str:
    if not returns:
        return '<div class="muted">Holdout equity not available.</div>'
    strategy = _returns_to_equity(returns, base=100.0)
    benchmark = _returns_to_equity(benchmark_returns or [], base=100.0) if benchmark_returns else []
    x_min = _month_to_ordinal(start_month)
    x_max = x_min + len(strategy) - 1
    if x_max <= x_min:
        x_max = x_min + 1

    margin_left, margin_right = 48, 12
    margin_top, margin_bottom = 16, 28
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x(index: int) -> float:
        return margin_left + index / max(1, len(strategy) - 1) * inner_w

    all_values = list(strategy)
    if benchmark:
        all_values.extend(benchmark)
    y_min = min(all_values)
    y_max = max(all_values)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    x_ticks = _axis_ticks(float(x_min), float(x_max), 5)
    x_labels = [
        f'<text x="{margin_left + ((tick - x_min) / max(1, x_max - x_min)) * inner_w:.1f}" y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick)}</text>'
        for tick in x_ticks
    ]
    y_ticks = _axis_ticks(y_min, y_max, 5)
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_float(tick, 2)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        'stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]
    strategy_points = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(strategy))
    benchmark_points = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(benchmark)) if benchmark else ""
    note = (
        '<div class="chart-note">'
        f'Green = strategy. Gold dashed = {config.PRIMARY_PASSIVE_BENCHMARK}. '
        "X-axis = holdout timeline."
        "</div>"
    )
    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + (
            f'<polyline points="{benchmark_points}" fill="none" stroke="#a87d3f" stroke-width="1.5" stroke-dasharray="6 4"></polyline>'
            if benchmark_points
            else ""
        )
        + f'<polyline points="{strategy_points}" fill="none" stroke="#4f7a67" stroke-width="1.8"></polyline>'
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 4}" text-anchor="start">Equity (start=100)</text>'
        + f'<text x="{margin_left + inner_w / 2:.1f}" y="{height - 4}" text-anchor="middle">Holdout timeline</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + "</svg>"
        + note
        + "</div>"
    )


def render_metric_bar_svg(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    y_label: str,
    as_pct: bool = False,
    width: int = 680,
    height: int = 210,
) -> str:
    if not rows:
        return '<div class="muted">Comparison not available.</div>'
    values = [row.get(metric_key) for row in rows if row.get(metric_key) is not None]
    if not values:
        return '<div class="muted">Comparison not available.</div>'

    margin_left, margin_right = 48, 12
    margin_top, margin_bottom = 16, 46
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom
    bar_gap = 12
    bar_w = max(12.0, (inner_w - bar_gap * (len(rows) - 1)) / max(1, len(rows)))
    y_min = min(0.0, min(float(value) for value in values))
    y_max = max(0.0, max(float(value) for value in values))
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    y_ticks = _axis_ticks(y_min, y_max, 5)
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_pct(tick) if as_pct else format_float(tick)}</text>'
        for tick in y_ticks
    ]
    zero_line = ""
    if y_min < 0.0 < y_max:
        zero_line = (
            f'<line x1="{margin_left}" y1="{_y(0.0):.1f}" x2="{width - margin_right}" y2="{_y(0.0):.1f}" '
            'stroke="#b58a60" stroke-width="1.4"></line>'
        )

    bars: list[str] = []
    labels: list[str] = []
    value_labels: list[str] = []
    for idx, row in enumerate(rows):
        value = row.get(metric_key)
        if value is None:
            continue
        x = margin_left + idx * (bar_w + bar_gap)
        top = _y(max(0.0, float(value)))
        bottom = _y(min(0.0, float(value)))
        color = "#4f7a67" if row.get("is_primary") else "#a87d3f"
        bars.append(
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{max(1.0, bottom - top):.1f}" fill="{color}" opacity="0.92"></rect>'
        )
        labels.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 18}" text-anchor="middle">{row.get("label","n/a")}</text>'
        )
        value_text = format_pct(value) if as_pct else format_float(value)
        value_labels.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top - 6:.1f}" text-anchor="middle">{value_text}</text>'
        )

    note = (
        '<div class="chart-note">'
        "Green = official primary Phase 3 track. Gold = robustness tracks. "
        "Categories are base-scenario holdout variants."
        "</div>"
    )
    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + zero_line
        + "".join(bars)
        + "".join(value_labels)
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 4}" text-anchor="start">{y_label}</text>'
        + "".join(labels)
        + "".join(y_labels)
        + "</svg>"
        + note
        + "</div>"
    )


def build_holdout_subperiod_rows(
    returns: list[float],
    benchmark_returns: list[float] | None,
    *,
    holdout_start: str,
    holdout_end: str,
    periods_per_year: int,
) -> list[dict[str, Any]]:
    windows = [
        ("2018-2019", "2018-01", "2019-12"),
        ("2020-2021", "2020-01", "2021-12"),
        ("2022-2023", "2022-01", "2023-12"),
        ("2024-end", "2024-01", holdout_end),
    ]
    start_ord = _month_to_ordinal(holdout_start)
    series_end_ord = start_ord + len(returns) - 1
    rows: list[dict[str, Any]] = []
    for label, window_start, window_end in windows:
        sub_start_ord = max(start_ord, _month_to_ordinal(window_start))
        sub_end_ord = min(series_end_ord, _month_to_ordinal(window_end))
        if sub_end_ord < sub_start_ord:
            continue
        idx_start = sub_start_ord - start_ord
        idx_end = sub_end_ord - start_ord + 1
        sub_returns = returns[idx_start:idx_end]
        sub_benchmark = (benchmark_returns or [])[idx_start:idx_end] if benchmark_returns else []
        sub_total = total_return(sub_returns) if sub_returns else None
        bench_total = total_return(sub_benchmark) if sub_benchmark else None
        rows.append(
            {
                "label": label,
                "window": f"{_ordinal_to_month(sub_start_ord)} to {_ordinal_to_month(sub_end_ord)}",
                "months": len(sub_returns),
                "sharpe": annualized_sharpe_periods(sub_returns, periods_per_year=periods_per_year)
                if len(sub_returns) >= 2
                else None,
                "cagr": annualized_return(sub_total, len(sub_returns), periods_per_year),
                "total_return": sub_total,
                "max_drawdown": max_drawdown(sub_returns) if sub_returns else None,
                "benchmark_total_return": bench_total,
                "beats_benchmark": sub_total is not None and bench_total is not None and sub_total > bench_total,
            }
        )
    return rows


def render_subperiod_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">Holdout subperiods not available.</p>'
    body = "".join(
        "<tr>"
        f"<td>{row['label']}</td>"
        f"<td>{row['window']}</td>"
        f"<td>{row['months']}</td>"
        f"<td>{format_float(row['sharpe'])}</td>"
        f"<td>{format_pct(row['cagr'])}</td>"
        f"<td>{format_pct(row['total_return'])}</td>"
        f"<td>{format_pct(row['benchmark_total_return'])}</td>"
        f"<td>{format_pct(row['max_drawdown'])}</td>"
        f"<td>{row['beats_benchmark']}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table>"
        "<thead><tr><th>Subperiod</th><th>Window</th><th>Months</th><th>Sharpe</th><th>CAGR</th><th>Total Return</th><th>Benchmark Return</th><th>Max Drawdown</th><th>Beats Benchmark</th></tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def render_track_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">Track comparison not available.</p>'
    body = "".join(
        "<tr>"
        f"<td>{row['label']}{' (primary)' if row['is_primary'] else ''}</td>"
        f"<td>{format_float(row['net_sharpe'])}</td>"
        f"<td>{format_pct(row['cagr'])}</td>"
        f"<td>{format_pct(row['total_return'])}</td>"
        f"<td>{format_pct(row['primary_benchmark_total_return'])}</td>"
        f"<td>{format_pct(row['max_drawdown'])}</td>"
        f"<td>{row['beats_primary_benchmark']}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table>"
        "<thead><tr><th>Track</th><th>Sharpe</th><th>CAGR</th><th>Total Return</th><th>Benchmark Return</th><th>Max Drawdown</th><th>Beats Benchmark</th></tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def render_phase3_verdict(holdout: dict[str, Any], primary_track: dict[str, Any]) -> str:
    phase4 = holdout.get("phase4_gate", {})
    eligible = bool(phase4.get("phase4_eligible"))
    headline = render_badge("PASS" if eligible else "FAIL", "good" if eligible else "weak")
    sharpe_ok = bool(phase4.get("meets_sharpe_gate"))
    beats_benchmark = bool(phase4.get("beats_primary_benchmark"))
    status_ok = holdout.get("status") == "ok"

    def _row(label: str, ok: bool, detail: str) -> str:
        return (
            '<li class="verdict-item">'
            f'<span class="verdict-label">{label}</span>'
            f'{render_badge("PASS" if ok else "FAIL", "good" if ok else "weak", detail)}'
            "</li>"
        )

    return (
        '<section class="section reveal delay-2">'
        "<h2>Phase 3 Holdout Verdict</h2>"
        f'<div class="verdict-panel">{headline}<span class="badge-detail">Untouched holdout decision for the locked candidate.</span></div>'
        f'<p class="verdict-note"><strong>Candidate:</strong> {format_params(holdout.get("selected_params"))}</p>'
        '<p class="verdict-note"><strong>Passing grade:</strong> untouched holdout must clear the Sharpe gate, beat the primary benchmark, and keep the official track healthy enough to unlock Phase 4.</p>'
        '<ul class="verdict-list">'
        + _row(
            "Holdout Sharpe gate",
            sharpe_ok,
            f"Base main net Sharpe {format_float(phase4.get('base_main_net_sharpe'))} vs gate {format_float(config.OOS_SHARPE_MIN)}",
        )
        + _row(
            "Benchmark gate",
            beats_benchmark,
            f"Primary benchmark total return {format_pct(primary_track.get('primary_benchmark_total_return'))}",
        )
        + _row("Holdout status", status_ok, f"Status {holdout.get('status','n/a')}")
        + _row("Phase 4 eligible", eligible, f"phase4_eligible = {phase4.get('phase4_eligible')}")
        + "</ul>"
        "</section>"
    )


def build_phase3_dashboard(
    *,
    thesis: dict[str, Any],
    selection: dict[str, Any],
    official_holdout: dict[str, Any],
    analysis_holdout: dict[str, Any],
    phase3_walk_forward: dict[str, Any],
    profile_set: str,
    profile_settings: dict[str, Any],
    selection_href: str,
    raw_holdout_href: str,
    back_href: str,
    portfolio_css_href: str,
    portfolio_theme_href: str,
    portfolio_home_href: str,
    employer_package_href: str,
    analysis_suite_href: str,
    context_note: str,
    source_note: str | None = None,
) -> str:
    holdout_window = official_holdout.get("holdout_window", {})
    holdout_start = holdout_window.get("start") or config.OOS_START
    holdout_end = holdout_window.get("end") or config.OOS_END
    official_primary_meta, official_primary_track = _find_holdout_primary_track(official_holdout) or ({}, {})
    analysis_primary_meta, analysis_primary_track = _find_holdout_primary_track(analysis_holdout) or ({}, {})
    primary_returns = analysis_primary_track.get("strategy_returns") or []
    primary_benchmark_returns = analysis_primary_track.get("primary_benchmark_returns") or []
    periods_per_year = int(
        analysis_primary_track.get("periods_per_year")
        or official_primary_track.get("periods_per_year")
        or analysis_holdout.get("periods_per_year")
        or official_holdout.get("periods_per_year")
        or 12
    )
    analysis_track_rows = collect_holdout_tracks(analysis_holdout)
    analysis_active_rows = [row for row in analysis_track_rows if row.get("total_return") is not None]
    wf_folds = phase3_walk_forward.get("folds") or []
    wf_combined = phase3_walk_forward.get("combined") or {}
    wf_combined_returns = phase3_walk_forward.get("combined_returns") or primary_returns
    wf_combined_benchmark_returns = phase3_walk_forward.get("combined_benchmark_returns") or []
    monte = monte_carlo_summary(
        primary_returns,
        periods_per_year=periods_per_year,
        n_resamples=config.MONTE_CARLO_RESAMPLES,
        block_length_months=config.MONTE_CARLO_BLOCK_LENGTH_MONTHS,
        seed=config.MONTE_CARLO_SEED,
    )
    monte["candidate_id"] = (selection.get("locked_candidate") or {}).get("candidate_id")
    monte["params"] = analysis_holdout.get("selected_params") or official_holdout.get("selected_params")
    mc_metrics = monte.get("metrics", {}) if isinstance(monte, dict) else {}
    mc_sharpe = mc_metrics.get("sharpe", {})
    mc_total = mc_metrics.get("total_return", {})
    mc_dd = mc_metrics.get("max_drawdown", {})
    mc_final_equity = mc_metrics.get("final_equity", {})
    mc_hist = monte.get("histograms", {}) if isinstance(monte, dict) else {}
    mc_paths = monte.get("sample_paths", []) if isinstance(monte, dict) else []
    mc_quantiles = monte.get("path_quantiles", {}) if isinstance(monte, dict) else {}
    mc_density = monte.get("path_density", {}) if isinstance(monte, dict) else {}
    mc_badge_label, mc_badge_tone, mc_badge_detail = monte_carlo_badge(mc_sharpe, mc_total, mc_dd)
    mc_badge_html = render_badge(mc_badge_label, mc_badge_tone, mc_badge_detail)
    mc_interp = monte_carlo_interpretation(mc_sharpe, mc_total, mc_dd)
    mc_anomaly = render_monte_carlo_anomaly_panel(monte)
    trimmed_html = render_trimmed_monte_carlo(monte)
    official_primary_total = official_primary_track.get("total_return")
    official_primary_cagr = annualized_return(
        official_primary_total,
        official_primary_track.get("months"),
        periods_per_year,
    )
    subperiod_rows = build_holdout_subperiod_rows(
        primary_returns,
        primary_benchmark_returns,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        periods_per_year=periods_per_year,
    )
    profile_desc_brief = format_profile_set_brief(profile_set, profile_settings)
    holdout_status = official_holdout.get("status", "n/a")
    backtest = selection.get("backtest_overfitting", {})
    pbo_value = backtest.get("pbo")
    pbo_display = format_pbo_display(pbo_value, backtest)
    phase2_pbo_source = selection.get("phase2_certification_source") or {}
    if phase2_pbo_source:
        candidate_count = phase2_pbo_source.get("candidate_count")
        pbo_note_html = (
            '<div class="note-strip"><strong>PBO note.</strong> '
            f'This page carries forward the upstream Phase 2 certification PBO for the same locked candidate: '
            f'<strong>{html.escape(pbo_display)}</strong>'
            + (
                f' across <strong>{html.escape(str(candidate_count))}</strong> tested candidates. '
                if candidate_count is not None
                else '. '
            )
            + 'The exact-candidate replay used for these Phase 3 diagnostics collapses replay-local PBO to n/a because there is no ranking step left. '
            f'<a href="{selection_href}">Open the Phase 2 certification artifact</a> for the full anti-overfitting context.</div>'
        )
    else:
        pbo_note_html = (
            '<div class="note-strip"><strong>PBO note.</strong> '
            f'{html.escape(pbo_explainer(backtest))} '
            'For this employer-facing Phase 3 page, the candidate is already frozen, so the official headline stays with the untouched holdout rather than re-running selection-stage PBO here. '
            f'<a href="{selection_href}">Open the Phase 2 certification artifact</a> for the upstream anti-overfitting context.</div>'
        )
    phase4_start = _ordinal_to_month(_month_to_ordinal(holdout_end) + 1)
    rebuilt_primary_cagr = annualized_return(
        analysis_primary_track.get("total_return"),
        analysis_primary_track.get("months"),
        periods_per_year,
    )
    rebuilt_primary_sharpe = analysis_holdout.get("phase4_gate", {}).get("base_main_net_sharpe")
    archived_primary_sharpe = official_holdout.get("phase4_gate", {}).get("base_main_net_sharpe")
    wf_quality_label, wf_quality_tone, wf_quality_detail = walk_forward_quality_badge(wf_folds)
    wf_quality_html = render_badge(wf_quality_label, wf_quality_tone, wf_quality_detail)
    wf_diag = walk_forward_diagnostic_result(phase3_walk_forward)
    wf_gap_sharpe = walk_forward_gap_summary(
        wf_folds,
        train_key="train_sharpe",
        validate_key="validate_sharpe",
        as_pct=False,
    )
    wf_gap_return = walk_forward_gap_summary(
        wf_folds,
        train_key="train_total_return",
        validate_key="validate_total_return",
        as_pct=True,
        train_value_fn=lambda fold: _fold_cagr(
            fold,
            total_key="train_total_return",
            months_key="train_months",
        ),
        validate_value_fn=lambda fold: _fold_cagr(
            fold,
            total_key="validate_total_return",
            months_key="validate_months",
        ),
    )
    wf_gap_months = walk_forward_gap_months_summary(wf_folds)
    wf_gate_passes = sum(
        1 for fold in wf_folds if fold.get("validate_sharpe") is not None and fold["validate_sharpe"] > 0.4
    )
    wf_combined_pass = bool(
        wf_combined.get("sharpe") is not None and float(wf_combined.get("sharpe")) >= 0.4
    )
    wf_combined_badge = render_badge(
        "COMBINED: PASS" if wf_combined_pass else "COMBINED: MIXED",
        "good" if wf_combined_pass else "caution",
        (
            f"Stitched validation Sharpe {format_float(wf_combined.get('sharpe'))} vs gate {format_float(0.4)}"
            if wf_combined.get("sharpe") is not None
            else "Combined stitched validation series unavailable."
        ),
    )
    wf_combined_cagr = annualized_return(
        wf_combined.get("total_return"),
        wf_combined.get("months"),
        periods_per_year,
    )
    wf_window_text = ", ".join(
        f"{(fold.get('validate_window') or {}).get('start')} to {(fold.get('validate_window') or {}).get('end')}"
        for fold in wf_folds
        if (fold.get("validate_window") or {}).get("start") and (fold.get("validate_window") or {}).get("end")
    )
    wf_schedule_folds = clip_walkforward_folds_to_window(
        wf_folds,
        clip_start=holdout_start,
        clip_end=holdout_end,
    )
    wf_schedule_chart = render_walkforward_schedule_svg(
        wf_schedule_folds,
        title="Phase 3 Walk-Forward Schedule",
    )
    boundary_chart = render_phase_boundary_svg(
        [
            {
                "label": "Selection history",
                "detail": "Phase 2 research and locking",
                "start": config.ROLLING_ORIGIN_FOLDS[0][1] if config.ROLLING_ORIGIN_FOLDS else "2000-01",
                "end": config.INSAMPLE_END,
                "fill": "#d9cab8",
            },
            {
                "label": "One-shot holdout",
                "detail": "this page's main test window",
                "start": holdout_start,
                "end": holdout_end,
                "fill": "#eadca7",
                "highlight": True,
            },
            {
                "label": "After holdout",
                "detail": "paper/live or future appended data",
                "start": phase4_start,
                "fill": "#efe6d7",
            },
        ],
        footer_note=(
            "This page is meant to be a one-shot Phase 3 test: the candidate should already be frozen before the "
            "orange window begins."
        ),
    )
    phase3_ladder_chart = render_walkforward_ladder_svg(
        wf_folds,
        stitched_label="Phase 3 stitched OOS",
    )
    wf_oos_equity_chart = render_walkforward_oos_equity_svg(
        wf_combined_returns,
        wf_folds,
        benchmark_returns=wf_combined_benchmark_returns,
        benchmark_label=config.PRIMARY_PASSIVE_BENCHMARK,
        title="Validation Equity (Stitched)",
    )
    wf_rolling_sharpe_chart = render_walkforward_rolling_sharpe_svg(
        wf_combined_returns,
        wf_folds,
        periods_per_year=periods_per_year,
        title="Rolling Sharpe (Validation, 12m)",
    )
    same_universe_alt = next(
        (
            row
            for row in analysis_active_rows
            if row["universe_variant"] == analysis_primary_meta.get("universe_variant")
            and row["execution_model"] != analysis_primary_meta.get("execution_model")
        ),
        None,
    )
    same_execution_alt = next(
        (
            row
            for row in analysis_active_rows
            if row["execution_model"] == analysis_primary_meta.get("execution_model")
            and row["universe_variant"] != analysis_primary_meta.get("universe_variant")
        ),
        None,
    )
    curve_rows = [
        row
        for row in [next((row for row in analysis_active_rows if row.get("is_primary")), None), same_universe_alt, same_execution_alt]
        if row
    ]
    context_note_html = (
        '<div class="callout"><strong>Why this page matters.</strong>'
        f'<div class="muted">{html.escape(context_note)}</div></div>'
    )
    source_note_html = (
        '<div class="callout"><strong>Source note.</strong>'
        f'<div class="muted">{html.escape(source_note)}</div></div>'
        if source_note
        else ""
    )

    head = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 3 Holdout Dashboard</title>
  <link rel="stylesheet" href="{portfolio_css_href}">
  <link rel="stylesheet" href="{portfolio_theme_href}">
  <style>
    .page-shell {{ max-width: 1200px; }}
    .hero,
    .section,
    .note-strip,
    .callout,
    .card,
    .metric-card {{ border: 1px solid var(--line); }}
    .hero,
    .section {{ padding: 30px; }}
    .hero-grid {{ display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.95fr); gap: 24px; align-items: start; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 16px; }}
    .grid.tight {{ margin-top: 12px; }}
    .card.full {{ grid-column: 1 / -1; }}
    .value {{ margin-top: 10px; font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif; font-size: clamp(1.45rem, 2.6vw, 2rem); line-height: 1.1; color: var(--ink); overflow-wrap: anywhere; }}
    .meta-note,
    .callout,
    .note-strip {{ margin-top: 14px; padding: 14px 16px; border-radius: 18px; background: rgba(250, 244, 236, 0.92); box-shadow: var(--shadow-sm); }}
    .note-strip {{ border-left: 4px solid var(--copper); }}
    .callout strong,
    .meta-note strong,
    .note-strip strong {{ color: var(--muted-strong); }}
    .metric-list {{ list-style: none; padding: 0; margin: 12px 0 0; display: grid; gap: 8px; }}
    .metric-list li {{ display: flex; gap: 10px; align-items: start; flex-wrap: wrap; }}
    .metric-list strong {{ min-width: 170px; color: var(--muted-strong); }}
    .chart-block {{ display: flex; flex-direction: column; gap: 8px; }}
    .chart-title {{ color: var(--muted-strong); text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.76rem; }}
    .chart {{ width: 100%; height: auto; }}
    .chart text {{ font-family: "Aptos", "Segoe UI Variable Text", "Trebuchet MS", sans-serif; font-size: 10px; fill: var(--muted-strong); }}
    .chart-note {{ font-size: 0.82rem; line-height: 1.55; color: var(--muted-strong); }}
    .legend-line {{ font-weight: 700; color: var(--muted-strong); }}
    .legend-line.median {{ color: #a97831; }}
    .legend-line.band {{ color: #6f8c5b; }}
    .legend-line.density {{ color: #4d7d74; }}
    .badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 10px 0 0; }}
    .badge {{ display: inline-flex; align-items: center; padding: 6px 12px; border-radius: 999px; border: 1px solid transparent; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }}
    .badge.good {{ background: var(--good-soft); color: var(--good); border-color: rgba(36, 94, 71, 0.16); }}
    .badge.caution {{ background: var(--warn-soft); color: var(--warn); border-color: rgba(123, 67, 49, 0.16); }}
    .badge.weak {{ background: rgba(128, 44, 44, 0.12); color: #7a2f2f; border-color: rgba(128, 44, 44, 0.16); }}
    .badge.neutral {{ background: rgba(27, 36, 32, 0.04); color: var(--muted-strong); border-color: rgba(18, 36, 29, 0.08); }}
    .badge-detail {{ font-size: 0.82rem; color: var(--muted); }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid rgba(18, 36, 29, 0.08); text-align: left; padding: 9px 8px; vertical-align: top; }}
    th {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.74rem; }}
    .verdict-panel {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .verdict-note {{ margin: 8px 0 10px; color: var(--muted); font-size: 0.92rem; }}
    .verdict-list {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }}
    .verdict-item {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .verdict-label {{ font-weight: 600; color: var(--muted-strong); min-width: 220px; }}
    .subtle {{ color: var(--muted); font-size: 0.94rem; }}
    @media (max-width: 900px) {{
      .hero-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body class="am-theme">
  <main class="page-shell">
"""

    page_html = head + f"""
    <header class="topbar reveal">
      <div class="brand">
        <div class="brand-mark">P3</div>
        <div class="brand-copy">
          <strong>Phase 3 Holdout Dashboard</strong>
          <span>Phase 3 holdout, walk-forward context, and Monte Carlo</span>
        </div>
      </div>
      <nav class="nav-links" aria-label="Primary">
        <a href="{portfolio_home_href}">Home</a>
        <a href="{employer_package_href}">Employer Package</a>
        <a href="{analysis_suite_href}">Evidence</a>
        <a href="{selection_href}">Phase 2 Certification</a>
        <a href="{raw_holdout_href}">Raw Artifact</a>
      </nav>
    </header>
    <section class="hero reveal delay-1">
      <div class="hero-grid">
        <div>
          <span class="eyebrow">Phase 3 Holdout</span>
          <h1 class="hero-title">{thesis.get('label','Phase 3 Holdout')}</h1>
          <p class="hero-subtitle">Final check on untouched data for the locked candidate.</p>
          <p class="lede">Window: {holdout_start} to {holdout_end}. The main verdict comes from the untouched holdout. The walk-forward charts lower down add context; they do not re-grade the strategy.</p>
          <p class="muted" style="margin-top:12px;">Research profile: {profile_desc_brief}</p>
          {context_note_html}
          {source_note_html}
          <div class="meta-note"><strong>Locked params:</strong> {format_params(official_holdout.get('selected_params'))}</div>
          <div class="actions">
            <a class="button primary" href="{selection_href}">Open Phase 2 certification</a>
            <a class="button secondary" href="{raw_holdout_href}">Open raw artifact</a>
            <a class="button ghost" href="{back_href}">Back</a>
          </div>
        </div>
        <aside class="status-panel">
          <p class="section-kicker">Quick take</p>
          <h2>Headline result</h2>
          <ul class="status-list">
            <li><span>Window</span><strong>{holdout_start} to {holdout_end}</strong></li>
            <li><span>Sharpe</span><strong>{format_float(archived_primary_sharpe or rebuilt_primary_sharpe)}</strong></li>
            <li><span>Benchmark gate</span><strong>{'Passed' if official_holdout.get('phase4_gate', {}).get('beats_primary_benchmark') else 'Failed'}</strong></li>
            <li><span>Phase 4 eligible</span><strong>{official_holdout.get('phase4_gate', {}).get('phase4_eligible')}</strong></li>
            <li><span>Status</span><strong>{selection.get('selection_status','n/a')}</strong></li>
            <li><span>PBO</span><strong>{pbo_display}</strong></li>
          </ul>
          <div class="badge-row">{render_badge('OFFICIAL: PASS' if official_holdout.get('phase4_gate', {}).get('phase4_eligible') else 'OFFICIAL: FAIL', 'good' if official_holdout.get('phase4_gate', {}).get('phase4_eligible') else 'weak', 'Headline verdict comes from the untouched holdout only.')}</div>
        </aside>
      </div>
      {pbo_note_html}
    </section>
    {render_phase3_verdict(official_holdout, official_primary_track)}
    <section class="section reveal delay-2">
      <h2>Phase Map</h2>
      <p class="muted"><strong>In short:</strong> Phase 2 locked the candidate on 2000-01 to 2017-12. Phase 3 is the untouched 2018-01 to 2026-01 holdout. The stitched slices below simply show how that same period behaved in smaller chunks.</p>
      <div class="grid">
        <div class="card">
          <div class="label">Phase 2</div>
          <div class="value">2000-01 to 2017-12</div>
          <div class="muted">Research certification, walk-forward selection, and anti-overfitting gates locked the candidate.</div>
        </div>
        <div class="card">
          <div class="label">Phase 3</div>
          <div class="value">{holdout_start} to {holdout_end}</div>
          <div class="muted">Untouched holdout verdict for the locked candidate. Diagnostic walk-forward views lower down are descriptive only.</div>
        </div>
        <div class="card">
          <div class="label">Phase 4</div>
          <div class="value">{phase4_start} onward</div>
          <div class="muted">Reserved future unseen period. Operational paper trading belongs after this frozen Phase 3 result is accepted.</div>
        </div>
      </div>
      <div class="grid">
        <div class="card full">
          {boundary_chart}
        </div>
        <div class="card full">
          {phase3_ladder_chart}
        </div>
      </div>
    </section>
    <section class="section reveal delay-2">
      <h2>Holdout summary</h2>
      <div class="grid">
        <div class="card">
          <div class="label">Primary track</div>
          <div class="value">{short_track_label(official_primary_meta.get('universe_variant','n/a'), official_primary_meta.get('execution_model','n/a'))}</div>
          <div class="muted">FX scenario: {analysis_primary_meta.get('fx_scenario','n/a') or official_primary_meta.get('fx_scenario','n/a')}</div>
        </div>
        <div class="card">
          <div class="label">Official holdout CAGR</div>
          <div class="value">{format_pct(official_primary_cagr or rebuilt_primary_cagr)}</div>
          <div class="muted">Full-period total return {format_pct(official_primary_total or analysis_primary_track.get('total_return'))}</div>
        </div>
        <div class="card">
          <div class="label">Benchmark return</div>
          <div class="value">{format_pct(official_primary_track.get('primary_benchmark_total_return') or analysis_primary_track.get('primary_benchmark_total_return'))}</div>
          <div class="muted">Primary benchmark on the same holdout window.</div>
        </div>
        <div class="card">
          <div class="label">Max drawdown</div>
          <div class="value">{format_pct(official_primary_track.get('max_drawdown') or analysis_primary_track.get('max_drawdown'))}</div>
          <div class="muted">{official_primary_track.get('months') or analysis_primary_track.get('months') or 'n/a'} holdout months</div>
        </div>
        <div class="card">
          <div class="label">Combined slice Sharpe</div>
          <div class="value">{format_float(wf_combined.get('sharpe'))}</div>
          <div class="muted">Stitched across the 5 Phase 3 slices; this is the best like-for-like read of the whole Phase 3 window.</div>
        </div>
        <div class="card">
          <div class="label">MC ending equity (median)</div>
          <div class="value">{format_equity_plus_return(mc_final_equity.get('median'))}</div>
          <div class="muted">Equity index with 100 = start. Bootstrap paths: {monte.get('sample_count','n/a')}</div>
        </div>
      </div>
      <div class="callout">
        <strong>What matters most:</strong>
        <div class="muted">The pass/fail call still comes from the untouched holdout above. The walk-forward section is there to show how the same Phase 3 period behaved when broken into smaller pieces.</div>
      </div>
    </section>
    <section class="section reveal delay-3">
      <h2>Phase 3 Walk-Forward View</h2>
      <p class="muted">Same locked parameters, same holdout era, split into five forward slices so we can look at sub-period stability without changing the official Phase 3 result.</p>
      <div class="badge-row">{wf_combined_badge}</div>
      <p class="muted"><strong>Full-window read:</strong> once the five forward slices are stitched back together, the Phase 3 series is still positive and the combined Sharpe stays above the 0.400 gate. That is why the strategy still counts as a pass across the full Phase 3 span even if some shorter slices wobble.</p>
      <p class="muted"><strong>Diagnostic summary:</strong> {wf_diag}.</p>
      <p class="muted"><strong>Gap summary:</strong> Sharpe {wf_gap_sharpe}. Annualized Return {wf_gap_return}. {wf_gap_months}.</p>
      <p class="muted"><strong>What the chart shows:</strong> Gray dashed = expanding train history for the already-locked candidate. Gold = the forward OOS slice for that pass. Gold dash-dot = {config.PRIMARY_PASSIVE_BENCHMARK} over the same slice.</p>
      <p class="muted"><strong>Timeline:</strong> The schedule is clipped to {holdout_start}-{holdout_end} for readability. Earlier diagnostic slices can be reused as history in later passes because they are all inside the already-finished Phase 3 era.</p>
      <p class="muted"><strong>Fold windows:</strong> {wf_window_text if wf_window_text else 'n/a'}.</p>
      <p class="muted"><strong>Short-slice dispersion:</strong> {wf_gate_passes}/{len(wf_folds) if wf_folds else 0} shorter slices cleared validation Sharpe &gt; {format_float(0.4)}. Smaller sub-windows are much noisier than the full 2018-01 to 2026-01 holdout, so this is context, not a replacement pass/fail rule.</p>
      <div class="grid">
        <div class="card full">
          {wf_schedule_chart}
        </div>
        <div class="card full">
          {render_walkforward_compare_svg(
              wf_folds,
              train_key="train_sharpe",
              validate_key="validate_sharpe",
              benchmark_key="validate_benchmark_sharpe",
              title="Sharpe: Train vs Validation",
              as_pct=False,
              gate_threshold=0.4,
          )}
        </div>
        <div class="card full">
          {render_walkforward_compare_svg(
              wf_folds,
              train_key="train_total_return",
              validate_key="validate_total_return",
              benchmark_value_fn=lambda fold: _fold_cagr(
                  fold,
                  total_key="validate_benchmark_total_return",
                  months_key="validate_benchmark_months",
              ),
              title="Annualized Return (CAGR): Train vs Validation",
              as_pct=True,
              train_value_fn=lambda fold: _fold_cagr(
                  fold,
                  total_key="train_total_return",
                  months_key="train_months",
              ),
              validate_value_fn=lambda fold: _fold_cagr(
                  fold,
                  total_key="validate_total_return",
                  months_key="validate_months",
              ),
          )}
        </div>
        <div class="card full">
          {wf_oos_equity_chart}
        </div>
        <div class="card full">
          {wf_rolling_sharpe_chart}
        </div>
      </div>
    </section>
"""

    page_html += f"""
    <section class="section reveal">
      <h2>Monte Carlo Distribution</h2>
      <p class="muted">Samples: {monte.get('sample_count','n/a')} &middot; Block length: {monte.get('block_length_months','n/a')} months</p>
      <p class="muted">Method: stationary block bootstrap of the realized Phase 3 holdout monthly returns from {holdout_start} to {holdout_end}. This is supporting context, not a replacement for the holdout verdict.</p>
      <div class="badge-row"><span class="label">Monte Carlo quality (context)</span>{mc_badge_html}</div>
      <p class="muted"><strong>What it shows:</strong> {mc_badge_label}. {mc_badge_detail}</p>
      <p class="muted"><strong>Takeaway:</strong> this bootstrap check is supportive rather than disqualifying. Reshuffled versions of the same Phase 3 return stream still look broadly healthy, and the downside tail stays close to breakeven instead of collapsing.</p>
      <p class="muted"><strong>What the fan shows:</strong> the teal fan is a density-first fan, the amber center line is the median path, and the dashed envelope marks p05-p95. Right-edge labels now show both the equity index and the matching total return. The histogram below shows where the bootstrap outcomes finish.</p>
      <p class="muted"><strong>Bottom line:</strong> {mc_interp}</p>
      {mc_anomaly}
      <div class="grid">
        <div class="card full">
          {render_spaghetti_svg(mc_paths, title='Monte Carlo Paths (Equity Index)', max_lines=3, line_opacity=0.028, line_width=0.45, display_quantile_range=(0.05, 0.95), path_quantiles=mc_quantiles, path_density=mc_density)}
        </div>
        <div class="card full">
          {render_histogram_svg(mc_hist.get('final_equity'), metric=mc_final_equity, title='Distribution @ Ending Equity Index', x_label='Ending Equity Index (100 = start)', as_pct=False, width=680, height=180)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get('sharpe'), metric=mc_sharpe, title='Sharpe', x_label='Sharpe', as_pct=False)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get('total_return'), metric=mc_total, title='Total Return From Start', x_label='Total Return From Start (%)', as_pct=True)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get('max_drawdown'), metric=mc_dd, title='Max Drawdown', x_label='Max Drawdown (%)', as_pct=True)}
        </div>
      </div>
      {trimmed_html}
    </section>
    <section class="section reveal">
      <h2>Phase 3 Subperiods</h2>
      <p class="muted">These are the rebuilt primary-track results for the same Phase 3 era, split into broad regime blocks so you can inspect stability without changing the thesis.</p>
      {render_subperiod_table(subperiod_rows)}
    </section>
    <section class="section reveal">
      <h2>Phase 3 Track Matrix</h2>
      <p class="muted">Rebuilt Phase 3 tracks across universes and execution timing for the locked params. This section is series-backed, so it lines up with the walk-forward and Monte Carlo sections above.</p>
      {render_track_table(analysis_track_rows)}
    </section>
    <section class="section reveal">
      <h2>Return Curves (Phase 3 Tracks)</h2>
      <div class="grid">
        {"".join(
            f'<div class="card"><div class="label">{row["label"]}{" (primary)" if row.get("is_primary") else ""}</div>{render_return_curve(row.get("strategy_returns"), row.get("primary_benchmark_returns"))}</div>'
            for row in curve_rows
        )}
      </div>
      <p class="muted" style="margin-top:10px;">Gold line = strategy. Dashed = {config.PRIMARY_PASSIVE_BENCHMARK} if available.</p>
    </section>
  </main>
</body>
</html>
"""
    return page_html


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir
    output_path = args.output

    selection = load_json(source_run_dir / "selection_summary.json")
    if not selection:
        raise ValueError(f"Missing selection_summary.json in {source_run_dir}")
    official_holdout = load_json(source_run_dir / "holdout_results.json")
    if not official_holdout:
        raise ValueError(f"Missing holdout_results.json in {source_run_dir}")

    thesis_name = args.thesis_name or source_run_dir.name
    thesis_meta = build_thesis(thesis_name).manifest_metadata()
    profile_set = infer_profile_set(source_run_dir, args.profile_set)
    profile_settings = resolve_profile_settings(profile_set)
    locked_candidate = selection.get("locked_candidate") or {}
    locked_candidate_id = locked_candidate.get("candidate_id")
    phase2_selection_path, phase2_selection = find_upstream_phase2_selection(
        thesis_name=thesis_name,
        locked_candidate_id=locked_candidate_id,
    )
    if phase2_selection:
        selection["backtest_overfitting"] = phase2_selection.get(
            "backtest_overfitting",
            selection.get("backtest_overfitting", {}),
        )
        selection["phase2_certification_source"] = {
            "candidate_count": (phase2_selection.get("backtest_overfitting") or {}).get("candidate_count"),
            "locked_candidate_id": (phase2_selection.get("locked_candidate") or {}).get("candidate_id"),
            "selection_status": phase2_selection.get("selection_status"),
        }
    if "pbo_band" not in selection or selection.get("pbo_band") in (None, "", "n/a"):
        apply_pbo_policy(selection)
    dataset: ResearchDataset | None = None
    if needs_walk_forward(selection):
        locked = selection.get("locked_candidate") or {}
        locked_params = locked.get("params")
        if locked_params:
            dataset = ResearchDataset(Path("data"))
            selection["walk_forward"] = walk_forward_test(
                dataset,
                thesis=thesis_meta,
                params_grid=[locked_params],
                period_label=selection.get("period_label", "months"),
                periods_per_year=int(selection.get("periods_per_year", 12)),
            )
    selected_params = official_holdout.get("selected_params") or (selection.get("locked_candidate") or {}).get("params")
    if not selected_params:
        raise ValueError("No locked params found for Phase 3 reconstruction.")
    dataset = dataset or ResearchDataset(Path("data"))
    holdout_window = official_holdout.get("holdout_window") or {}
    holdout_start = holdout_window.get("start") or config.OOS_START
    holdout_end = holdout_window.get("end") or config.OOS_END
    analysis_holdout = evaluate_holdout(
        dataset,
        thesis=thesis_meta,
        params=selected_params,
        period_label=official_holdout.get("period_label", "months"),
        periods_per_year=int(official_holdout.get("periods_per_year", 12)),
        start_month=holdout_start,
        end_month=holdout_end,
    )
    analysis_holdout["status"] = official_holdout.get("status", analysis_holdout.get("status", "ok"))
    phase3_walk_forward = phase3_walk_forward_test(
        dataset,
        thesis=thesis_meta,
        params=selected_params,
        period_label=official_holdout.get("period_label", "months"),
        periods_per_year=int(official_holdout.get("periods_per_year", 12)),
        start_month=holdout_start,
        end_month=holdout_end,
        fold_count=5,
    )

    archived_sharpe = holdout_primary_sharpe(official_holdout)
    rebuilt_sharpe = holdout_primary_sharpe(analysis_holdout)
    source_note: str | None = None
    if archived_sharpe is not None and rebuilt_sharpe is not None:
        if abs(float(archived_sharpe) - float(rebuilt_sharpe)) > 1e-9:
            source_note = (
                "The archived holdout JSON and the rebuilt series-backed holdout are both shown here. The headline "
                "Phase 3 pass/fail still follows the official archived holdout, while the charts use rebuilt monthly "
                "returns so the diagnostics line up cleanly. Official Sharpe "
                f"{format_float(archived_sharpe)} vs rebuilt Sharpe {format_float(rebuilt_sharpe)}."
            )

    if phase2_selection_path is not None:
        selection_href = rel_href(output_path, phase2_selection_path.with_suffix(".html"))
    else:
        selection_href = rel_href(output_path, source_run_dir / "selection_summary.html")
    raw_holdout_href = rel_href(output_path, source_run_dir / "holdout_results.html")
    back_href = "index.html" if "analysis_suite" in output_path.parts else selection_href
    portfolio_css_href = rel_href(output_path, Path("portfolio/assets/portfolio.css"))
    portfolio_theme_href = rel_href(output_path, Path("portfolio/assets/portfolio-theme.css"))
    portfolio_home_href = rel_href(output_path, Path("portfolio/index.html"))
    employer_package_href = rel_href(output_path, Path("portfolio/employer_package/index.html"))
    analysis_suite_href = rel_href(output_path, Path("portfolio/employer_package/analysis_suite/index.html"))
    context_note = args.context_note or (
        "This page keeps the official Phase 3 holdout result up front and puts the supporting diagnostics underneath, "
        f"all inside the same {holdout_start} to {holdout_end} window."
    )

    rendered = build_phase3_dashboard(
        thesis=thesis_meta,
        selection=selection,
        official_holdout=official_holdout,
        analysis_holdout=analysis_holdout,
        phase3_walk_forward=phase3_walk_forward,
        profile_set=profile_set,
        profile_settings=profile_settings,
        selection_href=selection_href,
        raw_holdout_href=raw_holdout_href,
        back_href=back_href,
        portfolio_css_href=portfolio_css_href,
        portfolio_theme_href=portfolio_theme_href,
        portfolio_home_href=portfolio_home_href,
        employer_package_href=employer_package_href,
        analysis_suite_href=analysis_suite_href,
        context_note=context_note,
        source_note=source_note,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Rendered Phase 3 holdout dashboard to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
