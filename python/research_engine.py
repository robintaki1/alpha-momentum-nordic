from __future__ import annotations

import argparse
import html
import itertools
import math
import os
import random
import statistics
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import validation_protocol
import strategy_variants
from phase1_lib import validate_phase1
from paper_trading_engine import (
    ResearchDataset,
    WindowSimulation,
    annualized_sharpe as annualized_sharpe_periods,
    build_thesis,
    max_drawdown,
    serialize_json,
    total_return,
)


PRIMARY_TRACK = validation_protocol.PRIMARY_TRACK
REQUIRED_UNIVERSE_VARIANTS = validation_protocol.REQUIRED_UNIVERSE_VARIANTS
REQUIRED_EXECUTION_MODELS = validation_protocol.REQUIRED_EXECUTION_MODELS
REQUIRED_FX_SCENARIOS = validation_protocol.REQUIRED_FX_SCENARIOS

MONTE_CARLO_HIST_BINS = 30
MONTE_CARLO_SAMPLE_PATHS = 150
MONTE_CARLO_TRIM_TOP_PCT = 0.05
MONTE_CARLO_PATH_QUANTILES = (
    0.01,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.35,
    0.45,
    0.50,
    0.55,
    0.65,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.99,
)
MONTE_CARLO_DENSITY_BINS = 72
MONTE_CARLO_DENSITY_QUANTILE_RANGE = (0.01, 0.99)


@dataclass(frozen=True)
class EvaluationWindow:
    start_month: str
    end_month: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research engine profiles and render dashboards.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--output-dir", default=Path("results/research_engine"), type=Path)
    parser.add_argument("--theses", nargs="+", default=list(config.RESEARCH_THESIS_SETTINGS))
    parser.add_argument("--profiles", nargs="+", default=["quick", "mega", "certification_baseline"])
    parser.add_argument("--profile-set", default="default")
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--walk-forward-profiles", nargs="+")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def pbo_is_not_applicable(backtest: dict[str, Any] | None) -> bool:
    if not isinstance(backtest, dict):
        return False
    interpretation = backtest.get("interpretation")
    candidate_count = backtest.get("candidate_count")
    return interpretation == "not_applicable_single_candidate" or candidate_count == 1


def format_pbo_display(value: float | None, backtest: dict[str, Any] | None = None) -> str:
    if value is not None:
        return format_pct(value)
    if pbo_is_not_applicable(backtest):
        return "n/a (single candidate)"
    return "n/a"


def pbo_explainer(backtest: dict[str, Any] | None) -> str:
    if pbo_is_not_applicable(backtest):
        return "PBO is not applicable here because this replay freezes one exact candidate rather than ranking a multi-candidate grid."
    return (
        f"PBO is evaluated with good < {format_pct(config.PBO_THRESHOLD_MAX)} and hard cutoff >= "
        f"{format_pct(config.PBO_HARD_CUTOFF)}."
    )


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def equity_index_to_total_return(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value) / 100.0 - 1.0


def format_signed_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:+.1f}%"


def format_equity_plus_return(value: float | None, digits: int = 1) -> str:
    total_return_value = equity_index_to_total_return(value)
    if total_return_value is None:
        return "n/a"
    return f"{format_float(float(value), digits)} ({format_pct(total_return_value)})"


def format_equity_plus_return_verbose(value: float | None, digits: int = 1) -> str:
    total_return_value = equity_index_to_total_return(value)
    if total_return_value is None:
        return "n/a"
    return f"{format_float(float(value), digits)} index ({format_signed_pct(total_return_value)} total return)"


def format_equity_plus_return_compact(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return format_float(float(value), digits)


def format_params(params: dict | None) -> str:
    if not params:
        return "n/a"
    strategy_id = params.get("strategy_id", "baseline")
    extras = []
    if params.get("band_buffer") is not None:
        extras.append(f"band={params.get('band_buffer')}")
    if params.get("min_hold_months") is not None:
        extras.append(f"min_hold={params.get('min_hold_months')}")
    if params.get("weighting") and params.get("weighting") != "equal":
        extras.append(f"weight={params.get('weighting')}")
    if params.get("trend_filter"):
        extras.append(f"trend_ma={params.get('ma_window', 'n/a')}")
    if params.get("liquidity_min_mdv"):
        extras.append(f"liq_mdv>={params.get('liquidity_min_mdv')}")
    extra_text = f" / {', '.join(extras)}" if extras else ""
    return (
        f"L={params.get('l')} / skip={params.get('skip')} / top_n={params.get('top_n')} / strat={strategy_id}"
        f"{extra_text}"
    )


def format_range(low: float | None, high: float | None, digits: int = 3) -> str:
    if low is None or high is None:
        return "n/a"
    return f"{low:.{digits}f} - {high:.{digits}f}"


def _clean_returns(returns: Sequence[float] | None) -> list[float]:
    if not returns:
        return []
    cleaned: list[float] = []
    for value in returns:
        if value is None:
            continue
        if not np.isfinite(value):
            continue
        cleaned.append(float(value))
    return cleaned


def format_mc(metric: dict | None, *, as_pct: bool = False) -> str:
    if not metric:
        return "n/a"
    if as_pct:
        return f"{format_pct(metric.get('median'))} ({format_pct(metric.get('p05'))} - {format_pct(metric.get('p95'))})"
    return f"{format_float(metric.get('median'))} ({format_float(metric.get('p05'))} - {format_float(metric.get('p95'))})"


def render_badge(label: str, tone: str = "neutral", detail: str | None = None) -> str:
    detail_html = f'<span class="badge-detail">{detail}</span>' if detail else ""
    return f'<span class="badge {tone}">{label}</span>{detail_html}'


def monte_carlo_badge(
    mc_sharpe: dict | None, mc_total: dict | None, mc_dd: dict | None
) -> tuple[str, str, str]:
    if not mc_sharpe and not mc_total and not mc_dd:
        return ("n/a", "neutral", "No Monte Carlo metrics.")
    score = 0
    max_score = 0
    detail_parts = []

    if mc_sharpe:
        p05 = mc_sharpe.get("p05")
        if p05 is not None and np.isfinite(p05):
            max_score += 1
            if p05 > 0:
                score += 1
            detail_parts.append(f"p05 Sharpe {format_float(p05)}")
    if mc_total:
        p05 = mc_total.get("p05")
        if p05 is not None and np.isfinite(p05):
            max_score += 1
            # A mildly negative 5th-percentile path can still be acceptable for a long
            # multi-year bootstrap. Treat deeper downside as the caution threshold.
            if p05 > -0.10:
                score += 1
            detail_parts.append(f"p05 Return {format_pct(p05)}")
    if mc_dd:
        p95 = mc_dd.get("p95")
        if p95 is not None and np.isfinite(p95):
            max_score += 1
            if p95 < 0.65:
                score += 1
            detail_parts.append(f"p95 Drawdown {format_pct(p95)}")

    if max_score == 0:
        return ("n/a", "neutral", "Monte Carlo metrics incomplete.")
    ratio = score / max_score
    if ratio >= 0.67:
        quality, tone = "GOOD", "good"
    elif ratio >= 0.34:
        quality, tone = "CAUTION", "caution"
    else:
        quality, tone = "WEAK", "weak"
    label = f"QUALITY: {quality}"
    detail = ("Not a pass/fail gate. " + " | ".join(detail_parts)).strip()
    return (label, tone, detail)


def selection_verdict(summary: dict[str, Any]) -> tuple[str, str, str]:
    status = summary.get("selection_status", "n/a")
    if status == "selected":
        return ("PASS", "good", "All hard gates cleared.")
    if status == "selected_with_caution":
        return ("PASS (CAUTION)", "caution", "Hard gates cleared; PBO in caution band.")
    if status == "pbo_hard_cutoff":
        return ("FAIL", "weak", "PBO hard cutoff triggered.")
    if status == "no_candidate_passed_hard_gates":
        return ("FAIL", "weak", "No candidate cleared all hard gates.")
    return ("FAIL", "weak", f"Status: {status}")


def negative_control_gate_info(negative_controls: dict[str, Any]) -> tuple[float | None, str]:
    if not negative_controls:
        return (None, "n/a")
    details: list[str] = []
    max_rate: float | None = None
    for key in config.NEGATIVE_CONTROL_GATED_KEYS:
        payload = negative_controls.get(key) or {}
        run_count = payload.get("run_count", 0)
        pass_count = payload.get("pass_count", 0)
        if run_count:
            rate = pass_count / run_count
            details.append(f"{key} {format_pct(rate)}")
            max_rate = rate if max_rate is None else max(max_rate, rate)
    detail_text = "; ".join(details) if details else "n/a"
    return (max_rate, detail_text)


def render_validation_verdict(summary: dict[str, Any]) -> str:
    if not summary:
        return ""
    candidate = summary.get("locked_candidate")
    if not candidate:
        ranked = summary.get("ranked_candidates") or []
        candidate = ranked[0] if ranked else {}
    verdict_label, verdict_tone, verdict_detail = selection_verdict(summary)
    verdict_badge = render_badge(verdict_label, verdict_tone, verdict_detail)
    candidate_id = candidate.get("candidate_id", "n/a")

    total_folds = len(config.ROLLING_ORIGIN_FOLDS)
    fold_passes = candidate.get("fold_pass_count")
    fold_gate = bool(candidate.get("gate_fold_count"))
    fold_detail = (
        f"{fold_passes}/{total_folds} folds >= 0.40 (need {config.MEGA_WF_PASSES_REQUIRED})"
        if fold_passes is not None
        else "n/a"
    )

    bootstrap_low = candidate.get("bootstrap_ci_low")
    bootstrap_gate = bool(candidate.get("gate_bootstrap"))
    bootstrap_detail = f"CI low {format_float(bootstrap_low)} > 0"

    deflated_score = candidate.get("deflated_sharpe_score")
    deflated_gate = bool(candidate.get("gate_deflated_sharpe"))
    deflated_detail = f"score {format_float(deflated_score)} > 0"

    neg_controls = summary.get("negative_controls", {})
    neg_rate, neg_detail = negative_control_gate_info(neg_controls)
    neg_gate = neg_rate is not None and neg_rate <= config.NEGATIVE_CONTROL_PASS_RATE_MAX
    neg_detail_full = f"{neg_detail} (max {format_pct(config.NEGATIVE_CONTROL_PASS_RATE_MAX)})"

    backtest = summary.get("backtest_overfitting", {})
    pbo_value = backtest.get("pbo")
    pbo_caution = summary.get("pbo_caution", False)
    pbo_hard = summary.get("pbo_hard_cutoff", False)
    if pbo_is_not_applicable(backtest):
        pbo_pass = True
        pbo_label = "N/A"
        pbo_tone = "neutral"
        pbo_detail = "PBO not applicable because this replay freezes one exact candidate instead of ranking a grid."
    else:
        pbo_pass = not pbo_hard
        pbo_label = "PASS" if pbo_pass else "FAIL"
        pbo_tone = "caution" if pbo_caution and pbo_pass else ("good" if pbo_pass else "weak")
        pbo_detail = (
            f"PBO {format_pbo_display(pbo_value, backtest)} (good < {format_pct(config.PBO_THRESHOLD_MAX)}; "
            f"hard cutoff >= {format_pct(config.PBO_HARD_CUTOFF)})"
        )

    def _gate_row(label: str, passed: bool, detail: str, tone_override: str | None = None) -> str:
        tone = tone_override if tone_override else ("good" if passed else "weak")
        badge = render_badge("PASS" if passed else "FAIL", tone, detail)
        return f'<li class="verdict-item"><span class="verdict-label">{label}</span>{badge}</li>'

    rows = [
        _gate_row("Walk-forward Sharpe gate", fold_gate, fold_detail),
        _gate_row("Bootstrap CI gate", bootstrap_gate, bootstrap_detail),
        _gate_row("Deflated Sharpe gate", deflated_gate, deflated_detail),
        _gate_row("Negative controls gate", neg_gate, neg_detail_full),
        f'<li class="verdict-item"><span class="verdict-label">PBO hard cutoff</span>{render_badge(pbo_label, pbo_tone, pbo_detail)}</li>',
    ]

    return (
        "<section>"
        "<h2>Validation Verdict (Hard Gates)</h2>"
        f'<div class="verdict-panel">{verdict_badge}</div>'
        f'<p class="verdict-note"><strong>Candidate:</strong> {candidate_id}</p>'
        f'<p class="verdict-note"><strong>Passing grade:</strong> all hard gates below must be PASS. '
        f'PBO >= {format_pct(config.PBO_HARD_CUTOFF)} is automatic FAIL; '
        f'{format_pct(config.PBO_THRESHOLD_MAX)} - {format_pct(config.PBO_HARD_CUTOFF)} = PASS with caution.</p>'
        f'<ul class="verdict-list">{"".join(rows)}</ul>'
        "</section>"
    )


def sensitivity_badge(info: dict[str, Any] | None) -> tuple[str, str, str]:
    if not info:
        return ("n/a", "neutral", "No sensitivity data.")
    by_value = info.get("by_value") or []
    if len(by_value) <= 1:
        return ("NOT TESTED", "neutral", "Only one value in grid.")
    spread = info.get("median_spread")
    if spread is None or not np.isfinite(spread):
        return ("n/a", "neutral", "Spread unavailable.")
    if spread <= 0.10:
        label, tone = "Robust", "good"
    elif spread <= 0.25:
        label, tone = "Caution", "caution"
    else:
        label, tone = "Sensitive", "weak"
    detail = f"Spread {format_float(spread)}"
    return (label, tone, detail)


def _axis_ticks(min_v: float, max_v: float, count: int) -> list[float]:
    if count < 2:
        return [min_v]
    if max_v <= min_v:
        return [min_v, max_v]
    step = (max_v - min_v) / (count - 1)
    return [min_v + step * i for i in range(count)]


def _month_to_ordinal(month: str) -> int:
    year, month_number = month.split("-")
    return int(year) * 12 + int(month_number) - 1


def _ordinal_to_year_label(ordinal: float) -> str:
    year = int(ordinal // 12)
    return str(year)


def _ordinal_to_month(ordinal: int) -> str:
    year = ordinal // 12
    month_number = ordinal % 12 + 1
    return f"{year:04d}-{month_number:02d}"


def _sample_paths_evenly(paths: list[list[float]], *, max_lines: int | None) -> list[list[float]]:
    if not max_lines or len(paths) <= max_lines:
        return paths
    order = sorted(range(len(paths)), key=lambda idx: paths[idx][-1])
    if max_lines <= 1:
        indices = [order[len(order) // 2]]
    else:
        step = (len(order) - 1) / (max_lines - 1)
        indices: list[int] = []
        seen: set[int] = set()
        for i in range(max_lines):
            pos = int(round(i * step))
            idx = order[pos]
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
        if len(indices) < max_lines:
            for idx in order:
                if idx in seen:
                    continue
                indices.append(idx)
                if len(indices) >= max_lines:
                    break
    return [paths[idx] for idx in indices]


def render_histogram_svg(
    hist: dict[str, Any] | None,
    *,
    title: str,
    metric: dict[str, Any] | None = None,
    x_label: str = "Value",
    as_pct: bool,
    width: int = 320,
    height: int = 180,
) -> str:
    if not hist:
        return '<div class="muted">Histogram not available.</div>'
    edges = hist.get("edges") or []
    counts = hist.get("counts") or []
    if len(edges) < 2 or len(counts) != len(edges) - 1:
        return '<div class="muted">Histogram not available.</div>'
    x_min = float(edges[0])
    x_max = float(edges[-1])
    y_max = max(counts) if counts else 1
    if y_max <= 0:
        y_max = 1
    margin_left, margin_right = 52, 14
    margin_top, margin_bottom = 18, 46
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * inner_w if x_max > x_min else margin_left

    def _y(value: float) -> float:
        return margin_top + inner_h - (value / y_max) * inner_h

    bar_color = "#6b8679"
    grid_color = "#ddd5c8"
    axis_color = "#8a7c69"
    marker_palette = {
        "p05": "#a9645d",
        "median": "#a97831",
        "p95": "#567c72",
    }

    bars = []
    for idx, count in enumerate(counts):
        x0 = _x(edges[idx])
        x1 = _x(edges[idx + 1])
        bar_w = max(1.0, x1 - x0 - 1.0)
        bar_h = inner_h - (_y(count) - margin_top)
        bars.append(
            f'<rect x="{x0:.1f}" y="{_y(count):.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'fill="{bar_color}" opacity="0.82"></rect>'
        )

    x_ticks = _axis_ticks(x_min, x_max, 4)
    y_ticks = _axis_ticks(0.0, float(y_max), 4)
    x_labels = [
        f'<text x="{_x(tick):.1f}" y="{height - 22}" text-anchor="middle">{format_pct(tick) if as_pct else format_float(tick)}</text>'
        for tick in x_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{int(round(tick))}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        f'stroke="{grid_color}" stroke-width="1"></line>'
        for tick in y_ticks
    ]

    marker_lines = []
    marker_labels = []
    if metric:
        for label, value, color, dash in (
            ("p05", metric.get("p05"), marker_palette["p05"], "4 4"),
            ("median", metric.get("median"), marker_palette["median"], "0"),
            ("p95", metric.get("p95"), marker_palette["p95"], "4 4"),
        ):
            if value is None or not np.isfinite(value):
                continue
            if value < x_min or value > x_max:
                continue
            x_pos = _x(float(value))
            dash_attr = f' stroke-dasharray="{dash}"' if dash != "0" else ""
            marker_lines.append(
                f'<line x1="{x_pos:.1f}" y1="{margin_top}" x2="{x_pos:.1f}" y2="{height - margin_bottom}" '
                f'stroke="{color}" stroke-width="1.4"{dash_attr}></line>'
            )
            marker_labels.append(f"{label} {format_pct(value) if as_pct else format_float(value)}")

    note_parts = [
        f"X-axis = {x_label}; Y-axis = number of bootstrap samples. Taller bars = more likely outcomes."
    ]
    if not as_pct and "Equity" in x_label:
        note_parts.append("On equity-index charts, 100 = start, so 329 means +229% total return from start.")
    if marker_labels:
        note_parts.append(f"Markers: {', '.join(marker_labels)}")
    marker_text = '<div class="chart-note">' + " ".join(note_parts) + "</div>"

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + "".join(bars)
        + "".join(marker_lines)
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="{axis_color}" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="{axis_color}" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 6}" text-anchor="start">Count</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 6}" text-anchor="middle">{x_label}</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + "</svg>"
        + marker_text
        + "</div>"
    )


def render_spaghetti_svg(
    paths: list[list[float]] | None,
    *,
    title: str,
    width: int = 680,
    height: int = 220,
    max_lines: int = 8,
    line_opacity: float = 0.045,
    line_width: float = 0.55,
    display_quantile_range: tuple[float, float] | None = None,
    fit_plotted_extents: bool = False,
    center_baseline: float | None = None,
    path_quantiles: dict[str, Any] | None = None,
    path_density: dict[str, Any] | None = None,
) -> str:
    if not paths:
        return '<div class="muted">Monte Carlo paths not available.</div>'
    series = [path for path in paths if path]
    if not series:
        return '<div class="muted">Monte Carlo paths not available.</div>'
    length = min(len(path) for path in series)
    if length < 2:
        return '<div class="muted">Monte Carlo paths not available.</div>'
    all_series = [path[:length] for path in series]
    plot_series = _sample_paths_evenly(all_series, max_lines=max_lines)
    margin_left, margin_right = 72, 104
    margin_top, margin_bottom = 18, 44
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x(idx: int) -> float:
        return margin_left + idx / (length - 1) * inner_w

    quantile_cache: dict[str, list[float]] = {}
    quantile_columns: list[list[float]] | None = None

    def _quantile_series(probability: float) -> list[float] | None:
        nonlocal quantile_columns
        key = _quantile_key(probability)
        cached = quantile_cache.get(key)
        if cached is not None:
            return cached
        if isinstance(path_quantiles, dict):
            series_values = path_quantiles.get(key)
            if isinstance(series_values, list) and len(series_values) >= length:
                quantile_cache[key] = series_values[:length]
                return quantile_cache[key]
        if quantile_columns is None:
            quantile_columns = [sorted(path[idx] for path in all_series) for idx in range(length)]
        derived = [_quantile(values, probability) for values in quantile_columns]
        quantile_cache[key] = derived
        return derived

    median_path = _quantile_series(0.5)
    display_low: list[float] | None = None
    display_high: list[float] | None = None
    if display_quantile_range is not None:
        lower_q, upper_q = display_quantile_range
        display_low = _quantile_series(lower_q)
        display_high = _quantile_series(upper_q)
    if display_quantile_range is None:
        display_low = _quantile_series(0.05)
        display_high = _quantile_series(0.95)
    if median_path is None or display_low is None or display_high is None:
        return '<div class="muted">Monte Carlo paths not available.</div>'

    if display_quantile_range is not None and display_low and display_high:
        y_min = min(display_low)
        y_max = max(display_high)
        padding = max((y_max - y_min) * 0.04, 1.0)
        y_min -= padding
        y_max += padding
        if fit_plotted_extents and plot_series:
            plotted_min = min(min(path) for path in plot_series)
            plotted_max = max(max(path) for path in plot_series)
            y_min = min(y_min, plotted_min)
            y_max = max(y_max, plotted_max)
            extra_padding = max((y_max - y_min) * 0.08, 4.0)
            y_min -= extra_padding
            y_max += extra_padding
    else:
        y_min = min(min(path) for path in all_series)
        y_max = max(max(path) for path in all_series)
    if center_baseline is not None:
        baseline_half_range = max(y_max - center_baseline, center_baseline - y_min, 1.0)
        y_min = center_baseline - baseline_half_range
        y_max = center_baseline + baseline_half_range
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    plot_bg = "#f5efe5"
    plot_bg_opacity = "1.0"
    grid_color = "#ddd5c8"
    axis_color = "#8a7c69"
    texture_color = "#7f8680"
    fan_color = "#4d7d74"
    concentration_band_specs = [
        (0.10, 0.90, "#bfd0c7", 0.18),
        (0.25, 0.75, "#8fb0a5", 0.22),
        (0.40, 0.60, "#5e857a", 0.28),
    ]
    band_polygons: list[str] = []
    band_lines: list[str] = []
    for low_q, high_q, fill_color, fill_opacity in concentration_band_specs:
        low_series = _quantile_series(low_q)
        high_series = _quantile_series(high_q)
        if low_series is None or high_series is None:
            continue
        band_points = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(high_series))
        band_points += " " + " ".join(
            f"{_x(i):.1f},{_y(v):.1f}" for i, v in reversed(list(enumerate(low_series)))
        )
        band_polygons.append(
            f'<polygon points="{band_points}" fill="{fill_color}" opacity="{fill_opacity:.3f}"></polygon>'
        )
        low_line = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(low_series))
        high_line = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(high_series))
        stroke_opacity = min(0.62, fill_opacity + 0.18)
        band_lines.append(
            f'<polyline points="{high_line}" fill="none" stroke="{fill_color}" stroke-width="1.0" '
            f'opacity="{stroke_opacity:.3f}" stroke-linecap="round"></polyline>'
        )
        band_lines.append(
            f'<polyline points="{low_line}" fill="none" stroke="{fill_color}" stroke-width="1.0" '
            f'opacity="{stroke_opacity:.3f}" stroke-linecap="round"></polyline>'
        )

    fan_pairs = [
        (0.01, 0.99),
        (0.03, 0.97),
        (0.05, 0.95),
        (0.10, 0.90),
        (0.20, 0.80),
        (0.35, 0.65),
    ]
    fan_polygons: list[str] = []
    for idx, (low_q, high_q) in enumerate(fan_pairs):
        low_series = _quantile_series(low_q)
        high_series = _quantile_series(high_q)
        if low_series is None or high_series is None:
            continue
        points = ' '.join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(high_series))
        points += ' ' + ' '.join(
            f"{_x(i):.1f},{_y(v):.1f}" for i, v in reversed(list(enumerate(low_series)))
        )
        alpha = 0.028 + idx * 0.018
        fan_polygons.append(
            f'<polygon points="{points}" fill="{fan_color}" opacity="{alpha:.3f}"></polygon>'
        )

    density_rects: list[str] = []
    density_mode = False
    if isinstance(path_density, dict):
        y_edges = path_density.get('y_edges') or []
        counts = path_density.get('counts') or []
        max_count = float(path_density.get('max_count') or 0)
        if (
            isinstance(y_edges, list)
            and isinstance(counts, list)
            and len(y_edges) >= 2
            and counts
            and max_count > 0
        ):
            density_mode = True
            usable_columns = min(len(counts), length)
            for idx in range(usable_columns):
                row = counts[idx] or []
                if not isinstance(row, list):
                    continue
                x0 = _x(idx)
                x1 = _x(idx + 1) if idx < length - 1 else (width - margin_right)
                rect_w = max(1.0, x1 - x0 + 0.6)
                usable_bins = min(len(row), len(y_edges) - 1)
                for bin_idx in range(usable_bins):
                    count = row[bin_idx]
                    if not count:
                        continue
                    y_low = float(y_edges[bin_idx])
                    y_high = float(y_edges[bin_idx + 1])
                    y0 = _y(y_high)
                    y1 = _y(y_low)
                    rect_y = min(y0, y1)
                    rect_h = abs(y1 - y0) + 0.6
                    if rect_y > margin_top + inner_h or rect_y + rect_h < margin_top:
                        continue
                    intensity = min(1.0, max(0.0, float(count) / max_count))
                    opacity = 0.01 + 0.62 * (intensity ** 1.8)
                    density_rects.append(
                        f'<rect x="{x0:.1f}" y="{rect_y:.1f}" width="{rect_w:.1f}" height="{rect_h:.1f}" '
                        f'fill="{fan_color}" opacity="{opacity:.3f}"></rect>'
                    )

    polylines = []
    for path in plot_series:
        points = ' '.join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(path))
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="{texture_color}" '
            f'stroke-width="{line_width}" opacity="{line_opacity}" stroke-linecap="round"></polyline>'
        )

    lower_q, upper_q = display_quantile_range or (0.05, 0.95)
    outer_high_points = ' '.join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(display_high))
    outer_low_points = ' '.join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(display_low))
    median_points = ' '.join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(median_path))
    final_p10 = _quantile_series(0.10)
    final_p25 = _quantile_series(0.25)
    final_p75 = _quantile_series(0.75)
    final_p90 = _quantile_series(0.90)

    x_ticks = _axis_ticks(0.0, float(length - 1), 5)
    y_ticks = _axis_ticks(y_min, y_max, 5)
    x_labels = [
        f'<text x="{_x(int(round(tick))):.1f}" y="{height - 20}" text-anchor="middle">{int(round(tick))}</text>'
        for tick in x_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_float(tick, 2)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        f'stroke="{grid_color}" stroke-width="1"></line>'
        for tick in y_ticks
    ]

    endpoint_annotations: list[tuple[str, float, str, str]] = []
    if final_p90 and len(final_p90) == length:
        endpoint_annotations.append(("P90:", final_p90[-1], "#6f8c5b", format_equity_plus_return_compact(final_p90[-1], 1)))
    if final_p75 and len(final_p75) == length:
        endpoint_annotations.append(("P75:", final_p75[-1], "#567c72", format_equity_plus_return_compact(final_p75[-1], 1)))
    endpoint_annotations.append(("Median:", median_path[-1], "#a97831", format_equity_plus_return_compact(median_path[-1], 1)))
    if final_p25 and len(final_p25) == length:
        endpoint_annotations.append(("P25:", final_p25[-1], "#567c72", format_equity_plus_return_compact(final_p25[-1], 1)))
    if final_p10 and len(final_p10) == length:
        endpoint_annotations.append(("P10:", final_p10[-1], "#a9645d", format_equity_plus_return_compact(final_p10[-1], 1)))

    endpoint_annotations.sort(key=lambda item: _y(item[1]))
    label_y_positions: list[float] = []
    min_gap = 12.0
    top_bound = margin_top + 10.0
    bottom_bound = height - margin_bottom - 10.0
    for _, value, _, _ in endpoint_annotations:
        y_pos = max(top_bound, _y(value))
        if label_y_positions:
            y_pos = max(y_pos, label_y_positions[-1] + min_gap)
        label_y_positions.append(y_pos)
    for idx in range(len(label_y_positions) - 2, -1, -1):
        if label_y_positions[idx + 1] > bottom_bound:
            label_y_positions[idx + 1] = bottom_bound
        label_y_positions[idx] = min(label_y_positions[idx], label_y_positions[idx + 1] - min_gap)
    if label_y_positions and label_y_positions[0] < top_bound:
        shift = top_bound - label_y_positions[0]
        label_y_positions = [min(bottom_bound, pos + shift) for pos in label_y_positions]

    endpoint_guides: list[str] = []
    endpoint_labels: list[str] = []
    end_x = _x(length - 1)
    guide_x = end_x + 8.0
    label_x = end_x + 12.0
    for (label, value, color, value_text), label_y in zip(endpoint_annotations, label_y_positions):
        path_y = _y(value)
        endpoint_guides.append(
            f'<line x1="{end_x:.1f}" y1="{path_y:.1f}" x2="{guide_x:.1f}" y2="{label_y:.1f}" '
            f'stroke="{color}" stroke-width="1.0" opacity="0.85"></line>'
        )
        endpoint_labels.append(
            f'<text x="{label_x:.1f}" y="{label_y + 3.0:.1f}" text-anchor="start" fill="{color}">{label} {value_text}</text>'
        )

    total_paths = len(all_series)
    shown_paths = len(plot_series)
    clipped_note = ''
    if display_quantile_range is not None:
        lower_label = f"p{int(round(lower_q * 100)):02d}"
        upper_label = f"p{int(round(upper_q * 100)):02d}"
        if fit_plotted_extents:
            clipped_note = (
                f"Display axis starts from the {lower_label}-{upper_label} path envelope and expands to fit the plotted sample paths; "
                'only rarer unshown extremes are clipped. '
            )
        else:
            clipped_note = (
                f"Display axis is capped to the {lower_label}-{upper_label} path envelope for readability; "
                'rare extremes are clipped. '
            )
    baseline_note = (
        f"Baseline {format_float(center_baseline)} is centered on the vertical axis for easier up/down comparison. "
        if center_baseline is not None
        else ''
    )
    concentration_note = ""
    if (
        final_p10
        and final_p25
        and final_p75
        and final_p90
        and len(final_p10) == length
        and len(final_p25) == length
        and len(final_p75) == length
        and len(final_p90) == length
    ):
        concentration_note = (
            f"By month {length - 1}, the middle 50% cluster ends around "
            f"{format_equity_plus_return_verbose(final_p25[-1])} to {format_equity_plus_return_verbose(final_p75[-1])}, "
            f"while the middle 80% spans about {format_equity_plus_return_verbose(final_p10[-1])} to {format_equity_plus_return_verbose(final_p90[-1])}. "
        )
    fan_desc = 'path density heatmap' if density_mode else 'layered density bands'
    legend = (
        '<div class="chart-note">'
        f'{concentration_note}'
        'X-axis = months from start; Y-axis = equity index (100 = start). '
        '<span class="legend-line density">Core ribbon</span> = middle 20% of outcomes &middot; '
        '<span class="legend-line density">Dark ribbon</span> = middle 50% of outcomes &middot; '
        '<span class="legend-line density">Light ribbon</span> = middle 80% of outcomes &middot; '
        '<span class="legend-line">Texture lines</span> = sampled bootstrap paths &middot; '
        f'<span class="legend-line density">Fan</span> = {fan_desc} &middot; '
        '<span class="legend-line median">Bold line</span> = median &middot; '
        f'<span class="legend-line band">Envelope</span> = p{int(round(lower_q * 100)):02d}-p{int(round(upper_q * 100)):02d}. '
        'Right-edge labels show ending index first, then total return from start. '
        f'{baseline_note}'
        f'{clipped_note}'
        'Darker fan = more likely outcomes. Wider fan = more uncertainty. '
        f'Showing {shown_paths} texture paths out of {total_paths} simulated paths.'
        '</div>'
    )
    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + f'<rect x="{margin_left}" y="{margin_top}" width="{inner_w:.1f}" height="{inner_h:.1f}" fill="{plot_bg}" opacity="{plot_bg_opacity}" rx="10"></rect>'
        + ''.join(y_grid)
        + (''.join(density_rects) if density_rects else ''.join(fan_polygons))
        + ''.join(band_polygons)
        + ''.join(band_lines)
        + ''.join(polylines)
        + f'<polyline points="{outer_high_points}" fill="none" stroke="#6f8c5b" stroke-width="1.8" opacity="0.95" stroke-dasharray="5 4"></polyline>'
        + f'<polyline points="{outer_low_points}" fill="none" stroke="#a9645d" stroke-width="1.8" opacity="0.95" stroke-dasharray="5 4"></polyline>'
        + f'<polyline points="{median_points}" fill="none" stroke="#a97831" stroke-width="2.5"></polyline>'
        + ''.join(endpoint_guides)
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="{axis_color}" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="{axis_color}" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 6}" text-anchor="start">Equity Index (100 = start)</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 6}" text-anchor="middle">Months from start</text>'
        + ''.join(x_labels)
        + ''.join(y_labels)
        + ''.join(endpoint_labels)
        + '</svg>'
        + legend
        + '</div>'
    )

def render_walkforward_svg(
    folds: list[dict[str, Any]] | None,
    *,
    metric_key: str,
    title: str,
    as_pct: bool,
    gate_threshold: float | None = None,
    width: int = 680,
    height: int = 200,
) -> str:
    if not folds:
        return '<div class="muted">Walk-forward data not available.</div>'
    points: list[tuple[float, float]] = []
    windows: list[tuple[int, int]] = []
    split_lines: list[int] = []
    for fold in folds:
        window = fold.get("validate_window") or {}
        start = window.get("start")
        end = window.get("end")
        value = fold.get(metric_key)
        if start and end and value is not None and np.isfinite(value):
            start_ord = _month_to_ordinal(start)
            end_ord = _month_to_ordinal(end)
            mid = (start_ord + end_ord) / 2.0
            points.append((mid, float(value)))
            windows.append((start_ord, end_ord))
            split_lines.append(start_ord)
    if not points:
        return '<div class="muted">Walk-forward data not available.</div>'
    x_min = min(start for start, _ in windows)
    x_max = max(end for _, end in windows)
    y_vals = [value for _, value in points]
    y_min = min(y_vals + [0.0])
    y_max = max(y_vals + [0.0])
    if y_max <= y_min:
        y_max = y_min + 1.0
    margin_left, margin_right = 48, 12
    margin_top, margin_bottom = 16, 28
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * inner_w if x_max > x_min else margin_left

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    window_rects = [
        f'<rect x="{_x(start):.1f}" y="{margin_top}" width="{max(1.0, _x(end) - _x(start)):.1f}" '
        f'height="{inner_h}" fill="#f1eadc" opacity="0.4"></rect>'
        for start, end in windows
    ]
    split_markers = [
        f'<line x1="{_x(split):.1f}" y1="{margin_top}" x2="{_x(split):.1f}" y2="{height - margin_bottom}" '
        'stroke="#bba789" stroke-width="1" stroke-dasharray="3 3"></line>'
        for split in split_lines
    ]
    points_sorted = sorted(points, key=lambda item: item[0])
    path_points = " ".join(f"{_x(x):.1f},{_y(y):.1f}" for x, y in points_sorted)
    markers = [
        f'<circle cx="{_x(x):.1f}" cy="{_y(y):.1f}" r="3" fill="#4f7a67"></circle>' for x, y in points_sorted
    ]
    x_ticks = _axis_ticks(float(x_min), float(x_max), 5)
    y_ticks = _axis_ticks(y_min, y_max, 5)
    x_labels = [
        f'<text x="{_x(tick):.1f}" y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick)}</text>'
        for tick in x_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_pct(tick) if as_pct else format_float(tick)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        'stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]
    zero_line = ""
    if y_min < 0.0 < y_max:
        zero_line = f'<line x1="{margin_left}" y1="{_y(0.0):.1f}" x2="{width - margin_right}" y2="{_y(0.0):.1f}" stroke="#b58a60" stroke-width="1.4"></line>'

    gate_line = ""
    gate_label = ""
    if gate_threshold is not None and np.isfinite(gate_threshold) and y_min < gate_threshold < y_max:
        gate_line = (
            f'<line x1="{margin_left}" y1="{_y(gate_threshold):.1f}" x2="{width - margin_right}" y2="{_y(gate_threshold):.1f}" '
            'stroke="#b06b1d" stroke-width="1.4" stroke-dasharray="5 4"></line>'
        )
        gate_label = (
            f'<text x="{width - margin_right}" y="{_y(gate_threshold) - 4:.1f}" text-anchor="end">Gate {format_float(gate_threshold)}</text>'
        )

    note_parts = [
        "X-axis = validation window midpoint (year).",
        f"Y-axis = {title} value.",
        "Dots = fold results; line connects folds in time.",
        "Shaded bands = validation window.",
        "Vertical dashed lines = train → validation split (validation start).",
        "Zero line = break-even when visible.",
        "More dots above zero = stronger out-of-sample.",
    ]
    if gate_threshold is not None:
        note_parts.append("Horizontal dashed line = Sharpe gate (if shown).")
    note = '<div class="chart-note">' + " ".join(note_parts) + "</div>"

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + "".join(window_rects)
        + "".join(split_markers)
        + zero_line
        + gate_line
        + f'<polyline points="{path_points}" fill="none" stroke="#4f7a67" stroke-width="1.2"></polyline>'
        + "".join(markers)
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 4}" text-anchor="start">{title}</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 4}" text-anchor="middle">Validation window (mid-year)</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + gate_label
        + "</svg>"
        + note
        + "</div>"
    )


def render_walkforward_compare_svg(
    folds: list[dict[str, Any]] | None,
    *,
    train_key: str,
    validate_key: str,
    benchmark_key: str | None = None,
    title: str,
    as_pct: bool,
    gate_threshold: float | None = None,
    train_value_fn: Callable[[dict[str, Any]], float | None] | None = None,
    validate_value_fn: Callable[[dict[str, Any]], float | None] | None = None,
    benchmark_value_fn: Callable[[dict[str, Any]], float | None] | None = None,
    benchmark_label_text: str = config.PRIMARY_PASSIVE_BENCHMARK,
    width: int = 680,
    height: int = 210,
) -> str:
    if not folds:
        return '<div class="muted">Walk-forward data not available.</div>'
    train_points: list[tuple[float, float]] = []
    validate_points: list[tuple[float, float]] = []
    benchmark_points: list[tuple[float, float]] = []
    windows: list[tuple[int, int]] = []
    split_lines: list[int] = []
    for fold in folds:
        window = fold.get("validate_window") or {}
        start = window.get("start")
        end = window.get("end")
        train_val = train_value_fn(fold) if train_value_fn else fold.get(train_key)
        validate_val = validate_value_fn(fold) if validate_value_fn else fold.get(validate_key)
        benchmark_val = benchmark_value_fn(fold) if benchmark_value_fn else (fold.get(benchmark_key) if benchmark_key else None)
        if start and end and validate_val is not None and np.isfinite(validate_val):
            start_ord = _month_to_ordinal(start)
            end_ord = _month_to_ordinal(end)
            mid = (start_ord + end_ord) / 2.0
            validate_points.append((mid, float(validate_val)))
            if train_val is not None and np.isfinite(train_val):
                train_points.append((mid, float(train_val)))
            if benchmark_val is not None and np.isfinite(benchmark_val):
                benchmark_points.append((mid, float(benchmark_val)))
            windows.append((start_ord, end_ord))
            split_lines.append(start_ord)
    if not validate_points:
        return '<div class="muted">Walk-forward data not available.</div>'
    x_min = min(start for start, _ in windows)
    x_max = max(end for _, end in windows)
    y_vals = [value for _, value in validate_points] + [value for _, value in train_points] + [value for _, value in benchmark_points]
    y_min = min(y_vals + [0.0])
    y_max = max(y_vals + [0.0])
    if y_max <= y_min:
        y_max = y_min + 1.0
    margin_left, margin_right = 48, 12
    margin_top, margin_bottom = 16, 28
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * inner_w if x_max > x_min else margin_left

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    window_rects = [
        f'<rect x="{_x(start):.1f}" y="{margin_top}" width="{max(1.0, _x(end) - _x(start)):.1f}" '
        f'height="{inner_h}" fill="#f1eadc" opacity="0.4"></rect>'
        for start, end in windows
    ]
    split_markers = [
        f'<line x1="{_x(split):.1f}" y1="{margin_top}" x2="{_x(split):.1f}" y2="{height - margin_bottom}" '
        'stroke="#bba789" stroke-width="1" stroke-dasharray="3 3"></line>'
        for split in split_lines
    ]

    train_sorted = sorted(train_points, key=lambda item: item[0])
    validate_sorted = sorted(validate_points, key=lambda item: item[0])
    benchmark_sorted = sorted(benchmark_points, key=lambda item: item[0])
    train_path = " ".join(f"{_x(x):.1f},{_y(y):.1f}" for x, y in train_sorted) if train_sorted else ""
    validate_path = " ".join(f"{_x(x):.1f},{_y(y):.1f}" for x, y in validate_sorted)
    benchmark_path = " ".join(f"{_x(x):.1f},{_y(y):.1f}" for x, y in benchmark_sorted) if benchmark_sorted else ""
    train_markers = [
        f'<circle cx="{_x(x):.1f}" cy="{_y(y):.1f}" r="2.6" fill="#7c6f8c"></circle>'
        for x, y in train_sorted
    ]
    validate_markers = [
        f'<circle cx="{_x(x):.1f}" cy="{_y(y):.1f}" r="3" fill="#4f7a67"></circle>'
        for x, y in validate_sorted
    ]
    benchmark_markers = [
        f'<rect x="{_x(x) - 2.2:.1f}" y="{_y(y) - 2.2:.1f}" width="4.4" height="4.4" fill="#a87d3f"></rect>'
        for x, y in benchmark_sorted
    ]
    x_ticks = _axis_ticks(float(x_min), float(x_max), 5)
    y_ticks = _axis_ticks(y_min, y_max, 5)
    x_labels = [
        f'<text x="{_x(tick):.1f}" y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick)}</text>'
        for tick in x_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_pct(tick) if as_pct else format_float(tick)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        'stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]

    zero_line = ""
    if y_min < 0.0 < y_max:
        zero_line = f'<line x1="{margin_left}" y1="{_y(0.0):.1f}" x2="{width - margin_right}" y2="{_y(0.0):.1f}" stroke="#b58a60" stroke-width="1.4"></line>'

    gate_line = ""
    if gate_threshold is not None and np.isfinite(gate_threshold) and y_min < gate_threshold < y_max:
        gate_line = (
            f'<line x1="{margin_left}" y1="{_y(gate_threshold):.1f}" x2="{width - margin_right}" y2="{_y(gate_threshold):.1f}" '
            'stroke="#b06b1d" stroke-width="1.4" stroke-dasharray="5 4"></line>'
        )

    legend_items = [
        '<span style="white-space:nowrap;display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;">'
        '<span style="display:inline-block;width:18px;border-top:3px solid #4f7a67;"></span>'
        'Validation (OOS)</span>',
        '<span style="white-space:nowrap;display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;">'
        '<span style="display:inline-block;width:18px;border-top:3px dashed #7c6f8c;"></span>'
        'Train (IS)</span>',
    ]
    if benchmark_path:
        legend_items.append(
            '<span style="white-space:nowrap;display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;">'
            '<span style="display:inline-block;width:18px;border-top:3px dashed #a87d3f;"></span>'
            f'{benchmark_label_text}</span>'
        )
    if gate_line:
        legend_items.append(
            '<span style="white-space:nowrap;display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;">'
            '<span style="display:inline-block;width:18px;border-top:3px dashed #b06b1d;"></span>'
            f'Gate {format_float(gate_threshold)}</span>'
        )

    context_parts = [
        "Shaded bands = validation window.",
        "Vertical dashed lines = train to validation split.",
    ]
    note = (
        '<div class="chart-note"><strong>Line key:</strong> '
        + "".join(legend_items)
        + "</div>"
        + '<div class="chart-note">'
        + " ".join(context_parts)
        + "</div>"
    )

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + "".join(window_rects)
        + "".join(split_markers)
        + zero_line
        + gate_line
        + (f'<polyline points="{train_path}" fill="none" stroke="#7c6f8c" stroke-width="1.2" stroke-dasharray="4 4"></polyline>' if train_path else "")
        + (f'<polyline points="{benchmark_path}" fill="none" stroke="#a87d3f" stroke-width="1.2" stroke-dasharray="7 3 1.5 3"></polyline>' if benchmark_path else "")
        + f'<polyline points="{validate_path}" fill="none" stroke="#4f7a67" stroke-width="1.6"></polyline>'
        + "".join(train_markers)
        + "".join(benchmark_markers)
        + "".join(validate_markers)
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 4}" text-anchor="start">{title}</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 4}" text-anchor="middle">Validation window (mid-year)</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + "</svg>"
        + note
        + "</div>"
    )


def walk_forward_gap_months_summary(folds: list[dict[str, Any]] | None) -> str:
    if not folds:
        return "n/a"
    gaps: list[int] = []
    for fold in folds:
        train_end = (fold.get("train_window") or {}).get("end")
        validate_start = (fold.get("validate_window") or {}).get("start")
        if not train_end or not validate_start:
            continue
        gap = _month_to_ordinal(validate_start) - _month_to_ordinal(train_end) - 1
        gaps.append(gap)
    if not gaps:
        return "n/a"
    gaps.sort()
    mid = gaps[len(gaps) // 2]
    return f"gap months: min {min(gaps)}, median {mid}, max {max(gaps)}"


def _build_holdout_walkforward_folds(
    *,
    start_month: str | None,
    end_month: str | None,
    fold_count: int = 5,
    train_start: str | None = None,
) -> list[dict[str, Any]]:
    if not start_month or not end_month:
        return []
    start_ord = _month_to_ordinal(start_month)
    end_ord = _month_to_ordinal(end_month)
    total_months = end_ord - start_ord + 1
    if total_months <= 0:
        return []
    fold_count = max(1, min(fold_count, total_months))
    base_len = total_months // fold_count
    remainder = total_months % fold_count
    current_start = start_ord
    if not train_start:
        train_start = config.ROLLING_ORIGIN_FOLDS[0][1] if config.ROLLING_ORIGIN_FOLDS else start_month
    train_start_ord = _month_to_ordinal(train_start)
    folds: list[dict[str, Any]] = []
    for idx in range(fold_count):
        length = base_len + (1 if idx < remainder else 0)
        validate_start_ord = current_start
        validate_end_ord = current_start + length - 1
        train_end_ord = max(train_start_ord, validate_start_ord - 1)
        folds.append(
            {
                "fold_id": f"holdout_fold_{idx + 1}",
                "train_window": {
                    "start": train_start,
                    "end": _ordinal_to_month(train_end_ord),
                },
                "validate_window": {
                    "start": _ordinal_to_month(validate_start_ord),
                    "end": _ordinal_to_month(validate_end_ord),
                },
            }
        )
        current_start = validate_end_ord + 1
    return folds


def clip_walkforward_folds_to_window(
    folds: list[dict[str, Any]] | None,
    *,
    clip_start: str | None,
    clip_end: str | None,
) -> list[dict[str, Any]]:
    if not folds or not clip_start or not clip_end:
        return list(folds or [])
    try:
        clip_start_ord = _month_to_ordinal(clip_start)
        clip_end_ord = _month_to_ordinal(clip_end)
    except (TypeError, ValueError):
        return list(folds)
    if clip_end_ord < clip_start_ord:
        return list(folds)
    clipped: list[dict[str, Any]] = []
    for fold in folds:
        train = fold.get("train_window") or {}
        validate = fold.get("validate_window") or {}
        if not train.get("start") or not train.get("end") or not validate.get("start") or not validate.get("end"):
            clipped.append(fold)
            continue
        try:
            train_start_ord = _month_to_ordinal(train["start"])
            train_end_ord = _month_to_ordinal(train["end"])
            validate_start_ord = _month_to_ordinal(validate["start"])
            validate_end_ord = _month_to_ordinal(validate["end"])
        except (TypeError, ValueError):
            clipped.append(fold)
            continue
        train_start_ord = max(train_start_ord, clip_start_ord)
        train_end_ord = min(train_end_ord, clip_end_ord)
        if train_end_ord < train_start_ord:
            train_end_ord = train_start_ord
        validate_start_ord = max(validate_start_ord, clip_start_ord)
        validate_end_ord = min(validate_end_ord, clip_end_ord)
        if validate_end_ord < validate_start_ord:
            continue
        clipped.append(
            {
                **fold,
                "train_window": {
                    "start": _ordinal_to_month(train_start_ord),
                    "end": _ordinal_to_month(train_end_ord),
                },
                "validate_window": {
                    "start": _ordinal_to_month(validate_start_ord),
                    "end": _ordinal_to_month(validate_end_ord),
                },
            }
        )
    return clipped


def render_walkforward_schedule_svg(
    folds: list[dict[str, Any]] | None,
    *,
    title: str = "Walk-Forward Schedule",
    width: int = 720,
    row_height: int = 16,
    row_gap: int = 8,
) -> str:
    if not folds:
        return '<div class="muted">Walk-forward schedule not available.</div>'
    valid_folds = []
    for fold in folds:
        train = fold.get("train_window") or {}
        validate = fold.get("validate_window") or {}
        if train.get("start") and train.get("end") and validate.get("start") and validate.get("end"):
            valid_folds.append(fold)
    if not valid_folds:
        return '<div class="muted">Walk-forward schedule not available.</div>'

    def _fold_start(fold: dict[str, Any]) -> int:
        return _month_to_ordinal(fold["train_window"]["start"])

    ordered = sorted(valid_folds, key=_fold_start)
    min_start = min(_month_to_ordinal(fold["train_window"]["start"]) for fold in ordered)
    max_end = max(_month_to_ordinal(fold["validate_window"]["end"]) for fold in ordered)
    span = max(1, max_end - min_start + 1)

    margin_left = 86
    margin_right = 12
    margin_top = 20
    margin_bottom = 38
    inner_w = width - margin_left - margin_right
    inner_h = len(ordered) * row_height + (len(ordered) - 1) * row_gap
    height = margin_top + inner_h + margin_bottom

    def _x(ordinal: int) -> float:
        return margin_left + (ordinal - min_start) / span * inner_w

    rows = []
    labels = []
    for idx, fold in enumerate(ordered, start=1):
        row_y = margin_top + (idx - 1) * (row_height + row_gap)
        train = fold["train_window"]
        validate = fold["validate_window"]
        train_start = _month_to_ordinal(train["start"])
        train_end = _month_to_ordinal(train["end"])
        validate_start = _month_to_ordinal(validate["start"])
        validate_end = _month_to_ordinal(validate["end"])

        dropped_w = max(0.0, _x(train_start) - _x(min_start))
        if dropped_w > 0.1:
            rows.append(
                f'<rect x="{_x(min_start):.1f}" y="{row_y:.1f}" width="{dropped_w:.1f}" height="{row_height}" fill="#f3eadb"></rect>'
            )
        train_w = max(1.0, _x(train_end + 1) - _x(train_start))
        rows.append(
            f'<rect x="{_x(train_start):.1f}" y="{row_y:.1f}" width="{train_w:.1f}" height="{row_height}" fill="#9b8872"></rect>'
        )
        validate_w = max(1.0, _x(validate_end + 1) - _x(validate_start))
        rows.append(
            f'<rect x="{_x(validate_start):.1f}" y="{row_y:.1f}" width="{validate_w:.1f}" height="{row_height}" fill="#c27c3a"></rect>'
        )
        labels.append(
            f'<text x="{margin_left - 8}" y="{row_y + row_height - 3:.1f}" text-anchor="end">Pass {idx}</text>'
        )

    x_ticks = _axis_ticks(float(min_start), float(max_end), 6)
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
        '<span style="display:inline-block;width:12px;height:12px;background:#f3eadb;margin-right:6px;border-radius:2px;"></span> Dropped '
        '<span style="display:inline-block;width:12px;height:12px;background:#9b8872;margin:0 6px 0 12px;border-radius:2px;"></span> Training (IS) '
        '<span style="display:inline-block;width:12px;height:12px;background:#c27c3a;margin:0 6px 0 12px;border-radius:2px;"></span> Validation (OOS) '
        "</div>"
    )

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(rows)
        + "".join(labels)
        + x_axis
        + "".join(x_labels)
        + "</svg>"
        + legend
        + "</div>"
    )


def render_phase_boundary_svg(
    segments: list[dict[str, Any]] | None,
    *,
    title: str = "Test Window vs What Comes Next",
    width: int = 760,
    future_months: int = 24,
    footer_note: str | None = None,
) -> str:
    if not segments:
        return '<div class="muted">Phase boundary map not available.</div>'

    normalized: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        start = segment.get("start")
        if not start:
            continue
        end = segment.get("end")
        if not end:
            end = _ordinal_to_month(_month_to_ordinal(start) + future_months - 1)
        normalized.append(
            {
                "label": str(segment.get("label") or f"Segment {idx + 1}"),
                "detail": str(segment.get("detail") or ""),
                "start": start,
                "end": end,
                "fill": str(segment.get("fill") or "#e7dcc9"),
                "stroke": str(segment.get("stroke") or "#8f7b5f"),
                "highlight": bool(segment.get("highlight")),
            }
        )
    if not normalized:
        return '<div class="muted">Phase boundary map not available.</div>'

    min_start = min(_month_to_ordinal(segment["start"]) for segment in normalized)
    max_end = max(_month_to_ordinal(segment["end"]) for segment in normalized)
    span = max(1, max_end - min_start + 1)

    margin_left = 28
    margin_right = 12
    margin_top = 22
    margin_bottom = 42
    bar_height = 54
    width_inner = width - margin_left - margin_right
    height = margin_top + bar_height + margin_bottom

    def _x(ordinal: int) -> float:
        return margin_left + (ordinal - min_start) / span * width_inner

    axis_ticks = _axis_ticks(float(min_start), float(max_end), 6)
    tick_lines = []
    tick_labels = []
    for tick in axis_ticks:
        x_value = _x(int(round(tick)))
        tick_lines.append(
            f'<line x1="{x_value:.1f}" y1="{margin_top + bar_height + 2}" '
            f'x2="{x_value:.1f}" y2="{height - margin_bottom + 8}" stroke="#d9cbb6" stroke-width="1"></line>'
        )
        tick_labels.append(
            f'<text x="{x_value:.1f}" y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick)}</text>'
        )

    rects = []
    boundary_lines = []
    inline_labels = []
    legend_rows = []
    for index, segment in enumerate(normalized, start=1):
        start_ord = _month_to_ordinal(segment["start"])
        end_ord = _month_to_ordinal(segment["end"])
        x_value = _x(start_ord)
        rect_width = max(1.0, _x(end_ord + 1) - x_value)
        stroke_width = 2 if segment["highlight"] else 1
        rects.append(
            f'<rect x="{x_value:.1f}" y="{margin_top:.1f}" width="{rect_width:.1f}" height="{bar_height}" '
            f'fill="{segment["fill"]}" stroke="{segment["stroke"]}" stroke-width="{stroke_width}"></rect>'
        )
        if index < len(normalized):
            boundary_x = _x(end_ord + 1)
            boundary_lines.append(
                f'<line x1="{boundary_x:.1f}" y1="{margin_top - 6}" x2="{boundary_x:.1f}" '
                f'y2="{margin_top + bar_height + 10}" stroke="#17211a" stroke-width="3"></line>'
            )
        if rect_width >= 112:
            center_x = x_value + rect_width / 2
            inline_labels.append(
                f'<text x="{center_x:.1f}" y="{margin_top + 24:.1f}" text-anchor="middle" '
                f'font-size="11" font-weight="700">{html.escape(segment["label"])}</text>'
            )
            if segment["detail"]:
                inline_labels.append(
                    f'<text x="{center_x:.1f}" y="{margin_top + 38:.1f}" text-anchor="middle" '
                    f'font-size="9">{html.escape(segment["detail"])}</text>'
                )
        legend_rows.append(
            '<div class="chart-note">'
            f'<span style="display:inline-block;width:12px;height:12px;background:{segment["fill"]};'
            'margin-right:6px;border-radius:2px;"></span>'
            f'<strong>{html.escape(segment["label"])}</strong>: {html.escape(segment["start"])} to '
            f'{html.escape(segment["end"])}'
            + (f" - {html.escape(segment['detail'])}" if segment["detail"] else "")
            + "</div>"
        )

    footer_html = (
        f'<div class="chart-note"><strong>What this means:</strong> {html.escape(footer_note)}</div>'
        if footer_note
        else ""
    )
    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(rects)
        + "".join(boundary_lines)
        + "".join(tick_lines)
        + "".join(inline_labels)
        + "".join(tick_labels)
        + "</svg>"
        + "".join(legend_rows)
        + footer_html
        + "</div>"
    )


def render_walkforward_ladder_svg(
    folds: list[dict[str, Any]] | None,
    *,
    title: str = "Rolling Walk-Forward Mechanics",
    stitched_label: str = "Stitched OOS record",
    width: int = 760,
    row_height: int = 16,
    row_gap: int = 8,
) -> str:
    if not folds:
        return '<div class="muted">Walk-forward ladder not available.</div>'
    valid_folds: list[dict[str, Any]] = []
    for fold in folds:
        train = fold.get("train_window") or {}
        validate = fold.get("validate_window") or {}
        if train.get("start") and train.get("end") and validate.get("start") and validate.get("end"):
            valid_folds.append(fold)
    if not valid_folds:
        return '<div class="muted">Walk-forward ladder not available.</div>'

    ordered = sorted(valid_folds, key=lambda fold: _month_to_ordinal((fold.get("validate_window") or {}).get("start")))
    min_start = min(_month_to_ordinal((fold["train_window"])["start"]) for fold in ordered)
    max_end = max(_month_to_ordinal((fold["validate_window"])["end"]) for fold in ordered)
    span = max(1, max_end - min_start + 1)

    margin_left = 118
    margin_right = 12
    margin_top = 20
    margin_bottom = 40
    row_count = len(ordered) + 1
    inner_w = width - margin_left - margin_right
    inner_h = row_count * row_height + (row_count - 1) * row_gap
    height = margin_top + inner_h + margin_bottom

    def _x(ordinal: int) -> float:
        return margin_left + (ordinal - min_start) / span * inner_w

    grid_lines = []
    grid_labels = []
    for tick in _axis_ticks(float(min_start), float(max_end), 6):
        x_value = _x(int(round(tick)))
        grid_lines.append(
            f'<line x1="{x_value:.1f}" y1="{margin_top - 2}" x2="{x_value:.1f}" '
            f'y2="{height - margin_bottom + 6}" stroke="#e1d5c3" stroke-width="1" stroke-dasharray="2 3"></line>'
        )
        grid_labels.append(
            f'<text x="{x_value:.1f}" y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick)}</text>'
        )

    rows = []
    labels = []
    for idx, fold in enumerate(ordered, start=1):
        row_y = margin_top + (idx - 1) * (row_height + row_gap)
        train = fold["train_window"]
        validate = fold["validate_window"]
        train_start = _month_to_ordinal(train["start"])
        train_end = _month_to_ordinal(train["end"])
        validate_start = _month_to_ordinal(validate["start"])
        validate_end = _month_to_ordinal(validate["end"])
        train_w = max(1.0, _x(train_end + 1) - _x(train_start))
        validate_w = max(1.0, _x(validate_end + 1) - _x(validate_start))
        rows.append(
            f'<rect x="{_x(train_start):.1f}" y="{row_y:.1f}" width="{train_w:.1f}" height="{row_height}" '
            'fill="#9b8872" opacity="0.95"></rect>'
        )
        rows.append(
            f'<rect x="{_x(validate_start):.1f}" y="{row_y:.1f}" width="{validate_w:.1f}" height="{row_height}" '
            'fill="#c27c3a"></rect>'
        )
        labels.append(
            f'<text x="{margin_left - 8}" y="{row_y + row_height - 3:.1f}" text-anchor="end">Pass {idx}</text>'
        )

    stitched_row_y = margin_top + (len(ordered)) * (row_height + row_gap)
    stitched_label_safe = html.escape(stitched_label)
    labels.append(
        f'<text x="{margin_left - 8}" y="{stitched_row_y + row_height - 3:.1f}" text-anchor="end">{stitched_label_safe}</text>'
    )
    for fold in ordered:
        validate = fold["validate_window"]
        validate_start = _month_to_ordinal(validate["start"])
        validate_end = _month_to_ordinal(validate["end"])
        validate_w = max(1.0, _x(validate_end + 1) - _x(validate_start))
        rows.append(
            f'<rect x="{_x(validate_start):.1f}" y="{stitched_row_y:.1f}" width="{validate_w:.1f}" height="{row_height}" '
            'fill="#d79a5c"></rect>'
        )

    note = (
        '<div class="chart-note"><strong>What this means:</strong> Each pass keeps the same anchor, expands train history, and '
        'tests only the next unseen slice. The bottom row is the realistic out-of-sample stream formed by stitching '
        'those orange slices together.</div>'
    )
    legend = (
        '<div class="chart-note">'
        '<span style="display:inline-block;width:12px;height:12px;background:#9b8872;margin-right:6px;border-radius:2px;"></span> Training history '
        '<span style="display:inline-block;width:12px;height:12px;background:#c27c3a;margin:0 6px 0 12px;border-radius:2px;"></span> New OOS slice '
        '<span style="display:inline-block;width:12px;height:12px;background:#d79a5c;margin:0 6px 0 12px;border-radius:2px;"></span> Stitched OOS record'
        '</div>'
    )
    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(grid_lines)
        + "".join(rows)
        + "".join(labels)
        + "".join(grid_labels)
        + "</svg>"
        + legend
        + note
        + "</div>"
    )


def _walk_forward_bounds(
    folds: list[dict[str, Any]] | None,
) -> tuple[int | None, int | None, str | None, str | None]:
    if not folds:
        return (None, None, None, None)
    starts: list[str] = []
    ends: list[str] = []
    for fold in folds:
        train_window = fold.get("train_window") or {}
        validate_window = fold.get("validate_window") or {}
        start = train_window.get("start") or validate_window.get("start")
        end = validate_window.get("end") or train_window.get("end")
        if start:
            starts.append(start)
        if end:
            ends.append(end)
    if not starts or not ends:
        return (None, None, None, None)
    start_ord = min(_month_to_ordinal(start) for start in starts)
    end_ord = max(_month_to_ordinal(end) for end in ends)
    return (start_ord, end_ord, min(starts), max(ends))


def _walk_forward_validate_bounds(
    folds: list[dict[str, Any]] | None,
) -> tuple[int | None, int | None, str | None, str | None]:
    if not folds:
        return (None, None, None, None)
    starts: list[str] = []
    ends: list[str] = []
    for fold in folds:
        validate_window = fold.get("validate_window") or {}
        start = validate_window.get("start")
        end = validate_window.get("end")
        if start:
            starts.append(start)
        if end:
            ends.append(end)
    if not starts or not ends:
        return _walk_forward_bounds(folds)
    start_ord = min(_month_to_ordinal(start) for start in starts)
    end_ord = max(_month_to_ordinal(end) for end in ends)
    return (start_ord, end_ord, min(starts), max(ends))


def render_walkforward_oos_equity_svg(
    returns: list[float] | None,
    folds: list[dict[str, Any]] | None,
    *,
    benchmark_returns: list[float] | None = None,
    benchmark_label: str | None = None,
    title: str = "Validation Equity (Stitched)",
    width: int = 680,
    height: int = 220,
) -> str:
    if not returns:
        return '<div class="muted">Validation equity not available.</div>'
    equity = _returns_to_equity(returns, base=100.0)
    if not equity:
        return '<div class="muted">Validation equity not available.</div>'
    benchmark_equity = _returns_to_equity(benchmark_returns, base=100.0) if benchmark_returns else []

    start_ord, end_ord, _, _ = _walk_forward_validate_bounds(folds)
    total_len = len(equity)
    x_min = start_ord if start_ord is not None else 0
    if start_ord is not None and end_ord is not None:
        x_max = max(end_ord, start_ord + total_len - 1)
    else:
        x_max = (start_ord + total_len - 1) if start_ord is not None else (total_len - 1)
    if x_max <= x_min:
        x_max = x_min + 1
    margin_left, margin_right = 48, 12
    margin_top, margin_bottom = 16, 28
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x_ordinal(ordinal: int) -> float:
        return margin_left + (ordinal - x_min) / (x_max - x_min) * inner_w

    def _x_index(index: int) -> float:
        ordinal = (start_ord + index) if start_ord is not None else index
        return _x_ordinal(ordinal)

    plotted_series = equity + benchmark_equity
    y_min = min(plotted_series)
    y_max = max(plotted_series)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    windows: list[tuple[int, int]] = []
    split_lines: list[int] = []
    if folds:
        for fold in folds:
            window = fold.get("validate_window") or {}
            start = window.get("start")
            end = window.get("end")
            if start and end:
                start_ord_window = _month_to_ordinal(start)
                end_ord_window = _month_to_ordinal(end)
                windows.append((start_ord_window, end_ord_window))
                split_lines.append(start_ord_window)

    window_rects = [
        f'<rect x="{_x_ordinal(start):.1f}" y="{margin_top}" width="{max(1.0, _x_ordinal(end) - _x_ordinal(start)):.1f}" '
        f'height="{inner_h}" fill="#f1eadc" opacity="0.35"></rect>'
        for start, end in windows
    ] if windows and start_ord is not None else []
    split_markers = [
        f'<line x1="{_x_ordinal(split):.1f}" y1="{margin_top}" x2="{_x_ordinal(split):.1f}" y2="{height - margin_bottom}" '
        'stroke="#bba789" stroke-width="1" stroke-dasharray="3 3"></line>'
        for split in split_lines
    ] if split_lines and start_ord is not None else []

    points = " ".join(f"{_x_index(i):.1f},{_y(v):.1f}" for i, v in enumerate(equity))
    benchmark_points = (
        " ".join(f"{_x_index(i):.1f},{_y(v):.1f}" for i, v in enumerate(benchmark_equity))
        if benchmark_equity
        else ""
    )
    x_ticks = _axis_ticks(float(x_min), float(x_max), 5)
    y_ticks = _axis_ticks(y_min, y_max, 5)
    x_labels = [
        f'<text x="{_x_ordinal(int(round(tick))):.1f}" y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick) if start_ord is not None else int(round(tick))}</text>'
        for tick in x_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_float(tick, 2)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        'stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]

    note = (
        '<div class="chart-note">'
        + "Stitched equity from validation windows only (out-of-sample). "
        + "Shaded bands mark validation folds."
        + (
            f" Dashed line = {benchmark_label or config.PRIMARY_PASSIVE_BENCHMARK} on the same stitched timeline."
            if benchmark_equity
            else ""
        )
        + "</div>"
    )

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + "".join(window_rects)
        + "".join(split_markers)
        + (
            f'<polyline points="{benchmark_points}" fill="none" stroke="#b06b1d" stroke-width="1.4" '
            'stroke-dasharray="5 4" opacity="0.95"></polyline>'
            if benchmark_points
            else ""
        )
        + f'<polyline points="{points}" fill="none" stroke="#4f7a67" stroke-width="1.8"></polyline>'
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 4}" text-anchor="start">Equity (start=100)</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 4}" text-anchor="middle">Validation timeline</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + "</svg>"
        + note
        + "</div>"
    )


def render_holdout_equity_svg(
    returns: list[float] | None,
    benchmark: list[float] | None = None,
    *,
    window_start: str | None = None,
    benchmark_label: str | None = None,
    title: str = "Holdout Equity (Primary Track)",
    width: int = 680,
    height: int = 220,
) -> str:
    if not returns:
        return '<div class="muted">Holdout equity not available.</div>'

    equity = _returns_to_equity(returns, base=100.0)
    benchmark_equity = _returns_to_equity(benchmark, base=100.0) if benchmark else []
    series = equity + (benchmark_equity or [])
    if not series:
        return '<div class="muted">Holdout equity not available.</div>'

    start_ord = _month_to_ordinal(window_start) if window_start else None
    x_min = start_ord if start_ord is not None else 0
    x_max = (start_ord + len(equity) - 1) if start_ord is not None else (len(equity) - 1)
    if x_max <= x_min:
        x_max = x_min + 1
    margin_left, margin_right = 72, 16
    margin_top, margin_bottom = 18, 44
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x_index(index: int) -> float:
        ordinal = (start_ord + index) if start_ord is not None else index
        return margin_left + (ordinal - x_min) / (x_max - x_min) * inner_w

    y_min = min(series)
    y_max = max(series)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    strategy_points = " ".join(f"{_x_index(i):.1f},{_y(v):.1f}" for i, v in enumerate(equity))
    benchmark_points = (
        " ".join(f"{_x_index(i):.1f},{_y(v):.1f}" for i, v in enumerate(benchmark_equity))
        if benchmark_equity
        else ""
    )
    x_ticks = _axis_ticks(float(x_min), float(x_max), 5)
    y_ticks = _axis_ticks(y_min, y_max, 5)
    x_labels = [
        f'<text x="{margin_left + (tick - x_min) / (x_max - x_min) * inner_w:.1f}" y="{height - 20}" text-anchor="middle">'
        f'{_ordinal_to_year_label(tick) if start_ord is not None else int(round(tick))}</text>'
        for tick in x_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_float(tick, 2)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        'stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]
    benchmark_poly = (
        f'<polyline points="{benchmark_points}" fill="none" stroke="#9a8c6a" stroke-width="2" stroke-dasharray="4 4"></polyline>'
        if benchmark_points
        else ""
    )
    note = (
        '<div class="chart-note">'
        f'Orange = strategy. Dashed line = {html.escape(benchmark_label or config.PRIMARY_PASSIVE_BENCHMARK)} when available. '
        'X-axis shows the holdout timeline; Y-axis is the equity index starting at 100.'
        '</div>'
    )

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + benchmark_poly
        + f'<polyline points="{strategy_points}" fill="none" stroke="#b06b1d" stroke-width="2.4"></polyline>'
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 6}" text-anchor="start">Equity (start=100)</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 6}" text-anchor="middle">Holdout timeline</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + "</svg>"
        + note
        + "</div>"
    )


def render_walkforward_rolling_sharpe_svg(
    returns: list[float] | None,
    folds: list[dict[str, Any]] | None,
    *,
    periods_per_year: int,
    window: int = 12,
    title: str = "Rolling Sharpe (Validation, 12m)",
    width: int = 680,
    height: int = 210,
    gate_threshold: float | None = 0.4,
) -> str:
    if not returns or len(returns) < window:
        return '<div class="muted">Rolling Sharpe not available.</div>'
    rolling: list[tuple[int, float]] = []
    for idx in range(window - 1, len(returns)):
        window_returns = returns[idx + 1 - window : idx + 1]
        sharpe = annualized_sharpe_periods(window_returns, periods_per_year=periods_per_year)
        if sharpe is not None and np.isfinite(sharpe):
            rolling.append((idx, float(sharpe)))
    if not rolling:
        return '<div class="muted">Rolling Sharpe not available.</div>'

    start_ord, end_ord, _, _ = _walk_forward_validate_bounds(folds)
    x_min = start_ord if start_ord is not None else 0
    if start_ord is not None and end_ord is not None:
        x_max = max(end_ord, start_ord + len(returns) - 1)
    else:
        x_max = (start_ord + len(returns) - 1) if start_ord is not None else (len(returns) - 1)
    if x_max <= x_min:
        x_max = x_min + 1
    margin_left, margin_right = 48, 12
    margin_top, margin_bottom = 16, 28
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x_ordinal(ordinal: int) -> float:
        return margin_left + (ordinal - x_min) / (x_max - x_min) * inner_w

    def _x_index(index: int) -> float:
        ordinal = (start_ord + index) if start_ord is not None else index
        return _x_ordinal(ordinal)

    y_vals = [value for _, value in rolling]
    y_min = min(y_vals + [0.0])
    y_max = max(y_vals + [0.0])
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    path_points = " ".join(f"{_x_index(i):.1f},{_y(v):.1f}" for i, v in rolling)
    x_ticks = _axis_ticks(float(x_min), float(x_max), 5)
    y_ticks = _axis_ticks(y_min, y_max, 5)
    x_labels = [
        f'<text x="{_x_ordinal(int(round(tick))):.1f}" y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick) if start_ord is not None else int(round(tick))}</text>'
        for tick in x_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_float(tick)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        'stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]
    gate_line = ""
    if gate_threshold is not None and np.isfinite(gate_threshold) and y_min < gate_threshold < y_max:
        gate_line = (
            f'<line x1="{margin_left}" y1="{_y(gate_threshold):.1f}" x2="{width - margin_right}" y2="{_y(gate_threshold):.1f}" '
            'stroke="#b06b1d" stroke-width="1.4" stroke-dasharray="5 4"></line>'
        )

    note = (
        '<div class="chart-note">'
        f"Rolling {window}m Sharpe on validation-only returns. "
        "Line updates monthly."
        "</div>"
    )

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + gate_line
        + f'<polyline points="{path_points}" fill="none" stroke="#4f7a67" stroke-width="1.6"></polyline>'
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 4}" text-anchor="start">Sharpe</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 4}" text-anchor="middle">Validation timeline</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + "</svg>"
        + note
        + "</div>"
    )


def render_forward_simulation_svg(
    history: list[float] | None,
    paths: list[list[float]] | None,
    *,
    start_ord: int | None,
    split_index: int,
    split_label: str | None,
    history_label: str | None = None,
    history_is_oos: bool = True,
    title: str = "Forward Simulation (Monte Carlo)",
    width: int = 680,
    height: int = 220,
    max_lines: int = 70,
) -> str:
    if not history or not paths:
        return '<div class="muted">Forward simulation not available.</div>'
    if split_index <= 0:
        split_index = len(history)
    history = history[:split_index]
    if not history:
        return '<div class="muted">Forward simulation not available.</div>'
    last_equity = history[-1]
    proj_len = min(len(path) for path in paths if path) if paths else 0
    if proj_len < 2:
        return '<div class="muted">Forward simulation not available.</div>'

    scaled_paths: list[list[float]] = []
    for path in paths:
        if len(path) < proj_len:
            continue
        scaled = [last_equity * (value / 100.0) for value in path[:proj_len]]
        scaled_paths.append([last_equity] + scaled)
    if not scaled_paths:
        return '<div class="muted">Forward simulation not available.</div>'

    plot_paths = _sample_paths_evenly(scaled_paths, max_lines=max_lines)
    total_len = split_index + proj_len
    x_min = start_ord if start_ord is not None else 0
    x_max = (start_ord + total_len - 1) if start_ord is not None else (total_len - 1)
    if x_max <= x_min:
        x_max = x_min + 1
    margin_left, margin_right = 48, 12
    margin_top, margin_bottom = 16, 28
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x_index(index: int) -> float:
        ordinal = (start_ord + index) if start_ord is not None else index
        return margin_left + (ordinal - x_min) / (x_max - x_min) * inner_w

    proj_values: list[float] = []
    for path in scaled_paths:
        proj_values.extend(path)
    proj_sorted = sorted(proj_values) if proj_values else []
    cap_high = _quantile(proj_sorted, 0.90) if proj_sorted else None
    cap_low = _quantile(proj_sorted, 0.05) if proj_sorted else None
    history_min = min(history)
    history_max = max(history)
    y_min = min(history_min, cap_low) if cap_low is not None else history_min
    y_max = max(history_max, cap_high) if cap_high is not None else history_max
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    history_points = " ".join(f"{_x_index(i):.1f},{_y(v):.1f}" for i, v in enumerate(history))
    split_x = _x_index(split_index - 1)
    projection_width = max(1.0, _x_index(total_len - 1) - split_x)
    history_rect = ""
    if history_is_oos:
        history_rect = (
            f'<rect x="{margin_left}" y="{margin_top}" width="{max(1.0, split_x - margin_left):.1f}" '
            f'height="{inner_h}" fill="#e6f1ea" opacity="0.35"></rect>'
        )
    projection_rect = (
        f'<rect x="{split_x:.1f}" y="{margin_top}" width="{projection_width:.1f}" '
        f'height="{inner_h}" fill="#f1eadc" opacity="0.45"></rect>'
    )
    split_line = (
        f'<line x1="{split_x:.1f}" y1="{margin_top}" x2="{split_x:.1f}" y2="{height - margin_bottom}" '
        'stroke="#bba789" stroke-width="1.2" stroke-dasharray="4 4"></line>'
    )

    polylines = []
    color_count = max(1, len(plot_paths))
    for idx, path in enumerate(plot_paths):
        hue = 210 + (idx / max(1, color_count - 1)) * 120
        color = f"hsl({hue:.1f}, 38%, 56%)"
        points = " ".join(
            f"{_x_index(split_index - 1 + i):.1f},{_y(v):.1f}" for i, v in enumerate(path)
        )
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="0.85" opacity="0.35" stroke-linecap="round"></polyline>'
        )

    median_path: list[float] = []
    band_low: list[float] = []
    band_high: list[float] = []
    proj_steps = len(scaled_paths[0])
    for idx in range(proj_steps):
        values = sorted(path[idx] for path in scaled_paths)
        median_path.append(_quantile(values, 0.5))
        band_low.append(_quantile(values, 0.1))
        band_high.append(_quantile(values, 0.9))

    band_high_points = " ".join(
        f"{_x_index(split_index - 1 + i):.1f},{_y(v):.1f}" for i, v in enumerate(band_high)
    )
    band_low_points = " ".join(
        f"{_x_index(split_index - 1 + i):.1f},{_y(v):.1f}" for i, v in enumerate(band_low)
    )
    band_points = band_high_points + " " + " ".join(
        f"{_x_index(split_index - 1 + i):.1f},{_y(v):.1f}" for i, v in reversed(list(enumerate(band_low)))
    )
    median_points = " ".join(
        f"{_x_index(split_index - 1 + i):.1f},{_y(v):.1f}" for i, v in enumerate(median_path)
    )

    x_ticks = _axis_ticks(float(x_min), float(x_max), 5)
    y_ticks = _axis_ticks(y_min, y_max, 5)
    x_labels = [
        f'<text x="{_x_index(int(round(tick - x_min)) if start_ord is None else int(round(tick - x_min))):.1f}" '
        f'y="{height - 8}" text-anchor="middle">{_ordinal_to_year_label(tick) if start_ord is not None else int(round(tick))}</text>'
        for tick in x_ticks
    ]
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_float(tick, 2)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        'stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]

    split_label_text = split_label or "final validation"
    history_label = history_label or ("validation history (OOS)" if history_is_oos else "history")
    legend = (
        '<div class="chart-note">'
        f"Solid line = {history_label}. "
        "Fan = Monte Carlo projection (not a gate). "
        "Band = p10-p90; bold line = median path. "
        "Y-axis capped to p05-p90 for readability. "
        f"Anchor: {split_label_text}."
        "</div>"
    )

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + history_rect
        + projection_rect
        + split_line
        + f'<polyline points="{history_points}" fill="none" stroke="#4f7a67" stroke-width="1.8"></polyline>'
        + f'<polygon points="{band_points}" fill="#bfa47a" opacity="0.35"></polygon>'
        + f'<polyline points="{band_high_points}" fill="none" stroke="#a87d3f" stroke-width="1.6" opacity="0.75"></polyline>'
        + f'<polyline points="{band_low_points}" fill="none" stroke="#a87d3f" stroke-width="1.6" opacity="0.75"></polyline>'
        + "".join(polylines)
        + f'<polyline points="{median_points}" fill="none" stroke="#4f7a67" stroke-width="2.4"></polyline>'
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 4}" text-anchor="start">Equity (start=100)</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 4}" text-anchor="middle">Timeline (year)</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + "</svg>"
        + legend
        + "</div>"
    )


def render_forward_simulation_panel(summary: dict[str, Any]) -> str:
    # Removed from dashboards to avoid any future-looking presentation that could be
    # mistaken for a predictive or validation-grade result.
    return ""


def render_sensitivity_line_svg(
    points: list[tuple[str, float]],
    *,
    title: str,
    x_label: str = "Parameter value",
    width: int = 520,
    height: int = 180,
) -> str:
    if not points:
        return '<div class="muted">Sensitivity data not available.</div>'
    labels = [label for label, _ in points]
    y_vals = [value for _, value in points]
    y_min = min(y_vals + [0.0])
    y_max = max(y_vals + [0.0])
    if y_max <= y_min:
        y_max = y_min + 1.0
    margin_left, margin_right = 48, 16
    margin_top, margin_bottom = 16, 44
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def _x(idx: int) -> float:
        if len(points) == 1:
            return margin_left + inner_w / 2.0
        return margin_left + idx / (len(points) - 1) * inner_w

    def _y(value: float) -> float:
        return margin_top + inner_h - (value - y_min) / (y_max - y_min) * inner_h

    path_points = " ".join(f"{_x(i):.1f},{_y(value):.1f}" for i, (_, value) in enumerate(points))
    best_index = max(range(len(points)), key=lambda i: points[i][1])
    markers = []
    for i, (_, value) in enumerate(points):
        radius = 4 if i == best_index else 3
        markers.append(f'<circle cx="{_x(i):.1f}" cy="{_y(value):.1f}" r="{radius}" fill="#4f7a67"></circle>')
    x_labels = [
        f'<text x="{_x(i):.1f}" y="{height - 18}" text-anchor="middle">{label}</text>'
        for i, label in enumerate(labels)
    ]
    y_ticks = _axis_ticks(y_min, y_max, 4)
    y_labels = [
        f'<text x="{margin_left - 6}" y="{_y(tick):.1f}" text-anchor="end">{format_float(tick)}</text>'
        for tick in y_ticks
    ]
    y_grid = [
        f'<line x1="{margin_left}" y1="{_y(tick):.1f}" x2="{width - margin_right}" y2="{_y(tick):.1f}" '
        'stroke="#e4dbc9" stroke-width="1"></line>'
        for tick in y_ticks
    ]
    zero_line = ""
    if y_min < 0.0 < y_max:
        zero_line = f'<line x1="{margin_left}" y1="{_y(0.0):.1f}" x2="{width - margin_right}" y2="{_y(0.0):.1f}" stroke="#b58a60" stroke-width="1.4"></line>'

    return (
        '<div class="chart-block">'
        f'<div class="chart-title">{title}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        + "".join(y_grid)
        + zero_line
        + f'<polyline points="{path_points}" fill="none" stroke="#4f7a67" stroke-width="1.2"></polyline>'
        + "".join(markers)
        + f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#8f7b5f" stroke-width="1"></line>'
        + f'<text x="{margin_left}" y="{margin_top - 4}" text-anchor="start">Median validation Sharpe</text>'
        + f'<text x="{(margin_left + inner_w / 2):.1f}" y="{height - 4}" text-anchor="middle">{x_label}</text>'
        + "".join(x_labels)
        + "".join(y_labels)
        + "</svg>"
        + '<div class="chart-note">X-axis = parameter value; Y-axis = median validation Sharpe. Flatter line = more robust; larger dot = best value.</div>'
        "</div>"
    )


def render_sensitivity_sections(sensitivity: dict[str, Any] | None) -> str:
    if not sensitivity or sensitivity.get("status") != "ok":
        return '<p class="muted">Sensitivity data not available.</p>'
    parameters = sensitivity.get("parameters", {})
    sections: list[str] = []
    coverage_tested: list[str] = []
    coverage_fixed: list[str] = []
    def _coverage(label: str, info: dict[str, Any] | None) -> None:
        if not info:
            return
        by_value = info.get("by_value") or []
        if not by_value:
            return
        if len(by_value) > 1:
            coverage_tested.append(label)
        else:
            coverage_fixed.append(label)

    def _section(label: str, info: dict[str, Any] | None) -> None:
        if not info:
            return
        by_value = info.get("by_value") or []
        if not by_value:
            return
        points: list[tuple[str, float]] = []
        rows: list[str] = []
        for row in by_value:
            value_label = str(row.get("value", "n/a"))
            median_value = row.get("median_validation_sharpe")
            if median_value is not None and np.isfinite(median_value):
                points.append((value_label, float(median_value)))
            rows.append(
                "<tr>"
                f"<td>{value_label}</td>"
                f"<td>{format_float(median_value)}</td>"
                f"<td>{format_pct(row.get('hard_gate_pass_rate'))}</td>"
                f"<td>{row.get('count','n/a')}</td>"
                "</tr>"
            )
        summary_line = (
            f"Best {info.get('best_value','n/a')} · Worst {info.get('worst_value','n/a')} · "
            f"Spread {format_float(info.get('median_spread'))} · Corr {format_float(info.get('correlation'))}"
        )
        badge_label, badge_tone, badge_detail = sensitivity_badge(info)
        badge_html = render_badge(badge_label, badge_tone, badge_detail)
        if len(by_value) <= 1:
            return
        sections.append(
            "<div class=\"sens-section\">"
            f"<h3>{label}</h3>"
            f"<p class=\"muted\">{summary_line}</p>"
            f"<div class=\"badge-row\">{badge_html}</div>"
            "<p class=\"muted\">X = parameter value · Y = median validation Sharpe (higher is better). Flatter lines = more robust.</p>"
            f"{render_sensitivity_line_svg(points, title=f'{label} median Sharpe', x_label=label)}"
            "<table>"
            "<thead><tr><th>Value</th><th>Median Sharpe</th><th>Hard-Gate Pass Rate</th><th>Count</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
            "</div>"
        )

    _section("Lookback (L)", parameters.get("l"))
    _section("Skip", parameters.get("skip"))
    _section("Top N", parameters.get("top_n"))
    _section("Strategy", sensitivity.get("strategy_id"))
    _coverage("Lookback (L)", parameters.get("l"))
    _coverage("Skip", parameters.get("skip"))
    _coverage("Top N", parameters.get("top_n"))
    _coverage("Strategy", sensitivity.get("strategy_id"))
    if sections:
        tested = ", ".join(coverage_tested) if coverage_tested else "none"
        fixed = ", ".join(coverage_fixed) if coverage_fixed else "none"
        coverage_note = (
            f'<p class="muted"><strong>Coverage:</strong> Tested: {tested}. '
            f'Not tested (single value): {fixed}.</p>'
        )
        return coverage_note + "\n" + "\n".join(sections)
    if coverage_fixed and not coverage_tested:
        fixed = ", ".join(coverage_fixed)
        return (
            f'<p class="muted"><strong>Sensitivity skipped:</strong> no parameters varied in this run. '
            f'Fixed: {fixed}.</p>'
        )
    return '<p class="muted">Sensitivity data not available.</p>'


def pbo_band(pbo: float | None) -> str:
    if pbo is None or not np.isfinite(pbo):
        return "n/a"
    if pbo < config.PBO_THRESHOLD_MAX:
        return "good"
    if pbo < config.PBO_HARD_CUTOFF:
        return "caution"
    return "hard_cutoff"


def apply_pbo_policy(summary: dict[str, Any]) -> None:
    pbo_value = summary.get("backtest_overfitting", {}).get("pbo")
    band = pbo_band(pbo_value)
    summary["pbo_band"] = band
    summary["pbo_policy"] = {
        "good_max": config.PBO_THRESHOLD_MAX,
        "hard_cutoff_min": config.PBO_HARD_CUTOFF,
    }
    summary["pbo_caution"] = band == "caution"
    summary["pbo_hard_cutoff"] = band == "hard_cutoff"
    if band == "caution" and summary.get("selection_status") == "selected":
        summary["selection_status"] = "selected_with_caution"
    if band == "hard_cutoff":
        summary["selection_status"] = "pbo_hard_cutoff"
        summary["locked_candidate"] = None
        for candidate in summary.get("ranked_candidates", []):
            candidate["selected"] = False


def format_profile_set(profile_set: str, profile_settings: dict[str, Any]) -> str:
    profile = profile_settings.get("certification_baseline") or next(iter(profile_settings.values()), {})
    lookbacks = ", ".join(str(value) for value in profile.get("lookbacks", ())) or "n/a"
    skips = ", ".join(str(value) for value in profile.get("skips", ())) or "n/a"
    top_ns = ", ".join(str(value) for value in profile.get("top_ns", ())) or "n/a"
    strategies = ", ".join(item.get("strategy_id", "baseline") for item in config.STRATEGY_VARIANTS) or "n/a"
    return f"{profile_set} (L={lookbacks} / skip={skips} / top_n={top_ns} / strat={strategies})"


CRISIS_WINDOWS = (
    {"label": "GFC 2008-2009", "start": "2008-01", "end": "2009-12"},
)


def _window_overlaps(start_a: str | None, end_a: str | None, start_b: str, end_b: str) -> bool:
    if not start_a or not end_a:
        return False
    return start_a <= end_b and end_a >= start_b


def _crisis_overlap_labels(start: str | None, end: str | None) -> list[str]:
    labels: list[str] = []
    for window in CRISIS_WINDOWS:
        if _window_overlaps(start, end, window["start"], window["end"]):
            labels.append(window["label"])
    return labels


def build_crisis_lens_rows(walk_forward: dict[str, Any] | None) -> tuple[str, bool]:
    if not isinstance(walk_forward, dict):
        return '<tr><td colspan="5">Crisis lens not available.</td></tr>', False
    folds = walk_forward.get("folds") or []
    if not folds:
        return '<tr><td colspan="5">Crisis lens not available.</td></tr>', False
    rows: list[str] = []
    flagged = False
    for fold in folds:
        window = fold.get("validate_window") or {}
        start = window.get("start")
        end = window.get("end")
        labels = _crisis_overlap_labels(start, end)
        star = "*" if labels else ""
        if labels:
            flagged = True
        label_text = ", ".join(labels) if labels else "none"
        rows.append(
            "<tr>"
            f"<td>{fold.get('fold_id','n/a')}{star}</td>"
            f"<td>{start or 'n/a'} to {end or 'n/a'}</td>"
            f"<td>{format_float(fold.get('validate_sharpe'))}</td>"
            f"<td>{format_pct(fold.get('validate_max_drawdown'))}</td>"
            f"<td>{label_text}</td>"
            "</tr>"
        )
    return "\n".join(rows), flagged


def _find_holdout_primary_track(holdout: dict[str, Any] | None) -> tuple[dict[str, str], dict[str, Any]] | None:
    if not isinstance(holdout, dict):
        return None
    results = holdout.get("results")
    if not isinstance(results, dict):
        return None
    universe = PRIMARY_TRACK.get("universe_variant")
    execution_model = PRIMARY_TRACK.get("execution_model")
    fx_scenario = PRIMARY_TRACK.get("fx_scenario")
    if universe is None or execution_model is None or fx_scenario is None:
        return None
    track = results.get(universe, {}).get(execution_model, {}).get(fx_scenario)
    if not isinstance(track, dict):
        return None
    return (
        {
            "universe_variant": str(universe),
            "execution_model": str(execution_model),
            "fx_scenario": str(fx_scenario),
            "cost_model_name": str(PRIMARY_TRACK.get("cost_model_name", "n/a")),
        },
        track,
    )


def _markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    html_lines: list[str] = []
    in_list = False
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if line.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{line[4:]}</h3>")
            continue
        if line.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{line[3:]}</h2>")
            continue
        if line.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h1>{line[2:]}</h1>")
            continue
        if line.lstrip().startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line.lstrip()[2:]}</li>")
            continue
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        html_lines.append(f"<p>{line}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def build_markdown_dashboard_html(
    *,
    title: str,
    subtitle: str,
    markdown_text: str,
    eyebrow: str = "Research Engine",
    back_href: str = "dashboard.html",
    back_label: str = "Back to research summary",
) -> str:
    body_html = _markdown_to_html(markdown_text)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:980px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.12em; font-size:.7rem; color:#b06b1d; }}
    a {{ color:#b06b1d; }}
  </style>
</head>
<body>
  <main>
    <section>
      <div class="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      <a href="{back_href}">{back_label}</a>
    </section>
    <section>
      {body_html}
    </section>
  </main>
</body>
</html>
"""


def gate_failures(candidate: dict) -> list[str]:
    failures = []
    if candidate.get("gate_fold_count") is False:
        failures.append("fold_count")
    if candidate.get("gate_bootstrap") is False:
        failures.append("bootstrap")
    if candidate.get("gate_deflated_sharpe") is False:
        failures.append("deflated_sharpe")
    if candidate.get("gate_negative_controls") is False:
        failures.append("negative_controls")
    if candidate.get("gate_pbo") is False:
        failures.append("pbo_hard_cutoff")
    return failures


def count_gate_passes(candidates: list[dict]) -> dict[str, int]:
    gates = ["gate_fold_count", "gate_bootstrap", "gate_deflated_sharpe", "gate_negative_controls"]
    counts = {gate: 0 for gate in gates}
    for candidate in candidates:
        for gate in gates:
            if candidate.get(gate) is True:
                counts[gate] += 1
    return counts


def count_gate_failures(candidates: list[dict]) -> dict[str, int]:
    counts = {"fold_count": 0, "bootstrap": 0, "deflated_sharpe": 0, "negative_controls": 0}
    for candidate in candidates:
        if candidate.get("gate_fold_count") is False:
            counts["fold_count"] += 1
        if candidate.get("gate_bootstrap") is False:
            counts["bootstrap"] += 1
        if candidate.get("gate_deflated_sharpe") is False:
            counts["deflated_sharpe"] += 1
        if candidate.get("gate_negative_controls") is False:
            counts["negative_controls"] += 1
    return counts


def compute_mcs(candidate: dict | None) -> int | None:
    if not candidate:
        return None
    score = 0
    score += 25 if candidate.get("gate_fold_count") else 0
    score += 25 if candidate.get("gate_bootstrap") else 0
    score += 25 if candidate.get("gate_deflated_sharpe") else 0
    score += 25 if candidate.get("gate_negative_controls") else 0
    return score


def render_evidence_stack_html(
    *,
    selection_summary: dict[str, Any] | None = None,
    holdout: dict[str, Any] | None = None,
    selection_href: str | None = None,
    holdout_href: str | None = None,
) -> str:
    selection_summary = selection_summary if isinstance(selection_summary, dict) else {}
    holdout = holdout if isinstance(holdout, dict) else {}
    backtest = selection_summary.get("backtest_overfitting", {})
    walk_forward = selection_summary.get("walk_forward", {})
    wf_combined = walk_forward.get("combined", {}) if isinstance(walk_forward, dict) else {}
    wf_folds = walk_forward.get("folds", []) if isinstance(walk_forward, dict) else []
    top = top_candidates(selection_summary.get("ranked_candidates") or [], limit=1)
    top_candidate = top[0] if top else (selection_summary.get("locked_candidate") or {})
    neg_controls = selection_summary.get("negative_controls", {})
    cross = neg_controls.get("cross_sectional_shuffle", {})
    block = neg_controls.get("block_shuffled_null", {})

    selection_window_start = None
    if getattr(config, "ROLLING_ORIGIN_FOLDS", None):
        try:
            selection_window_start = config.ROLLING_ORIGIN_FOLDS[0][1]
        except (IndexError, TypeError):
            selection_window_start = None
    selection_window_end = getattr(config, "INSAMPLE_END", None)
    if selection_window_start and selection_window_end:
        selection_window_text = f"{selection_window_start} to {selection_window_end}"
    else:
        selection_window_text = selection_window_start or selection_window_end or "n/a"

    validate_starts: list[str] = []
    validate_ends: list[str] = []
    validation_months = 0
    fold_pass_count = 0
    for fold in wf_folds:
        validate_window = fold.get("validate_window", {})
        if isinstance(validate_window, dict):
            start = validate_window.get("start")
            end = validate_window.get("end")
            if start:
                validate_starts.append(str(start))
            if end:
                validate_ends.append(str(end))
        months = fold.get("validate_months")
        if isinstance(months, (int, float)):
            validation_months += int(months)
        try:
            validate_sharpe = float(fold.get("validate_sharpe"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(validate_sharpe) and validate_sharpe >= 0.4:
            fold_pass_count += 1
    if validate_starts and validate_ends:
        validation_window_text = f"{min(validate_starts)} to {max(validate_ends)}"
    else:
        validation_window_text = "n/a"

    bootstrap_low = top_candidate.get("bootstrap_ci_low")
    deflated_score = top_candidate.get("deflated_sharpe_score")
    candidate_count = backtest.get("candidate_count")
    selection_status = selection_summary.get("selection_status", "n/a")

    holdout_window = holdout.get("holdout_window", {})
    holdout_start = holdout_window.get("start") if isinstance(holdout_window, dict) else None
    holdout_end = holdout_window.get("end") if isinstance(holdout_window, dict) else None
    if not holdout_start:
        holdout_start = getattr(config, "OOS_START", None)
    if not holdout_end:
        holdout_end = getattr(config, "OOS_END", None)
    if holdout_start and holdout_end:
        holdout_window_text = f"{holdout_start} to {holdout_end}"
    else:
        holdout_window_text = holdout_start or holdout_end or "n/a"
    holdout_status = holdout.get("status")
    holdout_phase4 = holdout.get("phase4_gate", {}) if isinstance(holdout, dict) else {}
    holdout_passed = bool(holdout_phase4.get("phase4_eligible"))

    selection_link_html = (
        f' <a href="{html.escape(selection_href, quote=True)}">Open Phase 2 dashboard</a>.'
        if selection_href
        else ""
    )
    holdout_link_html = (
        f' <a href="{html.escape(holdout_href, quote=True)}">Open holdout diagnostics</a>.'
        if holdout_href
        else ""
    )
    if holdout_passed:
        holdout_detail = (
            f"Holdout passed. Net Sharpe {format_float(holdout_phase4.get('base_main_net_sharpe'))}. "
            f"Phase 4 eligible: {holdout_phase4.get('phase4_eligible', 'n/a')}.{holdout_link_html}"
        )
    elif holdout_status == "blocked_by_pbo_hard_cutoff":
        holdout_detail = (
            f"Holdout not run by design because certification failed before Phase 3.{holdout_link_html}"
        )
    elif holdout_status:
        holdout_detail = (
            f"Status {holdout_status}. This remains the untouched final test and has not passed yet.{holdout_link_html}"
        )
    else:
        holdout_detail = (
            f"Reserved untouched final test; not used for tuning on this page.{holdout_link_html}"
        )

    if holdout_passed:
        paper_trading_value = "Unlocked"
        forward_detail = (
            f"Phase 4 may begin now because the untouched {holdout_window_text} holdout passed for this thesis. "
            "Operational paper-trading belongs in the forward-monitor dashboards, not this validation funnel."
        )
    else:
        paper_trading_value = "Locked"
        forward_detail = (
            f"Paper trading starts only after the untouched {holdout_window_text} holdout passes for this exact thesis."
        )

    return f"""
    <section>
      <h2>Phase 1-4 Validation Funnel</h2>
      <p class="muted">This report uses a strict phase ladder before any paper trading. Phase 1 is the infrastructure/data gate, Phase 2 is research certification on {selection_window_text}, Phase 3 is the untouched final holdout on {holdout_window_text}, and Phase 4 only starts after Phase 3 passes.</p>
      <div class="grid">
        <div class="card">
          <div class="label">Phase 1. Data / Engine Gate</div>
          <div class="value">{selection_window_text}</div>
          <div class="muted">This page assumes `validate.py` is already green. Parameters are chosen only after the Phase 1 data/engine gate is clear. Candidates tested: {candidate_count if candidate_count is not None else 'n/a'}.{selection_link_html}</div>
        </div>
        <div class="card">
          <div class="label">Phase 2A. Repeated OOS Folds</div>
          <div class="value">{f"{len(wf_folds)} folds · {validation_months} OOS months" if wf_folds else "n/a"}</div>
          <div class="muted">Validation windows {validation_window_text}. {fold_pass_count}/{len(wf_folds) if wf_folds else 'n/a'} folds clear Sharpe &gt;= 0.40. Combined Sharpe {format_float(wf_combined.get('sharpe'))}.</div>
        </div>
        <div class="card">
          <div class="label">Phase 2B. Anti-Overfitting Checks</div>
          <div class="value">PBO {format_pbo_display(backtest.get('pbo'), backtest)}</div>
          <div class="muted">Selection status {selection_status}. {pbo_explainer(backtest)} Bootstrap CI low {format_float(bootstrap_low)}. Deflated Sharpe {format_float(deflated_score)}. Cross shuffle {cross.get('pass_count', 'n/a')}/{cross.get('run_count', 'n/a')}; block null {block.get('pass_count', 'n/a')}/{block.get('run_count', 'n/a')}.</div>
        </div>
        <div class="card">
          <div class="label">Phase 3. Untouched Final Holdout</div>
          <div class="value">{holdout_window_text}</div>
          <div class="muted">{holdout_detail}</div>
        </div>
        <div class="card">
          <div class="label">Phase 4. Paper Trading</div>
          <div class="value">{paper_trading_value}</div>
          <div class="muted">{forward_detail}</div>
        </div>
      </div>
    </section>
    """


def phase2_selection_window_text() -> str:
    start = None
    if getattr(config, "ROLLING_ORIGIN_FOLDS", None):
        try:
            start = config.ROLLING_ORIGIN_FOLDS[0][1]
        except (IndexError, TypeError):
            start = None
    end = getattr(config, "INSAMPLE_END", None)
    if start and end:
        return f"{start} to {end}"
    return start or end or "n/a"


def phase2_validation_window_text(folds: list[dict[str, Any]] | None = None) -> str:
    starts: list[str] = []
    ends: list[str] = []
    if folds:
        for fold in folds:
            validate_window = fold.get("validate_window", {})
            if isinstance(validate_window, dict):
                start = validate_window.get("start")
                end = validate_window.get("end")
                if start:
                    starts.append(str(start))
                if end:
                    ends.append(str(end))
    elif getattr(config, "ROLLING_ORIGIN_FOLDS", None):
        for fold in config.ROLLING_ORIGIN_FOLDS:
            try:
                starts.append(str(fold[3]))
                ends.append(str(fold[4]))
            except (IndexError, TypeError):
                continue
    if starts and ends:
        return f"{min(starts)} to {max(ends)}"
    return "n/a"


def holdout_window_text(holdout: dict[str, Any] | None = None) -> str:
    holdout = holdout if isinstance(holdout, dict) else {}
    window = holdout.get("holdout_window", {}) if isinstance(holdout.get("holdout_window", {}), dict) else {}
    start = window.get("start") or getattr(config, "OOS_START", None)
    end = window.get("end") or getattr(config, "OOS_END", None)
    if start and end:
        return f"{start} to {end}"
    return start or end or "n/a"


def render_timeframe_note(text: str) -> str:
    return (
        '<div style="margin:10px 0 12px; padding:10px 12px; border-radius:14px; '
        'border:1px solid rgba(176,107,29,.28); background:#fff7eb; color:#5b6762;">'
        f'<strong style="color:#4b4031;">Timeframe:</strong> {text}</div>'
    )


def render_phase_map_html(
    *,
    phase1_href: str | None = None,
    selection_href: str | None = None,
    holdout_href: str | None = None,
    holdout: dict[str, Any] | None = None,
) -> str:
    phase1_link = (
        f' <a href="{html.escape(phase1_href, quote=True)}">Open Phase 1 validation.</a>'
        if phase1_href
        else ""
    )
    selection_link = (
        f' <a href="{html.escape(selection_href, quote=True)}">Open Phase 2 dashboard.</a>'
        if selection_href
        else ""
    )
    holdout_link = (
        f' <a href="{html.escape(holdout_href, quote=True)}">Open Phase 3 holdout.</a>'
        if holdout_href
        else ""
    )
    return (
        "<section>"
        "<h2>Phase Map</h2>"
        "<div class=\"grid\">"
        f"<div class=\"card\"><div class=\"label\">Phase 1</div><div class=\"value\">Infrastructure &amp; Data</div><div class=\"muted\">Gate this cycle with <code>validate.py</code>. This stage checks the engine and core artifacts before any research result is trusted.{phase1_link}</div></div>"
        f"<div class=\"card\"><div class=\"label\">Phase 2</div><div class=\"value\">Research Certification</div><div class=\"muted\">Selection boundary {phase2_selection_window_text()}. Validation windows {phase2_validation_window_text()}.{selection_link}</div></div>"
        f"<div class=\"card\"><div class=\"label\">Phase 3</div><div class=\"value\">Untouched Holdout</div><div class=\"muted\">Official holdout window {holdout_window_text(holdout)}.{holdout_link}</div></div>"
        f"<div class=\"card\"><div class=\"label\">Phase 4</div><div class=\"value\">Paper Trading</div><div class=\"muted\">Only unlock after the frozen Phase 3 result is accepted.</div></div>"
        "</div>"
        "</section>"
    )


def build_phase1_dashboard(
    *,
    success: bool,
    sections: list[Any],
    title: str,
    subtitle: str,
    back_href: str,
    back_label: str,
    input_dir: Path,
    selection_href: str = "selection_summary.html",
    holdout_href: str = "holdout_results.html",
) -> str:
    status_label = "green" if success else "not green"
    status_tone = "good" if success else "weak"
    section_rows = []
    for section in sections:
        section_status = str(getattr(section, "status", "n/a"))
        section_tone = "good" if section_status == "PASS" else ("weak" if section_status == "FAIL" else "caution")
        messages = "".join(f"<li>{html.escape(str(message))}</li>" for message in getattr(section, "messages", []))
        section_rows.append(
            "<tr>"
            f"<td>{html.escape(str(getattr(section, 'name', 'n/a')))}</td>"
            f"<td><span class=\"badge {section_tone}\">{html.escape(section_status)}</span></td>"
            f"<td><ul>{messages}</ul></td>"
            "</tr>"
        )
    section_table = "".join(section_rows) if section_rows else "<tr><td colspan=\"3\">No Phase 1 sections were returned.</td></tr>"
    coverage_start = config.ROLLING_ORIGIN_FOLDS[0][1] if config.ROLLING_ORIGIN_FOLDS else "n/a"
    exchanges = ", ".join(config.ACTIVE_NORDIC_EXCHANGE_CODES)
    benchmark_names = ", ".join(
        [
            config.PRIMARY_PASSIVE_BENCHMARK,
            config.SECONDARY_OPPORTUNITY_COST_BENCHMARK,
            config.TERTIARY_OPPORTUNITY_COST_BENCHMARK,
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:1080px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin-top:16px; }}
    .card {{ background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:20px; padding:18px; }}
    .label {{ color:#6c5842; text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; }}
    .value {{ font-size:1.2rem; margin-top:6px; overflow-wrap:anywhere; }}
    .muted {{ color:#5b6762; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; }}
    .badge.good {{ background:#dcefe3; color:#2f5c46; border:1px solid #a9c7b4; }}
    .badge.caution {{ background:#f4ead7; color:#7a5a25; border:1px solid #d9c59a; }}
    .badge.weak {{ background:#f3d7d7; color:#7a2f2f; border:1px solid #d6a3a3; }}
    .callout {{ background:#fff7eb; border:1px solid rgba(176,107,29,.28); border-radius:16px; padding:12px 14px; margin-top:12px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:10px 8px; vertical-align:top; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
    ul {{ margin:0; padding-left:18px; }}
    a {{ color:#b06b1d; }}
  </style>
</head>
<body>
  <main>
    <section>
      <p class="label">Phase 1</p>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      {render_timeframe_note(f"Phase 1 validates the local artifacts needed for Phase 2 research {phase2_selection_window_text()} and the untouched Phase 3 holdout {holdout_window_text(None)}.")}
      <a href="{back_href}">{back_label}</a>
    </section>
    {render_phase_map_html(phase1_href="phase1_summary.html", selection_href=selection_href, holdout_href=holdout_href)}
    <section>
      <h2>Validation Overview</h2>
      <div class="grid">
        <div class="card"><div class="label">Overall Status</div><div class="value"><span class="badge {status_tone}">{status_label}</span></div></div>
        <div class="card"><div class="label">Data Root</div><div class="value">{html.escape(str(input_dir))}</div></div>
        <div class="card"><div class="label">Active Exchanges</div><div class="value">{html.escape(exchanges)}</div></div>
        <div class="card"><div class="label">Benchmarks Required</div><div class="value">{html.escape(benchmark_names)}</div></div>
      </div>
      <div class="callout">
        <strong>What Phase 1 means here.</strong>
        <div class="muted">This page is not a performance backtest. It checks whether the engine, benchmark data, FX history, and point-in-time research artifacts are in a trustworthy state before any Phase 2 or Phase 3 result is treated as meaningful.</div>
      </div>
    </section>
    <section>
      <h2>Coverage Boundary</h2>
      {render_timeframe_note(f"Minimum research start implied by the rolling folds is {coverage_start}; the validation stack then reserves {holdout_window_text(None)} for the untouched holdout.")}
      <div class="grid">
        <div class="card"><div class="label">Phase 2 Research Start</div><div class="value">{coverage_start}</div></div>
        <div class="card"><div class="label">Phase 2 Ends</div><div class="value">{config.INSAMPLE_END}</div></div>
        <div class="card"><div class="label">Phase 3 Holdout</div><div class="value">{holdout_window_text(None)}</div></div>
        <div class="card"><div class="label">Primary Benchmark</div><div class="value">{config.PRIMARY_PASSIVE_BENCHMARK}</div></div>
      </div>
    </section>
    <section>
      <h2>Phase 1 Checks</h2>
      <table>
        <thead>
          <tr><th>Section</th><th>Status</th><th>Details</th></tr>
        </thead>
        <tbody>
          {section_table}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def walk_forward_gate_summary(folds: list[dict[str, Any]] | None, *, gate: float = 0.4) -> str:
    if not folds:
        return "n/a"
    values = [fold.get("validate_sharpe") for fold in folds if fold.get("validate_sharpe") is not None]
    if not values:
        return "n/a"
    passes = sum(1 for value in values if value > gate)
    return f"{passes}/{len(values)} folds with Sharpe > {format_float(gate)}"


def walk_forward_diagnostic_result(walk_forward: dict[str, Any] | None) -> str:
    if not isinstance(walk_forward, dict):
        return "n/a"
    combined = walk_forward.get("combined", {})
    folds = walk_forward.get("folds") or []
    values = [fold.get("validate_sharpe") for fold in folds if fold.get("validate_sharpe") is not None]
    neg_count = sum(1 for value in values if value < 0.0)
    total_folds = len(values) if values else 0
    positive_count = sum(1 for value in values if value > 0.0)
    combined_sharpe = combined.get("sharpe")
    if combined_sharpe is not None and np.isfinite(combined_sharpe) and combined_sharpe >= 0.4:
        stability_note = "the stitched Phase 3 read stays above the Sharpe gate even though shorter slices are noisier"
    else:
        stability_note = "short slices were mixed, so this is best read as context rather than a second gate"
    return (
        f"Combined Sharpe {format_float(combined_sharpe)}, "
        f"Total Return {format_pct(combined.get('total_return'))}, "
        f"{positive_count}/{total_folds} positive slices, {neg_count}/{total_folds} negative slices, "
        f"{stability_note}"
    )


def walk_forward_quality_badge(
    folds: list[dict[str, Any]] | None, *, gate_threshold: float = 0.4
) -> tuple[str, str, str]:
    if not folds:
        return ("n/a", "neutral", "No walk-forward folds.")
    values = []
    train_values = []
    for fold in folds:
        v = fold.get("validate_sharpe")
        t = fold.get("train_sharpe")
        if v is not None and np.isfinite(v):
            values.append(float(v))
            train_values.append(float(t) if t is not None and np.isfinite(t) else None)
    if not values:
        return ("n/a", "neutral", "Walk-forward metrics unavailable.")
    total = len(values)
    above_gate = sum(1 for v in values if v >= gate_threshold)
    positives = sum(1 for v in values if v > 0.0)
    negatives = sum(1 for v in values if v < 0.0)
    gaps = [
        v - t for v, t in zip(values, train_values) if t is not None and np.isfinite(t)
    ]
    median_gap = float(np.median(gaps)) if gaps else None

    score = 0
    max_score = 0
    if total:
        max_score += 1
        if above_gate / total >= 0.6:
            score += 1
    max_score += 1
    if positives / total >= 0.7:
        score += 1
    if median_gap is not None:
        max_score += 1
        if median_gap > -0.2:
            score += 1

    ratio = score / max_score if max_score else 0.0
    if ratio >= 0.67:
        quality, tone = "GOOD", "good"
    elif ratio >= 0.34:
        quality, tone = "CAUTION", "caution"
    else:
        quality, tone = "WEAK", "weak"
    detail = f"{above_gate}/{total} folds >= {format_float(gate_threshold)}; {negatives} negative"
    if median_gap is not None:
        detail += f"; median gap {format_float(median_gap)}"
    return (f"STABILITY: {quality}", tone, detail)


def walk_forward_gap_summary(
    folds: list[dict[str, Any]] | None,
    *,
    train_key: str,
    validate_key: str,
    as_pct: bool,
    train_value_fn: Callable[[dict[str, Any]], float | None] | None = None,
    validate_value_fn: Callable[[dict[str, Any]], float | None] | None = None,
) -> str:
    if not folds:
        return "n/a"
    gaps = []
    for fold in folds:
        train = train_value_fn(fold) if train_value_fn else fold.get(train_key)
        validate = validate_value_fn(fold) if validate_value_fn else fold.get(validate_key)
        if train is None or validate is None:
            continue
        if not np.isfinite(train) or not np.isfinite(validate):
            continue
        gaps.append(float(validate) - float(train))
    if not gaps:
        return "n/a"
    median_gap = float(np.median(gaps))
    worst_gap = min(gaps)
    formatter = format_pct if as_pct else format_float
    return f"Median gap (val - train) {formatter(median_gap)}; worst gap {formatter(worst_gap)}"


def monte_carlo_interpretation(mc_sharpe: dict, mc_total: dict, mc_dd: dict) -> str:
    if not mc_sharpe or not mc_total or not mc_dd:
        return "Monte Carlo summary not available."
    sharpe_median = mc_sharpe.get("median")
    sharpe_p05 = mc_sharpe.get("p05")
    total_median = mc_total.get("median")
    dd_median = mc_dd.get("median")
    sharpe_sign = "positive" if sharpe_median is not None and sharpe_median > 0 else "negative"
    return (
        f"Median Sharpe {format_float(sharpe_median)} ({sharpe_sign}); "
        f"p05 Sharpe {format_float(sharpe_p05)} marks the 1-in-20 downside. "
        f"Median total return {format_pct(total_median)}; median max drawdown {format_pct(dd_median)}."
    )


def ensure_summary_monte_carlo(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    monte = summary.get("monte_carlo")
    if isinstance(monte, dict) and monte.get("status") in {"ok", "unavailable"}:
        return summary
    locked_candidate = summary.get("locked_candidate")
    if not isinstance(locked_candidate, dict):
        return summary
    returns = locked_candidate.get("concatenated_returns")
    if not returns:
        return summary
    periods_per_year = int(summary.get("periods_per_year") or 12)
    enriched = dict(summary)
    enriched["monte_carlo"] = monte_carlo_summary(
        returns,
        periods_per_year=periods_per_year,
        n_resamples=config.MONTE_CARLO_RESAMPLES,
        block_length_months=config.MONTE_CARLO_BLOCK_LENGTH_MONTHS,
        seed=config.MONTE_CARLO_SEED,
    )
    return enriched


def render_trimmed_monte_carlo(
    monte: dict[str, Any],
    *,
    display_quantile_range: tuple[float, float] | None = (0.05, 0.95),
    center_baseline: float | None = None,
) -> str:
    trimmed = monte.get("trimmed") if isinstance(monte, dict) else None
    if not isinstance(trimmed, dict) or trimmed.get("status") != "ok":
        return ""
    trim_pct = trimmed.get("trim_top_pct", MONTE_CARLO_TRIM_TOP_PCT)
    trim_label = f"Trimmed Top {int(round(trim_pct * 100))}% (by total return)"
    metrics = trimmed.get("metrics", {})
    mc_sharpe = metrics.get("sharpe", {})
    mc_total = metrics.get("total_return", {})
    mc_dd = metrics.get("max_drawdown", {})
    mc_final_equity = metrics.get("final_equity", {})
    hist = trimmed.get("histograms", {})
    paths = trimmed.get("sample_paths", [])
    quantiles = trimmed.get("path_quantiles", {})
    density = trimmed.get("path_density", {})
    interp = monte_carlo_interpretation(mc_sharpe, mc_total, mc_dd)
    quality_label, quality_tone, quality_detail = monte_carlo_badge(mc_sharpe, mc_total, mc_dd)
    return (
        f"<div class=\"trimmed-monte-carlo\">"
        f"<h3>{trim_label} View</h3>"
        "<p class=\"muted\">Removes the top 5% of total-return outcomes to show performance without extreme winners.</p>"
        f"<div class=\"badge-row\"><span class=\"label\">Trimmed quality</span>{render_badge(quality_label, quality_tone, quality_detail)}</div>"
        f"<p class=\"muted\"><strong>Trimmed summary:</strong> {interp}</p>"
        "<div class=\"grid\">"
        f"<div class=\"card full\">{render_histogram_svg(hist.get('final_equity'), metric=mc_final_equity, title='Distribution @ Ending Equity Index (Trimmed)', x_label='Ending Equity Index (100 = start)', as_pct=False, width=680, height=180)}</div>"
        f"<div class=\"card\">{render_histogram_svg(hist.get('sharpe'), metric=mc_sharpe, title='Sharpe (Trimmed)', x_label='Sharpe', as_pct=False)}</div>"
        f"<div class=\"card\">{render_histogram_svg(hist.get('total_return'), metric=mc_total, title='Total Return From Start (Trimmed)', x_label='Total Return From Start (%)', as_pct=True)}</div>"
        f"<div class=\"card\">{render_histogram_svg(hist.get('max_drawdown'), metric=mc_dd, title='Max Drawdown (Trimmed)', x_label='Max Drawdown (%)', as_pct=True)}</div>"
        f"<div class=\"card full\">{render_spaghetti_svg(paths, title='Monte Carlo Paths (Trimmed Equity Index)', max_lines=3, line_opacity=0.028, line_width=0.45, display_quantile_range=display_quantile_range, fit_plotted_extents=True, center_baseline=center_baseline, path_quantiles=quantiles, path_density=density)}</div>"
        "</div></div>"
    )


def render_monte_carlo_anomaly_panel(monte: dict[str, Any]) -> str:
    if not isinstance(monte, dict) or monte.get("status") != "ok":
        return ""
    base = monte.get("base_return_stats") or {}
    metrics = monte.get("metrics", {}) if isinstance(monte, dict) else {}
    mc_sharpe = metrics.get("sharpe", {})
    mc_dd = metrics.get("max_drawdown", {})
    trimmed = monte.get("trimmed") or {}

    max_month = base.get("max_monthly")
    min_month = base.get("min_monthly")
    pct_pos = base.get("pct_months_gt_10")
    pct_neg = base.get("pct_months_lt_neg10")

    def _status_label(tone: str) -> str:
        if tone == "good":
            return "OK"
        if tone == "caution":
            return "CAUTION"
        return "FLAG"

    def _row(label: str, tone: str, detail: str) -> str:
        return (
            f'<li class="verdict-item"><span class="verdict-label">{label}</span>'
            f'{render_badge(_status_label(tone), tone, detail)}</li>'
        )

    rows: list[str] = []

    if max_month is not None and min_month is not None:
        tone = "good"
        if max_month > 0.30 or min_month < -0.30:
            tone = "weak"
        elif max_month > 0.20 or min_month < -0.20:
            tone = "caution"
        rows.append(
            _row(
                "Extreme months (historical)",
                tone,
                f"max {format_pct(max_month)}, min {format_pct(min_month)}",
            )
        )

    if pct_pos is not None and pct_neg is not None:
        tone = "good"
        if pct_pos > 0.30 or pct_neg > 0.25:
            tone = "weak"
        elif pct_pos > 0.20 or pct_neg > 0.15:
            tone = "caution"
        rows.append(
            _row(
                "Big month frequency",
                tone,
                f">{format_pct(0.10)}: {format_pct(pct_pos)} | < -{format_pct(0.10)}: {format_pct(pct_neg)}",
            )
        )

    if mc_sharpe and mc_dd:
        p05_sharpe = mc_sharpe.get("p05")
        p95_dd = mc_dd.get("p95")
        tone = "good"
        if p05_sharpe is not None and p05_sharpe < 0:
            tone = "weak"
        if p95_dd is not None and p95_dd > 0.65:
            tone = "weak"
        elif p95_dd is not None and p95_dd > 0.55:
            tone = "caution"
        rows.append(
            _row(
                "Tail risk (Monte Carlo)",
                tone,
                f"p05 Sharpe {format_float(p05_sharpe)} | p95 drawdown {format_pct(p95_dd)}",
            )
        )

    drop_pct = None
    if trimmed.get("status") == "ok":
        full_med = (metrics.get("total_return") or {}).get("median")
        trimmed_med = (trimmed.get("metrics", {}).get("total_return") or {}).get("median")
        if full_med is not None and trimmed_med is not None and full_med != 0:
            drop_pct = max(0.0, (full_med - trimmed_med) / abs(full_med))
            tone = "good"
            if drop_pct > 0.30:
                tone = "weak"
            elif drop_pct > 0.15:
                tone = "caution"
            rows.append(
                _row(
                    "Outlier dependence (trim top 5%)",
                    tone,
                    f"median return drop {format_pct(drop_pct)}",
                )
            )

    if not rows:
        return ""

    return (
        "<div class=\"card full\">"
        "<div class=\"label\">Monte Carlo Anomaly Check (Heuristic)</div>"
        "<p class=\"muted\">Checks whether the simulation is dominated by extreme months or rare winners.</p>"
        f"<ul class=\"verdict-list\">{''.join(rows)}</ul>"
        "</div>"
    )


def render_return_curve(returns: list[float] | None, benchmark: list[float] | None = None) -> str:
    if not returns:
        return "<div class=\"muted\">Return curve not available (rerun rebuild to populate return series).</div>"
    equity = [1.0]
    for value in returns:
        equity.append(equity[-1] * (1.0 + float(value)))
    bench_equity = []
    if benchmark:
        bench_equity = [1.0]
        for value in benchmark:
            bench_equity.append(bench_equity[-1] * (1.0 + float(value)))
    min_val = min(equity + (bench_equity or []))
    max_val = max(equity + (bench_equity or []))
    span = max(max_val - min_val, 1e-6)
    width = 560
    height = 160

    def _points(series: list[float]) -> str:
        return " ".join(
            f"{index/(len(series)-1)*width:.1f},{height - (value - min_val)/span*height:.1f}"
            for index, value in enumerate(series)
        )

    candidate_points = _points(equity)
    benchmark_points = _points(bench_equity) if bench_equity else ""
    benchmark_poly = (
        f"<polyline points=\"{benchmark_points}\" fill=\"none\" stroke=\"#9a8c6a\" stroke-width=\"2\" stroke-dasharray=\"4 4\" />"
        if benchmark_points
        else ""
    )
    return (
        "<svg viewBox=\"0 0 560 160\" width=\"100%\" height=\"160\" preserveAspectRatio=\"none\">"
        f"{benchmark_poly}<polyline points=\"{candidate_points}\" fill=\"none\" stroke=\"#b06b1d\" stroke-width=\"2.5\" />"
        "</svg>"
    )


def top_candidates(candidates: list[dict], limit: int = 6) -> list[dict]:
    ranked = sorted(candidates, key=lambda item: item.get("rank", 9999))
    return ranked[:limit]


def sensitivity_table_rows(sensitivity: dict[str, Any] | None) -> str:
    if not sensitivity or sensitivity.get("status") != "ok":
        return '<tr><td colspan="5">n/a</td></tr>'
    rows: list[str] = []
    params = sensitivity.get("parameters", {})
    for key, label in (("l", "Lookback"), ("skip", "Skip"), ("top_n", "Top N")):
        info = params.get(key, {})
        rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{info.get('best_value','n/a')}</td>"
            f"<td>{info.get('worst_value','n/a')}</td>"
            f"<td>{format_float(info.get('median_spread'))}</td>"
            f"<td>{format_float(info.get('correlation'))}</td>"
            "</tr>"
        )
    strategy = sensitivity.get("strategy_id", {})
    if strategy:
        rows.append(
            "<tr>"
            "<td>Strategy</td>"
            f"<td>{strategy.get('best_value','n/a')}</td>"
            f"<td>{strategy.get('worst_value','n/a')}</td>"
            f"<td>{format_float(strategy.get('median_spread'))}</td>"
            "<td>n/a</td>"
            "</tr>"
        )
    return "\n".join(rows) if rows else '<tr><td colspan="5">n/a</td></tr>'


def build_profile_dashboard(
    *,
    summary: dict,
    holdout: dict[str, Any] | None = None,
    title: str,
    subtitle: str,
    back_href: str,
    back_label: str,
    phase1_href: str = "phase1_summary.html",
    selection_href: str = "selection_summary.html",
    holdout_href: str = "holdout_results.html",
) -> str:
    summary = ensure_summary_monte_carlo(summary)
    candidates = summary.get("ranked_candidates") or []
    backtest = summary.get("backtest_overfitting", {})
    thesis = summary.get("thesis") or {}
    pbo_band_value = summary.get("pbo_band", "n/a")
    pbo_display = format_pbo_display(backtest.get("pbo"), backtest)
    pbo_policy = summary.get("pbo_policy", {})
    pbo_good_max = pbo_policy.get("good_max", config.PBO_THRESHOLD_MAX)
    pbo_hard_cutoff_min = pbo_policy.get("hard_cutoff_min", config.PBO_HARD_CUTOFF)
    neg_controls = summary.get("negative_controls", {})
    cross = neg_controls.get("cross_sectional_shuffle", {})
    block = neg_controls.get("block_shuffled_null", {})
    walk_forward = summary.get("walk_forward", {})
    wf_combined = walk_forward.get("combined", {}) if isinstance(walk_forward, dict) else {}
    monte = summary.get("monte_carlo", {})
    mc_metrics = monte.get("metrics", {}) if isinstance(monte, dict) else {}
    mc_sharpe = mc_metrics.get("sharpe", {})
    mc_total = mc_metrics.get("total_return", {})
    mc_dd = mc_metrics.get("max_drawdown", {})
    mc_hist = monte.get("histograms", {}) if isinstance(monte, dict) else {}
    mc_paths = monte.get("sample_paths", []) if isinstance(monte, dict) else []
    mc_quantiles = monte.get("path_quantiles", {}) if isinstance(monte, dict) else {}
    mc_density = monte.get("path_density", {}) if isinstance(monte, dict) else {}
    wf_folds = walk_forward.get("folds", []) if isinstance(walk_forward, dict) else []
    wf_combined_returns = walk_forward.get("combined_returns", []) if isinstance(walk_forward, dict) else []
    wf_combined_benchmark_returns = (
        walk_forward.get("combined_benchmark_returns", []) if isinstance(walk_forward, dict) else []
    )
    wf_periods_per_year = wf_combined.get("periods_per_year", 12) if isinstance(wf_combined, dict) else 12
    sensitivity_sections = render_sensitivity_sections(summary.get("parameter_sensitivity"))
    wf_gate_text = walk_forward_gate_summary(wf_folds)
    mc_interp = monte_carlo_interpretation(mc_sharpe, mc_total, mc_dd)
    mc_badge_label, mc_badge_tone, mc_badge_detail = monte_carlo_badge(mc_sharpe, mc_total, mc_dd)
    mc_badge_html = render_badge(mc_badge_label, mc_badge_tone, mc_badge_detail)
    verdict_panel = render_validation_verdict(summary)
    wf_diag = walk_forward_diagnostic_result(walk_forward)
    wf_quality_label, wf_quality_tone, wf_quality_detail = walk_forward_quality_badge(wf_folds, gate_threshold=0.4)
    wf_quality_html = render_badge(wf_quality_label, wf_quality_tone, wf_quality_detail)
    wf_gap_sharpe = walk_forward_gap_summary(
        wf_folds, train_key="train_sharpe", validate_key="validate_sharpe", as_pct=False
    )
    wf_gap_return = walk_forward_gap_summary(
        wf_folds,
        train_key="train_total_return",
        validate_key="validate_total_return",
        as_pct=True,
        train_value_fn=lambda fold: _fold_cagr(fold, total_key="train_total_return", months_key="train_months"),
        validate_value_fn=lambda fold: _fold_cagr(fold, total_key="validate_total_return", months_key="validate_months"),
    )
    wf_oos_equity_chart = render_walkforward_oos_equity_svg(
        wf_combined_returns,
        wf_folds,
        benchmark_returns=wf_combined_benchmark_returns,
        benchmark_label=config.PRIMARY_PASSIVE_BENCHMARK,
    )
    wf_rolling_sharpe_chart = render_walkforward_rolling_sharpe_svg(
        wf_combined_returns, wf_folds, periods_per_year=wf_periods_per_year
    )
    wf_schedule_chart = render_walkforward_schedule_svg(wf_folds)
    wf_gap_months = walk_forward_gap_months_summary(wf_folds)
    if thesis.get("name") == "ex_norway":
        benchmark_context = (
            f"<p class=\"muted\"><strong>Benchmark context:</strong> The gold line is "
            f"<strong>{config.PRIMARY_PASSIVE_BENCHMARK}</strong>, the project's broad Nordic passive proxy. "
            "For the ex_norway thesis this still includes Norway, so it is a house benchmark rather than a thesis-matched ex-Norway passive comparator.</p>"
        )
    else:
        benchmark_context = (
            f"<p class=\"muted\"><strong>Benchmark context:</strong> The gold line is "
            f"<strong>{config.PRIMARY_PASSIVE_BENCHMARK}</strong>, the project's primary passive benchmark proxy for the same validation windows.</p>"
        )
    trimmed_html = render_trimmed_monte_carlo(monte)
    mc_anomaly = render_monte_carlo_anomaly_panel(monte)
    top = top_candidates(candidates, limit=6)
    top_rows = "\n".join(
        f"<tr><td>{item.get('candidate_id','n/a')}</td><td>{format_params(item.get('params'))}</td>"
        f"<td>{format_float(item.get('median_validation_sharpe'))}</td><td>{item.get('fold_pass_count','n/a')}</td>"
        f"<td>{'yes' if item.get('gate_bootstrap') else 'no'}</td><td>{'yes' if item.get('gate_deflated_sharpe') else 'no'}</td></tr>"
        for item in top
    )
    evidence_stack_html = render_evidence_stack_html(
        selection_summary=summary,
        holdout=holdout,
        selection_href=selection_href,
        holdout_href=holdout_href,
    )
    selection_window_text = phase2_selection_window_text()
    validation_window_text = phase2_validation_window_text(wf_folds)
    holdout_window_text_value = holdout_window_text(holdout)
    phase2_start = config.ROLLING_ORIGIN_FOLDS[0][1] if config.ROLLING_ORIGIN_FOLDS else None
    holdout_window = holdout.get("holdout_window", {}) if holdout else {}
    holdout_start = holdout_window.get("start") or config.OOS_START
    holdout_end = holdout_window.get("end") or config.OOS_END
    after_holdout = _ordinal_to_month(_month_to_ordinal(holdout_end) + 1) if holdout_end else None
    protocol_chart = render_phase_boundary_svg(
        [
            {
                "label": "Phase 2 walk-forward",
                "detail": "robustness test across older regimes",
                "start": phase2_start,
                "end": config.INSAMPLE_END,
                "fill": "#d9cab8",
                "highlight": True,
            },
            {
                "label": "Phase 3 holdout",
                "detail": "after-test untouched data",
                "start": holdout_start,
                "end": holdout_end,
                "fill": "#eadca7",
            },
            {
                "label": "After holdout",
                "detail": "paper/live or future revalidation",
                "start": after_holdout,
                "fill": "#efe6d7",
            },
        ],
        footer_note=(
            "The first walk-forward starts far back on purpose: older folds test whether the idea survives multiple "
            "regimes before later holdout data is allowed into the story."
        ),
    )
    wf_ladder_chart = render_walkforward_ladder_svg(
        wf_folds,
        stitched_label="Stitched validation record",
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:980px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
    .card {{ background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:18px; padding:14px; }}
    .label {{ text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; color:#6c5842; }}
    .value {{ font-size:1.1rem; margin-top:4px; }}
    .muted {{ color:#5b6762; }}
    .card.full {{ grid-column: 1 / -1; }}
    .chart-block {{ display:flex; flex-direction:column; gap:6px; }}
    .chart-title {{ font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:#6c5842; }}
    .chart {{ width:100%; height:auto; }}
    .chart text {{ font-family:Georgia, serif; font-size:10px; fill:#5d685f; }}
    .chart-note {{ font-size:.78rem; color:#6c5842; }}
    .legend-line {{ font-weight:600; color:#7a6aa0; }}
    .legend-line.median {{ color:#4f7a67; }}
    .legend-line.band {{ color:#bfa47a; }}
    .legend-line.density {{ color:#2f9abf; }}
    .badge-row {{ display:flex; align-items:center; gap:8px; margin:6px 0 6px; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; }}
    .badge.good {{ background:#dcefe3; color:#2f5c46; border:1px solid #a9c7b4; }}
    .badge.caution {{ background:#f4ead7; color:#7a5a25; border:1px solid #d9c59a; }}
    .badge.weak {{ background:#f3d7d7; color:#7a2f2f; border:1px solid #d6a3a3; }}
    .badge.neutral {{ background:#ece7dd; color:#6c5842; border:1px solid #d7cdbc; }}
    .badge-detail {{ font-size:.78rem; color:#6c5842; }}
    .verdict-panel {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
    .verdict-panel .badge {{ font-size:.9rem; padding:6px 12px; }}
    .verdict-note {{ margin:6px 0 10px; color:#5b6762; font-size:.85rem; }}
    .verdict-list {{ list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:6px; }}
    .verdict-item {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
    .verdict-label {{ font-weight:600; color:#4b4031; min-width:240px; }}
    details.trimmed {{ margin-top:12px; background:#fffdf8; border:1px dashed rgba(23,33,26,.18); border-radius:16px; padding:10px 12px; }}
    details.trimmed summary {{ cursor:pointer; font-weight:600; color:#6c5842; }}
    details.trimmed {{ margin-top:12px; background:#fffdf8; border:1px dashed rgba(23,33,26,.18); border-radius:16px; padding:10px 12px; }}
    details.trimmed summary {{ cursor:pointer; font-weight:600; color:#6c5842; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
    a {{ color:#b06b1d; }}
  </style>
</head>
<body>
  <main>
    <section>
      <p class="label">Phase 2</p>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      {render_timeframe_note(f"This page is Phase 2 only: research boundary {selection_window_text}; stitched validation windows {validation_window_text}. The untouched Phase 3 holdout {holdout_window_text_value} appears here only as context and eligibility.")}
      <a href="{back_href}">{back_label}</a>
    </section>
    <section>
      <h2>Status Overview</h2>
      <div class="grid">
        <div class="card"><div class="label">Selection Status</div><div class="value">{summary.get('selection_status','n/a')}</div></div>
        <div class="card"><div class="label">PBO</div><div class="value">{pbo_display}</div></div>
        <div class="card"><div class="label">PBO Band</div><div class="value">{pbo_band_value}</div></div>
        <div class="card"><div class="label">PBO Thresholds</div><div class="value">{format_pct(pbo_good_max)} / {format_pct(pbo_hard_cutoff_min)}</div></div>
        <div class="card"><div class="label">Candidates Tested</div><div class="value">{backtest.get('candidate_count','n/a')}</div></div>
        <div class="card"><div class="label">Negative Controls</div><div class="value">{cross.get('pass_count','n/a')}/{cross.get('run_count','n/a')} · {block.get('pass_count','n/a')}/{block.get('run_count','n/a')}</div></div>
      </div>
      <div class="callout"><strong>PBO note.</strong><div class="muted">{pbo_explainer(backtest)}</div></div>
    </section>
    {verdict_panel}
    {render_phase_map_html(phase1_href=phase1_href, selection_href=selection_href, holdout_href=holdout_href, holdout=holdout)}
    {evidence_stack_html}
    <section>
      <h2>Protocol Insight</h2>
      {render_timeframe_note(f"Boundary charts below span the full validation story: Phase 2 research {selection_window_text}, Phase 2 validation {validation_window_text}, and reserved Phase 3 holdout {holdout_window_text_value}.")}
      <p class="muted"><strong>Why the validation starts years back:</strong> this page is asking whether the strategy behaves like a real repeatable edge across multiple old regimes, not just whether it looked good recently.</p>
      <p class="muted"><strong>What happens after the test period:</strong> the later holdout stays outside Phase 2 selection so you still have cleaner forward evidence after this walk-forward ends.</p>
      <div class="grid">
        <div class="card full">
          {protocol_chart}
        </div>
        <div class="card full">
          {wf_ladder_chart}
        </div>
      </div>
    </section>
    <section>
      <h2>Top Candidates</h2>
      {render_timeframe_note(f"Candidate ranks on this page are built from Phase 2 validation windows {validation_window_text} only.")}
      <table>
        <thead>
          <tr><th>Candidate</th><th>Params</th><th>Median Sharpe</th><th>Fold Passes</th><th>Bootstrap</th><th>Deflated</th></tr>
        </thead>
        <tbody>
          {top_rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Walk-Forward Timeline</h2>
      {render_timeframe_note(f"Every chart in this section uses the stitched Phase 2 validation record over {validation_window_text}. The untouched Phase 3 holdout {holdout_window_text_value} is excluded here.")}
      {render_timeframe_note(f"All charts in this section use the stitched Phase 2 validation series over {validation_window_text} ({len(wf_combined_returns)} monthly observations when available).")}
      <p class="muted">Status: {walk_forward.get('status','n/a') if isinstance(walk_forward, dict) else 'n/a'} · Combined Sharpe {format_float(wf_combined.get('sharpe'))} · Total Return {format_pct(wf_combined.get('total_return'))}</p>
      <p class="muted"><strong>Hard-gate pass/fail</strong> is shown in the Validation Verdict above. This timeline is diagnostic.</p>
      <div class="badge-row"><span class="label">Walk-forward quality (diagnostic)</span>{wf_quality_html}</div>
      <p class="muted"><strong>Result (diagnostic):</strong> {wf_diag}.</p>
      <p class="muted"><strong>Gap summary:</strong> Sharpe {wf_gap_sharpe}. Annualized Return {wf_gap_return}. {wf_gap_months}.</p>
      <p class="muted"><strong>What the chart shows:</strong> Green line = validation (out-of-sample). Gray dashed = train (in-sample). X-axis is the mid-year of each validation window; shaded bands show each validation window.</p>
      {benchmark_context}
      <p class="muted"><strong>Integrity boundary:</strong> These validation windows are out-of-sample relative to each fold's training span. The untouched final holdout starts at {config.OOS_START} and is not part of this walk-forward schedule.</p>
      <p class="muted">Diagnostic summary (not a hard gate): {wf_gate_text}. Dashed horizontal line is the 0.4 reference; solid horizontal line is zero (break-even) when visible.</p>
      <p class="muted"><strong>What it means:</strong> Validation close to train with few negatives = more stable; big gaps or repeated negatives = weaker robustness.</p>
      <p class="muted"><strong>Higher-frequency view:</strong> The stitched validation equity and rolling Sharpe below update monthly, so you can see movement inside folds.</p>
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
              train_value_fn=lambda fold: _fold_cagr(fold, total_key="train_total_return", months_key="train_months"),
              validate_value_fn=lambda fold: _fold_cagr(fold, total_key="validate_total_return", months_key="validate_months"),
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
    <section>
      <h2>Monte Carlo Distribution</h2>
      {render_timeframe_note(f"Monte Carlo uses the frozen Phase 2 validation return history over {validation_window_text}; it does not include or project the Phase 3 holdout {holdout_window_text_value}.")}
      <p class="muted">Samples: {monte.get('sample_count','n/a')} · Block length: {monte.get('block_length_months','n/a')} months</p>
      <p class="muted">Method: stationary block bootstrap of historical monthly returns (not IID random-walk).</p>
      <p class="muted"><strong>Integrity boundary:</strong> This bootstrap uses realized historical returns from the frozen candidate's validation record. It is diagnostic only and does not project or consume the untouched holdout.</p>
      <div class="badge-row"><span class="label">Monte Carlo quality (not a gate)</span>{mc_badge_html}</div>
      <p class="muted"><strong>Result (diagnostic):</strong> {mc_badge_label}. {mc_badge_detail}</p>
      <p class="muted"><strong>What the histograms show:</strong> X-axis = metric value, Y-axis = frequency across bootstrap samples. Taller bars = more likely outcomes. Higher Sharpe/return is better; lower drawdown is better. Markers show p05/median/p95.</p>
      <p class="muted"><strong>What the paths show:</strong> X-axis = months from start; Y-axis = equity index (start=100). Thin lines are sample bootstrap paths; the blue fan shows path density (darker = more likely); bold line is the median; shaded band is p10-p90.</p>
      <p class="muted"><strong>What it means:</strong> Distributions centered above zero with tight spread are stronger; large left tails or very wide fans signal more uncertainty.</p>
      <p class="muted"><strong>Bottom line:</strong> {mc_interp}</p>
      {mc_anomaly}
      <div class="grid">
        <div class="card">
          {render_histogram_svg(mc_hist.get("sharpe"), metric=mc_sharpe, title="Sharpe", x_label="Sharpe", as_pct=False)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get("total_return"), metric=mc_total, title="Total Return From Start", x_label="Total Return From Start (%)", as_pct=True)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get("max_drawdown"), metric=mc_dd, title="Max Drawdown", x_label="Max Drawdown (%)", as_pct=True)}
        </div>
        <div class="card full">
          {render_spaghetti_svg(mc_paths, title="Monte Carlo Paths (Equity Index)", display_quantile_range=(0.05, 0.95), path_quantiles=mc_quantiles, path_density=mc_density)}
        </div>
      </div>
      {trimmed_html}
    </section>
    <section>
      <h2>Parameter Sensitivity</h2>
      {render_timeframe_note(f"Sensitivity charts summarize Phase 2 fold outcomes across {validation_window_text}; they are not computed on the holdout.")}
      {sensitivity_sections}
      <p class="muted" style="margin-top:8px;"><strong>What the chart shows:</strong> X-axis = parameter value; Y-axis = median validation Sharpe. Flatter lines = more robust; steep slopes = sensitivity/overfit. Largest dot marks the best value. Table retains pass rates and counts.</p>
    </section>
  </main>
</body>
</html>
"""


def build_holdout_dashboard(
    *,
    holdout: dict,
    selection_summary: dict[str, Any] | None = None,
    title: str,
    subtitle: str,
    back_href: str,
    back_label: str,
    phase1_href: str = "phase1_summary.html",
    selection_href: str = "selection_summary.html",
    holdout_href: str = "holdout_results.html",
    diagnostic_holdout: dict[str, Any] | None = None,
    diagnostic_label: str | None = None,
    diagnostic_href: str | None = None,
) -> str:
    phase4 = holdout.get("phase4_gate", {})
    holdout_window = holdout.get("holdout_window", {}) if isinstance(holdout, dict) else {}
    window_start = holdout_window.get("start") or getattr(config, "OOS_START", None)
    window_end = holdout_window.get("end") or getattr(config, "OOS_END", None)
    window_text = f"{window_start} to {window_end}" if window_start and window_end else (window_start or window_end or "n/a")
    track_bundle = _find_holdout_primary_track(holdout)
    fallback_track_bundle = _find_holdout_primary_track(diagnostic_holdout)
    preserved_has_returns = bool(track_bundle and (track_bundle[1].get("strategy_returns") or []))
    fallback_has_returns = bool(fallback_track_bundle and (fallback_track_bundle[1].get("strategy_returns") or []))
    use_diagnostic_replay = (not preserved_has_returns) and fallback_has_returns
    if use_diagnostic_replay:
        track_bundle = fallback_track_bundle
    track_meta = track_bundle[0] if track_bundle else {}
    track_metrics = track_bundle[1] if track_bundle else {}
    track_source_holdout = diagnostic_holdout if use_diagnostic_replay else holdout
    diagnostic_phase4 = (diagnostic_holdout or {}).get("phase4_gate", {}) if isinstance(diagnostic_holdout, dict) else {}
    diagnostic_window = (diagnostic_holdout or {}).get("holdout_window", {}) if isinstance(diagnostic_holdout, dict) else {}
    diagnostic_window_start = diagnostic_window.get("start") or window_start
    diagnostic_window_end = diagnostic_window.get("end") or window_end
    diagnostic_window_text = (
        f"{diagnostic_window_start} to {diagnostic_window_end}"
        if diagnostic_window_start and diagnostic_window_end
        else (diagnostic_window_start or diagnostic_window_end or window_text)
    )
    track_label = (
        f"{track_meta.get('universe_variant','n/a')} / {track_meta.get('execution_model','n/a')} / "
        f"{track_meta.get('fx_scenario','n/a')} / cost={track_meta.get('cost_model_name','n/a')}"
        if track_meta
        else "n/a"
    )
    periods_per_year = int(
        track_metrics.get("periods_per_year")
        or track_source_holdout.get("periods_per_year")
        or holdout.get("periods_per_year")
        or 12
    )
    track_returns = track_metrics.get("strategy_returns") or []
    track_benchmark_returns = track_metrics.get("primary_benchmark_returns") or []
    monte = (
        monte_carlo_summary(
            track_returns,
            periods_per_year=periods_per_year,
            n_resamples=config.MONTE_CARLO_RESAMPLES,
            block_length_months=config.MONTE_CARLO_BLOCK_LENGTH_MONTHS,
            seed=config.MONTE_CARLO_SEED,
        )
        if track_returns
        else {}
    )
    mc_metrics = monte.get("metrics", {}) if isinstance(monte, dict) else {}
    mc_sharpe = mc_metrics.get("sharpe", {})
    mc_total = mc_metrics.get("total_return", {})
    mc_dd = mc_metrics.get("max_drawdown", {})
    mc_hist = monte.get("histograms", {}) if isinstance(monte, dict) else {}
    mc_paths = monte.get("sample_paths", []) if isinstance(monte, dict) else []
    mc_quantiles = monte.get("path_quantiles", {}) if isinstance(monte, dict) else {}
    mc_density = monte.get("path_density", {}) if isinstance(monte, dict) else {}
    mc_badge_label, mc_badge_tone, mc_badge_detail = monte_carlo_badge(mc_sharpe, mc_total, mc_dd)
    mc_badge_html = render_badge(mc_badge_label, mc_badge_tone, mc_badge_detail)
    mc_interp = monte_carlo_interpretation(mc_sharpe, mc_total, mc_dd)
    mc_anomaly = render_monte_carlo_anomaly_panel(monte)
    trimmed_html = render_trimmed_monte_carlo(
        monte,
        display_quantile_range=(0.01, 0.99),
    )
    selection_walk_forward = (selection_summary or {}).get("walk_forward", {}) if selection_summary else {}
    selection_folds = selection_walk_forward.get("folds", []) if isinstance(selection_walk_forward, dict) else []
    phase2_start = config.ROLLING_ORIGIN_FOLDS[0][1] if config.ROLLING_ORIGIN_FOLDS else None
    after_holdout = _ordinal_to_month(_month_to_ordinal(window_end) + 1) if window_end else None
    boundary_chart = render_phase_boundary_svg(
        [
            {
                "label": "Selection history",
                "detail": "Phase 2 research and locking",
                "start": phase2_start,
                "end": config.INSAMPLE_END,
                "fill": "#d9cab8",
            },
            {
                "label": "One-shot holdout",
                "detail": "this page's main test window",
                "start": window_start,
                "end": window_end,
                "fill": "#eadca7",
                "highlight": True,
            },
            {
                "label": "After holdout",
                "detail": "paper/live or future appended data",
                "start": after_holdout,
                "fill": "#efe6d7",
            },
        ],
        footer_note=(
            "This page is meant to be a one-shot test: the candidate should already be frozen before the orange "
            "window begins."
        ),
    )
    lineage_chart = render_walkforward_ladder_svg(
        selection_folds,
        title="Phase 2 Lineage (Context Only)",
        stitched_label="Phase 2 stitched OOS",
    )
    phase3_folds = _build_holdout_walkforward_folds(
        start_month=window_start,
        end_month=window_end,
        fold_count=5,
        train_start=phase2_start,
    )
    phase3_fold_count = len(phase3_folds)
    phase3_fold_count_display = phase3_fold_count if phase3_fold_count else "n/a"
    phase3_schedule_folds = clip_walkforward_folds_to_window(
        phase3_folds,
        clip_start=window_start,
        clip_end=window_end,
    )
    phase3_schedule_chart = render_walkforward_schedule_svg(
        phase3_schedule_folds,
        title="Phase 3 Walk-Forward Schedule",
    )
    phase3_oos_equity_chart = render_walkforward_oos_equity_svg(
        track_returns,
        phase3_folds,
        benchmark_returns=track_benchmark_returns,
        benchmark_label=config.PRIMARY_PASSIVE_BENCHMARK,
        title="Phase 3 Holdout Equity (Stitched)",
    )
    phase3_rolling_sharpe_chart = render_walkforward_rolling_sharpe_svg(
        track_returns,
        phase3_folds,
        periods_per_year=periods_per_year,
        title="Rolling Sharpe (Phase 3 Holdout, 12m)",
    )
    evidence_stack_html = render_evidence_stack_html(
        selection_summary=selection_summary,
        holdout=holdout,
        selection_href=selection_href,
        holdout_href=holdout_href,
    )
    track_rows: list[str] = []
    results = track_source_holdout.get("results", {}) if isinstance(track_source_holdout, dict) else {}
    if isinstance(results, dict):
        for universe_variant, execution_bundle in results.items():
            if not isinstance(execution_bundle, dict):
                continue
            for execution_model, fx_bundle in execution_bundle.items():
                if not isinstance(fx_bundle, dict):
                    continue
                for fx_scenario, row in fx_bundle.items():
                    if not isinstance(row, dict) or row.get("total_return") is None:
                        continue
                    track_rows.append(
                        "<tr>"
                        f"<td>{universe_variant} / {execution_model} / {fx_scenario}</td>"
                        f"<td>{format_float(row.get('net_sharpe'))}</td>"
                        f"<td>{format_pct(_annualized_return(row.get('total_return'), row.get('months'), periods_per_year))}</td>"
                        f"<td>{format_pct(row.get('total_return'))}</td>"
                        f"<td>{format_pct(row.get('primary_benchmark_total_return'))}</td>"
                        f"<td>{format_pct(row.get('max_drawdown'))}</td>"
                        f"<td>{row.get('beats_primary_benchmark')}</td>"
                        "</tr>"
                    )
    track_table_html = (
        "<table>"
        "<thead><tr><th>Track</th><th>Sharpe</th><th>CAGR</th><th>Total Return</th><th>Benchmark Return</th><th>Max Drawdown</th><th>Beats Benchmark</th></tr></thead>"
        f"<tbody>{''.join(track_rows)}</tbody>"
        "</table>"
        if track_rows
        else "<p class=\"muted\">No holdout track table is available.</p>"
    )
    blocked_holdout_html = ""
    phase3_walkforward_section = ""
    primary_track_section = ""
    track_comparison_section = ""
    monte_carlo_section = ""
    diagnostic_replay_html = ""
    equity_section_title = "Primary Track Equity"
    comparison_section_title = "Track Comparison"
    monte_carlo_section_title = "Monte Carlo Distribution"
    chart_window_text = diagnostic_window_text if use_diagnostic_replay else window_text
    if not track_returns:
        blocked_holdout_html = (
            "<section>"
            "<h2>Phase 3 Run Status</h2>"
            "<div style=\"margin-top:12px; padding:12px 14px; border-radius:16px; border:1px solid rgba(176,107,29,.28); background:#fff7eb; color:#5b6762;\">"
            "<strong style=\"color:#4b4031;\">No Phase 3 simulation was produced for this branch.</strong>"
            f"<div style=\"margin-top:6px;\">Status is <code>{html.escape(str(holdout.get('status', 'n/a')))}</code>. "
            "Phase 2 did not lock a candidate for this path, so there is no holdout equity curve, no track table, and no Monte Carlo sample to display. "
            "This page is only showing the reserved holdout window and the lineage boundary.</div>"
            "</div>"
            "</section>"
        )
    else:
        if use_diagnostic_replay:
            source_html = (
                f' <a href="{diagnostic_href}">Open the replay source</a>.'
                if diagnostic_href
                else ""
            )
            diagnostic_label_text = diagnostic_label or "a current exact-candidate replay"
            diagnostic_replay_html = (
                "<section>"
                "<h2>Diagnostic Replay Source</h2>"
                f"{render_timeframe_note(f'The preserved holdout window is {window_text}. The charts below use a replay over {diagnostic_window_text} because the rebuild snapshot only kept summary metrics, not monthly returns.')}"
                "<div style=\"margin-top:12px; padding:12px 14px; border-radius:16px; border:1px solid rgba(176,107,29,.28); background:#fff7eb; color:#5b6762;\">"
                "<strong style=\"color:#4b4031;\">The preserved rebuild snapshot did not store month-by-month holdout returns.</strong>"
                f"<div style=\"margin-top:6px;\">To make Phase 3 path risk visible, the charts below use {html.escape(diagnostic_label_text)}. Keep the historical snapshot metrics above separate from these replay diagnostics.{source_html}</div>"
                "</div>"
                "<div class=\"grid\">"
                f"<div class=\"card\"><div class=\"label\">Preserved Snapshot Sharpe</div><div class=\"value\">{format_float(phase4.get('base_main_net_sharpe'))}</div></div>"
                f"<div class=\"card\"><div class=\"label\">Diagnostic Replay Sharpe</div><div class=\"value\">{format_float(diagnostic_phase4.get('base_main_net_sharpe'))}</div></div>"
                f"<div class=\"card\"><div class=\"label\">Preserved Window</div><div class=\"value\">{window_text}</div></div>"
                f"<div class=\"card\"><div class=\"label\">Replay Window</div><div class=\"value\">{diagnostic_window_text}</div></div>"
                f"<div class=\"card\"><div class=\"label\">Replay Track</div><div class=\"value\">{track_label}</div></div>"
                "</div>"
                "</section>"
            )
            equity_section_title = "Diagnostic Replay Equity"
            comparison_section_title = "Diagnostic Replay Track Comparison"
            monte_carlo_section_title = "Diagnostic Replay Monte Carlo"
        phase3_walkforward_section = f"""
    <section>
      <h2>Phase 3 Walk-Forward View</h2>
      {render_timeframe_note(f"Charts below split the Phase 3 holdout window {window_text} into {phase3_fold_count_display} sequential slices for visualization. The candidate is frozen; this does not re-optimize or reselect.")}
      <p class="muted">Schedule, stitched equity, and rolling Sharpe are computed on the same holdout return series. The schedule axis is clipped to the Phase 3 window ({window_text}) so the earlier training history does not dominate the view.</p>
      <div class="grid">
        <div class="card full">{phase3_schedule_chart}</div>
        <div class="card full">{phase3_oos_equity_chart}</div>
        <div class="card full">{phase3_rolling_sharpe_chart}</div>
      </div>
    </section>
"""
        primary_track_section = f"""
    <section>
      <h2>{equity_section_title}</h2>
      {render_timeframe_note(f"Return curve uses the {'diagnostic replay' if use_diagnostic_replay else 'frozen holdout'} series from {chart_window_text} on the primary track.")}
      {render_holdout_equity_svg(
          track_returns,
          track_benchmark_returns,
          window_start=diagnostic_window_start if use_diagnostic_replay else window_start,
          benchmark_label=config.PRIMARY_PASSIVE_BENCHMARK,
      )}
    </section>
"""
        track_comparison_section = f"""
    <section>
      <h2>{comparison_section_title}</h2>
      {render_timeframe_note(f"Every row below is evaluated on the same {'diagnostic replay' if use_diagnostic_replay else 'Phase 3 holdout'} window {chart_window_text}.")}
      {track_table_html}
    </section>
"""
        monte_carlo_section = f"""
    <section>
      <h2>{monte_carlo_section_title}</h2>
      {render_timeframe_note(f"Monte Carlo uses the realized {'diagnostic replay' if use_diagnostic_replay else 'Phase 3 holdout'} monthly return series from {chart_window_text}. It is diagnostic only and does not project beyond this window.")}
      <p class="muted">Sample count: {monte.get('sample_count','n/a')} / Block length: {monte.get('block_length_months','n/a')} months</p>
      <p class="muted">Method: stationary block bootstrap of the primary-track monthly holdout returns.</p>
      <div class="badge-row"><span class="label">Monte Carlo quality (diagnostic)</span>{mc_badge_html}</div>
      <p class="muted"><strong>Result (diagnostic):</strong> {mc_badge_label}. {mc_badge_detail}</p>
      <p class="muted"><strong>Bottom line:</strong> {mc_interp}</p>
      {mc_anomaly}
      <div class="grid">
        <div class="card">
          {render_histogram_svg(mc_hist.get("sharpe"), metric=mc_sharpe, title="Sharpe", x_label="Sharpe", as_pct=False)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get("total_return"), metric=mc_total, title="Total Return From Start", x_label="Total Return From Start (%)", as_pct=True)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get("max_drawdown"), metric=mc_dd, title="Max Drawdown", x_label="Max Drawdown (%)", as_pct=True)}
        </div>
        <div class="card full">
          {render_spaghetti_svg(mc_paths, title="Monte Carlo Paths (Equity Index)", display_quantile_range=(0.03, 0.97), path_quantiles=mc_quantiles, path_density=mc_density)}
        </div>
      </div>
      {trimmed_html}
    </section>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:980px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
    .card {{ background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:18px; padding:14px; }}
    .label {{ text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; color:#6c5842; }}
    .value {{ font-size:1.1rem; margin-top:4px; }}
    .muted {{ color:#5b6762; }}
    .badge-row {{ display:flex; align-items:center; gap:8px; margin:6px 0 6px; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; }}
    .badge.good {{ background:#dcefe3; color:#2f5c46; border:1px solid #a9c7b4; }}
    .badge.caution {{ background:#f4ead7; color:#7a5a25; border:1px solid #d9c59a; }}
    .badge.weak {{ background:#f3d7d7; color:#7a2f2f; border:1px solid #d6a3a3; }}
    .badge.neutral {{ background:#ece7dd; color:#6c5842; border:1px solid #d7cdbc; }}
    .card.full {{ grid-column: 1 / -1; }}
    .chart-block {{ display:flex; flex-direction:column; gap:6px; }}
    .chart-title {{ font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:#6c5842; }}
    .chart {{ width:100%; height:auto; }}
    .chart text {{ font-family:Georgia, serif; font-size:10px; fill:#5d685f; }}
    .chart-note {{ font-size:.78rem; color:#6c5842; }}
    .legend-line {{ font-weight:600; color:#7a6aa0; }}
    .legend-line.median {{ color:#4f7a67; }}
    .legend-line.band {{ color:#bfa47a; }}
    .legend-line.density {{ color:#2f9abf; }}
    a {{ color:#b06b1d; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
  </style>
</head>
<body>
  <main>
    <section>
      <p class="label">Phase 3</p>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      {render_timeframe_note(f"This page is the untouched Phase 3 holdout for {window_text}. Any charts below use that window unless they explicitly say they are Phase 2 lineage charts.")}
      <a href="{back_href}">{back_label}</a>
    </section>
    {render_phase_map_html(phase1_href=phase1_href, selection_href=selection_href, holdout_href=holdout_href, holdout=holdout)}
    {evidence_stack_html}
    <section>
      <h2>Boundary Insight</h2>
      {render_timeframe_note(f"The boundary chart spans Phase 2 research {phase2_selection_window_text()}, the untouched Phase 3 holdout {window_text}, and the future Phase 4 period after {after_holdout or 'n/a'}.")}
      <p class="muted"><strong>What this page is testing:</strong> the orange holdout block is the single post-selection evaluation window. It is not another rolling walk-forward stage on this shared holdout page.</p>
      <div class="grid">
        <div class="card" style="grid-column:1 / -1;">{boundary_chart}</div>
        <div class="card" style="grid-column:1 / -1;">{lineage_chart}</div>
      </div>
    </section>
    <section>
      <h2>Holdout Status</h2>
      <div class="grid">
        <div class="card"><div class="label">Status</div><div class="value">{holdout.get('status','n/a')}</div></div>
        <div class="card"><div class="label">Phase 4 Eligible</div><div class="value">{phase4.get('phase4_eligible','n/a')}</div></div>
        <div class="card"><div class="label">Base Holdout Sharpe</div><div class="value">{format_float(phase4.get('base_main_net_sharpe'))}</div></div>
        <div class="card"><div class="label">Sharpe Gate</div><div class="value">{phase4.get('meets_sharpe_gate','n/a')}</div></div>
        <div class="card"><div class="label">Benchmark Gate</div><div class="value">{phase4.get('beats_primary_benchmark','n/a')}</div></div>
      </div>
    </section>
    <section>
      <h2>Holdout Window</h2>
      <div class="grid">
        <div class="card"><div class="label">Window</div><div class="value">{window_text}</div></div>
        <div class="card"><div class="label">Period Label</div><div class="value">{holdout.get('period_label','n/a')}</div></div>
        <div class="card"><div class="label">Periods per Year</div><div class="value">{holdout.get('periods_per_year','n/a')}</div></div>
      </div>
    </section>
    {blocked_holdout_html}
    {diagnostic_replay_html}
    {phase3_walkforward_section}
    {primary_track_section}
    {track_comparison_section}
    {monte_carlo_section}
  </main>
</body>
</html>
"""


def build_thesis_dashboard(
    *,
    thesis: dict[str, Any],
    quick: dict[str, Any],
    mega: dict[str, Any],
    certification: dict[str, Any],
    holdout: dict[str, Any] | None,
    profile_set: str,
    profile_settings: dict[str, Any],
    context_note: str | None = None,
    selection_href: str = "selection_summary.html",
    holdout_href: str = "holdout_results.html",
) -> str:
    certification = ensure_summary_monte_carlo(certification)
    selection_status = certification.get("selection_status", "n/a")
    backtest = certification.get("backtest_overfitting", {})
    pbo_value = backtest.get("pbo")
    pbo_text = format_pbo_display(pbo_value, backtest)
    pbo_threshold = backtest.get("pbo_threshold_max")
    pbo_pass = backtest.get("passes_pbo_threshold")
    pbo_band_value = certification.get("pbo_band", "n/a")
    pbo_policy = certification.get("pbo_policy", {})
    pbo_good_max = pbo_policy.get("good_max", config.PBO_THRESHOLD_MAX)
    pbo_hard_cutoff_min = pbo_policy.get("hard_cutoff_min", config.PBO_HARD_CUTOFF)
    holdout_gate = holdout.get("phase4_gate", {}) if holdout else {}
    holdout_text = format_float(holdout_gate.get("base_main_net_sharpe"))
    locked_params = (certification.get("locked_candidate") or {}).get("params")
    walk_forward = certification.get("walk_forward", {})
    wf_combined = walk_forward.get("combined", {}) if isinstance(walk_forward, dict) else {}
    monte = certification.get("monte_carlo", {})
    mc_metrics = monte.get("metrics", {}) if isinstance(monte, dict) else {}
    mc_sharpe = mc_metrics.get("sharpe", {})
    mc_total = mc_metrics.get("total_return", {})
    mc_dd = mc_metrics.get("max_drawdown", {})
    mc_hist = monte.get("histograms", {}) if isinstance(monte, dict) else {}
    mc_paths = monte.get("sample_paths", []) if isinstance(monte, dict) else []
    wf_folds = walk_forward.get("folds", []) if isinstance(walk_forward, dict) else []
    wf_combined_returns = walk_forward.get("combined_returns", []) if isinstance(walk_forward, dict) else []
    wf_combined_benchmark_returns = (
        walk_forward.get("combined_benchmark_returns", []) if isinstance(walk_forward, dict) else []
    )
    wf_periods_per_year = wf_combined.get("periods_per_year", 12) if isinstance(wf_combined, dict) else 12
    sensitivity_sections = render_sensitivity_sections(certification.get("parameter_sensitivity"))
    wf_gate_text = walk_forward_gate_summary(wf_folds)
    mc_interp = monte_carlo_interpretation(mc_sharpe, mc_total, mc_dd)
    verdict_panel = render_validation_verdict(certification)
    wf_diag = walk_forward_diagnostic_result(walk_forward)
    wf_quality_label, wf_quality_tone, wf_quality_detail = walk_forward_quality_badge(wf_folds, gate_threshold=0.4)
    wf_quality_html = render_badge(wf_quality_label, wf_quality_tone, wf_quality_detail)
    wf_gap_sharpe = walk_forward_gap_summary(
        wf_folds, train_key="train_sharpe", validate_key="validate_sharpe", as_pct=False
    )
    wf_gap_return = walk_forward_gap_summary(
        wf_folds,
        train_key="train_total_return",
        validate_key="validate_total_return",
        as_pct=True,
        train_value_fn=lambda fold: _fold_cagr(fold, total_key="train_total_return", months_key="train_months"),
        validate_value_fn=lambda fold: _fold_cagr(fold, total_key="validate_total_return", months_key="validate_months"),
    )
    wf_gap_months = walk_forward_gap_months_summary(wf_folds)
    wf_oos_equity_chart = render_walkforward_oos_equity_svg(
        wf_combined_returns,
        wf_folds,
        benchmark_returns=wf_combined_benchmark_returns,
        benchmark_label=config.PRIMARY_PASSIVE_BENCHMARK,
    )
    wf_rolling_sharpe_chart = render_walkforward_rolling_sharpe_svg(
        wf_combined_returns, wf_folds, periods_per_year=wf_periods_per_year
    )
    wf_schedule_chart = render_walkforward_schedule_svg(wf_folds)
    if thesis.get("name") == "ex_norway":
        benchmark_context = (
            f"<p class=\"muted\"><strong>Benchmark context:</strong> The gold line is "
            f"<strong>{config.PRIMARY_PASSIVE_BENCHMARK}</strong>, the project's broad Nordic passive proxy. "
            "For the ex_norway thesis this still includes Norway, so it is a house benchmark rather than a thesis-matched ex-Norway passive comparator.</p>"
        )
    else:
        benchmark_context = (
            f"<p class=\"muted\"><strong>Benchmark context:</strong> The gold line is "
            f"<strong>{config.PRIMARY_PASSIVE_BENCHMARK}</strong>, the project's primary passive benchmark proxy for the same validation windows.</p>"
        )
    mc_badge_label, mc_badge_tone, mc_badge_detail = monte_carlo_badge(mc_sharpe, mc_total, mc_dd)
    mc_badge_html = render_badge(mc_badge_label, mc_badge_tone, mc_badge_detail)
    trimmed_html = render_trimmed_monte_carlo(monte)
    mc_anomaly = render_monte_carlo_anomaly_panel(monte)

    cert_candidates = certification.get("ranked_candidates") or []
    top_cert = top_candidates(cert_candidates, limit=1)[0] if cert_candidates else {}
    top_cert_params = top_cert.get("params")
    pbo_blocks_progression = bool(certification.get("pbo_hard_cutoff")) or selection_status == "pbo_hard_cutoff"
    top_blocker_list = gate_failures(top_cert) if top_cert else []
    if pbo_blocks_progression and "pbo_hard_cutoff" not in top_blocker_list:
        top_blocker_list.append("pbo_hard_cutoff")
    top_blockers = ", ".join(top_blocker_list) if top_blocker_list else "n/a"
    gate_pass_counts = count_gate_passes(cert_candidates)
    gate_fail_counts = count_gate_failures(cert_candidates)
    hard_gate_winners = sum(
        1
        for candidate in cert_candidates
        if candidate.get("gate_fold_count")
        and candidate.get("gate_bootstrap")
        and candidate.get("gate_deflated_sharpe")
        and candidate.get("gate_negative_controls")
    )
    if pbo_blocks_progression:
        hard_gate_winners = 0
    neg_controls = certification.get("negative_controls", {})
    cross = neg_controls.get("cross_sectional_shuffle", {})
    block = neg_controls.get("block_shuffled_null", {})
    mcs_score = compute_mcs(top_cert)

    def _top_candidate(summary: dict[str, Any]) -> dict[str, Any]:
        ranked = summary.get("ranked_candidates") or []
        return sorted(ranked, key=lambda item: item.get("rank", 9999))[0] if ranked else {}

    top_quick = _top_candidate(quick)
    top_mega = _top_candidate(mega)
    profile_desc = format_profile_set(profile_set, profile_settings)

    frozen_single_candidate = (
        quick.get("backtest_overfitting", {}).get("candidate_count") == 1
        and mega.get("backtest_overfitting", {}).get("candidate_count") == 1
        and certification.get("backtest_overfitting", {}).get("candidate_count") == 1
        and top_quick.get("candidate_id")
        and top_quick.get("candidate_id") == top_mega.get("candidate_id") == top_cert.get("candidate_id")
    )
    if frozen_single_candidate:
        profile_rows = []
        for label, summary, top in (
            ("Quick", quick, top_quick),
            ("Mega", mega, top_mega),
            ("Certification", certification, top_cert),
        ):
            neg = summary.get("negative_controls", {})
            cross = neg.get("cross_sectional_shuffle", {})
            block = neg.get("block_shuffled_null", {})
            wf = summary.get("walk_forward", {})
            profile_rows.append(
                "<tr>"
                f"<td>{label}</td>"
                f"<td>{top.get('candidate_id','n/a')}</td>"
                f"<td>{cross.get('run_count','n/a')}</td>"
                f"<td>{block.get('run_count','n/a')}</td>"
                f"<td>{wf.get('status','n/a')}</td>"
                f"<td>{format_pct(summary.get('backtest_overfitting', {}).get('pbo'))}</td>"
                "</tr>"
            )
        profile_section_title = "Profile Diagnostics (Frozen Candidate)"
        profile_section_note = (
            "<p class=\"muted\" style=\"margin-top:10px;\">"
            "This rerun freezes one exact candidate tuple, so the top candidate is expected to be identical across "
            "Quick, Mega, and Certification. The real differences here are diagnostic depth and whether walk-forward ran."
            "</p>"
        )
        profile_table_head = (
            "<tr><th>Profile</th><th>Frozen Candidate</th><th>Cross Shuffle Runs</th>"
            "<th>Block-Null Runs</th><th>Walk-Forward</th><th>PBO</th></tr>"
        )
    else:
        profile_rows = []
        for label, summary, top in (
            ("Quick", quick, top_quick),
            ("Mega", mega, top_mega),
            ("Certification", certification, top_cert),
        ):
            profile_rows.append(
                "<tr>"
                f"<td>{label}</td>"
                f"<td>{top.get('candidate_id','n/a')}</td>"
                f"<td>{format_params(top.get('params'))}</td>"
                f"<td>{format_float(top.get('median_validation_sharpe'))}</td>"
                f"<td>{format_float(top.get('total_return'))}</td>"
                f"<td>{format_float(top.get('max_drawdown'))}</td>"
                f"<td>{format_pct(summary.get('backtest_overfitting', {}).get('pbo'))}</td>"
                "</tr>"
            )
        profile_section_title = "Profile Comparison (Top Candidate)"
        profile_section_note = ""
        profile_table_head = (
            "<tr><th>Profile</th><th>Candidate</th><th>Params</th><th>Median Sharpe</th>"
            "<th>Total Return</th><th>Max Drawdown</th><th>PBO</th></tr>"
        )
    profile_table = "\n".join(profile_rows)

    fold_sharpes = top_cert.get("per_fold_sharpes") or {}
    fold_drawdowns = top_cert.get("per_fold_drawdowns") or {}
    fold_rows = "\n".join(
        f"<tr><td>{fold_id}</td><td>{format_float(fold_sharpes.get(fold_id))}</td><td>{format_float(fold_drawdowns.get(fold_id))}</td></tr>"
        for fold_id in sorted(fold_sharpes.keys())
    )
    crisis_rows, crisis_flag = build_crisis_lens_rows(walk_forward if isinstance(walk_forward, dict) else None)
    crisis_callout = ""
    if crisis_flag:
        crisis_callout = (
            "<div class=\"callout\">"
            "<strong>Integrity note.</strong>"
            "<div class=\"muted\">Crisis windows (2008-2009) are retained in validation; asterisked folds below are a context marker only.</div>"
            "</div>"
        )
    evidence_stack_html = render_evidence_stack_html(
        selection_summary=certification,
        holdout=holdout,
        selection_href=selection_href,
        holdout_href=holdout_href,
    )
    context_note_html = (
        f'<div class="callout"><strong>Context.</strong><div class="muted">{html.escape(context_note)}</div></div>'
        if context_note
        else ""
    )
    phase2_start = config.ROLLING_ORIGIN_FOLDS[0][1] if config.ROLLING_ORIGIN_FOLDS else None
    holdout_window = holdout.get("holdout_window", {}) if holdout else {}
    holdout_start = holdout_window.get("start") or config.OOS_START
    holdout_end = holdout_window.get("end") or config.OOS_END
    after_holdout = _ordinal_to_month(_month_to_ordinal(holdout_end) + 1) if holdout_end else None
    protocol_chart = render_phase_boundary_svg(
        [
            {
                "label": "Phase 2 walk-forward",
                "detail": "older-regime validation",
                "start": phase2_start,
                "end": config.INSAMPLE_END,
                "fill": "#d9cab8",
                "highlight": True,
            },
            {
                "label": "Phase 3 holdout",
                "detail": "after-test untouched data",
                "start": holdout_start,
                "end": holdout_end,
                "fill": "#eadca7",
            },
            {
                "label": "After holdout",
                "detail": "paper/live or new data",
                "start": after_holdout,
                "fill": "#efe6d7",
            },
        ],
        footer_note=(
            "Old folds are not there to predict this month directly. They are there to prove the idea did not only "
            "work in one recent lucky regime before the later holdout was touched."
        ),
    )
    wf_ladder_chart = render_walkforward_ladder_svg(
        wf_folds,
        stitched_label="Stitched validation record",
    )
    selection_window_text = phase2_selection_window_text()
    validation_window_text = phase2_validation_window_text(wf_folds)
    holdout_window_text_value = holdout_window_text(holdout)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Engine Thesis</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:1080px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin-top:16px; }}
    .card {{ background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:20px; padding:18px; }}
    .label {{ color:#6c5842; text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; }}
    .value {{ font-size:1.3rem; margin-top:6px; overflow-wrap:anywhere; }}
    .muted {{ color:#5b6762; }}
    .callout {{ background:#fff7eb; border:1px solid rgba(176,107,29,.28); border-radius:16px; padding:12px 14px; margin-top:12px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
    a {{ color:#b06b1d; }}
    .card.full {{ grid-column: 1 / -1; }}
    .chart-block {{ display:flex; flex-direction:column; gap:6px; }}
    .chart-title {{ font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:#6c5842; }}
    .chart {{ width:100%; height:auto; }}
    .chart text {{ font-family:Georgia, serif; font-size:10px; fill:#5d685f; }}
    .chart-note {{ font-size:.78rem; color:#6c5842; }}
    .legend-line {{ font-weight:600; color:#7a6aa0; }}
    .legend-line.median {{ color:#4f7a67; }}
    .legend-line.band {{ color:#bfa47a; }}
    .legend-line.density {{ color:#2f9abf; }}
    .badge-row {{ display:flex; align-items:center; gap:8px; margin:6px 0 6px; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700; }}
    .badge.good {{ background:#dcefe3; color:#2f5c46; border:1px solid #a9c7b4; }}
    .badge.caution {{ background:#f4ead7; color:#7a5a25; border:1px solid #d9c59a; }}
    .badge.weak {{ background:#f3d7d7; color:#7a2f2f; border:1px solid #d6a3a3; }}
    .badge.neutral {{ background:#ece7dd; color:#6c5842; border:1px solid #d7cdbc; }}
    .badge-detail {{ font-size:.78rem; color:#6c5842; }}
    .verdict-panel {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
    .verdict-note {{ margin:6px 0 10px; color:#5b6762; font-size:.85rem; }}
    .verdict-list {{ list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:6px; }}
    .verdict-item {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
    .verdict-label {{ font-weight:600; color:#4b4031; min-width:240px; }}
    .sens-section {{ margin-top:16px; }}
  </style>
</head>
<body>
  <main>
    <section>
      <p class="label">Phase 1-3 Overview</p>
      <h1>{thesis.get('label','Research Thesis')}</h1>
      <p>{thesis.get('scope_note','')}</p>
      <p class="muted">Profile set: {profile_desc}</p>
      {render_timeframe_note(f"Phase 2 research boundary {selection_window_text}; Phase 2 validation windows {validation_window_text}; Phase 3 untouched holdout {holdout_window_text_value}.")}
      {context_note_html}
      <div class="grid">
        <div class="card"><div class="label">Selection Status</div><div class="value">{selection_status}</div></div>
        <div class="card"><div class="label">Certification PBO</div><div class="value">{pbo_text}</div></div>
        <div class="card"><div class="label">PBO Band</div><div class="value">{pbo_band_value}</div></div>
        <div class="card"><div class="label">Holdout Sharpe</div><div class="value">{holdout_text}</div></div>
        <div class="card"><div class="label">Locked Params</div><div class="value">{format_params(locked_params)}</div></div>
      </div>
      <div class="callout">
        <strong>PBO is one signal, not the verdict.</strong>
        <div class="muted">PBO band: {pbo_band_value} (good &lt; {format_pct(pbo_good_max)}, hard cutoff &gt;= {format_pct(pbo_hard_cutoff_min)}). Validation status is <strong>{selection_status}</strong>.</div>
      </div>
      {crisis_callout}
    </section>
    {verdict_panel}
    {render_phase_map_html(phase1_href="phase1_summary.html", selection_href=selection_href, holdout_href=holdout_href, holdout=holdout)}
    {evidence_stack_html}
    <section>
      <h2>Protocol Insight</h2>
      {render_timeframe_note(f"Boundary chart below spans Phase 2 research {selection_window_text}, Phase 2 validation {validation_window_text}, and the untouched Phase 3 holdout {holdout_window_text_value}.")}
      <p class="muted"><strong>Why the first validation starts far back:</strong> Phase 2 is testing structural robustness first. If the idea only works in the latest regime, it usually should not survive to the later holdout or live discussion.</p>
      <p class="muted"><strong>What counts as "after the test period" on this page:</strong> everything from {holdout_start} onward is outside the Phase 2 selector and is meant to stay cleaner than the walk-forward folds shown below.</p>
      <div class="grid">
        <div class="card full">
          {protocol_chart}
        </div>
        <div class="card full">
          {wf_ladder_chart}
        </div>
      </div>
    </section>
    <section>
      <h2>Diagnostics Overview</h2>
      {render_timeframe_note(f"Snapshot cards mix Phase 2 certification outputs over {validation_window_text} with the current Phase 3 status for {holdout_window_text_value}.")}
      <div class="grid">
        <div class="card">
          <div class="label">Certification PBO</div>
          <div class="value">{pbo_text}</div>
          <div class="muted">Band {pbo_band_value} · Good &lt; {format_pct(pbo_good_max)} · Hard cutoff &gt;= {format_pct(pbo_hard_cutoff_min)}</div>
          <div class="muted">Slices {backtest.get('slice_count')} · Slice length {format_float(backtest.get('slice_length_months'), 1)} months · Combos {backtest.get('combination_count')}</div>
          <div class="muted">OOS rank mean {format_float(backtest.get('mean_oos_rank_percentile'), 3)} · median {format_float(backtest.get('median_oos_rank_percentile'), 3)}</div>
        </div>
        <div class="card">
          <div class="label">Negative Controls</div>
          <div class="value">{cross.get('pass_count','n/a')}/{cross.get('run_count','n/a')} · {block.get('pass_count','n/a')}/{block.get('run_count','n/a')}</div>
          <div class="muted">Threshold max pass-rate {format_pct(config.NEGATIVE_CONTROL_PASS_RATE_MAX)}.</div>
        </div>
        <div class="card">
          <div class="label">Gate Pass Counts</div>
          <div class="value">fold {gate_pass_counts['gate_fold_count']} · boot {gate_pass_counts['gate_bootstrap']} · defl {gate_pass_counts['gate_deflated_sharpe']}</div>
          <div class="muted">Count of candidates clearing each gate.</div>
        </div>
        <div class="card">
          <div class="label">Gate Fail Counts</div>
          <div class="value">fold {gate_fail_counts['fold_count']} · boot {gate_fail_counts['bootstrap']} · defl {gate_fail_counts['deflated_sharpe']} · neg {gate_fail_counts['negative_controls']}</div>
          <div class="muted">Count of candidates failing each gate.</div>
        </div>
        <div class="card">
          <div class="label">Hard-Gate Winners</div>
          <div class="value">{hard_gate_winners}</div>
          <div class="muted">Candidates clearing all certification gates.</div>
        </div>
        <div class="card">
          <div class="label">Top Candidate Blockers</div>
          <div class="value">{top_blockers}</div>
          <div class="muted">Why the top candidate failed hard gates.</div>
        </div>
        <div class="card">
          <div class="label">Holdout Status</div>
          <div class="value">{holdout.get('status','n/a') if holdout else 'n/a'}</div>
          <div class="muted">Phase 4 eligible: {holdout_gate.get('phase4_eligible','n/a')}</div>
        </div>
        <div class="card">
          <div class="label">Model Confidence Score (Proxy)</div>
          <div class="value">{mcs_score if mcs_score is not None else 'n/a'} / 100</div>
          <div class="muted">Composite of fold/boot/deflated/controls gates.</div>
        </div>
        <div class="card">
          <div class="label">Walk-Forward Sharpe</div>
          <div class="value">{format_float(wf_combined.get('sharpe'))}</div>
          <div class="muted">Status: {walk_forward.get('status','n/a') if isinstance(walk_forward, dict) else 'n/a'}</div>
        </div>
        <div class="card">
          <div class="label">MC Sharpe (Median)</div>
          <div class="value">{format_mc(mc_sharpe)}</div>
          <div class="muted">Bootstrap paths: {monte.get('sample_count','n/a')}</div>
        </div>
      </div>
    </section>
    <section>
      <h2>Walk-Forward Timeline</h2>
      <p class="muted">Status: {walk_forward.get('status','n/a') if isinstance(walk_forward, dict) else 'n/a'} · Combined Sharpe {format_float(wf_combined.get('sharpe'))} · Total Return {format_pct(wf_combined.get('total_return'))}</p>
      <p class="muted"><strong>Hard-gate pass/fail</strong> is shown in the Validation Verdict above. This timeline is diagnostic.</p>
      <div class="badge-row"><span class="label">Walk-forward quality (diagnostic)</span>{wf_quality_html}</div>
      <p class="muted"><strong>Result (diagnostic):</strong> {wf_diag}.</p>
      <p class="muted"><strong>Gap summary:</strong> Sharpe {wf_gap_sharpe}. Annualized Return {wf_gap_return}. {wf_gap_months}.</p>
      <p class="muted"><strong>What the chart shows:</strong> Green line = validation (out-of-sample). Gray dashed = train (in-sample). X-axis is the mid-year of each validation window; shaded bands show each validation window.</p>
      {benchmark_context}
      <p class="muted"><strong>Integrity boundary:</strong> These validation windows are out-of-sample relative to each fold's training span. The untouched final holdout starts at {config.OOS_START} and is not part of this walk-forward schedule.</p>
      <p class="muted">Diagnostic summary (not a hard gate): {wf_gate_text}. Dashed horizontal line is the 0.4 reference; solid horizontal line is zero (break-even) when visible.</p>
      <p class="muted"><strong>What it means:</strong> Validation close to train with few negatives = more stable; big gaps or repeated negatives = weaker robustness.</p>
      <p class="muted"><strong>Higher-frequency view:</strong> The stitched validation equity and rolling Sharpe below update monthly, so you can see movement inside folds.</p>
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
              train_value_fn=lambda fold: _fold_cagr(fold, total_key="train_total_return", months_key="train_months"),
              validate_value_fn=lambda fold: _fold_cagr(fold, total_key="validate_total_return", months_key="validate_months"),
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
    <section>
      <h2>Monte Carlo Distribution</h2>
      <p class="muted">Samples: {monte.get('sample_count','n/a')} · Block length: {monte.get('block_length_months','n/a')} months</p>
      <p class="muted">Method: stationary block bootstrap of historical monthly returns (not IID random-walk).</p>
      <p class="muted"><strong>Integrity boundary:</strong> This bootstrap uses realized historical returns from the frozen candidate's validation record. It is diagnostic only and does not project or consume the untouched holdout.</p>
      <div class="badge-row"><span class="label">Monte Carlo quality (not a gate)</span>{mc_badge_html}</div>
      <p class="muted"><strong>Result (diagnostic):</strong> {mc_badge_label}. {mc_badge_detail}</p>
      <p class="muted"><strong>What the histograms show:</strong> X-axis = metric value, Y-axis = frequency across bootstrap samples. Taller bars = more likely outcomes. Higher Sharpe/return is better; lower drawdown is better. Markers show p05/median/p95.</p>
      <p class="muted"><strong>What the paths show:</strong> X-axis = months from start; Y-axis = equity index (start=100). Thin lines are sample bootstrap paths; the blue fan shows path density (darker = more likely); bold line is the median; shaded band is p10-p90.</p>
      <p class="muted"><strong>What it means:</strong> Distributions centered above zero with tight spread are stronger; large left tails or very wide fans signal more uncertainty.</p>
      <p class="muted"><strong>Bottom line:</strong> {mc_interp}</p>
      {mc_anomaly}
      <div class="grid">
        <div class="card">
          {render_histogram_svg(mc_hist.get("sharpe"), metric=mc_sharpe, title="Sharpe", x_label="Sharpe", as_pct=False)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get("total_return"), metric=mc_total, title="Total Return From Start", x_label="Total Return From Start (%)", as_pct=True)}
        </div>
        <div class="card">
          {render_histogram_svg(mc_hist.get("max_drawdown"), metric=mc_dd, title="Max Drawdown", x_label="Max Drawdown (%)", as_pct=True)}
        </div>
        <div class="card full">
          {render_spaghetti_svg(mc_paths, title="Monte Carlo Paths (Equity Index)", display_quantile_range=(0.05, 0.95), path_quantiles=mc_quantiles, path_density=mc_density)}
        </div>
      </div>
      {trimmed_html}
    </section>
    <section>
      <h2>Parameter Sensitivity (Certification)</h2>
      {render_timeframe_note(f"Sensitivity summaries below are computed from Phase 2 certification folds over {validation_window_text}.")}
      {sensitivity_sections}
      <p class="muted" style="margin-top:10px;"><strong>What the chart shows:</strong> X-axis = parameter value; Y-axis = median validation Sharpe. Flatter lines = more robust; steep slopes = sensitivity/overfit. Largest dot marks the best value. Table retains pass rates and counts.</p>
    </section>
    <section>
      <h2>Return Curves (Top Candidates)</h2>
      {render_timeframe_note(f"These return curves show Phase 2 stitched validation performance over {validation_window_text}; they are not the Phase 3 holdout curves.")}
      <div class="grid">
        <div class="card">
          <div class="label">Quick</div>
          {render_return_curve(top_quick.get('concatenated_returns'), top_quick.get('primary_benchmark_returns'))}
        </div>
        <div class="card">
          <div class="label">Mega</div>
          {render_return_curve(top_mega.get('concatenated_returns'), top_mega.get('primary_benchmark_returns'))}
        </div>
        <div class="card">
          <div class="label">Certification</div>
          {render_return_curve(top_cert.get('concatenated_returns'), top_cert.get('primary_benchmark_returns'))}
        </div>
      </div>
      <p class="muted" style="margin-top:10px;">Gold line = candidate; dashed = {config.PRIMARY_PASSIVE_BENCHMARK} if available.</p>
    </section>
    <section>
      <h2>Fold Diagnostics (Top Certification Candidate)</h2>
      {render_timeframe_note(f"Fold diagnostics cover the five fixed Phase 2 validation windows inside {validation_window_text}.")}
      <table>
        <thead>
          <tr><th>Fold</th><th>Sharpe</th><th>Max Drawdown</th></tr>
        </thead>
        <tbody>
          {fold_rows if fold_rows else '<tr><td colspan="3">Fold metrics not available yet.</td></tr>'}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Crisis Lens (Walk-Forward Validation)</h2>
      {render_timeframe_note(f"Crisis markers only annotate the Phase 2 validation windows over {validation_window_text}.")}
      <table>
        <thead>
          <tr><th>Fold</th><th>Validate Window</th><th>Sharpe</th><th>Max Drawdown</th><th>Overlap</th></tr>
        </thead>
        <tbody>
          {crisis_rows}
        </tbody>
      </table>
      <p class="muted" style="margin-top:10px;">Asterisk indicates overlap with 2008-2009. These windows remain included in validation; the marker is context only.</p>
    </section>
    <section>
      <h2>{profile_section_title}</h2>
      {render_timeframe_note(f"Profile rows summarize Phase 2 profile outputs over {validation_window_text}, with Phase 3 holdout {holdout_window_text_value} shown elsewhere.")}
      <table>
        <thead>
          {profile_table_head}
        </thead>
        <tbody>
          {profile_table}
        </tbody>
      </table>
      {profile_section_note}
    </section>
    <section>
      <h2>Top Candidates (Certification)</h2>
      {render_timeframe_note(f"Certification candidates below are ranked only on the Phase 2 validation evidence over {validation_window_text}.")}
      <table>
        <thead>
          <tr><th>Candidate</th><th>Params</th><th>Median Sharpe</th><th>Fold Passes</th><th>Bootstrap</th><th>Deflated</th></tr>
        </thead>
        <tbody>
          {"".join(
              f"<tr><td>{item.get('candidate_id','n/a')}</td><td>{format_params(item.get('params'))}</td>"
              f"<td>{format_float(item.get('median_validation_sharpe'))}</td><td>{item.get('fold_pass_count','n/a')}</td>"
              f"<td>{'yes' if item.get('gate_bootstrap') else 'no'}</td><td>{'yes' if item.get('gate_deflated_sharpe') else 'no'}</td></tr>"
              for item in top_candidates(cert_candidates, limit=8)
          )}
        </tbody>
      </table>
      <p class="muted" style="margin-top:10px;">This is the easiest way to compare nearby candidates and see which gates are stopping them.</p>
    </section>
  </main>
</body>
</html>
"""


def build_summary_dashboard(summary: dict[str, Any]) -> str:
    profile_set = summary.get("profile_set", "default")
    profile_settings = config.RESEARCH_PROFILE_SETS.get(profile_set, config.RESEARCH_PROFILE_SETTINGS)
    profile_desc = format_profile_set(profile_set, profile_settings)
    profile_list = ", ".join(summary.get("profiles", [])) or "n/a"
    summary_dir = Path(summary.get("results_root", "results/research_engine")) / "summary"
    thesis_rows = summary.get("theses", [])
    phase1_green = bool(summary.get("phase1_green"))
    selected_count = 0
    phase4_count = 0
    rows = []
    for thesis_row in thesis_rows:
        thesis = thesis_row.get("thesis", {})
        certification = thesis_row.get("certification", {})
        holdout = thesis_row.get("holdout", {})
        phase4 = holdout.get("phase4_gate", {})
        backtest = certification.get("backtest_overfitting", {})
        pbo = backtest.get("pbo")
        holdout_sharpe = phase4.get("base_main_net_sharpe")
        if certification.get("selection_status") == "selected":
            selected_count += 1
        if phase4.get("phase4_eligible"):
            phase4_count += 1
        thesis_dir = Path(thesis_row.get("output_dir", summary_dir.parent / thesis.get("name", "")))
        overview_href = Path(os.path.relpath(thesis_dir / "dashboard.html", summary_dir)).as_posix()
        phase1_href = Path(os.path.relpath(thesis_dir / "phase1_summary.html", summary_dir)).as_posix()
        selection_href = Path(os.path.relpath(thesis_dir / "selection_summary.html", summary_dir)).as_posix()
        holdout_href = Path(os.path.relpath(thesis_dir / "holdout_results.html", summary_dir)).as_posix()
        rows.append(
            f"<tr><td>{thesis.get('label','n/a')}</td>"
            f"<td>{certification.get('selection_status','n/a')}</td>"
            f"<td>{format_pbo_display(pbo, backtest)}</td>"
            f"<td>{certification.get('pbo_band','n/a')}</td>"
            f"<td>{format_float(holdout_sharpe)}</td>"
            f"<td><a href=\"{overview_href}\">overview</a> &middot; <a href=\"{phase1_href}\">phase 1</a> &middot; <a href=\"{selection_href}\">phase 2</a> &middot; <a href=\"{holdout_href}\">phase 3</a></td></tr>"
        )
    table_rows = "\n".join(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Engine Summary</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:1200px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin-top:16px; }}
    .card {{ background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:20px; padding:18px; }}
    .label {{ color:#6c5842; text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; }}
    .value {{ font-size:1.3rem; margin-top:6px; }}
    .muted {{ color:#5b6762; }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ padding:10px 8px; border-bottom:1px solid rgba(23,33,26,.08); text-align:left; }}
    th {{ font: .78rem Consolas, monospace; text-transform:uppercase; letter-spacing:.05em; color:#5d685f; }}
    a {{ color:#b06b1d; }}
  </style>
</head>
<body>
  <main>
    <section>
      <p class="label">Phase 1-3 Overview</p>
      <h1>Research Engine Summary</h1>
      <p>This is the runnable research-engine output (separate from the live package).</p>
      <p><strong>Profile set:</strong> {profile_desc}</p>
      <p><strong>Profiles:</strong> {profile_list}</p>
      {render_timeframe_note(f"Phase 2 research boundary {phase2_selection_window_text()}; Phase 2 validation windows {phase2_validation_window_text()}; Phase 3 untouched holdout {holdout_window_text(None)}.")}
    </section>
    {render_phase_map_html(phase1_href="phase1_summary.html")}
    <section>
      <h2>Cycle Overview</h2>
      <div class="grid">
        <div class="card"><div class="label">Phase 1 Status</div><div class="value">{'green' if phase1_green else 'not green'}</div></div>
        <div class="card"><div class="label">Theses</div><div class="value">{len(thesis_rows)}</div></div>
        <div class="card"><div class="label">Phase 2 Selected</div><div class="value">{selected_count}</div></div>
        <div class="card"><div class="label">Phase 4 Eligible</div><div class="value">{phase4_count}</div></div>
      </div>
      <p class="muted" style="margin-top:12px;">Use the artifact links below to jump directly into the summary overview or the dedicated Phase 1, Phase 2, and Phase 3 pages for each thesis.</p>
    </section>
    <section>
      <h2>Thesis Results</h2>
      <table>
        <thead>
          <tr><th>Thesis</th><th>Certification</th><th>PBO</th><th>PBO Band</th><th>Holdout Sharpe</th><th>Artifacts</th></tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def resolve_profile_settings(profile_set: str) -> dict[str, Any]:
    settings = config.RESEARCH_PROFILE_SETS.get(profile_set)
    if settings is None:
        raise ValueError(f"Unknown profile set '{profile_set}'. Available: {', '.join(config.RESEARCH_PROFILE_SETS)}")
    return settings


def build_param_grid(
    profile_name: str,
    profile_settings: dict[str, Any],
    strategy_variants: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    profile = profile_settings[profile_name]
    strategies = strategy_variants or config.STRATEGY_VARIANTS
    params: list[dict[str, int]] = []
    for lookback in profile["lookbacks"]:
        for skip in profile["skips"]:
            for top_n in profile["top_ns"]:
                for strategy in strategies:
                    payload: dict[str, Any] = {
                        "l": int(lookback),
                        "skip": int(skip),
                        "top_n": int(top_n),
                        "strategy_id": strategy.get("strategy_id", "baseline"),
                    }
                    for key, value in strategy.items():
                        if key == "strategy_id":
                            continue
                        payload[key] = value
                    params.append(payload)
    return params


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


def _concatenate_returns(
    evaluations: Sequence[dict[str, Any]],
    ordered_fold_ids: Sequence[str],
) -> list[float]:
    fold_order = {fold_id: index for index, fold_id in enumerate(ordered_fold_ids)}
    ordered = sorted(evaluations, key=lambda item: fold_order[item["fold_id"]])
    returns: list[float] = []
    for evaluation in ordered:
        returns.extend(evaluation["monthly_returns"])
    return returns


def _concatenate_series(
    evaluations: Sequence[dict[str, Any]],
    ordered_fold_ids: Sequence[str],
    key: str,
) -> list[float]:
    fold_order = {fold_id: index for index, fold_id in enumerate(ordered_fold_ids)}
    ordered = sorted(evaluations, key=lambda item: fold_order[item["fold_id"]])
    series: list[float] = []
    for evaluation in ordered:
        values = evaluation.get(key)
        if values is None:
            return []
        series.extend(values)
    return series


def active_selection_variants(dataset: Any, excluded_countries: Sequence[str]) -> list[str]:
    variants: list[str] = []
    for variant in REQUIRED_UNIVERSE_VARIANTS:
        try:
            mask = dataset.variant_mask(variant, excluded_countries=excluded_countries)
        except Exception:
            continue
        if np.any(mask):
            variants.append(variant)
    return variants


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


def _quantile_key(probability: float) -> str:
    if abs(probability - 0.5) < 1e-9:
        return "median"
    return f"p{int(round(probability * 100)):02d}"


def _path_quantiles(
    paths: list[list[float]], quantiles: Sequence[float]
) -> dict[str, list[float]]:
    if not paths:
        return {}
    length = min(len(path) for path in paths)
    if length < 2:
        return {}
    matrix = np.array([path[:length] for path in paths], dtype=float)
    result: dict[str, list[float]] = {}
    for q in quantiles:
        key = _quantile_key(float(q))
        series = np.quantile(matrix, q, axis=0)
        result[key] = [float(value) for value in series]
    return result


def _density_range_from_quantiles(
    path_quantiles: dict[str, list[float]],
    q_low: float,
    q_high: float,
    *,
    fallback_paths: list[list[float]],
) -> tuple[float, float]:
    low_series = path_quantiles.get(_quantile_key(q_low))
    high_series = path_quantiles.get(_quantile_key(q_high))
    if low_series and high_series:
        y_min = min(low_series)
        y_max = max(high_series)
    else:
        y_min = min(min(path) for path in fallback_paths)
        y_max = max(max(path) for path in fallback_paths)
    if y_max <= y_min:
        y_max = y_min + 1.0
    return (y_min, y_max)


def _path_density(
    paths: list[list[float]], y_min: float, y_max: float, bins: int
) -> dict[str, Any]:
    if not paths or bins < 2:
        return {}
    length = min(len(path) for path in paths)
    if length < 2:
        return {}
    if y_max <= y_min:
        y_max = y_min + 1.0
    edges = np.linspace(y_min, y_max, bins + 1)
    counts: list[list[int]] = []
    max_count = 1
    for idx in range(length):
        values = [path[idx] for path in paths]
        hist, _ = np.histogram(values, bins=edges)
        row = [int(value) for value in hist]
        counts.append(row)
        if row:
            max_count = max(max_count, max(row))
    return {
        "y_edges": [float(edge) for edge in edges],
        "counts": counts,
        "max_count": int(max_count),
    }


def stationary_bootstrap_sharpe_ci(
    returns: Sequence[float],
    *,
    periods_per_year: int,
    mean_block_length_months: int = config.BOOTSTRAP_BLOCK_LENGTH_MONTHS,
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = 7,
) -> tuple[float, float]:
    if len(returns) < 2:
        return (0.0, 0.0)
    block_length_periods = max(1, int(round(mean_block_length_months * periods_per_year / 12.0)))
    rng = random.Random(seed)
    sharpe_values = [
        annualized_sharpe_periods(
            _stationary_bootstrap_sample(returns, block_length_periods, rng),
            periods_per_year=periods_per_year,
        )
        for _ in range(n_resamples)
    ]
    sharpe_values.sort()
    return (_quantile(sharpe_values, 0.025), _quantile(sharpe_values, 0.975))


def _metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values.sort()
    return {
        "mean": statistics.fmean(values),
        "median": _quantile(values, 0.5),
        "p05": _quantile(values, 0.05),
        "p95": _quantile(values, 0.95),
        "min": values[0],
        "max": values[-1],
    }


def _returns_to_equity(returns: Sequence[float], *, base: float = 100.0) -> list[float]:
    value = base
    series: list[float] = []
    for ret in returns:
        value *= 1.0 + float(ret)
        series.append(value)
    return series


def _annualized_return(
    total: float | None, months: int | None, periods_per_year: int | None
) -> float | None:
    if total is None or months is None:
        return None
    if not np.isfinite(total) or months <= 0:
        return None
    periods_per_year = periods_per_year or 12
    years = months / periods_per_year
    if years <= 0:
        return None
    if 1.0 + total <= 0:
        return None
    return (1.0 + float(total)) ** (1.0 / years) - 1.0


def _fold_cagr(
    fold: dict[str, Any], *, total_key: str, months_key: str
) -> float | None:
    return _annualized_return(
        fold.get(total_key),
        fold.get(months_key),
        fold.get("periods_per_year", 12),
    )


def _histogram(values: list[float], bins: int) -> dict[str, Any]:
    if not values:
        return {}
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [float(edge) for edge in edges], "counts": [int(count) for count in counts]}


def monte_carlo_summary(
    returns: Sequence[float] | None,
    *,
    periods_per_year: int,
    n_resamples: int = config.MONTE_CARLO_RESAMPLES,
    block_length_months: int = config.MONTE_CARLO_BLOCK_LENGTH_MONTHS,
    seed: int = config.MONTE_CARLO_SEED,
) -> dict[str, Any]:
    cleaned = _clean_returns(returns)
    if len(cleaned) < 2 or n_resamples < 1:
        return {
            "status": "unavailable",
            "sample_count": 0,
            "block_length_months": block_length_months,
        }
    base_return_stats = {
        "count": len(cleaned),
        "max_monthly": max(cleaned),
        "min_monthly": min(cleaned),
        "pct_months_gt_10": sum(1 for value in cleaned if value > 0.10) / len(cleaned),
        "pct_months_lt_neg10": sum(1 for value in cleaned if value < -0.10) / len(cleaned),
        "mean": statistics.fmean(cleaned),
        "stdev": statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0,
    }
    block_length_periods = max(1, int(round(block_length_months * periods_per_year / 12.0)))
    rng = random.Random(seed)
    sharpe_values: list[float] = []
    total_values: list[float] = []
    drawdown_values: list[float] = []
    all_paths: list[list[float]] = []
    for _ in range(n_resamples):
        sample = _stationary_bootstrap_sample(cleaned, block_length_periods, rng)
        if len(sample) < 2:
            continue
        sharpe_values.append(annualized_sharpe_periods(sample, periods_per_year=periods_per_year))
        total_values.append(total_return(sample))
        drawdown_values.append(max_drawdown(sample))
        all_paths.append(_returns_to_equity(sample, base=100.0))
    if not sharpe_values:
        return {
            "status": "unavailable",
            "sample_count": 0,
            "block_length_months": block_length_months,
        }
    histograms = {
        "sharpe": _histogram(sharpe_values, MONTE_CARLO_HIST_BINS),
        "total_return": _histogram(total_values, MONTE_CARLO_HIST_BINS),
        "max_drawdown": _histogram(drawdown_values, MONTE_CARLO_HIST_BINS),
        "final_equity": _histogram([path[-1] for path in all_paths if path], MONTE_CARLO_HIST_BINS),
    }
    path_quantiles = _path_quantiles(all_paths, MONTE_CARLO_PATH_QUANTILES)
    path_density: dict[str, Any] = {}
    if path_quantiles:
        density_min, density_max = _density_range_from_quantiles(
            path_quantiles,
            MONTE_CARLO_DENSITY_QUANTILE_RANGE[0],
            MONTE_CARLO_DENSITY_QUANTILE_RANGE[1],
            fallback_paths=all_paths,
        )
        path_density = _path_density(all_paths, density_min, density_max, MONTE_CARLO_DENSITY_BINS)
    sample_paths = _sample_paths_evenly(all_paths, max_lines=MONTE_CARLO_SAMPLE_PATHS)
    sample_length = len(sample_paths[0]) if sample_paths else 0
    trimmed: dict[str, Any] | None = None
    trim_top_n = int(len(total_values) * MONTE_CARLO_TRIM_TOP_PCT)
    if trim_top_n > 0 and len(total_values) > trim_top_n:
        order = sorted(range(len(total_values)), key=lambda idx: total_values[idx])
        keep = order[: len(total_values) - trim_top_n]
        trimmed_sharpe = [sharpe_values[idx] for idx in keep]
        trimmed_total = [total_values[idx] for idx in keep]
        trimmed_dd = [drawdown_values[idx] for idx in keep]
        trimmed_paths_all = [all_paths[idx] for idx in keep]
        trimmed_quantiles = _path_quantiles(trimmed_paths_all, MONTE_CARLO_PATH_QUANTILES)
        trimmed_density: dict[str, Any] = {}
        if trimmed_quantiles:
            density_min, density_max = _density_range_from_quantiles(
                trimmed_quantiles,
                MONTE_CARLO_DENSITY_QUANTILE_RANGE[0],
                MONTE_CARLO_DENSITY_QUANTILE_RANGE[1],
                fallback_paths=trimmed_paths_all,
            )
            trimmed_density = _path_density(
                trimmed_paths_all, density_min, density_max, MONTE_CARLO_DENSITY_BINS
            )
        trimmed_paths = _sample_paths_evenly(
            trimmed_paths_all,
            max_lines=MONTE_CARLO_SAMPLE_PATHS,
        )
        trimmed = {
            "status": "ok",
            "trim_top_pct": MONTE_CARLO_TRIM_TOP_PCT,
            "sample_count": len(trimmed_sharpe),
            "histograms": {
                "sharpe": _histogram(trimmed_sharpe, MONTE_CARLO_HIST_BINS),
                "total_return": _histogram(trimmed_total, MONTE_CARLO_HIST_BINS),
                "max_drawdown": _histogram(trimmed_dd, MONTE_CARLO_HIST_BINS),
                "final_equity": _histogram([path[-1] for path in trimmed_paths_all if path], MONTE_CARLO_HIST_BINS),
            },
            "metrics": {
                "sharpe": _metric_summary(trimmed_sharpe),
                "total_return": _metric_summary(trimmed_total),
                "max_drawdown": _metric_summary(trimmed_dd),
                "final_equity": _metric_summary([path[-1] for path in trimmed_paths_all if path]),
            },
            "sample_paths": trimmed_paths,
            "path_quantiles": trimmed_quantiles,
            "path_density": trimmed_density,
        }
    return {
        "status": "ok",
        "sample_count": len(sharpe_values),
        "block_length_months": block_length_months,
        "sample_paths": sample_paths,
        "sample_length": sample_length,
        "path_base": 100.0,
        "path_quantiles": path_quantiles,
        "path_density": path_density,
        "histograms": histograms,
        "base_return_stats": base_return_stats,
        "metrics": {
            "sharpe": _metric_summary(sharpe_values),
            "total_return": _metric_summary(total_values),
            "max_drawdown": _metric_summary(drawdown_values),
            "final_equity": _metric_summary([path[-1] for path in all_paths if path]),
        },
        "trimmed": trimmed,
    }


def parameter_sensitivity(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    if not aggregates:
        return {"status": "unavailable"}

    def _param_group(key: str) -> dict[str, Any]:
        values = sorted({item["params"].get(key) for item in aggregates if key in item.get("params", {})})
        rows: list[dict[str, Any]] = []
        for value in values:
            subset = [item for item in aggregates if item["params"].get(key) == value]
            medians = [
                item["median_validation_sharpe"]
                for item in subset
                if item.get("median_validation_sharpe") is not None
                and np.isfinite(item.get("median_validation_sharpe"))
            ]
            median_value = statistics.median(medians) if medians else None
            hard_gate_pass = sum(
                1
                for item in subset
                if item.get("gate_fold_count")
                and item.get("gate_bootstrap")
                and item.get("gate_deflated_sharpe")
                and item.get("gate_negative_controls")
            )
            rows.append(
                {
                    "value": value,
                    "median_validation_sharpe": median_value,
                    "hard_gate_pass_rate": hard_gate_pass / len(subset) if subset else None,
                    "count": len(subset),
                }
            )
        filtered = [row for row in rows if row.get("median_validation_sharpe") is not None]
        correlation = None
        if len(filtered) >= 2:
            xs = [row["value"] for row in filtered]
            ys = [row["median_validation_sharpe"] for row in filtered]
            try:
                correlation = float(np.corrcoef(xs, ys)[0, 1])
            except Exception:
                correlation = None
        best = max(filtered, key=lambda row: row["median_validation_sharpe"], default=None)
        worst = min(filtered, key=lambda row: row["median_validation_sharpe"], default=None)
        spread = None
        if best and worst:
            spread = best["median_validation_sharpe"] - worst["median_validation_sharpe"]
        return {
            "by_value": rows,
            "best_value": best["value"] if best else None,
            "worst_value": worst["value"] if worst else None,
            "median_spread": spread,
            "correlation": correlation,
        }

    def _strategy_group() -> dict[str, Any]:
        values = sorted(
            {item["params"].get("strategy_id", "baseline") for item in aggregates if item.get("params")}
        )
        rows: list[dict[str, Any]] = []
        for value in values:
            subset = [item for item in aggregates if item["params"].get("strategy_id", "baseline") == value]
            medians = [
                item["median_validation_sharpe"]
                for item in subset
                if item.get("median_validation_sharpe") is not None
                and np.isfinite(item.get("median_validation_sharpe"))
            ]
            median_value = statistics.median(medians) if medians else None
            rows.append(
                {
                    "value": value,
                    "median_validation_sharpe": median_value,
                    "count": len(subset),
                }
            )
        filtered = [row for row in rows if row.get("median_validation_sharpe") is not None]
        best = max(filtered, key=lambda row: row["median_validation_sharpe"], default=None)
        worst = min(filtered, key=lambda row: row["median_validation_sharpe"], default=None)
        spread = None
        if best and worst:
            spread = best["median_validation_sharpe"] - worst["median_validation_sharpe"]
        return {
            "by_value": rows,
            "best_value": best["value"] if best else None,
            "worst_value": worst["value"] if worst else None,
            "median_spread": spread,
        }

    return {
        "status": "ok",
        "parameters": {
            "l": _param_group("l"),
            "skip": _param_group("skip"),
            "top_n": _param_group("top_n"),
        },
        "strategy_id": _strategy_group(),
    }


def _candidate_sort_key(aggregate: dict[str, Any]) -> tuple[Any, ...]:
    hard_gate = all(
        (
            aggregate["gate_fold_count"],
            aggregate["gate_deflated_sharpe"],
            aggregate["gate_bootstrap"],
            aggregate["gate_negative_controls"],
        )
    )
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
    )


def _attach_plateau_diagnostics(aggregates: list[dict[str, Any]]) -> None:
    candidates_by_params = {
        (item["params"]["l"], item["params"]["skip"], item["params"]["top_n"]): item for item in aggregates
    }
    l_values = sorted({item["params"]["l"] for item in aggregates})
    skip_values = sorted({item["params"]["skip"] for item in aggregates})
    top_n_values = sorted({item["params"]["top_n"] for item in aggregates})
    grids = {"l": l_values, "skip": skip_values, "top_n": top_n_values}

    for aggregate in aggregates:
        params = aggregate["params"]
        neighbors: list[dict[str, Any]] = []
        for key, values in grids.items():
            current_index = values.index(params[key])
            for offset in (-1, 1):
                neighbor_index = current_index + offset
                if neighbor_index < 0 or neighbor_index >= len(values):
                    continue
                neighbor_params = dict(params)
                neighbor_params[key] = values[neighbor_index]
                lookup_key = (neighbor_params["l"], neighbor_params["skip"], neighbor_params["top_n"])
                neighbor = candidates_by_params.get(lookup_key)
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


def aggregate_candidates(
    *,
    candidates: list[dict[str, Any]],
    negative_controls: dict[str, Any],
    periods_per_year: int,
) -> list[dict[str, Any]]:
    ordered_folds = [fold.fold_id for fold in validation_protocol.fixed_folds()]
    n_trials = len(candidates)
    aggregates: list[dict[str, Any]] = []
    neg_controls_gate = validation_protocol.negative_controls_pass(negative_controls)

    for candidate in candidates:
        params = candidate["params"]
        candidate_name = validation_protocol.candidate_id(params)
        evaluations = candidate["evaluations"]
        validation_protocol.validate_phase2_evaluations(evaluations)

        main_track_evaluations = _filter_evaluations(evaluations, **PRIMARY_TRACK)
        fold_map = {row["fold_id"]: row for row in main_track_evaluations}
        missing_folds = [fold_id for fold_id in ordered_folds if fold_id not in fold_map]
        if missing_folds:
            raise ValueError(f"Candidate {candidate_name} is missing required folds: {', '.join(missing_folds)}")

        per_fold_sharpes = {
            fold_id: annualized_sharpe_periods(
                fold_map[fold_id]["monthly_returns"],
                periods_per_year=periods_per_year,
            )
            for fold_id in ordered_folds
        }
        per_fold_drawdowns = {
            fold_id: max_drawdown(fold_map[fold_id]["monthly_returns"]) for fold_id in ordered_folds
        }
        concatenated_returns = _concatenate_returns([fold_map[fold_id] for fold_id in ordered_folds], ordered_folds)
        primary_benchmark_returns = _concatenate_series(
            [fold_map[fold_id] for fold_id in ordered_folds],
            ordered_folds,
            "primary_benchmark_returns",
        )
        secondary_benchmark_returns = _concatenate_series(
            [fold_map[fold_id] for fold_id in ordered_folds],
            ordered_folds,
            "secondary_benchmark_returns",
        )
        tertiary_benchmark_returns = _concatenate_series(
            [fold_map[fold_id] for fold_id in ordered_folds],
            ordered_folds,
            "tertiary_benchmark_returns",
        )
        median_validation_sharpe = statistics.median(per_fold_sharpes.values())
        overall_sharpe = annualized_sharpe_periods(concatenated_returns, periods_per_year=periods_per_year)
        fold_pass_count = sum(value > 0.4 for value in per_fold_sharpes.values())
        bootstrap_ci_low, bootstrap_ci_high = stationary_bootstrap_sharpe_ci(
            concatenated_returns,
            periods_per_year=periods_per_year,
        )
        dsr_metrics = validation_protocol.deflated_sharpe_metrics(concatenated_returns, n_trials)

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
            universe_variant_sharpes.append(
                annualized_sharpe_periods(
                    _concatenate_returns(rows, ordered_folds),
                    periods_per_year=periods_per_year,
                )
            )
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
            "overall_sharpe": overall_sharpe,
            "bootstrap_ci_low": bootstrap_ci_low,
            "bootstrap_ci_high": bootstrap_ci_high,
            "deflated_sharpe_score": dsr_metrics["score"],
            "deflated_sharpe_probability": dsr_metrics["probability"],
            "max_drawdown": max_drawdown(concatenated_returns),
            "total_return": total_return(concatenated_returns),
            "concatenated_returns": concatenated_returns,
            "primary_benchmark_returns": primary_benchmark_returns,
            "secondary_benchmark_returns": secondary_benchmark_returns,
            "tertiary_benchmark_returns": tertiary_benchmark_returns,
            "primary_benchmark_total_return": total_return(primary_benchmark_returns) if primary_benchmark_returns else None,
            "secondary_benchmark_total_return": total_return(secondary_benchmark_returns)
            if secondary_benchmark_returns
            else None,
            "tertiary_benchmark_total_return": total_return(tertiary_benchmark_returns)
            if tertiary_benchmark_returns
            else None,
            "universe_sensitivity_std": universe_sensitivity_std,
            "gate_fold_count": fold_pass_count >= config.MEGA_WF_PASSES_REQUIRED,
            "gate_deflated_sharpe": dsr_metrics["score"] > 0.0,
            "gate_bootstrap": bootstrap_ci_low > 0.0,
            "gate_negative_controls": neg_controls_gate,
        }
        aggregates.append(aggregate)

    _attach_plateau_diagnostics(aggregates)
    _attach_ranks(aggregates)
    return aggregates


def selection_summary(
    *,
    aggregates: list[dict[str, Any]],
    negative_controls: dict[str, Any],
    mode: str,
    period_label: str,
    periods_per_year: int,
    active_variants: list[str],
) -> dict[str, Any]:
    ranked = sorted(aggregates, key=lambda item: item["rank"])
    selected = next((item for item in ranked if item["selected"]), None)
    summary = {
        "mode": mode,
        "selection_status": "selected" if selected is not None else "no_candidate_passed_hard_gates",
        "locked_candidate": None,
        "ranked_candidates": [],
        "negative_controls": negative_controls,
        "neighbor_diagnostics": [],
        "period_label": period_label,
        "periods_per_year": periods_per_year,
    }
    for aggregate in ranked:
        candidate_summary = {
            "candidate_id": aggregate["candidate_id"],
            "params": aggregate["params"],
            "rank": aggregate["rank"],
            "fold_pass_count": aggregate["fold_pass_count"],
            "median_validation_sharpe": aggregate["median_validation_sharpe"],
            "overall_sharpe": aggregate["overall_sharpe"],
            "deflated_sharpe_score": aggregate["deflated_sharpe_score"],
            "deflated_sharpe_probability": aggregate["deflated_sharpe_probability"],
            "bootstrap_ci_low": aggregate["bootstrap_ci_low"],
            "bootstrap_ci_high": aggregate["bootstrap_ci_high"],
            "universe_sensitivity_std": aggregate["universe_sensitivity_std"],
            "plateau_neighbor_median_sharpe": aggregate["plateau_neighbor_median_sharpe"],
            "plateau_neighbor_ratio": aggregate["plateau_neighbor_ratio"],
            "max_drawdown": aggregate["max_drawdown"],
            "total_return": aggregate["total_return"],
            "concatenated_returns": aggregate["concatenated_returns"],
            "primary_benchmark_returns": aggregate["primary_benchmark_returns"],
            "secondary_benchmark_returns": aggregate["secondary_benchmark_returns"],
            "tertiary_benchmark_returns": aggregate["tertiary_benchmark_returns"],
            "primary_benchmark_total_return": aggregate["primary_benchmark_total_return"],
            "secondary_benchmark_total_return": aggregate["secondary_benchmark_total_return"],
            "tertiary_benchmark_total_return": aggregate["tertiary_benchmark_total_return"],
            "per_fold_sharpes": aggregate["per_fold_sharpes"],
            "per_fold_drawdowns": aggregate["per_fold_drawdowns"],
            "gate_fold_count": aggregate["gate_fold_count"],
            "gate_deflated_sharpe": aggregate["gate_deflated_sharpe"],
            "gate_bootstrap": aggregate["gate_bootstrap"],
            "gate_negative_controls": aggregate["gate_negative_controls"],
            "selected": aggregate["selected"],
            "period_label": period_label,
            "periods_per_year": periods_per_year,
            "active_selection_universe_variants": list(active_variants),
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
                failure_reasons.append("negative-control pass rate exceeded 5%")
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


def _selection_distribution(selected_params: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        "l": {},
        "skip": {},
        "top_n": {},
        "strategy_id": {},
    }
    for params in selected_params:
        for key in ("l", "skip", "top_n", "strategy_id"):
            value = params.get(key)
            if value is None:
                continue
            bucket = counts[key]
            label = str(value)
            bucket[label] = bucket.get(label, 0) + 1
    return counts


def walk_forward_test(
    dataset: Any,
    *,
    thesis: dict[str, Any],
    params_grid: list[dict[str, int]],
    period_label: str,
    periods_per_year: int,
) -> dict[str, Any]:
    folds = validation_protocol.fixed_folds()
    if not folds:
        return {"status": "unavailable", "reason": "no_folds"}

    track = PRIMARY_TRACK
    excluded_countries = tuple(thesis.get("excluded_countries", ()))
    fold_results: list[dict[str, Any]] = []
    combined_returns: list[float] = []
    combined_benchmark_returns: list[float] = []
    selected_params: list[dict[str, Any]] = []

    for fold in folds:
        best_params: dict[str, int] | None = None
        best_train_sharpe = float("-inf")
        best_train_total = None
        best_train_months = 0
        for params in params_grid:
            sim_train: WindowSimulation = dataset.simulate_window(
                params=params,
                universe_variant=track["universe_variant"],
                execution_model=track["execution_model"],
                fx_scenario=track["fx_scenario"],
                start_month=fold.train_start,
                end_month=fold.train_end,
                excluded_countries=excluded_countries,
            )
            train_returns = _clean_returns(sim_train.monthly_returns)
            if len(train_returns) < 2:
                continue
            train_sharpe = annualized_sharpe_periods(train_returns, periods_per_year=periods_per_year)
            if train_sharpe > best_train_sharpe:
                best_train_sharpe = train_sharpe
                best_params = params
                best_train_total = total_return(train_returns)
                best_train_months = len(train_returns)

        if best_params is None:
            fold_results.append(
                {
                    "fold_id": fold.fold_id,
                    "status": "no_training_data",
                    "train_window": {"start": fold.train_start, "end": fold.train_end},
                    "validate_window": {"start": fold.validate_start, "end": fold.validate_end},
                }
            )
            continue

        sim_validate: WindowSimulation = dataset.simulate_window(
            params=best_params,
            universe_variant=track["universe_variant"],
            execution_model=track["execution_model"],
            fx_scenario=track["fx_scenario"],
            start_month=fold.validate_start,
            end_month=fold.validate_end,
            excluded_countries=excluded_countries,
        )
        validate_returns = _clean_returns(sim_validate.monthly_returns)
        validate_benchmark_returns = _clean_returns(sim_validate.primary_benchmark_returns or [])
        combined_returns.extend(validate_returns)
        combined_benchmark_returns.extend(validate_benchmark_returns)
        selected_params.append(best_params)

        fold_results.append(
            {
                "fold_id": fold.fold_id,
                "status": "ok",
                "train_window": {"start": fold.train_start, "end": fold.train_end},
                "validate_window": {"start": fold.validate_start, "end": fold.validate_end},
                "selected_candidate_id": validation_protocol.candidate_id(best_params),
                "selected_params": best_params,
                "train_sharpe": best_train_sharpe if best_train_sharpe != float("-inf") else None,
                "train_total_return": best_train_total,
                "train_months": best_train_months,
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
        return {
            "status": "unavailable",
            "reason": "no_validation_returns",
            "folds": fold_results,
        }

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
        "selection_metric": "train_sharpe",
        "track": dict(track),
        "folds": fold_results,
        "combined": combined,
        "combined_returns": combined_returns,
        "combined_benchmark_returns": combined_benchmark_returns,
        "selection_distribution": _selection_distribution(selected_params),
    }


def compute_cscv_pbo(
    candidates: list[dict[str, Any]],
    *,
    periods_per_year: int,
    period_label: str,
) -> dict[str, Any]:
    if not candidates:
        return {
            "status": "unavailable",
            "method": "cscv_pbo_main_track_sharpe",
            "score_function": "annualized_sharpe",
            "pbo_threshold_max": config.PBO_THRESHOLD_MAX,
            "passes_pbo_threshold": False,
            "passes_threshold": False,
            "period_label": period_label,
            "periods_per_year": periods_per_year,
        }

    ordered_folds = [fold.fold_id for fold in validation_protocol.fixed_folds()]
    candidate_returns: list[list[float]] = []
    for candidate in candidates:
        evaluations = candidate["evaluations"]
        main_track = _filter_evaluations(evaluations, **PRIMARY_TRACK)
        if len(main_track) != len(ordered_folds):
            return {
                "status": "unavailable",
                "method": "cscv_pbo_main_track_sharpe",
                "score_function": "annualized_sharpe",
                "pbo_threshold_max": config.PBO_THRESHOLD_MAX,
                "passes_pbo_threshold": False,
                "passes_threshold": False,
                "period_label": period_label,
                "periods_per_year": periods_per_year,
            }
        candidate_returns.append(_concatenate_returns(main_track, ordered_folds))

    total_periods = len(candidate_returns[0])
    if total_periods == 0:
        return {
            "status": "unavailable",
            "method": "cscv_pbo_main_track_sharpe",
            "score_function": "annualized_sharpe",
            "pbo_threshold_max": config.PBO_THRESHOLD_MAX,
            "passes_pbo_threshold": False,
            "passes_threshold": False,
            "period_label": period_label,
            "periods_per_year": periods_per_year,
        }
    if any(len(series) != total_periods for series in candidate_returns):
        return {
            "status": "unavailable",
            "method": "cscv_pbo_main_track_sharpe",
            "score_function": "annualized_sharpe",
            "pbo_threshold_max": config.PBO_THRESHOLD_MAX,
            "passes_pbo_threshold": False,
            "passes_threshold": False,
            "period_label": period_label,
            "periods_per_year": periods_per_year,
        }

    if len(candidate_returns) == 1:
        return {
            "status": "ok",
            "method": "cscv_pbo_main_track_sharpe",
            "score_function": "annualized_sharpe",
            "pbo": None,
            "pbo_threshold_max": config.PBO_THRESHOLD_MAX,
            "passes_pbo_threshold": True,
            "passes_threshold": True,
            "slice_count": None,
            "slice_length_periods": None,
            "slice_length_months": None,
            "combination_count": 0,
            "candidate_count": 1,
            "total_periods": total_periods,
            "total_months": total_periods / periods_per_year * 12.0,
            "lambda_logit_mean": None,
            "lambda_logit_median": None,
            "mean_oos_rank_percentile": None,
            "median_oos_rank_percentile": None,
            "interpretation": "not_applicable_single_candidate",
            "period_label": period_label,
            "periods_per_year": periods_per_year,
            "threshold_max": config.PBO_THRESHOLD_MAX,
        }

    target_slices = config.PBO_TARGET_SLICE_COUNT
    min_slice_months = config.PBO_MIN_SLICE_LENGTH_MONTHS
    slice_count = None
    if total_periods % target_slices == 0:
        candidate_slice_length = total_periods // target_slices
        slice_months = candidate_slice_length / periods_per_year * 12.0
        if slice_months >= min_slice_months:
            slice_count = target_slices
    if slice_count is None:
        even_divisors = [
            value
            for value in range(2, target_slices + 1, 2)
            if total_periods % value == 0
        ]
        even_divisors.sort(reverse=True)
        for value in even_divisors:
            slice_length = total_periods // value
            slice_months = slice_length / periods_per_year * 12.0
            if slice_months >= min_slice_months:
                slice_count = value
                break
    if slice_count is None:
        return {
            "status": "unavailable",
            "method": "cscv_pbo_main_track_sharpe",
            "score_function": "annualized_sharpe",
            "pbo_threshold_max": config.PBO_THRESHOLD_MAX,
            "passes_pbo_threshold": False,
            "passes_threshold": False,
            "period_label": period_label,
            "periods_per_year": periods_per_year,
        }

    slice_length_periods = total_periods // slice_count
    total_months = total_periods / periods_per_year * 12.0
    slice_length_months = slice_length_periods / periods_per_year * 12.0

    slices = []
    for series in candidate_returns:
        slices.append(
            [
                series[index * slice_length_periods : (index + 1) * slice_length_periods]
                for index in range(slice_count)
            ]
        )

    combination_count = 0
    logit_values: list[float] = []
    percentile_values: list[float] = []
    half = slice_count // 2
    epsilon = 1e-6
    for combo in itertools.combinations(range(slice_count), half):
        combination_count += 1
        in_indices = set(combo)
        out_indices = [index for index in range(slice_count) if index not in in_indices]

        in_scores: list[float] = []
        out_scores: list[float] = []
        for candidate_slices in slices:
            in_returns = [value for idx in in_indices for value in candidate_slices[idx]]
            out_returns = [value for idx in out_indices for value in candidate_slices[idx]]
            in_scores.append(annualized_sharpe_periods(in_returns, periods_per_year=periods_per_year))
            out_scores.append(annualized_sharpe_periods(out_returns, periods_per_year=periods_per_year))

        ranked = sorted(
            enumerate(in_scores),
            key=lambda item: (-item[1], item[0]),
        )
        best_candidate_index = ranked[0][0]
        oos_scores = list(out_scores)
        oos_ranked = sorted(
            enumerate(oos_scores),
            key=lambda item: (-item[1], item[0]),
        )
        oos_rank = next(rank for rank, item in enumerate(oos_ranked, start=1) if item[0] == best_candidate_index)
        if len(oos_scores) <= 1:
            percentile = 0.5
        else:
            # Define relative rank so that 1.0 = best, 0.0 = worst (PRD convention).
            percentile = 1.0 - (oos_rank - 1) / (len(oos_scores) - 1)
        percentile_values.append(percentile)
        safe_percentile = min(1.0 - epsilon, max(epsilon, percentile))
        logit_values.append(math.log(safe_percentile / (1.0 - safe_percentile)))

    if combination_count == 0:
        return {
            "status": "unavailable",
            "method": "cscv_pbo_main_track_sharpe",
            "score_function": "annualized_sharpe",
            "pbo_threshold_max": config.PBO_THRESHOLD_MAX,
            "passes_pbo_threshold": False,
            "passes_threshold": False,
            "period_label": period_label,
            "periods_per_year": periods_per_year,
        }

    pbo = sum(value <= 0.0 for value in logit_values) / float(combination_count)
    mean_percentile = statistics.fmean(percentile_values)
    median_percentile = statistics.median(percentile_values)
    mean_logit = statistics.fmean(logit_values)
    median_logit = statistics.median(logit_values)
    passes = pbo <= config.PBO_THRESHOLD_MAX
    interpretation = "healthy" if passes else "overfitting_warning"
    return {
        "status": "ok",
        "method": "cscv_pbo_main_track_sharpe",
        "score_function": "annualized_sharpe",
        "pbo": pbo,
        "pbo_threshold_max": config.PBO_THRESHOLD_MAX,
        "passes_pbo_threshold": passes,
        "passes_threshold": passes,
        "slice_count": slice_count,
        "slice_length_periods": slice_length_periods,
        "slice_length_months": slice_length_months,
        "combination_count": combination_count,
        "candidate_count": len(candidate_returns),
        "total_periods": total_periods,
        "total_months": total_months,
        "lambda_logit_mean": mean_logit,
        "lambda_logit_median": median_logit,
        "mean_oos_rank_percentile": mean_percentile,
        "median_oos_rank_percentile": median_percentile,
        "interpretation": interpretation,
        "period_label": period_label,
        "periods_per_year": periods_per_year,
        "threshold_max": config.PBO_THRESHOLD_MAX,
    }


def _build_negative_control_months(
    dataset: Any,
    *,
    params: dict[str, int],
    universe_variant: str,
    execution_model: str,
    fx_scenario: str,
    start_month: str,
    end_month: str,
    excluded_countries: Sequence[str],
) -> list[dict[str, Any]]:
    if hasattr(dataset, "negative_control_months"):
        return dataset.negative_control_months(
            params=params,
            universe_variant=universe_variant,
            execution_model=execution_model,
            fx_scenario=fx_scenario,
            start_month=start_month,
            end_month=end_month,
            excluded_countries=excluded_countries,
        )

    strategy = strategy_variants.resolve_strategy(params)
    lookback = int(params["l"])
    skip = int(params["skip"])
    top_n = int(params["top_n"])
    weighting = strategy.get("weighting", "equal")
    trend_filter = bool(strategy.get("trend_filter", False))
    ma_window = int(strategy.get("ma_window", 10) or 10)
    liquidity_min_mdv = strategy.get("liquidity_min_mdv")
    vol_window = int(strategy.get("vol_window", 12) or 12)
    vol_skip = int(strategy.get("vol_skip", 1) or 1)
    signal_indices = dataset._window_signal_indices(start_month, end_month)
    base_mask = dataset.variant_mask(
        universe_variant,
        excluded_countries=excluded_countries,
    )
    rankable = (
        base_mask
        & dataset.asof_matches_anchor
        & dataset.capacity_masks[top_n]
        & dataset.entry_available[execution_model]
        & np.isfinite(dataset.holding_returns[execution_model])
        & np.isfinite(dataset.scores[(lookback, skip)])
    )
    months: list[dict[str, Any]] = []
    for signal_index in signal_indices:
        filter_on = True
        if trend_filter:
            filter_on = dataset.trend_filter_on(signal_index, ma_window)
        positions: list[dict[str, float]] = []
        if signal_index >= rankable.shape[0]:
            continue
        valid = rankable[signal_index]
        if liquidity_min_mdv is not None:
            valid = valid & (dataset.median_daily_value_60d_sek[signal_index] >= float(liquidity_min_mdv))
        scores = dataset.scores[(lookback, skip)][signal_index]
        returns = dataset.holding_returns[execution_model][signal_index]
        weight_values: np.ndarray | None = None
        if weighting == "cap":
            weight_values = dataset.market_cap_sek[signal_index]
        elif weighting == "inv_vol":
            weight_values = dataset.volatility_for_index(signal_index, vol_window, vol_skip)
        for security_index in np.flatnonzero(valid):
            score = scores[security_index]
            next_return = returns[security_index]
            if not np.isfinite(score) or not np.isfinite(next_return):
                continue
            payload = {"score": float(score), "next_return": float(next_return)}
            if weighting in ("cap", "inv_vol") and weight_values is not None:
                payload["weight_value"] = float(weight_values[security_index])
            positions.append(payload)
        if positions:
            baseline_return = statistics.fmean(item["next_return"] for item in positions)
        else:
            baseline_return = 0.0
        months.append(
            {
                "positions": positions,
                "baseline_return": baseline_return,
                "weighting": weighting,
                "filter_on": filter_on,
            }
        )
    return months


def _select_control_candidate(candidates: list[dict[str, Any]], periods_per_year: int) -> dict[str, Any]:
    ordered_folds = [fold.fold_id for fold in validation_protocol.fixed_folds()]
    best_candidate = candidates[0]
    best_score = float("-inf")
    for candidate in candidates:
        evaluations = candidate["evaluations"]
        main_track = _filter_evaluations(evaluations, **PRIMARY_TRACK)
        fold_map = {row["fold_id"]: row for row in main_track}
        fold_scores = [
            annualized_sharpe_periods(
                fold_map[fold_id]["monthly_returns"],
                periods_per_year=periods_per_year,
            )
            for fold_id in ordered_folds
        ]
        median_score = statistics.median(fold_scores)
        if median_score > best_score:
            best_score = median_score
            best_candidate = candidate
    return best_candidate


def compute_negative_controls(
    dataset: Any,
    *,
    candidates: list[dict[str, Any]],
    periods_per_year: int,
    profile_name: str,
    profile_settings: dict[str, Any],
    thesis_exclusions: Sequence[str],
) -> dict[str, Any]:
    if not candidates:
        return {
            "cross_sectional_shuffle": {"pass_count": 0, "run_count": 0},
            "block_shuffled_null": {"pass_count": 0, "run_count": 0},
        }

    profile = profile_settings[profile_name]
    control_candidate = _select_control_candidate(candidates, periods_per_year)
    params = control_candidate["params"]
    folds = validation_protocol.fixed_folds()
    window = EvaluationWindow(start_month=folds[0].validate_start, end_month=folds[-1].validate_end)

    months = _build_negative_control_months(
        dataset,
        params=params,
        universe_variant=PRIMARY_TRACK["universe_variant"],
        execution_model=PRIMARY_TRACK["execution_model"],
        fx_scenario=PRIMARY_TRACK["fx_scenario"],
        start_month=window.start_month,
        end_month=window.end_month,
        excluded_countries=thesis_exclusions,
    )

    def _metrics(series: list[float]) -> tuple[float, float, float]:
        cleaned: list[float] = []
        for value in series:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                cleaned.append(numeric)
        if len(cleaned) < 2:
            return (float("nan"), float("nan"), float("nan"))
        sharpe = annualized_sharpe_periods(cleaned, periods_per_year=periods_per_year)
        bootstrap_low, _ = stationary_bootstrap_sharpe_ci(
            cleaned,
            periods_per_year=periods_per_year,
            n_resamples=config.NEGATIVE_CONTROL_BOOTSTRAP_RESAMPLES,
        )
        dsr_score = validation_protocol.deflated_sharpe_metrics(cleaned, len(candidates))["score"]
        return sharpe, bootstrap_low, dsr_score

    def _valid(metrics: tuple[float, float, float]) -> bool:
        return all(math.isfinite(value) for value in metrics)

    shuffle_runs = validation_protocol.cross_sectional_score_shuffle_runs(
        months,
        params["top_n"],
        profile["cross_sectional_shuffle_runs"],
    )
    cross_sectional_pass_count = 0
    actual_cross_sectional = validation_protocol.cross_sectional_score_actual_run(months, params["top_n"])
    actual_cross_metrics = _metrics(actual_cross_sectional)
    if not _valid(actual_cross_metrics):
        cross_sectional_pass_count = profile["cross_sectional_shuffle_runs"]
    else:
        for run in shuffle_runs:
            run_metrics = _metrics(run)
            if _valid(run_metrics) and all(
                metric >= actual for metric, actual in zip(run_metrics, actual_cross_metrics, strict=True)
            ):
                cross_sectional_pass_count += 1

    ordered_folds = [fold.fold_id for fold in folds]
    main_track = _filter_evaluations(control_candidate["evaluations"], **PRIMARY_TRACK)
    main_returns = _concatenate_returns(main_track, ordered_folds)
    benchmark_returns = _concatenate_series(main_track, ordered_folds, "primary_benchmark_returns")

    def _excess_series(strategy: list[float], benchmark: list[float]) -> list[float]:
        if not benchmark or len(benchmark) != len(strategy):
            return strategy
        excess: list[float] = []
        for strategy_value, benchmark_value in zip(strategy, benchmark):
            if strategy_value is None or benchmark_value is None:
                return strategy
            try:
                excess.append(float(strategy_value) - float(benchmark_value))
            except (TypeError, ValueError):
                return strategy
        return excess

    block_series = _excess_series(main_returns, benchmark_returns)
    block_length_periods = max(1, int(round(config.BOOTSTRAP_BLOCK_LENGTH_MONTHS * periods_per_year / 12.0)))
    block_runs = validation_protocol.block_shuffled_return_path_runs(
        block_series,
        block_length_periods,
        profile["block_shuffled_null_runs"],
    )
    block_pass_count = 0
    actual_block_metrics = _metrics(block_series)
    if not _valid(actual_block_metrics):
        block_pass_count = profile["block_shuffled_null_runs"]
    else:
        for run in block_runs:
            run_metrics = _metrics(run)
            if _valid(run_metrics) and all(
                metric >= actual for metric, actual in zip(run_metrics, actual_block_metrics, strict=True)
            ):
                block_pass_count += 1

    return {
        "cross_sectional_shuffle": {
            "pass_count": cross_sectional_pass_count,
            "run_count": profile["cross_sectional_shuffle_runs"],
        },
        "block_shuffled_null": {
            "pass_count": block_pass_count,
            "run_count": profile["block_shuffled_null_runs"],
        },
    }


def build_candidate_evaluations(
    dataset: Any,
    *,
    thesis: dict[str, Any],
    params_grid: list[dict[str, int]],
    periods_per_year: int,
) -> list[dict[str, Any]]:
    folds = validation_protocol.fixed_folds()
    excluded_countries = tuple(thesis.get("excluded_countries", ()))
    candidates: list[dict[str, Any]] = []
    for params in params_grid:
        evaluations: list[dict[str, Any]] = []
        for fold in folds:
            for universe_variant in REQUIRED_UNIVERSE_VARIANTS:
                for execution_model in REQUIRED_EXECUTION_MODELS:
                    for fx_scenario in REQUIRED_FX_SCENARIOS:
                        simulation: WindowSimulation = dataset.simulate_window(
                            params=params,
                            universe_variant=universe_variant,
                            execution_model=execution_model,
                            fx_scenario=fx_scenario,
                            start_month=fold.validate_start,
                            end_month=fold.validate_end,
                            excluded_countries=excluded_countries,
                        )
                        evaluations.append(
                            {
                                "fold_id": fold.fold_id,
                                "validate_start": fold.validate_start,
                                "validate_end": fold.validate_end,
                                "monthly_returns": simulation.monthly_returns,
                                "primary_benchmark_returns": simulation.primary_benchmark_returns,
                                "secondary_benchmark_returns": simulation.secondary_benchmark_returns,
                                "tertiary_benchmark_returns": simulation.tertiary_benchmark_returns,
                                "universe_variant": universe_variant,
                                "execution_model": execution_model,
                                "fx_scenario": fx_scenario,
                                "cost_model_name": config.PRIMARY_SELECTION_COST_MODEL,
                                "periods_per_year": periods_per_year,
                            }
                        )
        candidates.append({"params": params, "evaluations": evaluations})
    return candidates


def run_profile(
    dataset: Any,
    *,
    thesis: dict[str, Any],
    profile_name: str,
    profile_settings: dict[str, Any],
    period_label: str,
    periods_per_year: int,
    strategy_variants: list[dict[str, Any]] | None = None,
    run_walk_forward: bool = True,
    walk_forward_profiles: set[str] | None = None,
    run_monte_carlo: bool = True,
    run_sensitivity: bool = True,
) -> dict[str, Any]:
    params_grid = build_param_grid(
        profile_name,
        profile_settings,
        strategy_variants=strategy_variants,
    )
    candidates = build_candidate_evaluations(
        dataset,
        thesis=thesis,
        params_grid=params_grid,
        periods_per_year=periods_per_year,
    )
    negative_controls = compute_negative_controls(
        dataset,
        candidates=candidates,
        periods_per_year=periods_per_year,
        profile_name=profile_name,
        profile_settings=profile_settings,
        thesis_exclusions=tuple(thesis.get("excluded_countries", ())),
    )
    aggregates = aggregate_candidates(
        candidates=candidates,
        negative_controls=negative_controls,
        periods_per_year=periods_per_year,
    )
    active_variants = active_selection_variants(dataset, tuple(thesis.get("excluded_countries", ())))
    summary = selection_summary(
        aggregates=aggregates,
        negative_controls=negative_controls,
        mode=profile_name,
        period_label=period_label,
        periods_per_year=periods_per_year,
        active_variants=active_variants,
    )
    summary["backtest_overfitting"] = compute_cscv_pbo(
        candidates,
        periods_per_year=periods_per_year,
        period_label=period_label,
    )
    apply_pbo_policy(summary)
    if run_sensitivity:
        summary["parameter_sensitivity"] = parameter_sensitivity(aggregates)
    else:
        summary["parameter_sensitivity"] = {"status": "skipped"}

    if run_monte_carlo and config.RUN_MONTE_CARLO_DIAGNOSTICS:
        ranked_candidates = summary.get("ranked_candidates") or []
        locked = summary.get("locked_candidate")
        mc_candidate = locked or (ranked_candidates[0] if ranked_candidates else None)
        if mc_candidate:
            monte = monte_carlo_summary(
                mc_candidate.get("concatenated_returns"),
                periods_per_year=periods_per_year,
                n_resamples=config.MONTE_CARLO_RESAMPLES,
                block_length_months=config.MONTE_CARLO_BLOCK_LENGTH_MONTHS,
                seed=config.MONTE_CARLO_SEED,
            )
            monte["candidate_id"] = mc_candidate.get("candidate_id")
            monte["params"] = mc_candidate.get("params")
            monte["source"] = "locked_candidate" if locked else "top_candidate"
            summary["monte_carlo"] = monte
        else:
            summary["monte_carlo"] = {"status": "unavailable", "reason": "no_candidates"}
    else:
        summary["monte_carlo"] = {"status": "skipped"}

    walk_forward_profiles = walk_forward_profiles or set(config.WALK_FORWARD_PROFILES)
    if run_walk_forward and config.RUN_WALK_FORWARD_DIAGNOSTICS and profile_name in walk_forward_profiles:
        summary["walk_forward"] = walk_forward_test(
            dataset,
            thesis=thesis,
            params_grid=params_grid,
            period_label=period_label,
            periods_per_year=periods_per_year,
        )
    else:
        summary["walk_forward"] = {"status": "skipped"}
    return summary


def evaluate_holdout(
    dataset: Any,
    *,
    thesis: dict[str, Any],
    params: dict[str, int],
    period_label: str,
    periods_per_year: int,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict[str, Any]:
    start_month = start_month or config.OOS_START
    end_month = end_month or config.OOS_END
    excluded_countries = tuple(thesis.get("excluded_countries", ()))
    results: dict[str, Any] = {}
    for variant in REQUIRED_UNIVERSE_VARIANTS:
        results[variant] = {}
        for execution_model in REQUIRED_EXECUTION_MODELS:
            results[variant][execution_model] = {}
            for fx_scenario in REQUIRED_FX_SCENARIOS:
                simulation: WindowSimulation = dataset.simulate_window(
                    params=params,
                    universe_variant=variant,
                    execution_model=execution_model,
                    fx_scenario=fx_scenario,
                    start_month=start_month,
                    end_month=end_month,
                    excluded_countries=excluded_countries,
                )
                returns = simulation.monthly_returns
                primary_benchmark_returns = simulation.primary_benchmark_returns
                secondary_benchmark_returns = simulation.secondary_benchmark_returns
                tertiary_benchmark_returns = simulation.tertiary_benchmark_returns
                result_row = {
                    "strategy_returns": returns,
                    "net_sharpe": annualized_sharpe_periods(returns, periods_per_year=periods_per_year),
                    "max_drawdown": max_drawdown(returns),
                    "total_return": total_return(returns),
                    "months": len(returns),
                    "period_count": len(returns),
                    "period_label": period_label,
                    "periods_per_year": periods_per_year,
                }
                if primary_benchmark_returns is not None:
                    result_row["primary_benchmark_returns"] = primary_benchmark_returns
                    result_row["primary_benchmark_total_return"] = total_return(primary_benchmark_returns)
                    result_row["beats_primary_benchmark"] = (
                        result_row["total_return"] > result_row["primary_benchmark_total_return"]
                    )
                if secondary_benchmark_returns is not None:
                    result_row["secondary_benchmark_returns"] = secondary_benchmark_returns
                    result_row["secondary_benchmark_total_return"] = total_return(secondary_benchmark_returns)
                if tertiary_benchmark_returns is not None:
                    result_row["tertiary_benchmark_returns"] = tertiary_benchmark_returns
                    result_row["tertiary_benchmark_total_return"] = total_return(tertiary_benchmark_returns)
                results[variant][execution_model][fx_scenario] = result_row

    base_main = results["Full Nordics"]["next_open"]["base"]
    phase4_eligible = bool(
        base_main.get("beats_primary_benchmark")
        and base_main["net_sharpe"] >= config.OOS_SHARPE_MIN
    )
    return {
        "selected_params": params,
        "holdout_window": {"start": start_month, "end": end_month},
        "results": results,
        "phase4_gate": {
            "base_main_net_sharpe": base_main["net_sharpe"],
            "meets_sharpe_gate": base_main["net_sharpe"] >= config.OOS_SHARPE_MIN,
            "beats_primary_benchmark": base_main.get("beats_primary_benchmark", False),
            "phase4_eligible": phase4_eligible,
        },
        "period_label": period_label,
        "periods_per_year": periods_per_year,
    }


def run_research_engine(
    *,
    data_dir: Path,
    output_dir: Path,
    thesis_names: Sequence[str],
    profiles: Sequence[str],
    profile_set: str,
    skip_holdout: bool,
    skip_walk_forward: bool,
    skip_monte_carlo: bool,
    skip_sensitivity: bool,
    walk_forward_profiles: Sequence[str] | None,
    render_only: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    period_label = "months"
    periods_per_year = 12
    profile_settings = resolve_profile_settings(profile_set)
    phase1_success, phase1_sections = validate_phase1(
        input_dir=data_dir,
        require_cpp=False,
        require_benchmarks=True,
    )

    summary_payload: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results_root": str(output_dir),
        "research_cycle": "research_engine",
        "profiles": list(profiles),
        "profile_set": profile_set,
        "phase1_green": phase1_success,
        "theses": [],
    }

    def profile_page_metadata(profile_name: str) -> tuple[str, str, str]:
        if profile_name == "quick":
            return (
                "quick_summary",
                "Quick Profile Summary",
                "Exploratory small-grid diagnostics for this thesis.",
            )
        if profile_name == "mega":
            return (
                "mega_summary",
                "Mega Profile Summary",
                "Broader-grid diagnostics for this thesis.",
            )
        if profile_name == "certification_baseline":
            return (
                "selection_summary",
                "Certification Summary",
                "Certification gate outputs for this thesis.",
            )
        return (
            f"{profile_name}_summary",
            f"{profile_name.replace('_', ' ').title()} Summary",
            f"{profile_name} diagnostics for this thesis.",
        )

    if render_only:
        for thesis_name in thesis_names:
            thesis_dir = output_dir / thesis_name
            quick_path = thesis_dir / "quick_summary.json"
            mega_path = thesis_dir / "mega_summary.json"
            cert_path = thesis_dir / "selection_summary.json"
            holdout_path = thesis_dir / "holdout_results.json"
            if not cert_path.exists():
                raise ValueError(f"Missing certification summary for {thesis_name} at {cert_path}.")
            quick = validation_protocol.load_json(quick_path) if quick_path.exists() else {}
            mega = validation_protocol.load_json(mega_path) if mega_path.exists() else {}
            certification = validation_protocol.load_json(cert_path)
            holdout = validation_protocol.load_json(holdout_path) if holdout_path.exists() else {"status": "missing"}
            thesis_meta = build_thesis(thesis_name).manifest_metadata()
            render_summaries = {
                "quick": quick,
                "mega": mega,
                "certification_baseline": certification,
            }
            (thesis_dir / "phase1_summary.html").write_text(
                build_phase1_dashboard(
                    success=phase1_success,
                    sections=phase1_sections,
                    title="Phase 1 Validation",
                    subtitle="Infrastructure, benchmark, FX, and point-in-time artifact checks for this research cycle.",
                    back_href="dashboard.html",
                    back_label="Back to thesis dashboard",
                    input_dir=data_dir,
                    selection_href="selection_summary.html",
                    holdout_href="holdout_results.html",
                ),
                encoding="utf-8",
            )
            for profile_name, profile_summary in render_summaries.items():
                filename, page_title, page_subtitle = profile_page_metadata(profile_name)
                if profile_name == "certification_baseline":
                    (thesis_dir / f"{filename}.html").write_text(
                        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=dashboard.html">
  <script>window.location.replace("dashboard.html");</script>
  <title>Phase 2 Dashboard</title>
  <style>body { display:none; }</style>
</head>
<body>Redirecting to the Phase 2 dashboard...</body>
</html>
""",
                        encoding="utf-8",
                    )
                else:
                    (thesis_dir / f"{filename}.html").write_text(
                        build_profile_dashboard(
                            summary=profile_summary,
                            holdout=holdout,
                            title=page_title,
                            subtitle=page_subtitle,
                            back_href="dashboard.html",
                            back_label="Back to thesis dashboard",
                        ),
                        encoding="utf-8",
                    )
            (thesis_dir / "holdout_results.html").write_text(
                build_holdout_dashboard(
                    holdout=holdout,
                    selection_summary=certification,
                    title="Holdout Diagnostics",
                    subtitle="Phase 3: untouched holdout evaluation for the locked certification candidate.",
                    back_href="dashboard.html",
                    back_label="Back to thesis dashboard",
                ),
                encoding="utf-8",
            )
            (thesis_dir / "dashboard.html").write_text(
                build_thesis_dashboard(
                    thesis=thesis_meta,
                    quick=quick,
                    mega=mega,
                    certification=certification,
                    holdout=holdout,
                    profile_set=profile_set,
                    profile_settings=profile_settings,
                ),
                encoding="utf-8",
            )
            summary_payload["theses"].append(
                {
                    "thesis": thesis_meta,
                    "quick": quick,
                    "mega": mega,
                    "certification": certification,
                    "holdout": holdout,
                    "output_dir": str(thesis_dir),
                }
            )
        serialize_json(summary_dir / "research_engine_summary.json", summary_payload)
        (summary_dir / "phase1_summary.html").write_text(
            build_phase1_dashboard(
                success=phase1_success,
                sections=phase1_sections,
                title="Phase 1 Validation",
                subtitle="Infrastructure, benchmark, FX, and point-in-time artifact checks for this research cycle.",
                back_href="dashboard.html",
                back_label="Back to research summary",
                input_dir=data_dir,
                selection_href=None,
                holdout_href=None,
            ),
            encoding="utf-8",
        )
        (summary_dir / "dashboard.html").write_text(build_summary_dashboard(summary_payload), encoding="utf-8")
        return summary_payload

    dataset = ResearchDataset(data_dir)
    for thesis_name in thesis_names:
        thesis_meta = build_thesis(thesis_name).manifest_metadata()
        thesis_dir = output_dir / thesis_name
        thesis_dir.mkdir(parents=True, exist_ok=True)

        summaries: dict[str, dict[str, Any]] = {}
        for profile in profiles:
            summary = run_profile(
                dataset,
                thesis=thesis_meta,
                profile_name=profile,
                profile_settings=profile_settings,
                period_label=period_label,
                periods_per_year=periods_per_year,
                run_walk_forward=not skip_walk_forward,
                walk_forward_profiles=set(walk_forward_profiles) if walk_forward_profiles else None,
                run_monte_carlo=not skip_monte_carlo,
                run_sensitivity=not skip_sensitivity,
            )
            summary["thesis"] = thesis_meta
            summary["profile_name"] = profile
            summaries[profile] = summary

            filename, page_title, page_subtitle = profile_page_metadata(profile)
            serialize_json(thesis_dir / f"{filename}.json", summary)
            (thesis_dir / f"{filename}.html").write_text(
                build_profile_dashboard(
                    summary=summary,
                    title=page_title,
                    subtitle=page_subtitle,
                    back_href="dashboard.html",
                    back_label="Back to thesis dashboard",
                ),
                encoding="utf-8",
            )

        (thesis_dir / "phase1_summary.html").write_text(
            build_phase1_dashboard(
                success=phase1_success,
                sections=phase1_sections,
                title="Phase 1 Validation",
                subtitle="Infrastructure, benchmark, FX, and point-in-time artifact checks for this research cycle.",
                back_href="dashboard.html",
                back_label="Back to thesis dashboard",
                input_dir=data_dir,
                selection_href="selection_summary.html",
                holdout_href="holdout_results.html",
            ),
            encoding="utf-8",
        )

        certification = summaries.get("certification_baseline", {})
        locked = certification.get("locked_candidate")
        if not skip_holdout and locked is not None:
            holdout = evaluate_holdout(
                dataset,
                thesis=thesis_meta,
                params=locked["params"],
                period_label=period_label,
                periods_per_year=periods_per_year,
            )
            holdout["status"] = "ok"
        else:
            status = "skipped" if skip_holdout else "blocked_by_missing_selection"
            if not skip_holdout and certification.get("pbo_hard_cutoff"):
                status = "blocked_by_pbo_hard_cutoff"
            holdout = {
                "status": status,
                "phase4_gate": {},
            }
        for profile_name, profile_summary in summaries.items():
            filename, page_title, page_subtitle = profile_page_metadata(profile_name)
            if profile_name == "certification_baseline":
                (thesis_dir / f"{filename}.html").write_text(
                    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=dashboard.html">
  <script>window.location.replace("dashboard.html");</script>
  <title>Phase 2 Dashboard</title>
  <style>body { display:none; }</style>
</head>
<body>Redirecting to the Phase 2 dashboard...</body>
</html>
""",
                    encoding="utf-8",
                )
            else:
                (thesis_dir / f"{filename}.html").write_text(
                    build_profile_dashboard(
                        summary=profile_summary,
                        holdout=holdout,
                        title=page_title,
                        subtitle=page_subtitle,
                        back_href="dashboard.html",
                        back_label="Back to thesis dashboard",
                    ),
                    encoding="utf-8",
                )
        serialize_json(thesis_dir / "holdout_results.json", holdout)
        (thesis_dir / "holdout_results.html").write_text(
            build_holdout_dashboard(
                holdout=holdout,
                selection_summary=certification,
                title="Holdout Diagnostics",
                subtitle="Phase 3: untouched holdout evaluation for the locked certification candidate.",
                back_href="dashboard.html",
                back_label="Back to thesis dashboard",
            ),
            encoding="utf-8",
        )

        (thesis_dir / "dashboard.html").write_text(
            build_thesis_dashboard(
                thesis=thesis_meta,
                quick=summaries.get("quick", {}),
                mega=summaries.get("mega", {}),
                certification=certification,
                holdout=holdout,
                profile_set=profile_set,
                profile_settings=profile_settings,
            ),
            encoding="utf-8",
        )
        summary_payload["theses"].append(
            {
                "thesis": thesis_meta,
                "quick": summaries.get("quick", {}),
                "mega": summaries.get("mega", {}),
                "certification": certification,
                "holdout": holdout,
                "output_dir": str(thesis_dir),
            }
        )

    serialize_json(summary_dir / "research_engine_summary.json", summary_payload)
    (summary_dir / "phase1_summary.html").write_text(
        build_phase1_dashboard(
            success=phase1_success,
            sections=phase1_sections,
            title="Phase 1 Validation",
            subtitle="Infrastructure, benchmark, FX, and point-in-time artifact checks for this research cycle.",
            back_href="dashboard.html",
            back_label="Back to research summary",
            input_dir=data_dir,
            selection_href=None,
            holdout_href=None,
        ),
        encoding="utf-8",
    )
    (summary_dir / "dashboard.html").write_text(build_summary_dashboard(summary_payload), encoding="utf-8")
    return summary_payload


def main() -> int:
    args = parse_args()
    run_research_engine(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        thesis_names=args.theses,
        profiles=args.profiles,
        profile_set=args.profile_set,
        skip_holdout=bool(args.skip_holdout),
        skip_walk_forward=bool(args.skip_walk_forward),
        skip_monte_carlo=bool(args.skip_monte_carlo),
        skip_sensitivity=bool(args.skip_sensitivity),
        walk_forward_profiles=args.walk_forward_profiles,
        render_only=bool(args.render_only),
    )
    print(f"Research engine ready. Open {args.output_dir / 'summary' / 'dashboard.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
