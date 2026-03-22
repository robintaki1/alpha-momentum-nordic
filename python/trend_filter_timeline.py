from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import validation_protocol
from paper_trading_engine import ResearchDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trend-filter on/off timeline for a locked candidate.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--window",
        choices=("validation", "holdout", "full"),
        default="validation",
        help="Which months to include (validation folds, holdout window, or full history).",
    )
    return parser.parse_args()


def _unique_ordered(indices: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    return ordered


def _window_indices(dataset: ResearchDataset, window: str) -> list[int]:
    if window == "full":
        return [idx for idx, month in enumerate(dataset.holding_months) if month is not None]
    if window == "holdout":
        return dataset._window_signal_indices(config.OOS_START, config.OOS_END)

    indices: list[int] = []
    for fold in validation_protocol.fixed_folds():
        indices.extend(dataset._window_signal_indices(fold.validate_start, fold.validate_end))
    return _unique_ordered(indices)


def _trend_filter_meta(
    series: list[float] | None,
    *,
    signal_index: int,
    ma_window: int,
) -> tuple[bool, str, float | None, float | None]:
    if not series:
        return True, "missing_benchmark_series", None, None
    price_index = signal_index - 1  # use last fully known month to avoid look-ahead
    if price_index < 0:
        current = series[price_index] if 0 <= price_index < len(series) else None
        return True, "insufficient_history", current, None
    window_start = price_index - int(ma_window) + 1
    if window_start < 0 or price_index >= len(series):
        current = series[price_index] if 0 <= price_index < len(series) else None
        return True, "insufficient_history", current, None
    window = series[window_start : price_index + 1]
    if any(not math.isfinite(value) for value in window):
        current = series[price_index] if 0 <= price_index < len(series) else None
        return True, "missing_benchmark_data", current, None
    current = series[price_index]
    if not math.isfinite(current):
        return True, "missing_benchmark_data", current, None
    average = float(sum(window)) / float(len(window))
    if current >= average:
        return True, "price_above_ma", current, average
    return False, "price_below_ma", current, average


def _coerce_series(raw: Iterable[object], *, limit: int) -> list[float]:
    series: list[float] = []
    for value in raw:
        if len(series) >= limit:
            break
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not math.isfinite(numeric):
            numeric = 0.0
        series.append(numeric)
    return series


def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    curve: list[float] = []
    for value in returns:
        equity *= 1.0 + float(value)
        curve.append(equity)
    return curve


def _build_return_chart(
    *,
    months: list[str],
    filter_on: list[bool],
    returns: list[float],
    benchmark_returns: list[float] | None,
) -> str:
    if not returns:
        return '<div class="muted">Return curve not available.</div>'

    bench_series = benchmark_returns or []
    n = len(returns)
    if bench_series:
        n = min(n, len(bench_series))
    n = min(n, len(months), len(filter_on))
    if n < 2:
        return '<div class="muted">Return curve not available.</div>'

    months = months[:n]
    filter_on = filter_on[:n]
    returns = returns[:n]
    bench_series = bench_series[:n] if bench_series else []

    portfolio_curve = _equity_curve(returns)
    benchmark_curve = _equity_curve(bench_series) if bench_series else []
    min_val = min(portfolio_curve + (benchmark_curve or []))
    max_val = max(portfolio_curve + (benchmark_curve or []))
    if math.isclose(min_val, max_val):
        max_val = min_val + 1.0

    width = 920.0
    height = 260.0
    margin_left = 52.0
    margin_right = 18.0
    margin_top = 20.0
    margin_bottom = 42.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    step = plot_width / float(max(1, n - 1))

    def _xy(index: int, value: float) -> tuple[float, float]:
        x = margin_left + index * step
        y = margin_top + (max_val - value) / (max_val - min_val) * plot_height
        return x, y

    def _points(series: list[float]) -> str:
        return " ".join("{:.1f},{:.1f}".format(*_xy(idx, value)) for idx, value in enumerate(series))

    def _tick(value: float) -> str:
        return "{:.2f}x".format(value)

    # Shade months when the filter is OFF (cash).
    shade_rects: list[str] = []
    if filter_on:
        start = None
        for idx, state in enumerate(filter_on):
            if not state and start is None:
                start = idx
            if (state or idx == n - 1) and start is not None:
                end = idx - 1 if state else idx
                half = step * 0.5
                x0 = max(margin_left, margin_left + start * step - half)
                x1 = min(margin_left + plot_width, margin_left + end * step + half)
                shade_rects.append(
                    '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
                    'fill="#f4d6cc" opacity="0.45" />'.format(
                        x0, margin_top, max(0.0, x1 - x0), plot_height
                    )
                )
                start = None

    # Y-axis ticks.
    tick_count = 4
    ticks = []
    for idx in range(tick_count + 1):
        frac = idx / float(tick_count)
        value = max_val - frac * (max_val - min_val)
        y = margin_top + frac * plot_height
        ticks.append(
            '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="#d3c7b7" stroke-width="1" />'.format(margin_left, y, width - margin_right, y)
        )
        ticks.append(
            '<text x="{:.1f}" y="{:.1f}" text-anchor="end" font-size="10" fill="#6c5842">{}</text>'.format(
                margin_left - 6.0, y + 3.0, _tick(value)
            )
        )

    # X-axis labels at Januarys plus first/last.
    label_indices: list[int] = []
    for idx, month in enumerate(months):
        if month.endswith("-01"):
            label_indices.append(idx)
    if 0 not in label_indices:
        label_indices.insert(0, 0)
    if (n - 1) not in label_indices:
        label_indices.append(n - 1)
    label_indices = sorted(set(label_indices))
    x_labels = []
    for idx in label_indices:
        x = margin_left + idx * step
        label = months[idx]
        x_labels.append(
            '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" font-size="10" fill="#6c5842">{}</text>'.format(
                x, height - 16.0, label
            )
        )

    portfolio_points = _points(portfolio_curve)
    benchmark_points = _points(benchmark_curve) if benchmark_curve else ""
    benchmark_poly = (
        '<polyline points="{}" fill="none" stroke="#9a8c6a" stroke-width="2" stroke-dasharray="4 4" />'.format(
            benchmark_points
        )
        if benchmark_points
        else ""
    )
    portfolio_poly = (
        '<polyline points="{}" fill="none" stroke="#b06b1d" stroke-width="2.5" />'.format(
            portfolio_points
        )
    )

    return (
        '<svg class="return-curve" viewBox="0 0 {width:.0f} {height:.0f}" width="100%" height="260" '
        'preserveAspectRatio="none">'
        '{shade}'
        '<rect x="{ml:.1f}" y="{mt:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
        'fill="none" stroke="#d9cdbd" stroke-width="1" />'
        '{ticks}'
        '{bench}'
        '{port}'
        '{xlabel}'
        '<text x="{ml:.1f}" y="{yt:.1f}" text-anchor="start" font-size="10" fill="#6c5842">Equity multiple</text>'
        "</svg>"
    ).format(
        width=width,
        height=height,
        shade="".join(shade_rects),
        ml=margin_left,
        mt=margin_top,
        pw=plot_width,
        ph=plot_height,
        ticks="".join(ticks),
        bench=benchmark_poly,
        port=portfolio_poly,
        xlabel="".join(x_labels),
        yt=margin_top - 6.0,
    )


def main() -> int:
    args = parse_args()
    selection_path = args.selection_summary
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selection summary at {selection_path}")

    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    locked = payload.get("locked_candidate") or {}
    params = locked.get("params") or {}
    if not params:
        raise ValueError("Selection summary has no locked candidate params.")

    ma_window = int(params.get("ma_window", 10) or 10)
    dataset = ResearchDataset(args.data_dir)
    indices = _window_indices(dataset, args.window)

    series = dataset.benchmark_monthly_prices.get(config.PRIMARY_PASSIVE_BENCHMARK_ID)
    rows: list[dict[str, object]] = []
    for signal_index in indices:
        signal_month = dataset.signal_months[signal_index]
        holding_month = dataset.holding_months[signal_index]
        filter_on, reason, current_price, ma_value = _trend_filter_meta(
            series,
            signal_index=signal_index,
            ma_window=ma_window,
        )
        price_minus_ma = None
        if current_price is not None and ma_value is not None:
            price_minus_ma = float(current_price) - float(ma_value)
        rows.append(
            {
                "signal_month": signal_month,
                "holding_month": holding_month,
                "filter_on": bool(filter_on),
                "reason": reason,
                "benchmark_price": current_price,
                "benchmark_ma": ma_value,
                "price_minus_ma": price_minus_ma,
                "ma_window": ma_window,
                "thesis": payload.get("thesis", {}).get("name") if isinstance(payload.get("thesis"), dict) else None,
                "candidate_id": locked.get("candidate_id"),
            }
        )

    locked_returns = locked.get("concatenated_returns") or []
    locked_benchmark = locked.get("primary_benchmark_returns") or []

    timeline_months = [str(row.get("holding_month") or "") for row in rows]
    timeline_filter = [bool(row.get("filter_on")) for row in rows]
    return_limit = len(rows) if rows else 0
    portfolio_returns = _coerce_series(locked_returns, limit=return_limit)
    benchmark_returns = _coerce_series(locked_benchmark, limit=return_limit) if locked_benchmark else []

    out_dir = args.out_dir or selection_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"trend_filter_timeline_{args.window}.csv"
    html_path = out_dir / f"trend_filter_timeline_{args.window}.html"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    on_count = sum(1 for row in rows if row.get("filter_on"))
    off_count = sum(1 for row in rows if row.get("filter_on") is False)
    total = len(rows)
    on_pct = (on_count / total * 100.0) if total else 0.0
    title = "Trend Filter Timeline"
    subtitle = (
        f"{locked.get('candidate_id','n/a')} - MA={ma_window} - window={args.window} - "
        f"on={on_count} ({on_pct:.1f}%) / off={off_count}"
    )
    chart_html = _build_return_chart(
        months=timeline_months,
        filter_on=timeline_filter,
        returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    def _fmt(value: object) -> str:
        if value is None:
            return ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(numeric):
            return ""
        return f"{numeric:.4f}"

    table_rows = "\n".join(
        "<tr class=\"{row_class}\">"
        "<td>{signal_month}</td>"
        "<td>{holding_month}</td>"
        "<td>{filter_state}</td>"
        "<td>{reason}</td>"
        "<td>{benchmark_price}</td>"
        "<td>{benchmark_ma}</td>"
        "<td>{price_minus_ma}</td>"
        "</tr>".format(
            row_class="on" if row.get("filter_on") else "off",
            signal_month=row.get("signal_month") or "",
            holding_month=row.get("holding_month") or "",
            filter_state="on" if row.get("filter_on") else "off",
            reason=row.get("reason") or "",
            benchmark_price=_fmt(row.get("benchmark_price")),
            benchmark_ma=_fmt(row.get("benchmark_ma")),
            price_minus_ma=_fmt(row.get("price_minus_ma")),
        )
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:1100px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .legend {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; font-size:.9rem; color:#5b6762; }}
    .legend-line {{ display:inline-block; width:32px; height:3px; border-radius:999px; margin-right:6px; vertical-align:middle; }}
    .legend-line.portfolio {{ background:#b06b1d; }}
    .legend-line.benchmark {{ background:#9a8c6a; border-top:2px dashed #9a8c6a; height:0; }}
    .legend-box {{ display:inline-block; width:18px; height:10px; background:#f4d6cc; border-radius:4px; margin-right:6px; vertical-align:middle; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
    .off {{ background:#f7ddd3; }}
    .on {{ background:#dcebdd; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </section>
    <section>
      <h2>Portfolio vs Benchmark</h2>
      {chart_html}
      <div class="legend">
        <div><span class="legend-line portfolio"></span>Portfolio equity</div>
        <div><span class="legend-line benchmark"></span>Primary benchmark</div>
        <div><span class="legend-box"></span>Filter OFF (cash)</div>
      </div>
    </section>
    <section>
      <h2>Timeline</h2>
      <table>
        <thead>
          <tr>
            <th>Signal Month</th>
            <th>Holding Month</th>
            <th>Filter</th>
            <th>Reason</th>
            <th>Benchmark Price</th>
            <th>Benchmark MA</th>
            <th>Price - MA</th>
          </tr>
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
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
