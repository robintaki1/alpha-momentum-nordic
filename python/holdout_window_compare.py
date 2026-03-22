from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

import validation_protocol
from paper_trading_engine import ResearchDataset, build_thesis
from research_engine import evaluate_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run holdout diagnostics for custom windows.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--output-dir", default=Path("results/holdout_window_compare"), type=Path)
    parser.add_argument("--thesis", default="baseline")
    parser.add_argument(
        "--params-file",
        required=True,
        help="Path to a JSON file containing selected_params or locked_candidate params.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        required=True,
        help="Holdout windows as YYYY-MM:YYYY-MM (start:end).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of trials for deflated Sharpe metrics (defaults to 1).",
    )
    return parser.parse_args()


def load_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "selected_params" in payload:
        return payload["selected_params"]
    locked = payload.get("locked_candidate") or {}
    if "params" in locked:
        return locked["params"]
    if "params" in payload:
        return payload["params"]
    raise ValueError(f"No params found in {path}. Expected selected_params or locked_candidate.params.")


def parse_windows(values: list[str]) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for raw in values:
        if ":" not in raw:
            raise ValueError(f"Invalid window '{raw}'. Use YYYY-MM:YYYY-MM.")
        start, end = raw.split(":", 1)
        windows.append((start.strip(), end.strip()))
    return windows


def _equity_curve(returns: list[float]) -> list[float]:
    equity = [1.0]
    for value in returns:
        equity.append(equity[-1] * (1.0 + float(value)))
    return equity


def _cagr(total_return: float, months: int) -> float:
    if months <= 0:
        return 0.0
    return (1.0 + total_return) ** (12.0 / months) - 1.0


def _volatility(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns) * math.sqrt(12.0)


def _sortino(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_value = statistics.fmean(returns)
    downside = [value for value in returns if value < 0.0]
    if len(downside) < 2:
        return 0.0
    downside_std = statistics.stdev(downside)
    if downside_std == 0:
        return 0.0
    return mean_value / downside_std * math.sqrt(12.0)


def _win_rate(returns: list[float]) -> float:
    if not returns:
        return 0.0
    wins = sum(1 for value in returns if value > 0.0)
    return wins / len(returns)


def _render_chart(
    *,
    strategy_returns: list[float],
    benchmark_returns: list[float],
    title: str,
    output_path: Path,
) -> None:
    strategy_equity = _equity_curve(strategy_returns)
    benchmark_equity = _equity_curve(benchmark_returns) if benchmark_returns else []
    plt.figure(figsize=(9, 3.6))
    plt.plot(strategy_equity, color="#b06b1d", linewidth=2.0, label="Strategy")
    if benchmark_equity:
        plt.plot(benchmark_equity, color="#9a8c6a", linewidth=1.8, linestyle="--", label="Primary benchmark")
    plt.title(title)
    plt.xlabel("Months")
    plt.ylabel("Equity multiple")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140)
    plt.close()


def _build_html_report(
    *,
    title: str,
    window_label: str,
    metrics: dict[str, Any],
    chart_path: Path,
    output_path: Path,
) -> None:
    rows = "\n".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in metrics.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:980px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <p>{window_label}</p>
    </section>
    <section>
      <h2>Equity Curve</h2>
      <img src="{chart_path.name}" alt="{title}" style="width:100%; border-radius:16px; border:1px solid rgba(23,33,26,.12);">
    </section>
    <section>
      <h2>Metrics & Robustness Checks</h2>
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    params_path = Path(args.params_file)
    params = load_params(params_path)
    windows = parse_windows(args.windows)

    thesis = build_thesis(args.thesis).manifest_metadata()
    dataset = ResearchDataset(args.data_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    summary_cards = []

    for start_month, end_month in windows:
        holdout = evaluate_holdout(
            dataset,
            thesis=thesis,
            params=params,
            period_label="months",
            periods_per_year=12,
            start_month=start_month,
            end_month=end_month,
        )
        holdout["status"] = "ok"
        tag = f"{start_month}_to_{end_month}".replace("-", "")
        out_path = output_dir / f"holdout_{tag}.json"
        out_path.write_text(json.dumps(holdout, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        base_main = holdout.get("results", {}).get("Full Nordics", {}).get("next_open", {}).get("base", {})
        strategy_returns = base_main.get("strategy_returns", [])
        benchmark_returns = base_main.get("primary_benchmark_returns", [])
        if not strategy_returns:
            # fallback to recompute returns from totals
            strategy_returns = []
        months = int(base_main.get("months", len(strategy_returns)))
        total_return = float(base_main.get("total_return", 0.0))
        benchmark_total_return = float(base_main.get("primary_benchmark_total_return", 0.0))
        cagr = _cagr(total_return, months)
        bench_cagr = _cagr(benchmark_total_return, months)
        excess_returns = []
        if strategy_returns and benchmark_returns and len(strategy_returns) == len(benchmark_returns):
            excess_returns = [float(a) - float(b) for a, b in zip(strategy_returns, benchmark_returns)]
        dsr_metrics = validation_protocol.deflated_sharpe_metrics(strategy_returns or [0.0], args.n_trials)
        bootstrap_low, bootstrap_high = validation_protocol.stationary_bootstrap_sharpe_ci(
            strategy_returns or [0.0],
            n_resamples=validation_protocol.config.BOOTSTRAP_RESAMPLES,
        )
        info_ratio = validation_protocol.annualized_sharpe(excess_returns) if excess_returns else 0.0
        max_dd = validation_protocol.max_drawdown(strategy_returns) if strategy_returns else 0.0
        vol = _volatility(strategy_returns) if strategy_returns else 0.0
        sortino = _sortino(strategy_returns) if strategy_returns else 0.0
        win_rate = _win_rate(strategy_returns) if strategy_returns else 0.0
        calmar = (cagr / max_dd) if max_dd > 0 else 0.0

        metrics = {
            "Strategy total return": f"{total_return:.3f}",
            "Benchmark total return": f"{benchmark_total_return:.3f}",
            "Strategy CAGR": f"{cagr:.3%}",
            "Benchmark CAGR": f"{bench_cagr:.3%}",
            "CAGR alpha": f"{(cagr - bench_cagr):.3%}",
            "Net Sharpe": f"{base_main.get('net_sharpe', 0.0):.3f}",
            "Max drawdown": f"{max_dd:.3f}",
            "Volatility (annualized)": f"{vol:.3f}",
            "Sortino": f"{sortino:.3f}",
            "Calmar": f"{calmar:.3f}",
            "Win rate": f"{win_rate:.1%}",
            "Info ratio (excess)": f"{info_ratio:.3f}",
            "Bootstrap Sharpe CI (low)": f"{bootstrap_low:.3f}",
            "Bootstrap Sharpe CI (high)": f"{bootstrap_high:.3f}",
            "Deflated Sharpe score": f"{dsr_metrics['score']:.3f}",
            "Deflated Sharpe prob": f"{dsr_metrics['probability']:.3f}",
            "Beats primary benchmark": str(base_main.get("beats_primary_benchmark", False)),
        }

        chart_path = output_dir / f"holdout_{tag}_chart.png"
        if strategy_returns and benchmark_returns:
            _render_chart(
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                title=f"Portfolio vs Benchmark ({start_month} to {end_month})",
                output_path=chart_path,
            )

        html_path = output_dir / f"holdout_{tag}.html"
        _build_html_report(
            title="Holdout Window Report",
            window_label=f"{start_month} to {end_month} (months={months})",
            metrics=metrics,
            chart_path=chart_path,
            output_path=html_path,
        )

        summary_rows.append(
            {
                "window": f"{start_month} to {end_month}",
                "net_sharpe": base_main.get("net_sharpe"),
                "total_return": total_return,
                "primary_benchmark_total_return": benchmark_total_return,
                "strategy_cagr": cagr,
                "benchmark_cagr": bench_cagr,
                "beats_primary_benchmark": base_main.get("beats_primary_benchmark"),
            }
        )
        summary_cards.append(
            {
                "window": f"{start_month} to {end_month}",
                "report_html": html_path.name,
                "chart_png": chart_path.name,
            }
        )

    summary_path = output_dir / "holdout_window_summary.json"
    summary_payload = {
        "thesis": thesis,
        "params": params,
        "windows": summary_rows,
        "window_reports": summary_cards,
        "source_params_file": str(params_path),
        "n_trials": args.n_trials,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_html_rows = "\n".join(
        f"<tr><td>{row['window']}</td>"
        f"<td>{row['strategy_cagr']:.3%}</td>"
        f"<td>{row['benchmark_cagr']:.3%}</td>"
        f"<td>{row['net_sharpe']:.3f}</td>"
        f"<td>{row['beats_primary_benchmark']}</td>"
        f"<td><a href=\"{card['report_html']}\">Open report</a></td></tr>"
        for row, card in zip(summary_rows, summary_cards, strict=True)
    )
    summary_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Holdout Window Summary</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:980px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Holdout Window Summary</h1>
      <p>Summary across custom holdout windows.</p>
    </section>
    <section>
      <table>
        <thead>
          <tr><th>Window</th><th>Strategy CAGR</th><th>Benchmark CAGR</th><th>Sharpe</th><th>Beats Benchmark</th><th>Report</th></tr>
        </thead>
        <tbody>
          {summary_html_rows}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "holdout_window_summary.html").write_text(summary_html, encoding="utf-8")
    print(f"Wrote holdout window summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
