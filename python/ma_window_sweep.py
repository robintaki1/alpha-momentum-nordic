from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import validation_protocol
from paper_trading_engine import ResearchDataset, build_thesis, serialize_json
from research_engine import (
    active_selection_variants,
    aggregate_candidates,
    build_candidate_evaluations,
    compute_cscv_pbo,
    compute_negative_controls,
    selection_summary,
)


DEFAULT_MA_WINDOWS = (6, 9, 10, 12, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trend-filter MA windows without touching core outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ma_window_sweep"))
    parser.add_argument("--thesis", default=None)
    parser.add_argument("--ma-windows", nargs="+", type=int, default=list(DEFAULT_MA_WINDOWS))
    parser.add_argument("--lookback", type=int, default=config.L)
    parser.add_argument("--skip", type=int, default=config.SKIP)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--cross-section-runs", type=int, default=None)
    parser.add_argument("--block-runs", type=int, default=None)
    return parser.parse_args()


def resolve_default_thesis() -> str:
    summary_path = ROOT / "results" / "cadence_compare_rebuild" / "summary" / "cadence_comparison.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            winner = payload.get("winner") or {}
            thesis = winner.get("thesis") or {}
            name = thesis.get("name")
            if isinstance(name, str) and name:
                return name
        except (json.JSONDecodeError, OSError):
            pass
    return config.DEFAULT_RESEARCH_THESIS


def _profile_settings(cross_section_runs: int | None, block_runs: int | None) -> dict[str, Any]:
    baseline = config.RESEARCH_PROFILE_SETTINGS["certification_baseline"]
    return {
        "ma_sweep": {
            "lookbacks": (config.L,),
            "skips": (config.SKIP,),
            "top_ns": (8,),
            "bootstrap_resamples": baseline["bootstrap_resamples"],
            "cross_sectional_shuffle_runs": cross_section_runs
            if cross_section_runs is not None
            else baseline["cross_sectional_shuffle_runs"],
            "block_shuffled_null_runs": block_runs
            if block_runs is not None
            else baseline["block_shuffled_null_runs"],
        }
    }


def build_params_grid(
    *,
    lookback: int,
    skip: int,
    top_n: int,
    ma_windows: Sequence[int],
) -> list[dict[str, Any]]:
    params_grid: list[dict[str, Any]] = []
    for ma_window in sorted({int(value) for value in ma_windows if int(value) > 0}):
        params_grid.append(
            {
                "l": int(lookback),
                "skip": int(skip),
                "top_n": int(top_n),
                "strategy_id": f"trend_filter_ma{int(ma_window)}",
                "trend_filter": True,
                "ma_window": int(ma_window),
                "rebalance": "full",
                "weighting": "equal",
            }
        )
    return params_grid


def build_report_html(summary: dict[str, Any], *, title: str, subtitle: str) -> str:
    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not numeric == numeric:
            return "n/a"
        return f"{numeric:.3f}"

    rows: list[str] = []
    candidates = summary.get("ranked_candidates") or []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.get("params", {}).get("ma_window") or 0,
            item.get("rank", 9999),
        ),
    ):
        params = candidate.get("params") or {}
        rows.append(
            "<tr>"
            f"<td>{params.get('ma_window','n/a')}</td>"
            f"<td>{candidate.get('rank','n/a')}</td>"
            f"<td>{candidate.get('candidate_id','n/a')}</td>"
            f"<td>{_fmt(candidate.get('median_validation_sharpe'))}</td>"
            f"<td>{_fmt(candidate.get('overall_sharpe'))}</td>"
            f"<td>{_fmt(candidate.get('max_drawdown'))}</td>"
            f"<td>{_fmt(candidate.get('total_return'))}</td>"
            f"<td>{_fmt(candidate.get('bootstrap_ci_low'))}</td>"
            f"<td>{_fmt(candidate.get('deflated_sharpe_score'))}</td>"
            f"<td>{candidate.get('fold_pass_count','n/a')}</td>"
            f"<td>{'yes' if candidate.get('gate_bootstrap') else 'no'}</td>"
            f"<td>{'yes' if candidate.get('gate_deflated_sharpe') else 'no'}</td>"
            f"<td>{'yes' if candidate.get('gate_negative_controls') else 'no'}</td>"
            f"<td>{'yes' if candidate.get('selected') else 'no'}</td>"
            "</tr>"
        )
    backtest = summary.get("backtest_overfitting") or {}
    negative = summary.get("negative_controls") or {}
    cross = negative.get("cross_sectional_shuffle") or {}
    block = negative.get("block_shuffled_null") or {}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:1100px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
    .card {{ background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:18px; padding:14px; }}
    .label {{ text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; color:#6c5842; }}
    .value {{ font-size:1.05rem; margin-top:4px; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </section>
    <section>
      <h2>Diagnostics</h2>
      <div class="grid">
        <div class="card"><div class="label">Selection Status</div><div class="value">{summary.get('selection_status','n/a')}</div></div>
        <div class="card"><div class="label">PBO</div><div class="value">{backtest.get('pbo','n/a')}</div></div>
        <div class="card"><div class="label">PBO Threshold</div><div class="value">{backtest.get('pbo_threshold_max','n/a')}</div></div>
        <div class="card"><div class="label">Candidates Tested</div><div class="value">{backtest.get('candidate_count','n/a')}</div></div>
        <div class="card"><div class="label">Neg Controls</div><div class="value">{cross.get('pass_count','n/a')}/{cross.get('run_count','n/a')} · {block.get('pass_count','n/a')}/{block.get('run_count','n/a')}</div></div>
      </div>
    </section>
    <section>
      <h2>MA Window Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>MA Window</th>
            <th>Rank</th>
            <th>Candidate</th>
            <th>Median Sharpe</th>
            <th>Overall Sharpe</th>
            <th>Max DD</th>
            <th>Total Return</th>
            <th>Bootstrap Low</th>
            <th>Deflated Sharpe</th>
            <th>Fold Pass</th>
            <th>Gate Boot</th>
            <th>Gate DSR</th>
            <th>Gate Neg</th>
            <th>Selected</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    thesis_name = args.thesis or resolve_default_thesis()
    profile_settings = _profile_settings(args.cross_section_runs, args.block_runs)
    period_label = "months"
    periods_per_year = 12

    params_grid = build_params_grid(
        lookback=args.lookback,
        skip=args.skip,
        top_n=args.top_n,
        ma_windows=args.ma_windows,
    )
    if not params_grid:
        raise ValueError("No MA windows provided for the sweep.")

    output_dir = args.output_dir / thesis_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ResearchDataset(args.data_dir)
    thesis_meta = build_thesis(thesis_name).manifest_metadata()
    candidates = build_candidate_evaluations(
        dataset,
        thesis=thesis_meta,
        params_grid=params_grid,
        periods_per_year=periods_per_year,
    )
    negative_controls = compute_negative_controls(
        dataset,
        candidates=candidates,
        periods_per_year=periods_per_year,
        profile_name="ma_sweep",
        profile_settings=profile_settings,
        thesis_exclusions=tuple(thesis_meta.get("excluded_countries", ())),
    )
    aggregates = aggregate_candidates(
        candidates=candidates,
        negative_controls=negative_controls,
        periods_per_year=periods_per_year,
    )
    active_variants = active_selection_variants(dataset, tuple(thesis_meta.get("excluded_countries", ())))
    summary = selection_summary(
        aggregates=aggregates,
        negative_controls=negative_controls,
        mode="ma_sweep",
        period_label=period_label,
        periods_per_year=periods_per_year,
        active_variants=active_variants,
    )
    summary["backtest_overfitting"] = compute_cscv_pbo(
        candidates,
        periods_per_year=periods_per_year,
        period_label=period_label,
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["thesis"] = thesis_meta
    summary["params_grid"] = params_grid
    summary["profile_settings"] = profile_settings["ma_sweep"]

    serialize_json(output_dir / "ma_window_sweep_summary.json", summary)

    rows: list[dict[str, Any]] = []
    for candidate in summary.get("ranked_candidates") or []:
        params = candidate.get("params") or {}
        rows.append(
            {
                "ma_window": params.get("ma_window"),
                "rank": candidate.get("rank"),
                "candidate_id": candidate.get("candidate_id"),
                "median_validation_sharpe": candidate.get("median_validation_sharpe"),
                "overall_sharpe": candidate.get("overall_sharpe"),
                "max_drawdown": candidate.get("max_drawdown"),
                "total_return": candidate.get("total_return"),
                "bootstrap_ci_low": candidate.get("bootstrap_ci_low"),
                "deflated_sharpe_score": candidate.get("deflated_sharpe_score"),
                "fold_pass_count": candidate.get("fold_pass_count"),
                "gate_bootstrap": candidate.get("gate_bootstrap"),
                "gate_deflated_sharpe": candidate.get("gate_deflated_sharpe"),
                "gate_negative_controls": candidate.get("gate_negative_controls"),
                "selected": candidate.get("selected"),
            }
        )
    write_csv(rows, output_dir / "ma_window_comparison.csv")

    report_html = build_report_html(
        summary,
        title="Trend Filter MA Window Sweep",
        subtitle=(
            f"Thesis: {thesis_meta.get('name','n/a')} · "
            f"L={args.lookback} / skip={args.skip} / top_n={args.top_n} · "
            f"MA windows: {', '.join(str(row['ma_window']) for row in rows)}"
        ),
    )
    (output_dir / "ma_window_report.html").write_text(report_html, encoding="utf-8")

    print(f"MA window sweep complete. Outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
