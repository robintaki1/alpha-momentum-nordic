from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a side-by-side HTML dashboard from two frozen candidate dashboard artifacts."
    )
    parser.add_argument("--left-run-dir", type=Path, required=True)
    parser.add_argument("--right-run-dir", type=Path, required=True)
    parser.add_argument("--search-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Frozen Candidate Comparison")
    parser.add_argument("--subtitle", default="Two frozen candidates re-run under the same validation protocol.")
    parser.add_argument("--left-label", default="Incumbent")
    parser.add_argument("--right-label", default="Challenger")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_href(output_path: Path, target_path: Path) -> str:
    return Path(
        os.path.relpath(
            Path(target_path).resolve(),
            start=output_path.resolve().parent,
        )
    ).as_posix()


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def format_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def yes_no(value: Any) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def find_search_candidate(search_summary: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for item in search_summary.get("ranked_candidates", []):
        if item.get("candidate_id") == candidate_id:
            return item
    locked = search_summary.get("locked_candidate") or {}
    if locked.get("candidate_id") == candidate_id:
        return locked
    return {}


def escape_text(value: Any) -> str:
    return html.escape(str(value if value is not None else "n/a"))


def series_correlation(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = (
        sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)
    ) ** 0.5
    if denominator == 0:
        return None
    return numerator / denominator


def median_abs_diff(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    diffs = sorted(abs(a - b) for a, b in zip(left, right))
    return diffs[len(diffs) // 2]


def parse_candidate_id(candidate_id: str) -> dict[str, str | None]:
    match = re.match(r"^l(?P<lookback>\d+)_s(?P<skip>\d+)_n(?P<top_n>\d+)_strat=(?P<strategy>.+)$", candidate_id or "")
    if match is None:
        return {"lookback": None, "skip": None, "top_n": None, "strategy": None}
    return {
        "lookback": match.group("lookback"),
        "skip": match.group("skip"),
        "top_n": match.group("top_n"),
        "strategy": match.group("strategy"),
    }


def build_candidate_context(
    *,
    run_dir: Path,
    label: str,
    search_summary: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    selection_path = run_dir / "selection_summary.json"
    holdout_path = run_dir / "holdout_results.json"
    dashboard_path = run_dir / "dashboard.html"

    selection = load_json(selection_path)
    holdout = load_json(holdout_path)

    locked = selection.get("locked_candidate") or {}
    candidate_id = locked.get("candidate_id", "n/a")
    walk_forward = selection.get("walk_forward") or {}
    wf_combined = walk_forward.get("combined") or {}
    monte = selection.get("monte_carlo") or {}
    mc_metrics = monte.get("metrics") or {}
    mc_sharpe = mc_metrics.get("sharpe") or {}
    search_candidate = find_search_candidate(search_summary, candidate_id)
    candidate_meta = parse_candidate_id(candidate_id)

    return {
        "label": label,
        "candidate_id": candidate_id,
        "candidate_meta": candidate_meta,
        "selection_status": selection.get("selection_status", "n/a"),
        "original_rank": search_candidate.get("rank"),
        "original_selected": search_candidate.get("selected"),
        "fold_pass_count": locked.get("fold_pass_count"),
        "bootstrap_ci_low": locked.get("bootstrap_ci_low"),
        "deflated_sharpe_score": locked.get("deflated_sharpe_score"),
        "gate_bootstrap": locked.get("gate_bootstrap"),
        "gate_deflated_sharpe": locked.get("gate_deflated_sharpe"),
        "gate_fold_count": locked.get("gate_fold_count"),
        "gate_negative_controls": locked.get("gate_negative_controls"),
        "walk_forward_sharpe": wf_combined.get("sharpe"),
        "median_validation_sharpe": locked.get("median_validation_sharpe"),
        "mc_sharpe_median": mc_sharpe.get("median"),
        "pbo_band": selection.get("pbo_band", "n/a"),
        "holdout_status": holdout.get("status", "n/a"),
        "search_selected": search_candidate.get("selected"),
        "validation_returns": walk_forward.get("combined_returns") or [],
        "dashboard_path": dashboard_path,
        "selection_path": selection_path,
        "holdout_path": holdout_path,
        "dashboard_href": rel_href(output_path, dashboard_path),
        "selection_href": rel_href(output_path, selection_path),
        "holdout_href": rel_href(output_path, holdout_path),
    }


def render_metric_card(label: str, value: str, note: str) -> str:
    return (
        '<div class="card">'
        f'<div class="label">{escape_text(label)}</div>'
        f'<div class="value">{escape_text(value)}</div>'
        f'<div class="muted">{escape_text(note)}</div>'
        "</div>"
    )


def render_candidate_card(candidate: dict[str, Any]) -> str:
    rows = [
        ("Frozen status", candidate["selection_status"]),
        ("Original search rank", f"#{candidate['original_rank']}" if candidate["original_rank"] is not None else "n/a"),
        ("Fold passes", f"{candidate['fold_pass_count']}/5" if candidate["fold_pass_count"] is not None else "n/a"),
        ("Bootstrap CI low", format_float(candidate["bootstrap_ci_low"])),
        ("Deflated Sharpe", format_float(candidate["deflated_sharpe_score"])),
        ("Walk-forward Sharpe", format_float(candidate["walk_forward_sharpe"])),
        ("Median validation Sharpe", format_float(candidate["median_validation_sharpe"])),
        ("MC Sharpe (median)", format_float(candidate["mc_sharpe_median"])),
        ("Neg-controls gate", yes_no(candidate["gate_negative_controls"])),
    ]
    stats_html = "".join(
        f'<div class="stat-row"><span>{escape_text(label)}</span><strong>{escape_text(value)}</strong></div>'
        for label, value in rows
    )
    return f"""
    <div class="card candidate-card">
      <div class="eyebrow">{escape_text(candidate["label"])}</div>
      <h2>{escape_text(candidate["candidate_id"])}</h2>
      <p class="muted">Frozen rerun under the same ex_norway validation protocol.</p>
      <div class="stats">
        {stats_html}
      </div>
      <div class="actions">
        <a href="{escape_text(candidate["dashboard_href"])}">Open standalone dashboard</a>
        <a href="{escape_text(candidate["selection_href"])}">selection_summary.json</a>
        <a href="{escape_text(candidate["holdout_href"])}">holdout_results.json</a>
      </div>
      <p class="muted">Final holdout remains linked below as its own artifact and is intentionally not ranked side by side here.</p>
    </div>
    """


def render_compare_table(left: dict[str, Any], right: dict[str, Any]) -> str:
    rows = [
        ("Original search rank", f"#{left['original_rank']}" if left["original_rank"] is not None else "n/a", f"#{right['original_rank']}" if right["original_rank"] is not None else "n/a"),
        ("Originally selected", yes_no(left["original_selected"]), yes_no(right["original_selected"])),
        ("Frozen status", left["selection_status"], right["selection_status"]),
        ("top_n", escape_text(left["candidate_meta"].get("top_n")), escape_text(right["candidate_meta"].get("top_n"))),
        ("Fold passes", f"{left['fold_pass_count']}/5" if left["fold_pass_count"] is not None else "n/a", f"{right['fold_pass_count']}/5" if right["fold_pass_count"] is not None else "n/a"),
        ("Bootstrap gate", yes_no(left["gate_bootstrap"]), yes_no(right["gate_bootstrap"])),
        ("Bootstrap CI low", format_float(left["bootstrap_ci_low"]), format_float(right["bootstrap_ci_low"])),
        ("Deflated Sharpe gate", yes_no(left["gate_deflated_sharpe"]), yes_no(right["gate_deflated_sharpe"])),
        ("Deflated Sharpe score", format_float(left["deflated_sharpe_score"]), format_float(right["deflated_sharpe_score"])),
        ("Fold-count gate", yes_no(left["gate_fold_count"]), yes_no(right["gate_fold_count"])),
        ("Neg-controls gate", yes_no(left["gate_negative_controls"]), yes_no(right["gate_negative_controls"])),
        ("Median validation Sharpe", format_float(left["median_validation_sharpe"]), format_float(right["median_validation_sharpe"])),
        ("Walk-forward Sharpe", format_float(left["walk_forward_sharpe"]), format_float(right["walk_forward_sharpe"])),
        ("MC Sharpe (median)", format_float(left["mc_sharpe_median"]), format_float(right["mc_sharpe_median"])),
        ("Frozen rerun PBO band", left["pbo_band"], right["pbo_band"]),
    ]
    row_html = "".join(
        "<tr>"
        f"<td>{escape_text(metric)}</td>"
        f"<td>{escape_text(left_value)}</td>"
        f"<td>{escape_text(right_value)}</td>"
        "</tr>"
        for metric, left_value, right_value in rows
    )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>{escape_text(left["candidate_id"])}</th>
          <th>{escape_text(right["candidate_id"])}</th>
        </tr>
      </thead>
      <tbody>
        {row_html}
      </tbody>
    </table>
    """


def build_html(
    *,
    title: str,
    subtitle: str,
    left: dict[str, Any],
    right: dict[str, Any],
    search_summary: dict[str, Any],
) -> str:
    shared_pbo = format_pct(search_summary.get("backtest_overfitting", {}).get("pbo"))
    candidate_count = search_summary.get("backtest_overfitting", {}).get("candidate_count", "n/a")
    combo_count = search_summary.get("backtest_overfitting", {}).get("combination_count", "n/a")
    cadence = search_summary.get("cadence_label", "n/a")
    compare_table = render_compare_table(left, right)
    left_card = render_candidate_card(left)
    right_card = render_candidate_card(right)
    validation_corr = series_correlation(left["validation_returns"], right["validation_returns"])
    validation_median_diff = median_abs_diff(left["validation_returns"], right["validation_returns"])
    left_top_n = left["candidate_meta"].get("top_n")
    right_top_n = right["candidate_meta"].get("top_n")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape_text(title)}</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin-top:16px; }}
    .card {{ background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:20px; padding:18px; }}
    .label {{ color:#6c5842; text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; }}
    .value {{ font-size:1.3rem; margin-top:6px; overflow-wrap:anywhere; }}
    .muted {{ color:#5b6762; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; color:#b06b1d; }}
    .callout {{ background:#fff7eb; border:1px solid rgba(176,107,29,.28); border-radius:16px; padding:12px 14px; margin-top:12px; }}
    .actions {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }}
    a {{ color:#b06b1d; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
    .candidate-picks {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
    .candidate-card h2 {{ margin-bottom:8px; overflow-wrap:anywhere; }}
    .stats {{ display:flex; flex-direction:column; gap:8px; margin-top:12px; }}
    .stat-row {{ display:flex; justify-content:space-between; gap:16px; border-bottom:1px solid rgba(23,33,26,.08); padding-bottom:6px; }}
    .stat-row span {{ color:#5b6762; }}
  </style>
</head>
<body>
  <main>
    <section>
      <div class="eyebrow">Research Engine Comparison</div>
      <h1>{escape_text(title)}</h1>
      <p>{escape_text(subtitle)}</p>
      <div class="callout">
        <strong>Integrity note.</strong>
        <div class="muted">Both candidates were re-run as frozen hypotheses under the same ex_norway protocol. In these single-candidate reruns, PBO is intentionally n/a; the meaningful overfitting diagnostic is the original shared search PBO of {shared_pbo} across {escape_text(candidate_count)} candidates and {escape_text(combo_count)} CSCV combinations.</div>
      </div>
      <div class="callout">
        <strong>Decision note.</strong>
        <div class="muted">This page compares validation-safe diagnostics only. Final holdout artifacts remain linked for each frozen rerun, but they are not ranked side by side here because that would encourage post-hoc selection on the last untouched OOS window.</div>
      </div>
      <div class="grid">
        {render_metric_card("Search Cadence", cadence, "Original selection search context")}
        {render_metric_card("Shared Search PBO", shared_pbo, f"{candidate_count} candidates / {combo_count} combos")}
        {render_metric_card(left["label"], f"rank #{left['original_rank'] if left['original_rank'] is not None else 'n/a'}", f"Originally selected: {yes_no(left['original_selected'])}")}
        {render_metric_card(right["label"], f"rank #{right['original_rank'] if right['original_rank'] is not None else 'n/a'}", f"Originally selected: {yes_no(right['original_selected'])}")}
      </div>
    </section>
    <section>
      <h2>Quick Read</h2>
      <p class="muted">Some protocol visuals are expected to match exactly, especially the walk-forward schedule and fold windows, because both frozen reruns use the same dates and validation structure. The useful differences are in the candidate-specific metrics and the strategy-specific curves inside each dashboard.</p>
      <div class="callout">
        <strong>Why they look so similar.</strong>
        <div class="muted">These are both the same ex_norway thesis, with the same lookback, skip, and trend_filter family. The practical change is portfolio breadth: top_n {escape_text(left_top_n)} versus top_n {escape_text(right_top_n)}. Their monthly validation return correlation is {format_float(validation_corr)} and the median month-by-month validation return gap is {format_pct(validation_median_diff)}, so many validation charts naturally track each other closely.</div>
      </div>
      {compare_table}
    </section>
    <section>
      <h2>Candidate Artifacts</h2>
      <div class="candidate-picks">
        {left_card}
        {right_card}
      </div>
    </section>
    <section>
      <h2>Standalone Dashboards</h2>
      <p class="muted">Each frozen rerun still has its full native dashboard and its own holdout artifact. Open them individually from the cards above when you want the complete view without turning the final holdout into a side-by-side selector.</p>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    search_summary = load_json(args.search_summary)
    left = build_candidate_context(
        run_dir=args.left_run_dir.resolve(),
        label=args.left_label,
        search_summary=search_summary,
        output_path=output_path,
    )
    right = build_candidate_context(
        run_dir=args.right_run_dir.resolve(),
        label=args.right_label,
        search_summary=search_summary,
        output_path=output_path,
    )
    html_text = build_html(
        title=args.title,
        subtitle=args.subtitle,
        left=left,
        right=right,
        search_summary=search_summary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
