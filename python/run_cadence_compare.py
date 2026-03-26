from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from cadence_engine import CadenceDataset, cadence_period_label, load_cadence_spec
from paper_trading_engine import build_thesis
from phase1_lib import validate_phase1
from research_engine import (
    build_holdout_dashboard as shared_build_holdout_dashboard,
    build_phase1_dashboard as shared_build_phase1_dashboard,
    build_profile_dashboard as shared_build_profile_dashboard,
    evaluate_holdout,
    format_pbo_display,
    holdout_window_text,
    phase2_selection_window_text,
    phase2_validation_window_text,
    pbo_explainer,
    render_phase_map_html,
    render_timeframe_note,
    resolve_profile_settings,
    run_profile,
)


REBALANCE_LOGIC = "legacy_entries_exits"
REBALANCE_LOGIC_LABEL = "Legacy Entry/Exit Only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cadence comparison across schedules.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--results-root",
        default="results/cadence_compare_rebuild",
        help="Root folder for cadence compare outputs (default keeps the live snapshot untouched).",
    )
    parser.add_argument("--theses", nargs="+", default=list(config.RESEARCH_THESIS_SETTINGS))
    parser.add_argument("--cadences", nargs="+", default=list(config.DEFAULT_CADENCE_COMPARE_CADENCES))
    parser.add_argument("--profile-set", default="default")
    parser.add_argument(
        "--protocol-file",
        help="Path to a JSON protocol file defining folds and holdout window overrides.",
    )
    parser.add_argument(
        "--strategy-ids",
        nargs="+",
        help="Restrict the strategy grid to the provided strategy_id values.",
    )
    parser.add_argument(
        "--strategy-filter",
        choices=["all", "combo"],
        default="all",
        help="Apply a built-in filter over strategy variants (e.g., 'combo' keeps trend+rebalance/weighting combos).",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Only regenerate dashboards from existing cadence_comparison.json without rerunning profiles.",
    )
    return parser.parse_args()


def _serialize_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


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


def format_profile_set(
    profile_set: str,
    profile_settings: dict[str, Any],
    *,
    strategy_ids: list[str] | None = None,
    strategy_variants: list[dict[str, Any]] | None = None,
) -> str:
    profile = profile_settings.get("certification_baseline") or next(iter(profile_settings.values()), {})
    lookbacks = ", ".join(str(value) for value in profile.get("lookbacks", ())) or "n/a"
    skips = ", ".join(str(value) for value in profile.get("skips", ())) or "n/a"
    top_ns = ", ".join(str(value) for value in profile.get("top_ns", ())) or "n/a"
    if strategy_ids is not None:
        strategies = ", ".join(strategy_ids) or "n/a"
    else:
        source = strategy_variants or config.STRATEGY_VARIANTS
        strategies = ", ".join(item.get("strategy_id", "baseline") for item in source) or "n/a"
    return f"{profile_set} (L={lookbacks} / skip={skips} / top_n={top_ns} / strat={strategies})"


def _is_combo_strategy(strategy: dict[str, Any]) -> bool:
    if not strategy.get("trend_filter"):
        return False
    rebalance = strategy.get("rebalance", "full")
    weighting = strategy.get("weighting", "equal")
    if rebalance != "full" or weighting != "equal":
        return True
    if strategy.get("band_buffer") is not None:
        return True
    if strategy.get("min_hold_months") is not None:
        return True
    if strategy.get("vol_window") is not None and weighting != "equal":
        return True
    return False


def select_strategy_variants(
    *,
    strategy_ids: list[str] | None,
    strategy_filter: str,
) -> list[dict[str, Any]]:
    variants = list(config.STRATEGY_VARIANTS)
    if strategy_ids:
        allowed = {value.strip() for value in strategy_ids if value.strip()}
        variants = [item for item in variants if item.get("strategy_id") in allowed]
    if strategy_filter == "combo":
        variants = [item for item in variants if _is_combo_strategy(item)]
    if not variants:
        raise SystemExit("No strategy variants matched the provided filters.")
    return variants


def apply_protocol_overrides(protocol_path: str | None) -> dict[str, Any] | None:
    if not protocol_path:
        return None
    path = Path(protocol_path)
    if not path.exists():
        raise SystemExit(f"Protocol file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("rolling_origin_folds")
    if not folds:
        raise SystemExit("Protocol file missing rolling_origin_folds.")
    config.ROLLING_ORIGIN_FOLDS = [tuple(fold) for fold in folds]
    if "oos_start" in payload:
        config.OOS_START = payload["oos_start"]
    if "oos_end" in payload:
        config.OOS_END = payload["oos_end"]
    if "insample_end" in payload:
        config.INSAMPLE_END = payload["insample_end"]
    if "oos_sharpe_min" in payload:
        config.OOS_SHARPE_MIN = payload["oos_sharpe_min"]
    return payload


def compute_mcs(candidate: dict | None) -> int | None:
    if not candidate:
        return None
    score = 0
    score += 25 if candidate.get("gate_fold_count") else 0
    score += 25 if candidate.get("gate_bootstrap") else 0
    score += 25 if candidate.get("gate_deflated_sharpe") else 0
    score += 25 if candidate.get("gate_negative_controls") else 0
    return score


def render_return_curve(returns: list[float] | None, benchmark: list[float] | None = None) -> str:
    if not returns:
        return "<div class=\"muted\">Return curve not available (rerun rebuild to populate return series).</div>"
    equity = [1.0]
    for value in returns:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric != numeric:
            continue
        equity.append(equity[-1] * (1.0 + numeric))
    bench_equity = []
    if benchmark:
        bench_equity = [1.0]
        for value in benchmark:
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric != numeric:
                continue
            bench_equity.append(bench_equity[-1] * (1.0 + numeric))
    if len(equity) < 2:
        return "<div class=\"muted\">Return curve not available (insufficient numeric data).</div>"
    min_val = min(equity + (bench_equity or []))
    max_val = max(equity + (bench_equity or []))
    span = max(max_val - min_val, 1e-6)
    width = 560
    height = 160
    def _points(series: list[float]) -> str:
        if len(series) < 2:
            return ""
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
    eyebrow: str = "Cadence Report",
    back_href: str = "dashboard.html",
    back_label: str = "Back to cadence summary",
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


def top_candidates(candidates: list[dict], limit: int = 6) -> list[dict]:
    ranked = sorted(candidates, key=lambda item: item.get("rank", 9999))
    return ranked[:limit]


def build_profile_dashboard(
    *,
    summary: dict,
    holdout: dict[str, Any] | None = None,
    title: str,
    subtitle: str,
    back_href: str,
    back_label: str,
) -> str:
    return shared_build_profile_dashboard(
        summary=summary,
        holdout=holdout,
        title=title,
        subtitle=subtitle,
        back_href=back_href,
        back_label=back_label,
        phase1_href="phase1_summary.html",
        selection_href="selection_summary.html",
        holdout_href="walk_forward_results.html",
    )


def build_holdout_dashboard(
    *,
    holdout: dict,
    selection_summary: dict[str, Any] | None = None,
    title: str,
    subtitle: str,
    back_href: str,
    back_label: str,
) -> str:
    return shared_build_holdout_dashboard(
        holdout=holdout,
        selection_summary=selection_summary,
        title=title,
        subtitle=subtitle,
        back_href=back_href,
        back_label=back_label,
        phase1_href="phase1_summary.html",
        selection_href="selection_summary.html",
        holdout_href="walk_forward_results.html",
    )


def _attach_metadata(summary: dict, *, thesis: dict, cadence: dict) -> dict:
    summary = dict(summary)
    summary["thesis"] = thesis
    summary["cadence_id"] = cadence["cadence_id"]
    summary["cadence_label"] = cadence["cadence_label"]
    summary["canonical_offset_id"] = cadence["canonical_offset_id"]
    summary["schedule_type"] = cadence["schedule_type"]
    summary["rebalance_logic"] = cadence["rebalance_logic"]
    summary["rebalance_logic_label"] = cadence["rebalance_logic_label"]
    return summary


def build_summary_report(summary: dict) -> str:
    lines = [
        "# Rebalance Cadence Comparison",
        "",
        "- This is a separate research cycle.",
        "- This rerun uses the entry/exit-only cost model for comparison.",
        f"- Rebalance logic: `{REBALANCE_LOGIC_LABEL}` (`{REBALANCE_LOGIC}`)",
        "",
        "## Pair Summary",
        "",
    ]
    for pair in summary.get("pairs", []):
        thesis_label = pair["thesis"]["label"]
        cadence_label = pair["cadence"]["cadence_label"]
        certification = pair.get("certification", {})
        holdout = pair.get("holdout", {})
        pbo = certification.get("backtest_overfitting", {}).get("pbo")
        holdout_sharpe = holdout.get("phase4_gate", {}).get("base_main_net_sharpe")
        phase4 = holdout.get("phase4_gate", {}).get("phase4_eligible")
        lines.extend(
            [
                f"### {thesis_label} / {cadence_label}",
                f"- Certification status: `{certification.get('selection_status')}`",
                f"- Certification PBO: `{pbo}`",
                f"- Holdout Sharpe: `{holdout_sharpe}`",
                f"- Phase 4 eligible: `{phase4}`",
                "",
            ]
        )

    winner = summary.get("winner")
    lines.append("## Current Winner")
    lines.append("")
    if winner:
        lines.append(f"- Thesis: `{winner['thesis']['label']}`")
        lines.append(f"- Cadence: `{winner['cadence']['cadence_label']}`")
        holdout = winner.get("holdout", {})
        holdout_sharpe = holdout.get("phase4_gate", {}).get("base_main_net_sharpe")
        lines.append(f"- Holdout Sharpe: `{holdout_sharpe}`")
    else:
        lines.append("- No validated winner yet.")
    lines.append("")
    return "\n".join(lines)


def build_summary_dashboard(summary: dict) -> str:
    profile_set = summary.get("profile_set", "default")
    profile_settings = config.RESEARCH_PROFILE_SETS.get(profile_set, config.RESEARCH_PROFILE_SETTINGS)
    profile_desc = format_profile_set(
        profile_set,
        profile_settings,
        strategy_ids=summary.get("strategy_ids"),
    )
    summary_dir = Path(summary.get("results_root", "results/cadence_compare_rebuild")) / "summary"
    pairs = summary.get("pairs", [])
    phase1_green = bool(summary.get("phase1_green"))
    selected_count = 0
    phase4_count = 0
    winner = summary.get("winner")
    if winner:
        thesis_label = winner["thesis"]["label"]
        cadence_label = winner["cadence"]["cadence_label"]
        holdout_sharpe = winner.get("holdout", {}).get("phase4_gate", {}).get("base_main_net_sharpe")
        verdict = f"Current research winner: {thesis_label} / {cadence_label} with holdout Sharpe {holdout_sharpe}."
    else:
        verdict = "No cadence pair currently passes certification plus holdout."

    rows = []
    cadence_root = Path(summary["results_root"])
    for pair in pairs:
        thesis_label = pair["thesis"]["label"]
        cadence_label = pair["cadence"]["cadence_label"]
        cert = pair.get("certification", {})
        backtest = cert.get("backtest_overfitting", {})
        pbo = backtest.get("pbo")
        pbo_text = format_pbo_display(pbo, backtest)
        holdout_sharpe = pair.get("holdout", {}).get("phase4_gate", {}).get("base_main_net_sharpe")
        holdout_text = "n/a" if holdout_sharpe is None else f"{holdout_sharpe}"
        phase4 = pair.get("holdout", {}).get("phase4_gate", {}).get("phase4_eligible")
        if cert.get("selection_status") == "selected":
            selected_count += 1
        if phase4:
            phase4_count += 1
        output_dir = Path(pair.get("output_dir", cadence_root))
        overview_href = Path(os.path.relpath(output_dir / "dashboard.html", summary_dir)).as_posix()
        phase1_href = Path(os.path.relpath(output_dir / "phase1_summary.html", summary_dir)).as_posix()
        selection_href = Path(os.path.relpath(output_dir / "selection_summary.html", summary_dir)).as_posix()
        holdout_href = Path(os.path.relpath(output_dir / "walk_forward_results.html", summary_dir)).as_posix()
        rows.append(
            f"<tr><td>{thesis_label}</td><td>{cadence_label}</td><td>{cert.get('selection_status')}</td>"
            f"<td>{pbo_text}</td><td>{holdout_text}</td><td>{phase4}</td>"
            f"<td><a href=\"{overview_href}\">overview</a> &middot; <a href=\"{phase1_href}\">phase 1</a> &middot; <a href=\"{selection_href}\">phase 2</a> &middot; <a href=\"{holdout_href}\">phase 3</a></td></tr>"
        )

    table_rows = "".join(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Systematic Nordic Equity Research & Validation - Cadence Compare</title>
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
      <h1>Rebalance Cadence Comparison</h1>
      <p>This dashboard re-runs the cadence sweep with the entry/exit-only cost logic for comparison.</p>
      <p><strong>Rebalance logic:</strong> {REBALANCE_LOGIC_LABEL}</p>
      <p><strong>Profile set:</strong> {profile_desc}</p>
      <p><strong>{verdict}</strong></p>
      {render_timeframe_note(f"Phase 2 research boundary {phase2_selection_window_text()}; Phase 2 validation windows {phase2_validation_window_text()}; Phase 3 untouched holdout {holdout_window_text(None)}.")}
    </section>
    {render_phase_map_html(phase1_href="phase1_summary.html")}
    <section>
      <h2>Cycle Overview</h2>
      <div class="grid">
        <div class="card"><div class="label">Phase 1 Status</div><div class="value">{'green' if phase1_green else 'not green'}</div></div>
        <div class="card"><div class="label">Pairs</div><div class="value">{len(pairs)}</div></div>
        <div class="card"><div class="label">Phase 2 Selected</div><div class="value">{selected_count}</div></div>
        <div class="card"><div class="label">Phase 4 Eligible</div><div class="value">{phase4_count}</div></div>
      </div>
      <p class="muted" style="margin-top:12px;">Use the artifact links below to jump from this cycle summary into the detailed pair overview or the dedicated Phase 1, Phase 2, and Phase 3 pages for each thesis/cadence pair.</p>
    </section>
    <section>
      <h2>Comparison Table</h2>
      <table>
        <thead>
          <tr><th>Thesis</th><th>Cadence</th><th>Certification</th><th>PBO</th><th>Holdout Sharpe</th><th>Phase 4 Eligible</th><th>Artifacts</th></tr>
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


def build_pair_dashboard(pair: dict, cadence_root: Path, base_results_root: Path) -> str:
    thesis = pair.get("thesis", {})
    cadence = pair.get("cadence", {})
    certification = pair.get("certification", {})
    quick = pair.get("quick", {})
    mega = pair.get("mega", {})
    holdout = pair.get("holdout", {})
    phase4_gate = holdout.get("phase4_gate", {})
    profile_set = pair.get("profile_set", "default")
    profile_settings = config.RESEARCH_PROFILE_SETS.get(profile_set, config.RESEARCH_PROFILE_SETTINGS)
    profile_desc = format_profile_set(
        profile_set,
        profile_settings,
        strategy_ids=pair.get("strategy_ids"),
    )

    thesis_name = thesis.get("name") or "n/a"
    thesis_label = thesis.get("label") or "n/a"
    cadence_label = cadence.get("cadence_label") or "n/a"
    cadence_id = cadence.get("cadence_id") or "n/a"
    selection_status = certification.get("selection_status", "n/a")
    backtest = certification.get("backtest_overfitting", {})
    pbo = backtest.get("pbo")
    pbo_text = format_pbo_display(pbo, backtest)
    holdout_sharpe = phase4_gate.get("base_main_net_sharpe")
    holdout_text = f"{holdout_sharpe:.3f}" if isinstance(holdout_sharpe, (int, float)) else "n/a"
    locked = certification.get("locked_candidate") or {}
    locked_params = locked.get("params")

    ranked_candidates = certification.get("ranked_candidates") or []
    gate_counts = count_gate_passes(ranked_candidates)
    gate_fail_counts = count_gate_failures(ranked_candidates)
    hard_gate_winners = sum(
        1
        for candidate in ranked_candidates
        if candidate.get("gate_fold_count")
        and candidate.get("gate_bootstrap")
        and candidate.get("gate_deflated_sharpe")
        and candidate.get("gate_negative_controls")
    )
    top_ranked = top_candidates(ranked_candidates, limit=1)
    top_candidate = top_ranked[0] if top_ranked else {}
    top_table_rows = "\n".join(
        f"<tr><td>{item.get('candidate_id','n/a')}</td>"
        f"<td>{format_params(item.get('params'))}</td>"
        f"<td>{format_float(item.get('median_validation_sharpe'))}</td>"
        f"<td>{item.get('fold_pass_count','n/a')}</td>"
        f"<td>{'yes' if item.get('gate_bootstrap') else 'no'}</td>"
        f"<td>{'yes' if item.get('gate_deflated_sharpe') else 'no'}</td></tr>"
        for item in top_candidates(ranked_candidates, limit=5)
    )
    top_returns = top_candidate.get("concatenated_returns") or []
    top_primary_benchmark = top_candidate.get("primary_benchmark_returns") or []
    fold_sharpes = top_candidate.get("per_fold_sharpes") or {}
    fold_drawdowns = top_candidate.get("per_fold_drawdowns") or {}
    fold_rows = "\n".join(
        f"<tr><td>{fold_id}</td><td>{format_float(fold_sharpes.get(fold_id))}</td><td>{format_float(fold_drawdowns.get(fold_id))}</td></tr>"
        for fold_id in sorted(fold_sharpes.keys())
    )
    mcs_score = compute_mcs(top_candidate)
    profile_rows = []
    def _top_from(summary: dict) -> dict:
        ranked = summary.get("ranked_candidates") or []
        top_ranked = sorted(ranked, key=lambda item: item.get("rank", 9999))
        return top_ranked[0] if top_ranked else {}
    top_quick_candidate = _top_from(quick)
    top_mega_candidate = _top_from(mega)
    top_cert_candidate = _top_from(certification)
    for label, summary, top in (
        ("Quick", quick, top_quick_candidate),
        ("Mega", mega, top_mega_candidate),
        ("Certification", certification, top_cert_candidate),
    ):
        summary_backtest = summary.get("backtest_overfitting", {})
        pbo_value = summary_backtest.get("pbo")
        profile_rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{top.get('candidate_id','n/a')}</td>"
            f"<td>{format_params(top.get('params'))}</td>"
            f"<td>{format_float(top.get('median_validation_sharpe'))}</td>"
            f"<td>{format_float(top.get('total_return'))}</td>"
            f"<td>{format_float(top.get('max_drawdown'))}</td>"
            f"<td>{format_pbo_display(pbo_value, summary_backtest)}</td>"
            "</tr>"
        )
    profile_table = "\n".join(profile_rows)
    top_params_text = format_params(top_candidate.get("params"))
    top_failure_text = ", ".join(gate_failures(top_candidate)) if top_candidate else "n/a"
    neg_controls = certification.get("negative_controls", {})
    cross = neg_controls.get("cross_sectional_shuffle", {})
    block = neg_controls.get("block_shuffled_null", {})
    pbo_threshold = backtest.get("pbo_threshold_max")
    pbo_pass = backtest.get("passes_pbo_threshold")
    pbo_slice_count = backtest.get("slice_count")
    pbo_slice_length = backtest.get("slice_length_months")
    pbo_combo_count = backtest.get("combination_count")
    pbo_mean_percentile = backtest.get("mean_oos_rank_percentile")
    pbo_median_percentile = backtest.get("median_oos_rank_percentile")
    pbo_mean_logit = backtest.get("lambda_logit_mean")
    pbo_median_logit = backtest.get("lambda_logit_median")
    quick_backtest = quick.get("backtest_overfitting", {})
    mega_backtest = mega.get("backtest_overfitting", {})
    quick_pbo = quick_backtest.get("pbo")
    mega_pbo = mega_backtest.get("pbo")

    pair_dir = Path(pair.get("output_dir", cadence_root))
    summary_path = cadence_root / "summary" / "dashboard.html"
    summary_report_path = cadence_root / "summary" / "cadence_comparison_report.html"
    package_path = base_results_root.parent / "portfolio" / "alpha_momentum_validation_package" / "dashboard.html"
    dossier_path = base_results_root.parent / "portfolio" / "alpha_momentum_validation_package" / "research_audit_dossier.html"
    full_report_path = base_results_root.parent / "portfolio" / "alpha_momentum_validation_package" / "full_research_report.html"
    selection_path = pair_dir / "selection_summary.html"
    quick_path = pair_dir / "quick_summary.html"
    mega_path = pair_dir / "mega_summary.html"
    holdout_path = pair_dir / "walk_forward_results.html"

    summary_href = Path(os.path.relpath(summary_path, pair_dir)).as_posix()
    summary_report_href = Path(os.path.relpath(summary_report_path, pair_dir)).as_posix()
    package_href = Path(os.path.relpath(package_path, pair_dir)).as_posix()
    dossier_href = Path(os.path.relpath(dossier_path, pair_dir)).as_posix()
    full_report_href = Path(os.path.relpath(full_report_path, pair_dir)).as_posix()
    selection_href = Path(os.path.relpath(selection_path, pair_dir)).as_posix() if selection_path.exists() else None
    quick_href = Path(os.path.relpath(quick_path, pair_dir)).as_posix() if quick_path.exists() else None
    mega_href = Path(os.path.relpath(mega_path, pair_dir)).as_posix() if mega_path.exists() else None
    holdout_href = Path(os.path.relpath(holdout_path, pair_dir)).as_posix() if holdout_path.exists() else None

    selection_card = (
        f'<a class="card" href="{selection_href}"><div class="label">Selection Summary</div><div class="value">Open selection summary</div></a>'
        if selection_href
        else '<div class="card"><div class="label">Selection Summary</div><div class="value">not available</div></div>'
    )
    quick_card = (
        f'<a class="card" href="{quick_href}"><div class="label">Quick Profile</div><div class="value">Open quick diagnostics</div></a>'
        if quick_href
        else '<div class="card"><div class="label">Quick Profile</div><div class="value">not available</div></div>'
    )
    mega_card = (
        f'<a class="card" href="{mega_href}"><div class="label">Mega Profile</div><div class="value">Open mega diagnostics</div></a>'
        if mega_href
        else '<div class="card"><div class="label">Mega Profile</div><div class="value">not available</div></div>'
    )
    holdout_card = (
        f'<a class="card" href="{holdout_href}"><div class="label">Holdout</div><div class="value">Open holdout diagnostics</div></a>'
        if holdout_href
        else '<div class="card"><div class="label">Holdout</div><div class="value">not available</div></div>'
    )
    phase1_card = (
        '<a class="card" href="phase1_summary.html"><div class="label">Phase 1</div><div class="value">Open validation checks</div></a>'
    )
    validation_window_text = phase2_validation_window_text(
        (certification.get("walk_forward", {}) or {}).get("folds", [])
        if isinstance(certification.get("walk_forward", {}), dict)
        else []
    )
    holdout_window_text_value = holdout_window_text(holdout)
    phase3_ready = isinstance(holdout_sharpe, (int, float))
    phase3_callout = (
        f"Phase 3 summary diagnostics are available for the untouched holdout window {holdout_window_text_value}. "
        "If a historical rebuild snapshot did not preserve monthly return arrays, the holdout page will say so and use a clearly labeled replay only for path diagnostics."
        if phase3_ready
        else (
            f"Phase 3 is blocked for this pair. The reserved holdout window is still {holdout_window_text_value}, "
            "but Phase 2 did not lock a candidate, so the holdout page is a boundary/status page rather than a full equity and Monte Carlo report."
        )
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pair Dashboard</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:980px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin-top:16px; }}
    .card {{ display:block; text-decoration:none; color:inherit; background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:20px; padding:18px; }}
    .label {{ color:#6c5842; text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; }}
    .value {{ font-size:1.4rem; margin-top:6px; }}
    .muted {{ color:#5b6762; }}
    .callout {{ background:#fff7eb; border:1px solid rgba(176,107,29,.28); border-radius:16px; padding:12px 14px; margin-top:12px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
    th, td {{ border-bottom:1px solid rgba(23,33,26,.08); text-align:left; padding:8px; }}
    th {{ font-size:.75rem; text-transform:uppercase; color:#5d685f; letter-spacing:.05em; }}
    a {{ color:#b06b1d; }}
  </style>
</head>
<body>
  <main>
    <section>
      <p class="label">Phase 1-3 Overview</p>
      <h1>{thesis_label} - {cadence_label}</h1>
      <p>Thesis: <strong>{thesis_name}</strong> / Cadence: <strong>{cadence_id}</strong></p>
      <p class="muted">Profile set: {profile_desc}</p>
      {render_timeframe_note(f"Phase 2 research boundary {phase2_selection_window_text()}; Phase 2 validation windows {validation_window_text}; Phase 3 holdout {holdout_window_text_value}.")}
      <div class="grid">
        <div class="card">
          <div class="label">Selection Status</div>
          <div class="value">{selection_status}</div>
        </div>
        <div class="card">
          <div class="label">PBO</div>
          <div class="value">{pbo_text}</div>
        </div>
        <div class="card">
          <div class="label">Holdout Sharpe</div>
          <div class="value">{holdout_text}</div>
        </div>
      </div>
      <p class="muted" style="margin-top:12px;">Locked params: {format_params(locked_params)}. Top ranked params: {top_params_text}.</p>
      <div class="callout">
        <strong>PBO is one signal, not the verdict.</strong>
        <div class="muted">
          PBO passes: {pbo_pass}. Validation status is <strong>{selection_status}</strong> because hard-gate blockers remain: {top_failure_text or "see gate counts below"}. {pbo_explainer(backtest)}
        </div>
      </div>
      <div class="callout">
        <strong>Phase 3 status.</strong>
        <div class="muted">{phase3_callout}</div>
      </div>
    </section>
    {render_phase_map_html(phase1_href="phase1_summary.html", selection_href=selection_href, holdout_href=holdout_href, holdout=holdout)}
    <section>
      <h2>Artifact Map</h2>
      <p class="muted">These links jump between the shared Phase 1 page, the richer Phase 2 diagnostics, and the dedicated Phase 3 holdout page for this cadence pair.</p>
      <div class="grid">
        {phase1_card}
        {selection_card}
        {quick_card}
        {mega_card}
        {holdout_card}
      </div>
    </section>
    <section>
      <h2>Diagnostics Overview (Rebuild)</h2>
      {render_timeframe_note(f"These cards combine Phase 2 certification outputs over {validation_window_text} with the current Phase 3 status for {holdout_window_text_value}.")}
      <div class="grid">
        <div class="card">
          <div class="label">Certification PBO</div>
          <div class="value">{pbo_text}</div>
          <div class="muted">Threshold {format_pct(pbo_threshold)} · Passes: {pbo_pass}</div>
          <div class="muted">Slices {pbo_slice_count} · Slice length {format_float(pbo_slice_length, 1)} months · Combos {pbo_combo_count}</div>
          <div class="muted">OOS rank mean {format_float(pbo_mean_percentile, 3)} · median {format_float(pbo_median_percentile, 3)}</div>
        </div>
        <div class="card">
          <div class="label">Quick / Mega PBO</div>
          <div class="value">{format_pbo_display(quick_pbo, quick_backtest)} / {format_pbo_display(mega_pbo, mega_backtest)}</div>
          <div class="muted">Exploratory signals only.</div>
        </div>
        <div class="card">
          <div class="label">Negative Controls</div>
          <div class="value">{cross.get('pass_count','n/a')}/{cross.get('run_count','n/a')} · {block.get('pass_count','n/a')}/{block.get('run_count','n/a')}</div>
          <div class="muted">Cross-sectional / block-shuffled null.</div>
          <div class="muted">Threshold max pass-rate {format_pct(config.NEGATIVE_CONTROL_PASS_RATE_MAX)}.</div>
        </div>
        <div class="card">
          <div class="label">Gate Pass Counts</div>
          <div class="value">fold {gate_counts['gate_fold_count']} · boot {gate_counts['gate_bootstrap']} · defl {gate_counts['gate_deflated_sharpe']}</div>
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
          <div class="value">{top_failure_text or 'n/a'}</div>
          <div class="muted">Why the top candidate failed hard gates.</div>
        </div>
        <div class="card">
          <div class="label">Holdout Status</div>
          <div class="value">{holdout.get('status','n/a')}</div>
          <div class="muted">Phase 4 eligible: {phase4_gate.get('phase4_eligible','n/a')}</div>
        </div>
        <div class="card">
          <div class="label">PBO Logit Diagnostics</div>
          <div class="value">mean {format_float(pbo_mean_logit, 3)} · median {format_float(pbo_median_logit, 3)}</div>
          <div class="muted">Lower is healthier; >0 implies OOS rank below median.</div>
        </div>
        <div class="card">
          <div class="label">Model Confidence Score (Proxy)</div>
          <div class="value">{mcs_score if mcs_score is not None else 'n/a'} / 100</div>
          <div class="muted">Composite of the four certification gates (fold/boot/deflated/controls).</div>
        </div>
      </div>
    </section>
    <section>
      <h2>Return Curve (Top Candidate)</h2>
      {render_timeframe_note(f"This curve shows the stitched Phase 2 validation record over {validation_window_text}; it is not the dedicated Phase 3 holdout curve.")}
      {render_return_curve(top_returns, top_primary_benchmark)}
      <p class="muted" style="margin-top:10px;">Gold line = candidate; dashed = primary benchmark if available.</p>
    </section>
    <section>
      <h2>Return Curves (Top Candidates by Profile)</h2>
      {render_timeframe_note(f"Each small multiple below is a Phase 2 stitched validation curve over {validation_window_text}.")}
      <div class="grid">
        <div class="card">
          <div class="label">Quick</div>
          {render_return_curve(top_quick_candidate.get('concatenated_returns'), top_quick_candidate.get('primary_benchmark_returns'))}
        </div>
        <div class="card">
          <div class="label">Mega</div>
          {render_return_curve(top_mega_candidate.get('concatenated_returns'), top_mega_candidate.get('primary_benchmark_returns'))}
        </div>
        <div class="card">
          <div class="label">Certification</div>
          {render_return_curve(top_cert_candidate.get('concatenated_returns'), top_cert_candidate.get('primary_benchmark_returns'))}
        </div>
      </div>
      <p class="muted" style="margin-top:10px;">Each curve shows the top-ranked candidate for that profile.</p>
    </section>
    <section>
      <h2>Fold Diagnostics (Top Candidate)</h2>
      {render_timeframe_note(f"Fold metrics below come from the fixed Phase 2 validation windows inside {validation_window_text}.")}
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
      <h2>Profile Comparison (Top Candidate)</h2>
      {render_timeframe_note(f"Profile rows compare Phase 2 outputs only; the separate Phase 3 holdout lives on the holdout page for {holdout_window_text_value}.")}
      <table>
        <thead>
          <tr><th>Profile</th><th>Candidate</th><th>Params</th><th>Median Sharpe</th><th>Total Return</th><th>Max Drawdown</th><th>PBO</th></tr>
        </thead>
        <tbody>
          {profile_table}
        </tbody>
      </table>
      <p class="muted" style="margin-top:10px;">This shows how the top candidate changes across the quick, mega, and certification profiles.</p>
    </section>
    <section>
      <h2>Top Candidates (Certification)</h2>
      {render_timeframe_note(f"Certification candidates below are ranked only on the Phase 2 validation evidence over {validation_window_text}.")}
      <table>
        <thead>
          <tr><th>Candidate</th><th>Params</th><th>Median Sharpe</th><th>Fold Passes</th><th>Bootstrap</th><th>Deflated</th></tr>
        </thead>
        <tbody>
          {top_table_rows}
        </tbody>
      </table>
      <p class="muted" style="margin-top:10px;">This lets you compare nearby candidates and see which gates are stopping progress.</p>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    cadence_root = Path(args.results_root)
    cadence_root.mkdir(parents=True, exist_ok=True)
    summary_dir = cadence_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    base_results_root = cadence_root.parent
    pairs: list[dict] = []
    protocol_payload = apply_protocol_overrides(args.protocol_file)
    profile_settings = resolve_profile_settings(args.profile_set)
    phase1_success, phase1_sections = validate_phase1(
        input_dir=data_dir,
        require_cpp=False,
        require_benchmarks=True,
    )

    if args.render_only:
        summary_path = summary_dir / "cadence_comparison.json"
        if not summary_path.exists():
            raise SystemExit(f"Missing summary JSON at {summary_path}. Run a full cadence compare first.")
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["phase1_green"] = phase1_success
        for pair in summary.get("pairs", []):
            pair["profile_set"] = summary.get("profile_set", args.profile_set)
            output_dir = Path(pair.get("output_dir", cadence_root))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "phase1_summary.html").write_text(
                shared_build_phase1_dashboard(
                    success=phase1_success,
                    sections=phase1_sections,
                    title="Phase 1 Validation",
                    subtitle="Infrastructure, benchmark, FX, and point-in-time artifact checks for this research cycle.",
                    back_href="dashboard.html",
                    back_label="Back to pair dashboard",
                    input_dir=data_dir,
                    selection_href="selection_summary.html",
                    holdout_href="walk_forward_results.html",
                ),
                encoding="utf-8",
            )
            (output_dir / "selection_summary.html").write_text(
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
            (output_dir / "quick_summary.html").write_text(
                build_profile_dashboard(
                    summary=pair.get("quick", {}),
                    holdout=pair.get("holdout", {}),
                    title="Quick Profile Summary",
                    subtitle="Phase 2: exploratory small-grid diagnostics for this cadence pair.",
                    back_href="dashboard.html",
                    back_label="Back to pair dashboard",
                ),
                encoding="utf-8",
            )
            (output_dir / "mega_summary.html").write_text(
                build_profile_dashboard(
                    summary=pair.get("mega", {}),
                    holdout=pair.get("holdout", {}),
                    title="Mega Profile Summary",
                    subtitle="Phase 2: broader-grid diagnostics for this cadence pair.",
                    back_href="dashboard.html",
                    back_label="Back to pair dashboard",
                ),
                encoding="utf-8",
            )
            (output_dir / "walk_forward_results.html").write_text(
                build_holdout_dashboard(
                    holdout=pair.get("holdout", {}),
                    selection_summary=pair.get("certification", {}),
                    title="Holdout Diagnostics",
                    subtitle="Phase 3: untouched holdout evaluation for the locked certification candidate (if any).",
                    back_href="dashboard.html",
                    back_label="Back to pair dashboard",
                ),
                encoding="utf-8",
            )
            (output_dir / "dashboard.html").write_text(
                build_pair_dashboard(pair, cadence_root, base_results_root),
                encoding="utf-8",
            )
        report_text = build_summary_report(summary)
        (summary_dir / "cadence_comparison_report.html").write_text(
            build_markdown_dashboard_html(
                title="Cadence Comparison Report",
                subtitle="Readable summary of the rebuild cadence sweep.",
                markdown_text=report_text,
                back_href="dashboard.html",
                back_label="Back to cadence summary",
            ),
            encoding="utf-8",
        )
        (summary_dir / "phase1_summary.html").write_text(
            shared_build_phase1_dashboard(
                success=phase1_success,
                sections=phase1_sections,
                title="Phase 1 Validation",
                subtitle="Infrastructure, benchmark, FX, and point-in-time artifact checks for this research cycle.",
                back_href="dashboard.html",
                back_label="Back to cadence summary",
                input_dir=data_dir,
                selection_href=None,
                holdout_href=None,
            ),
            encoding="utf-8",
        )
        (summary_dir / "dashboard.html").write_text(build_summary_dashboard(summary), encoding="utf-8")
        md_report = summary_dir / "cadence_comparison_report.md"
        if md_report.exists():
            md_report.unlink()
        return 0

    strategy_variants = select_strategy_variants(
        strategy_ids=args.strategy_ids,
        strategy_filter=args.strategy_filter,
    )
    strategy_ids = [item.get("strategy_id", "baseline") for item in strategy_variants]

    for thesis_name in args.theses:
        thesis = build_thesis(thesis_name).manifest_metadata()
        for cadence_id in args.cadences:
            spec = load_cadence_spec(cadence_id)
            dataset = CadenceDataset(data_dir, cadence_id=cadence_id, offset_id=spec.canonical_offset_id)
            period_label = cadence_period_label(spec.periods_per_year)

            quick = run_profile(
                dataset,
                thesis=thesis,
                profile_name="quick",
                profile_settings=profile_settings,
                period_label=period_label,
                periods_per_year=spec.periods_per_year,
                strategy_variants=strategy_variants,
            )
            mega = run_profile(
                dataset,
                thesis=thesis,
                profile_name="mega",
                profile_settings=profile_settings,
                period_label=period_label,
                periods_per_year=spec.periods_per_year,
                strategy_variants=strategy_variants,
            )
            certification = run_profile(
                dataset,
                thesis=thesis,
                profile_name="certification_baseline",
                profile_settings=profile_settings,
                period_label=period_label,
                periods_per_year=spec.periods_per_year,
                strategy_variants=strategy_variants,
            )

            cadence_meta = {
                "cadence_id": spec.cadence_id,
                "cadence_label": spec.cadence_label,
                "canonical_offset_id": spec.canonical_offset_id,
                "schedule_type": spec.schedule_type,
                "rebalance_logic": REBALANCE_LOGIC,
                "rebalance_logic_label": REBALANCE_LOGIC_LABEL,
            }

            quick = _attach_metadata(quick, thesis=thesis, cadence=cadence_meta)
            mega = _attach_metadata(mega, thesis=thesis, cadence=cadence_meta)
            certification = _attach_metadata(certification, thesis=thesis, cadence=cadence_meta)

            locked = certification.get("locked_candidate")
            if locked is not None:
                holdout = evaluate_holdout(
                    dataset,
                    thesis=thesis,
                    params=locked["params"],
                    period_label=period_label,
                    periods_per_year=spec.periods_per_year,
                    start_month=config.OOS_START,
                    end_month=config.OOS_END,
                )
                holdout.update(cadence_meta)
                holdout["selection_mode"] = "certification_baseline"
                holdout["thesis"] = thesis
            else:
                holdout = {
                    "status": "blocked_by_missing_selection",
                    **cadence_meta,
                    "thesis": thesis,
                }

            output_dir = cadence_root / thesis_name / cadence_id
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "phase1_summary.html").write_text(
                shared_build_phase1_dashboard(
                    success=phase1_success,
                    sections=phase1_sections,
                    title="Phase 1 Validation",
                    subtitle="Infrastructure, benchmark, FX, and point-in-time artifact checks for this research cycle.",
                    back_href="dashboard.html",
                    back_label="Back to pair dashboard",
                    input_dir=data_dir,
                    selection_href="selection_summary.html",
                    holdout_href="walk_forward_results.html",
                ),
                encoding="utf-8",
            )
            _serialize_json(output_dir / "selection_summary.json", certification)
            _serialize_json(output_dir / "quick_summary.json", quick)
            _serialize_json(output_dir / "mega_summary.json", mega)
            _serialize_json(output_dir / "walk_forward_results.json", holdout)
            (output_dir / "selection_summary.html").write_text(
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
            (output_dir / "quick_summary.html").write_text(
                build_profile_dashboard(
                    summary=quick,
                    holdout=holdout,
                    title="Quick Profile Summary",
                    subtitle="Phase 2: exploratory small-grid diagnostics for this cadence pair.",
                    back_href="dashboard.html",
                    back_label="Back to pair dashboard",
                ),
                encoding="utf-8",
            )
            (output_dir / "mega_summary.html").write_text(
                build_profile_dashboard(
                    summary=mega,
                    holdout=holdout,
                    title="Mega Profile Summary",
                    subtitle="Phase 2: broader-grid diagnostics for this cadence pair.",
                    back_href="dashboard.html",
                    back_label="Back to pair dashboard",
                ),
                encoding="utf-8",
            )
            (output_dir / "walk_forward_results.html").write_text(
                build_holdout_dashboard(
                    holdout=holdout,
                    selection_summary=certification,
                    title="Holdout Diagnostics",
                    subtitle="Phase 3: untouched holdout evaluation for the locked certification candidate (if any).",
                    back_href="dashboard.html",
                    back_label="Back to pair dashboard",
                ),
                encoding="utf-8",
            )

            pair = {
                "thesis": thesis,
                "cadence": cadence_meta,
                "quick": quick,
                "mega": mega,
                "certification": certification,
                "holdout": holdout,
                "output_dir": str(output_dir),
                "profile_set": args.profile_set,
                "strategy_ids": strategy_ids,
            }
            pairs.append(pair)

            dashboard_html = build_pair_dashboard(pair, cadence_root, base_results_root)
            (output_dir / "dashboard.html").write_text(dashboard_html, encoding="utf-8")

    winner = None
    best_sharpe = float("-inf")
    for pair in pairs:
        certification = pair.get("certification", {})
        holdout = pair.get("holdout", {})
        if certification.get("selection_status") != "selected":
            continue
        phase4 = holdout.get("phase4_gate", {})
        if not phase4.get("phase4_eligible"):
            continue
        sharpe = phase4.get("base_main_net_sharpe")
        if sharpe is not None and sharpe > best_sharpe:
            best_sharpe = sharpe
            winner = pair

    summary = {
        "authoritative_status": "validated_winner_found" if winner else "no_validated_winner",
        "authoritative_validation_model": "legacy_entry_exit_costs",
        "phase1_green": phase1_success,
        "pairs": pairs,
        "winner": winner,
        "rebalance_logic": REBALANCE_LOGIC,
        "rebalance_logic_label": REBALANCE_LOGIC_LABEL,
        "research_cycle": "rebalance_cadence_compare",
        "profile_set": args.profile_set,
        "results_root": str(cadence_root),
        "strategy_ids": strategy_ids,
        "strategy_filter": args.strategy_filter,
        "strategy_id_overrides": args.strategy_ids or [],
        "protocol_file": args.protocol_file,
        "protocol_payload": protocol_payload or {},
        "oos_start": config.OOS_START,
        "oos_end": config.OOS_END,
        "insample_end": config.INSAMPLE_END,
    }

    frozen_manifest = base_results_root / "forward_monitor" / "frozen_strategy_manifest.json"
    if frozen_manifest.exists():
        summary["preserved_frozen_manifest"] = str(frozen_manifest)

    _serialize_json(summary_dir / "cadence_comparison.json", summary)
    report_text = build_summary_report(summary)
    (summary_dir / "cadence_comparison_report.html").write_text(
        build_markdown_dashboard_html(
            title="Cadence Comparison Report",
            subtitle="Readable summary of the rebuild cadence sweep.",
            markdown_text=report_text,
            back_href="dashboard.html",
            back_label="Back to cadence summary",
        ),
        encoding="utf-8",
    )
    (summary_dir / "phase1_summary.html").write_text(
        shared_build_phase1_dashboard(
            success=phase1_success,
            sections=phase1_sections,
            title="Phase 1 Validation",
            subtitle="Infrastructure, benchmark, FX, and point-in-time artifact checks for this research cycle.",
            back_href="dashboard.html",
            back_label="Back to cadence summary",
            input_dir=data_dir,
            selection_href=None,
            holdout_href=None,
        ),
        encoding="utf-8",
    )
    (summary_dir / "dashboard.html").write_text(build_summary_dashboard(summary), encoding="utf-8")
    md_report = summary_dir / "cadence_comparison_report.md"
    if md_report.exists():
        md_report.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
