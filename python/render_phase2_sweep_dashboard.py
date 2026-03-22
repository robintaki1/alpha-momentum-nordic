from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from research_engine import (
    ResearchDataset,
    apply_pbo_policy,
    build_thesis,
    build_thesis_dashboard,
    config,
    monte_carlo_summary,
    parameter_sensitivity,
    resolve_profile_settings,
    walk_forward_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a second full thesis dashboard from an existing Phase 2 sweep run."
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


def needs_monte_carlo(summary: dict[str, Any]) -> bool:
    monte = summary.get("monte_carlo", {})
    if not isinstance(monte, dict):
        return True
    if monte.get("status") in {"skipped", "unavailable"}:
        return True
    metrics = monte.get("metrics")
    return not isinstance(metrics, dict) or not metrics


def needs_parameter_sensitivity(summary: dict[str, Any]) -> bool:
    sensitivity = summary.get("parameter_sensitivity", {})
    if not isinstance(sensitivity, dict):
        return True
    if sensitivity.get("status") in {"skipped", "unavailable"}:
        return True
    return sensitivity.get("status") != "ok"


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir
    output_path = args.output

    selection = load_json(source_run_dir / "selection_summary.json")
    if not selection:
        raise ValueError(f"Missing selection_summary.json in {source_run_dir}")

    holdout = load_json(source_run_dir / "holdout_results.json", default={"status": "missing"})
    quick = load_json(source_run_dir / "quick_summary.json", default={})
    mega = load_json(source_run_dir / "mega_summary.json", default={})

    thesis_name = args.thesis_name or source_run_dir.name
    thesis_meta = build_thesis(thesis_name).manifest_metadata()
    profile_set = infer_profile_set(source_run_dir, args.profile_set)
    profile_settings = resolve_profile_settings(profile_set)
    periods_per_year = int(selection.get("periods_per_year", 12))
    locked = selection.get("locked_candidate") or {}
    locked_params = locked.get("params")
    if "pbo_band" not in selection or selection.get("pbo_band") in (None, "", "n/a"):
        apply_pbo_policy(selection)
    dataset: ResearchDataset | None = None
    if needs_walk_forward(selection):
        if locked_params:
            dataset = ResearchDataset(Path("data"))
            selection["walk_forward"] = walk_forward_test(
                dataset,
                thesis=thesis_meta,
                params_grid=[locked_params],
                period_label=selection.get("period_label", "months"),
                periods_per_year=periods_per_year,
            )
    if needs_monte_carlo(selection):
        monte_candidate = locked or ((selection.get("ranked_candidates") or [{}])[0] if selection.get("ranked_candidates") else {})
        monte_returns = monte_candidate.get("concatenated_returns")
        if monte_returns:
            selection["monte_carlo"] = monte_carlo_summary(
                monte_returns,
                periods_per_year=periods_per_year,
                n_resamples=config.MONTE_CARLO_RESAMPLES,
                block_length_months=config.MONTE_CARLO_BLOCK_LENGTH_MONTHS,
                seed=config.MONTE_CARLO_SEED,
            )
            selection["monte_carlo"]["candidate_id"] = monte_candidate.get("candidate_id")
            selection["monte_carlo"]["params"] = monte_candidate.get("params")
            selection["monte_carlo"]["source"] = "locked_candidate" if locked else "top_candidate"
    if needs_parameter_sensitivity(selection):
        ranked_candidates = selection.get("ranked_candidates") or []
        if ranked_candidates:
            selection["parameter_sensitivity"] = parameter_sensitivity(ranked_candidates)

    selection_href = rel_href(output_path, source_run_dir / "selection_summary.html")
    holdout_href = rel_href(output_path, source_run_dir / "holdout_results.html")
    context_note = args.context_note or (
        "This is the full Phase 2 sweep dashboard for the exact "
        "l12_s1_n15_strat=trend_filter candidate. The walk-forward charts below are the "
        "Phase 2 out-of-sample validation folds by design, with validation windows from "
        "2008-01 to 2017-12. The untouched 2018-01 to 2026-01 holdout is shown separately "
        "in the validation funnel. Because this older sweep JSON omitted some rich "
        "diagnostics, this page hydrates the missing walk-forward, Monte Carlo, and "
        "sensitivity sections from the locked candidate and sweep rankings."
    )

    rendered = build_thesis_dashboard(
        thesis=thesis_meta,
        quick=quick,
        mega=mega,
        certification=selection,
        holdout=holdout,
        profile_set=profile_set,
        profile_settings=profile_settings,
        context_note=context_note,
        selection_href=selection_href,
        holdout_href=holdout_href,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Rendered Phase 2 sweep dashboard to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
