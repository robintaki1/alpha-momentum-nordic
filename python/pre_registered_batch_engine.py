from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from research_engine import run_research_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pre-registered batch of hypotheses.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results/pre_registered_batch"))
    parser.add_argument("--theses", nargs="+", default=list(config.RESEARCH_THESIS_SETTINGS))
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["quick", "mega", "certification_baseline"],
        help="Profiles to run (default: quick mega certification_baseline).",
    )
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    return parser.parse_args()


def hypotheses() -> list[dict[str, object]]:
    return [
        {
            "id": "baseline_full",
            "label": "Baseline (no trend filter), full rebalance",
            "strategy_variants": [
                {"strategy_id": "baseline", "rebalance": "full", "weighting": "equal"},
            ],
        },
        {
            "id": "trend_filter_ma12",
            "label": "Trend filter MA12, full rebalance",
            "strategy_variants": [
                {
                    "strategy_id": "trend_filter_ma12",
                    "rebalance": "full",
                    "weighting": "equal",
                    "trend_filter": True,
                    "ma_window": 12,
                }
            ],
        },
        {
            "id": "banded2",
            "label": "Banded rebalance (buffer 2), no trend filter",
            "strategy_variants": [
                {"strategy_id": "banded2", "rebalance": "banded", "band_buffer": 2, "weighting": "equal"},
            ],
        },
        {
            "id": "minhold3",
            "label": "Min-hold 3 months, no trend filter",
            "strategy_variants": [
                {"strategy_id": "minhold3", "rebalance": "min_hold", "min_hold_months": 3, "weighting": "equal"},
            ],
        },
    ]


def register_profile_set(*, name: str, profiles: list[str], top_n_grid: tuple[int, ...]) -> None:
    profile_settings: dict[str, dict[str, int]] = {}
    for profile in profiles:
        base = config.RESEARCH_PROFILE_SETTINGS.get(profile)
        if base is None:
            raise ValueError(f"Unknown profile '{profile}'. Available: {', '.join(config.RESEARCH_PROFILE_SETTINGS)}")
        profile_settings[profile] = {
            **base,
            "lookbacks": config.PRE_REGISTERED_LOOKBACK_GRID,
            "skips": config.PRE_REGISTERED_SKIP_GRID,
            "top_ns": top_n_grid,
        }
    config.RESEARCH_PROFILE_SETS[name] = profile_settings


def write_manifest(
    *,
    output_root: Path,
    top_n_grid: tuple[int, ...],
    thesis_names: list[str],
    profiles: list[str],
    hypothesis_list: list[dict[str, object]],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_n_grid": list(top_n_grid),
        "lookbacks": list(config.PRE_REGISTERED_LOOKBACK_GRID),
        "skips": list(config.PRE_REGISTERED_SKIP_GRID),
        "theses": list(thesis_names),
        "profiles": list(profiles),
        "pbo_policy": {
            "good_max": config.PBO_THRESHOLD_MAX,
            "hard_cutoff_min": config.PBO_HARD_CUTOFF,
        },
        "holdout_policy": {
            "sharpe_min": config.OOS_SHARPE_MIN,
            "must_beat_primary_benchmark": True,
        },
        "decision_rule": (
            "Eligible candidates must avoid PBO hard cutoff and pass holdout "
            "(Sharpe >= OOS_SHARPE_MIN and beat primary benchmark). "
            "Among eligible candidates, choose the one with the highest holdout Sharpe; "
            "if tied, choose the higher holdout total return."
        ),
        "hypotheses": hypothesis_list,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def summarize_runs(
    *,
    output_root: Path,
    hypothesis_list: list[dict[str, object]],
    theses: list[str],
) -> None:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypotheses": [],
    }
    for hypothesis in hypothesis_list:
        hypothesis_id = hypothesis["id"]
        hypothesis_dir = output_root / hypothesis_id
        summary_path = hypothesis_dir / "summary" / "research_engine_summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        thesis_rows = payload.get("theses", [])
        rows = []
        for row in thesis_rows:
            thesis_meta = row.get("thesis", {})
            if thesis_meta.get("name") not in theses:
                continue
            certification = row.get("certification", {})
            holdout = row.get("holdout", {})
            phase4 = holdout.get("phase4_gate", {})
            rows.append(
                {
                    "thesis": thesis_meta.get("name"),
                    "selection_status": certification.get("selection_status"),
                    "pbo": certification.get("backtest_overfitting", {}).get("pbo"),
                    "pbo_band": certification.get("pbo_band"),
                    "locked_candidate": (certification.get("locked_candidate") or {}).get("candidate_id"),
                    "holdout_status": holdout.get("status"),
                    "holdout_sharpe": phase4.get("base_main_net_sharpe"),
                    "beats_primary_benchmark": phase4.get("beats_primary_benchmark"),
                    "phase4_eligible": phase4.get("phase4_eligible"),
                }
            )
        summary["hypotheses"].append(
            {
                "id": hypothesis_id,
                "label": hypothesis.get("label"),
                "output_dir": str(hypothesis_dir),
                "results": rows,
            }
        )
    (output_root / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    thesis_names = list(args.theses)
    profiles = list(args.profiles)
    top_n_grid = (8, 10, 12)
    hypothesis_list = hypotheses()

    write_manifest(
        output_root=output_root,
        top_n_grid=top_n_grid,
        thesis_names=thesis_names,
        profiles=profiles,
        hypothesis_list=hypothesis_list,
    )

    original_variants = list(config.STRATEGY_VARIANTS)
    try:
        for hypothesis in hypothesis_list:
            hypothesis_id = hypothesis["id"]
            profile_set_name = f"pre_registered_batch_{hypothesis_id}"
            register_profile_set(name=profile_set_name, profiles=profiles, top_n_grid=top_n_grid)
            config.STRATEGY_VARIANTS = list(hypothesis["strategy_variants"])

            output_dir = output_root / hypothesis_id
            run_research_engine(
                data_dir=args.data_dir,
                output_dir=output_dir,
                thesis_names=thesis_names,
                profiles=profiles,
                profile_set=profile_set_name,
                skip_holdout=args.skip_holdout,
                skip_walk_forward=args.skip_walk_forward,
                skip_monte_carlo=args.skip_monte_carlo,
                skip_sensitivity=args.skip_sensitivity,
                walk_forward_profiles=None,
                render_only=False,
            )
    finally:
        config.STRATEGY_VARIANTS = original_variants

    summarize_runs(output_root=output_root, hypothesis_list=hypothesis_list, theses=thesis_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
