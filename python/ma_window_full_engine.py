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
    parser = argparse.ArgumentParser(description="Run full research engine for specific MA windows.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results/ma_window_full_engine"))
    parser.add_argument("--thesis", default=None)
    parser.add_argument("--ma-windows", nargs="+", type=int, default=[10, 12])
    parser.add_argument("--lookback", type=int, default=config.L)
    parser.add_argument("--skip", type=int, default=config.SKIP)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["quick", "mega", "certification_baseline"],
        help="Profiles to run (default: quick mega certification_baseline).",
    )
    parser.add_argument("--skip-holdout", action="store_true")
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


def build_profile_settings(
    *,
    lookback: int,
    skip: int,
    top_n: int,
    profiles: list[str],
) -> dict[str, dict[str, int]]:
    settings: dict[str, dict[str, int]] = {}
    for profile in profiles:
        base = config.RESEARCH_PROFILE_SETTINGS.get(profile)
        if base is None:
            raise ValueError(f"Unknown profile '{profile}'. Available: {', '.join(config.RESEARCH_PROFILE_SETTINGS)}")
        settings[profile] = {
            **base,
            "lookbacks": (int(lookback),),
            "skips": (int(skip),),
            "top_ns": (int(top_n),),
        }
    return settings


def build_strategy_variant(ma_window: int) -> dict[str, int | str | bool]:
    return {
        "strategy_id": f"trend_filter_ma{int(ma_window)}",
        "rebalance": "full",
        "weighting": "equal",
        "trend_filter": True,
        "ma_window": int(ma_window),
    }


def main() -> int:
    args = parse_args()
    thesis_name = args.thesis or resolve_default_thesis()
    profiles = list(args.profiles)
    profile_settings = build_profile_settings(
        lookback=args.lookback,
        skip=args.skip,
        top_n=args.top_n,
        profiles=profiles,
    )

    profile_set_name = "ma_window_full_engine"
    config.RESEARCH_PROFILE_SETS[profile_set_name] = profile_settings

    original_variants = list(config.STRATEGY_VARIANTS)
    try:
        for ma_window in sorted({int(value) for value in args.ma_windows if int(value) > 0}):
            config.STRATEGY_VARIANTS = [build_strategy_variant(ma_window)]
            output_dir = args.output_root / f"ma{ma_window}"
            output_dir.mkdir(parents=True, exist_ok=True)

            run_research_engine(
                data_dir=args.data_dir,
                output_dir=output_dir,
                thesis_names=[thesis_name],
                profiles=profiles,
                profile_set=profile_set_name,
                skip_holdout=args.skip_holdout,
                skip_walk_forward=False,
                skip_monte_carlo=False,
                skip_sensitivity=False,
                walk_forward_profiles=None,
                render_only=False,
            )

            meta = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "thesis": thesis_name,
                "ma_window": ma_window,
                "lookback": args.lookback,
                "skip": args.skip,
                "top_n": args.top_n,
                "profiles": profiles,
            }
            (output_dir / "summary" / "ma_window_meta.json").write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )
            print(f"Completed full engine run for MA={ma_window}. Output: {output_dir}")
    finally:
        config.STRATEGY_VARIANTS = original_variants
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
