from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import run_cadence_compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cadence compare with specific MA windows.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results/cadence_compare_ma_window_promo"))
    parser.add_argument("--theses", nargs="+", default=list(config.RESEARCH_THESIS_SETTINGS))
    parser.add_argument("--cadences", nargs="+", default=list(config.DEFAULT_CADENCE_COMPARE_CADENCES))
    parser.add_argument("--ma-windows", nargs="+", type=int, default=[7, 10])
    parser.add_argument("--lookback", type=int, default=config.L)
    parser.add_argument("--skip", type=int, default=config.SKIP)
    parser.add_argument("--top-ns", nargs="+", type=int, default=list(config.CERTIFICATION_BASELINE_TOP_N_GRID))
    return parser.parse_args()


def build_strategy_variant(ma_window: int) -> dict[str, int | str | bool]:
    return {
        "strategy_id": f"trend_filter_ma{int(ma_window)}",
        "rebalance": "full",
        "weighting": "equal",
        "trend_filter": True,
        "ma_window": int(ma_window),
    }


def build_profile_settings(
    *,
    lookback: int,
    skip: int,
    top_ns: Sequence[int],
    profiles: Sequence[str],
) -> dict[str, dict]:
    settings: dict[str, dict] = {}
    for profile in profiles:
        base = config.RESEARCH_PROFILE_SETTINGS.get(profile)
        if base is None:
            raise ValueError(f"Unknown profile '{profile}'. Available: {', '.join(config.RESEARCH_PROFILE_SETTINGS)}")
        settings[profile] = {
            **base,
            "lookbacks": (int(lookback),),
            "skips": (int(skip),),
            "top_ns": tuple(int(value) for value in top_ns),
        }
    return settings


def main() -> int:
    args = parse_args()
    profile_set_name = "ma_window_promo"
    profiles = ["quick", "mega", "certification_baseline"]

    original_variants = list(config.STRATEGY_VARIANTS)
    original_profile_sets = dict(config.RESEARCH_PROFILE_SETS)
    try:
        config.STRATEGY_VARIANTS = [build_strategy_variant(value) for value in args.ma_windows]
        config.RESEARCH_PROFILE_SETS[profile_set_name] = build_profile_settings(
            lookback=args.lookback,
            skip=args.skip,
            top_ns=args.top_ns,
            profiles=profiles,
        )

        argv = [
            "run_cadence_compare.py",
            "--data-dir",
            str(args.data_dir),
            "--results-root",
            str(args.results_root),
            "--profile-set",
            profile_set_name,
        ]
        if args.theses:
            argv.extend(["--theses", *args.theses])
        if args.cadences:
            argv.extend(["--cadences", *args.cadences])

        sys.argv = argv
        return run_cadence_compare.main()
    finally:
        config.STRATEGY_VARIANTS = original_variants
        config.RESEARCH_PROFILE_SETS = original_profile_sets
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
