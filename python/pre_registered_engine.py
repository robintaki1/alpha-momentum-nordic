from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from research_engine import run_research_engine

_CANDIDATE_ID_PATTERN = re.compile(
    r"^l(?P<l>\d+)_s(?P<skip>\d+)_n(?P<top_n>\d+)_strat=(?P<strategy_id>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pre-registered (low-DoF) research engine.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pre_registered_engine"))
    parser.add_argument("--thesis", default=None)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["quick", "mega", "certification_baseline"],
        help="Profiles to run (default: quick mega certification_baseline).",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Freeze an exact candidate tuple, e.g. l12_s1_n15_strat=trend_filter.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="If set, pre-register a single Top-N value (overrides the grid unless --candidate-id is used).",
    )
    parser.add_argument(
        "--top-n-grid",
        nargs="+",
        type=int,
        default=None,
        help="If set, pre-register an explicit Top-N grid (overrides --top-n and default grid unless --candidate-id is used).",
    )
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    return parser.parse_args()


def parse_candidate_id(candidate_id: str) -> dict[str, Any]:
    match = _CANDIDATE_ID_PATTERN.fullmatch(candidate_id.strip())
    if match is None:
        raise ValueError(
            "Candidate id must match the format l<lookback>_s<skip>_n<top_n>_strat=<strategy_id>."
        )
    return {
        "l": int(match.group("l")),
        "skip": int(match.group("skip")),
        "top_n": int(match.group("top_n")),
        "strategy_id": match.group("strategy_id"),
    }


def resolve_strategy_variant(strategy_id: str | None) -> dict[str, Any]:
    if strategy_id is None:
        # Default single strategy family with a bear-market safety net.
        return {
            "strategy_id": "trend_filter_ma12",
            "rebalance": "full",
            "weighting": "equal",
            "trend_filter": True,
            "ma_window": 12,
        }
    for item in config.STRATEGY_VARIANTS:
        if item.get("strategy_id") == strategy_id:
            return dict(item)
    raise ValueError(f"Unknown strategy_id '{strategy_id}'.")


def pre_registered_strategies(strategy_id: str | None = None) -> list[dict[str, Any]]:
    # Single strategy family with a bear-market safety net, aligned to the 12-month lookback.
    return [resolve_strategy_variant(strategy_id)]


def build_profile_settings(
    *,
    lookback: int | None,
    skip: int | None,
    top_n: int | None,
    top_n_grid: list[int] | None,
    profiles: list[str],
) -> str:
    use_default_grid = lookback is None and skip is None and top_n is None and not top_n_grid
    if top_n_grid:
        grid = sorted({int(value) for value in top_n_grid if int(value) > 0})
        if not grid:
            grid = list(config.PRE_REGISTERED_TOP_N_GRID)
        top_ns = tuple(grid)
    elif top_n is not None:
        top_ns = (int(top_n),)
    else:
        top_ns = config.PRE_REGISTERED_TOP_N_GRID
    lookbacks = (int(lookback),) if lookback is not None else config.PRE_REGISTERED_LOOKBACK_GRID
    skips = (int(skip),) if skip is not None else config.PRE_REGISTERED_SKIP_GRID
    if use_default_grid:
        return "pre_registered"
    profile_set_name = (
        "pre_registered_"
        f"l{'_'.join(str(value) for value in lookbacks)}_"
        f"s{'_'.join(str(value) for value in skips)}_"
        f"n{'_'.join(str(value) for value in top_ns)}"
    )
    profile_settings: dict[str, dict[str, int]] = {}
    for profile in profiles:
        base = config.RESEARCH_PROFILE_SETTINGS.get(profile)
        if base is None:
            raise ValueError(f"Unknown profile '{profile}'. Available: {', '.join(config.RESEARCH_PROFILE_SETTINGS)}")
        profile_settings[profile] = {
            **base,
            "lookbacks": lookbacks,
            "skips": skips,
            "top_ns": top_ns,
        }
    config.RESEARCH_PROFILE_SETS[profile_set_name] = profile_settings
    return profile_set_name


def main() -> int:
    args = parse_args()
    thesis_names = [args.thesis] if args.thesis else list(config.RESEARCH_THESIS_SETTINGS)
    profiles = list(args.profiles)
    candidate = parse_candidate_id(args.candidate_id) if args.candidate_id else None
    profile_set = build_profile_settings(
        lookback=candidate["l"] if candidate else None,
        skip=candidate["skip"] if candidate else None,
        top_n=candidate["top_n"] if candidate else args.top_n,
        top_n_grid=None if candidate else args.top_n_grid,
        profiles=profiles,
    )

    original_variants = list(config.STRATEGY_VARIANTS)
    try:
        config.STRATEGY_VARIANTS = pre_registered_strategies(
            candidate["strategy_id"] if candidate else None
        )
        run_research_engine(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            thesis_names=thesis_names,
            profiles=profiles,
            profile_set=profile_set,
            skip_holdout=args.skip_holdout,
            skip_walk_forward=args.skip_walk_forward,
            skip_monte_carlo=args.skip_monte_carlo,
            skip_sensitivity=args.skip_sensitivity,
            walk_forward_profiles=None,
            render_only=False,
        )
    finally:
        config.STRATEGY_VARIANTS = original_variants
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
