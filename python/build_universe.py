from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from phase1_lib import Phase1Error, build_phase1_universe, parse_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 1 point-in-time universe artifacts.")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--universe-mode", default=config.UNIVERSE_MODE)
    parser.add_argument("--emit-se-only", default=str(config.RUN_SE_ONLY_DIAGNOSTIC).lower())
    parser.add_argument("--emit-liquid-subset", default=str(config.RUN_LIQUID_SUBSET_DIAGNOSTIC).lower())
    parser.add_argument("--out", default="data/universe_pti.parquet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.universe_mode != "full_nordics_main_markets":
        raise Phase1Error("Phase 1 currently supports only --universe-mode full_nordics_main_markets.")

    outputs = build_phase1_universe(
        input_dir=Path(args.input_dir),
        emit_se_only=parse_bool(args.emit_se_only),
        emit_liquid_subset=parse_bool(args.emit_liquid_subset),
        out_path=Path(args.out),
    )
    for artifact_name, frame in outputs.items():
        print(f"{artifact_name}.parquet: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
