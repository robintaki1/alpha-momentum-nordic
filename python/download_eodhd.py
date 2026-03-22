from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from phase1_lib import Phase1Error, download_eodhd_artifacts, parse_bool, resolve_end_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Phase 1 EODHD artifacts.")
    parser.add_argument("--start", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default="auto", help="Inclusive end date in YYYY-MM-DD format or 'auto'.")
    parser.add_argument("--universe-mode", default=config.UNIVERSE_MODE)
    parser.add_argument("--include-delisted", default=str(config.INCLUDE_DELISTED).lower())
    parser.add_argument("--main-market-allowlist", default=config.MAIN_MARKET_ALLOWLIST_PATH)
    parser.add_argument("--out-dir", default="data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.universe_mode != "full_nordics_main_markets":
        raise Phase1Error("Phase 1 currently supports only --universe-mode full_nordics_main_markets.")

    start = resolve_end_date(args.start)
    end = resolve_end_date(args.end)
    if start > end:
        raise Phase1Error("--start must be on or before --end.")

    artifacts = download_eodhd_artifacts(
        start=start,
        end=end,
        include_delisted=parse_bool(args.include_delisted),
        out_dir=Path(args.out_dir),
        main_market_allowlist_path=Path(args.main_market_allowlist) if args.main_market_allowlist else None,
        progress_callback=lambda message: print(message, flush=True),
    )
    for artifact_name, frame in artifacts.items():
        print(f"{artifact_name}.parquet: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
