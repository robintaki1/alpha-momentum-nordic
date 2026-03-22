from __future__ import annotations

import argparse
from pathlib import Path

from phase1_lib import Phase1Error, RIKSBANK_SERIES, download_riksbank_fx_artifact, resolve_end_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Phase 1 Riksbank FX artifacts.")
    parser.add_argument("--start", required=True, help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default="auto", help="Inclusive end date in YYYY-MM-DD format or 'auto'.")
    parser.add_argument("--base-currency", default="SEK")
    parser.add_argument("--pairs", default=",".join(RIKSBANK_SERIES))
    parser.add_argument("--out", default="data/riksbank_fx_daily.parquet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.base_currency != "SEK":
        raise Phase1Error("Phase 1 FX download only supports --base-currency SEK.")
    requested_pairs = {value.strip().upper() for value in args.pairs.split(",") if value.strip()}
    if requested_pairs != set(RIKSBANK_SERIES):
        raise Phase1Error(f"Phase 1 FX download requires --pairs {','.join(RIKSBANK_SERIES)}.")

    start = resolve_end_date(args.start)
    end = resolve_end_date(args.end)
    if start > end:
        raise Phase1Error("--start must be on or before --end.")

    frame = download_riksbank_fx_artifact(start=start, end=end, out_path=Path(args.out))
    print(f"riksbank_fx_daily.parquet: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
