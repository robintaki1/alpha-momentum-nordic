from __future__ import annotations

import argparse
from pathlib import Path

from phase1_lib import format_validation_report, parse_bool, validate_phase1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 1 artifacts.")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--require-cpp", default="false")
    parser.add_argument("--require-benchmarks", default="true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    success, sections = validate_phase1(
        input_dir=Path(args.input_dir),
        require_cpp=parse_bool(args.require_cpp),
        require_benchmarks=parse_bool(args.require_benchmarks),
    )
    print(format_validation_report(sections))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
