from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_protocol import evaluate_holdout_candidate, load_json, load_locked_candidate, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the untouched holdout using only the locked Phase 2 parameter tuple."
    )
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--holdout-panel", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results"), type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    locked_candidate = load_locked_candidate(args.selection)
    manifest = load_json(args.holdout_panel)
    candidates = manifest.get("candidates", [])
    target = next(
        (
            candidate
            for candidate in candidates
            if candidate.get("params") == locked_candidate.get("params")
        ),
        None,
    )
    if target is None:
        raise ValueError("Holdout panel does not contain the locked parameter tuple from selection_summary.json.")

    results = evaluate_holdout_candidate(target)
    write_json(args.output_dir / "walk_forward_results.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
