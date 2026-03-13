from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_protocol import candidate_aggregates, csv_rows, load_json, selection_summary, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a locked Phase 2 parameter tuple using in-sample rolling-origin validation only."
    )
    parser.add_argument("--mode", choices=("quick", "mega"), required=True)
    parser.add_argument("--candidate-panel", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results"), type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "l",
        "skip",
        "top_n",
        "fold_id",
        "universe_variant",
        "execution_model",
        "fx_scenario",
        "cost_model_name",
        "validation_sharpe",
        "validation_max_drawdown",
        "candidate_fold_pass_count",
        "candidate_median_validation_sharpe",
        "candidate_deflated_sharpe_score",
        "candidate_deflated_sharpe_probability",
        "candidate_bootstrap_ci_low",
        "candidate_bootstrap_ci_high",
        "candidate_universe_sensitivity_std",
        "candidate_plateau_neighbor_median_sharpe",
        "candidate_plateau_neighbor_ratio",
        "gate_fold_count",
        "gate_deflated_sharpe",
        "gate_bootstrap",
        "gate_negative_controls",
        "rank",
        "selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    manifest = load_json(args.candidate_panel)
    manifest_mode = manifest.get("mode")
    if manifest_mode and manifest_mode != args.mode:
        raise ValueError(f"Candidate panel mode '{manifest_mode}' does not match CLI mode '{args.mode}'.")

    aggregates, negative_controls = candidate_aggregates(manifest)
    write_csv(args.output_dir / f"grid_results_{args.mode}.csv", csv_rows(aggregates))
    write_json(
        args.output_dir / "selection_summary.json",
        selection_summary(aggregates, negative_controls, args.mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
