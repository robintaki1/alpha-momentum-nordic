from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward_monitor import build_forward_monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frozen n8 lead / n15 shadow forward-trial monitor and optionally update the tandem paper-trading books."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/forward_trials/ex_norway_n8_n15"),
        help="Destination for the forward-trial dashboard, picks CSV, and manifest.",
    )
    parser.add_argument(
        "--papertrading-dir",
        type=Path,
        default=Path("papertrading/ex_norway_n8_n15"),
        help="Dedicated paper-trading folder for the tandem lead/shadow books.",
    )
    parser.add_argument("--capital-sek", type=float, default=50_000.0)
    parser.add_argument("--history-months", type=int, default=6)
    parser.add_argument("--trade-date", type=str, default=None)
    parser.add_argument("--backfill-history", action="store_true")
    parser.add_argument("--start-holding-month", type=str, default=None)
    parser.add_argument("--run-paper-trades", action="store_true")
    parser.add_argument("--lead-selection-summary", type=Path, default=None)
    parser.add_argument("--shadow-selection-summary", type=Path, default=None)
    return parser.parse_args()


def resolve_latest_selection_summary(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern), key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No selection summaries matched pattern: {pattern}")
    return matches[0]


def resolve_default_lead_selection() -> Path:
    return resolve_latest_selection_summary(
        "results/current_engine_replay_ex_norway_l12_s1_n8_trend_filter_*/ex_norway/selection_summary.json"
    )


def resolve_default_shadow_selection() -> Path:
    return resolve_latest_selection_summary(
        "results/current_engine_replay_ex_norway_l12_s1_n15_trend_filter_*/ex_norway/selection_summary.json"
    )


def main() -> int:
    args = parse_args()
    lead_selection = args.lead_selection_summary or resolve_default_lead_selection()
    shadow_selection = args.shadow_selection_summary or resolve_default_shadow_selection()

    payload = build_forward_monitor(
        data_dir=args.data_dir,
        results_root=args.results_root,
        output_dir=args.output_dir,
        theses=[],
        history_months=int(args.history_months),
        selection_summaries=[lead_selection, shadow_selection],
        export_package=False,
    )
    print(
        f"Frozen forward trial ready for {payload['monitoring_window']['current_pick_month']}. "
        f"Open {args.output_dir / 'dashboard.html'}",
        flush=True,
    )

    if not args.run_paper_trades:
        return 0

    cmd = [
        sys.executable,
        str(ROOT / "python" / "live_paper_engine.py"),
        "--capital-sek",
        str(float(args.capital_sek)),
        "--data-dir",
        str(args.data_dir),
        "--results-root",
        str(args.results_root),
        "--monitor-dir",
        str(args.output_dir),
        "--papertrading-dir",
        str(args.papertrading_dir),
        "--roles",
        "lead",
        "shadow",
    ]
    if args.trade_date:
        cmd.extend(["--trade-date", args.trade_date])
    if args.backfill_history:
        cmd.append("--backfill-history")
    if args.start_holding_month:
        cmd.extend(["--start-holding-month", args.start_holding_month])
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
