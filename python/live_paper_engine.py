from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-trading for multiple forward-monitor roles.")
    parser.add_argument("--capital-sek", type=float, default=50_000.0, help="Capital per role.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--papertrading-dir", type=Path, default=Path("papertrading"))
    parser.add_argument("--roles", nargs="+", default=["lead", "shadow"])
    parser.add_argument("--trade-date", type=str, default=None)
    parser.add_argument("--backfill-history", action="store_true")
    parser.add_argument("--start-holding-month", type=str, default=None)
    return parser.parse_args()


def run_role(*, role: str, args: argparse.Namespace) -> None:
    role_tag = role.strip().lower()
    paper_dir = args.papertrading_dir
    paper_dir.mkdir(parents=True, exist_ok=True)
    state_path = paper_dir / f"{role_tag}_portfolio_state.json"
    history_path = paper_dir / f"{role_tag}_portfolio_history.csv"
    ledger_path = paper_dir / f"{role_tag}_trade_ledger.csv"

    cmd = [
        sys.executable,
        str(ROOT / "python" / "paper_trade_tracker.py"),
        "--role",
        role_tag,
        "--capital-sek",
        str(float(args.capital_sek)),
        "--data-dir",
        str(args.data_dir),
        "--results-root",
        str(args.results_root),
        "--state-path",
        str(state_path),
        "--history-path",
        str(history_path),
        "--ledger-path",
        str(ledger_path),
    ]
    if args.trade_date:
        cmd.extend(["--trade-date", args.trade_date])
    if args.backfill_history:
        cmd.append("--backfill-history")
    if args.start_holding_month:
        cmd.extend(["--start-holding-month", args.start_holding_month])

    subprocess.run(cmd, check=True)
    print(f"Live paper-trade updated for role={role_tag}.")


def main() -> int:
    args = parse_args()
    for role in args.roles:
        run_role(role=role, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
