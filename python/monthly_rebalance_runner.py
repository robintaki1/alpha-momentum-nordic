from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forward_monitor import build_forward_monitor, resolve_default_theses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the monthly forward monitor and apply paper-trade rebalances only when a new holding month is available."
        )
    )
    parser.add_argument("--capital-sek", type=float, default=50_000.0, help="Capital per role.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--monitor-dir", type=Path, default=Path("results/forward_monitor"))
    parser.add_argument("--papertrading-dir", type=Path, default=Path("papertrading"))
    parser.add_argument("--history-months", type=int, default=6)
    parser.add_argument("--roles", nargs="+", default=["lead", "shadow"])
    parser.add_argument("--trade-date", type=str, default=None)
    parser.add_argument("--backfill-history", action="store_true")
    parser.add_argument("--start-holding-month", type=str, default=None)
    parser.add_argument(
        "--force-paper-trades",
        action="store_true",
        help="Run the paper-trade command even when all requested roles already match the current holding month.",
    )
    return parser.parse_args()


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def state_holding_month(papertrading_dir: Path, role: str) -> str | None:
    state_path = papertrading_dir / f"{role.strip().lower()}_portfolio_state.json"
    state = load_state(state_path)
    if not state:
        return None
    value = state.get("last_holding_month")
    return str(value) if value else None


def build_monitor(args: argparse.Namespace) -> dict[str, Any]:
    theses = resolve_default_theses(args.results_root)
    return build_forward_monitor(
        data_dir=args.data_dir,
        results_root=args.results_root,
        output_dir=args.monitor_dir,
        theses=theses,
        history_months=int(args.history_months),
        export_package=True,
    )


def should_run_paper_trades(args: argparse.Namespace, current_holding_month: str | None) -> tuple[bool, list[str]]:
    if args.backfill_history or args.force_paper_trades:
        return True, []
    if not current_holding_month:
        return False, list(args.roles)

    stale_roles: list[str] = []
    for role in args.roles:
        last_holding_month = state_holding_month(args.papertrading_dir, role)
        if last_holding_month != current_holding_month:
            stale_roles.append(role)
    return bool(stale_roles), stale_roles


def run_paper_trades(args: argparse.Namespace) -> None:
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
        str(args.monitor_dir),
        "--papertrading-dir",
        str(args.papertrading_dir),
        "--roles",
        *[str(role).strip().lower() for role in args.roles],
    ]
    if args.trade_date:
        cmd.extend(["--trade-date", args.trade_date])
    if args.backfill_history:
        cmd.append("--backfill-history")
    if args.start_holding_month:
        cmd.extend(["--start-holding-month", args.start_holding_month])
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    payload = build_monitor(args)
    monitoring = payload["monitoring_window"]
    current_holding_month = monitoring.get("current_pick_month")
    latest_available = monitoring.get("latest_available_holding_month")
    latest_completed = monitoring.get("latest_completed_holding_month")

    print(
        "Monitor rebuilt: "
        f"latest_available={latest_available}, latest_completed={latest_completed}, current_pick={current_holding_month}",
        flush=True,
    )

    should_run, stale_roles = should_run_paper_trades(args, current_holding_month)
    if not should_run:
        print(
            "Paper-trade step skipped: requested roles already match the current holding month "
            f"({current_holding_month or 'n/a'}).",
            flush=True,
        )
        return 0

    if stale_roles:
        print(
            "New month detected for roles: " + ", ".join(stale_roles),
            flush=True,
        )
    else:
        print("Running paper-trade step by explicit override.", flush=True)

    run_paper_trades(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
