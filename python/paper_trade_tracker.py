from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from phase1_lib import enrich_with_fx, read_parquet


@dataclass
class PriceRow:
    price_sek: float
    execution_date: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a paper-trading portfolio from the current forward-monitor picks."
    )
    parser.add_argument("--capital-sek", type=float, default=50_000.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--state-path", type=Path, default=Path("papertrading/portfolio_state.json"))
    parser.add_argument("--history-path", type=Path, default=Path("papertrading/portfolio_history.csv"))
    parser.add_argument("--ledger-path", type=Path, default=Path("papertrading/trade_ledger.csv"))
    parser.add_argument(
        "--role",
        type=str,
        default="lead",
        help="Role in forward_monitor_picks.csv to paper trade (lead or shadow).",
    )
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="Override execution date (YYYY-MM-DD). Defaults to the next_execution_date in the picks.",
    )
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Process historical holding months in forward_monitor_picks.csv before the current month.",
    )
    parser.add_argument(
        "--start-holding-month",
        type=str,
        default=None,
        help="Optional earliest holding_month (YYYY-MM) to process when backfilling.",
    )
    return parser.parse_args()


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_history_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "trade_date,holding_month,portfolio_value_before_sek,portfolio_value_after_sek,period_return,"
        "total_cost_sek,cash_sek,notes\n",
        encoding="utf-8",
    )


def ensure_ledger_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "trade_date,holding_month,security_id,action,trade_value_sek,price_sek,shares_delta,"
        "cost_bps,cost_sek,execution_date\n",
        encoding="utf-8",
    )


def load_prices(
    *,
    data_dir: Path,
    security_ids: list[str],
    trade_date: pd.Timestamp,
) -> dict[str, PriceRow]:
    raw_prices = read_parquet(data_dir / "prices_raw_daily.parquet", "prices_raw_daily")[
        ["security_id", "date", "open_raw", "currency"]
    ].copy()
    raw_prices["date"] = pd.to_datetime(raw_prices["date"])
    raw_prices = raw_prices[raw_prices["security_id"].isin(security_ids)]
    raw_prices = raw_prices.sort_values(["security_id", "date"])

    adjusted = read_parquet(data_dir / "prices_adjusted_daily.parquet", "prices_adjusted_daily")[
        ["security_id", "date", "adj_factor"]
    ].copy()
    adjusted["date"] = pd.to_datetime(adjusted["date"])
    adjusted_index = adjusted.set_index(["security_id", "date"])

    fx_frame = read_parquet(data_dir / "riksbank_fx_daily.parquet", "riksbank_fx_daily")

    price_map: dict[str, PriceRow] = {}
    for security_id in security_ids:
        sec_rows = raw_prices.loc[raw_prices["security_id"] == security_id]
        if sec_rows.empty:
            raise ValueError(f"Missing price history for {security_id}.")
        dates = sec_rows["date"].to_numpy()
        idx = int(np.searchsorted(dates, np.datetime64(trade_date), side="left"))
        if idx >= len(sec_rows):
            raise ValueError(f"No trade-date pricing found for {security_id} at or after {trade_date.date()}.")
        row = sec_rows.iloc[idx]
        exec_date = pd.Timestamp(row["date"])
        try:
            adj_factor = float(adjusted_index.loc[(security_id, exec_date), "adj_factor"])
        except KeyError as exc:
            raise ValueError(f"Missing adjusted factor for {security_id} on {exec_date.date()}.") from exc
        fx_row = enrich_with_fx(
            pd.DataFrame(
                {
                    "security_id": [security_id],
                    "date": [exec_date],
                    "currency": [row["currency"]],
                    "open_raw": [row["open_raw"]],
                }
            ),
            fx_frame,
            date_col="date",
            currency_col="currency",
        ).iloc[0]
        price_sek = float(fx_row["open_raw"]) * adj_factor * float(fx_row["sek_per_ccy"])
        price_map[security_id] = PriceRow(price_sek=price_sek, execution_date=exec_date)

    return price_map


def portfolio_value(holdings: dict[str, Any], prices: dict[str, PriceRow], cash_sek: float) -> float:
    total = float(cash_sek)
    for security_id, holding in holdings.items():
        if security_id not in prices:
            continue
        total += float(holding["shares"]) * prices[security_id].price_sek
    return float(total)


def main() -> int:
    args = parse_args()

    picks_path = args.results_root / "forward_monitor" / "forward_monitor_picks.csv"
    if not picks_path.exists():
        raise FileNotFoundError(f"Missing picks file: {picks_path}")

    all_picks = pd.read_csv(picks_path)
    role = (args.role or "lead").strip().lower()
    role_picks = all_picks[all_picks["role"].fillna("").str.lower() == role].copy()
    if role_picks.empty:
        raise ValueError(f"No {role} picks found in forward_monitor_picks.csv.")

    state = load_state(args.state_path)
    if state is None:
        state = {
            "capital_sek": float(args.capital_sek),
            "last_trade_date": None,
            "last_holding_month": None,
            "last_portfolio_value_sek": float(args.capital_sek),
            "cash_sek": float(args.capital_sek),
            "holdings": {},
        }

    holdings: dict[str, Any] = {k: dict(v) for k, v in state.get("holdings", {}).items()}
    cash_sek = float(state.get("cash_sek", args.capital_sek))
    last_value = float(state.get("last_portfolio_value_sek", args.capital_sek))

    def process_month(picks: pd.DataFrame, execution_date: pd.Timestamp, holding_month: str) -> None:
        nonlocal holdings, cash_sek, last_value

        last_trade_date = state.get("last_trade_date")
        last_holding_month = state.get("last_holding_month")
        if last_trade_date == str(execution_date.date()) and last_holding_month == holding_month:
            print(f"Paper-trade already recorded for {holding_month} (trade date {execution_date.date()}).")
            return

        picks = picks.copy()
        picks["target_weight"] = picks["target_weight"].fillna(1.0 / float(len(picks)))
        picks["one_way_cost_bps"] = picks["one_way_cost_bps"].fillna(0.0)

        current_ids = picks["security_id"].tolist()
        all_ids = sorted(set(current_ids) | set(holdings.keys()))
        prices = load_prices(data_dir=args.data_dir, security_ids=all_ids, trade_date=execution_date)

        value_before = portfolio_value(holdings, prices, cash_sek)
        period_return = (value_before / last_value - 1.0) if last_value else 0.0

        target_total = value_before
        target_weights = {row.security_id: float(row.target_weight) for row in picks.itertuples(index=False)}
        cost_bps_map = {row.security_id: float(row.one_way_cost_bps) for row in picks.itertuples(index=False)}

        def simulate_rebalance(
            *,
            holdings_in: dict[str, Any],
            cash_in: float,
            target_total_value: float,
        ) -> tuple[dict[str, Any], float, list[dict[str, Any]], float]:
            holdings_local: dict[str, Any] = {k: dict(v) for k, v in holdings_in.items()}
            cash_local = float(cash_in)
            trades_local: list[dict[str, Any]] = []
            total_cost_local = 0.0

            for security_id in all_ids:
                current_value = 0.0
                if security_id in holdings_local:
                    current_value = float(holdings_local[security_id]["shares"]) * prices[security_id].price_sek

                target_value = 0.0
                if security_id in target_weights:
                    target_value = target_total_value * target_weights[security_id]

                trade_value = target_value - current_value
                if abs(trade_value) < 1.0:
                    continue

                price_sek = prices[security_id].price_sek
                shares_delta = trade_value / price_sek
                action = "buy" if trade_value > 0 else "sell"

                cost_bps = cost_bps_map.get(security_id)
                if cost_bps is None and security_id in holdings_local:
                    cost_bps = float(holdings_local[security_id].get("cost_bps", 0.0))
                cost_bps = float(cost_bps or 0.0)
                cost_sek = abs(trade_value) * cost_bps / 10_000.0

                total_cost_local += cost_sek
                cash_local += -trade_value - cost_sek

                new_shares = float(holdings_local.get(security_id, {}).get("shares", 0.0)) + shares_delta
                if abs(new_shares) < 1e-6:
                    holdings_local.pop(security_id, None)
                else:
                    holdings_local[security_id] = {
                        "shares": new_shares,
                        "entry_price_sek": holdings_local.get(security_id, {}).get("entry_price_sek", price_sek),
                        "entry_date": holdings_local.get(
                            security_id, {}
                        ).get("entry_date", str(prices[security_id].execution_date.date())),
                        "cost_bps": cost_bps,
                    }

                trades_local.append(
                    {
                        "trade_date": str(execution_date.date()),
                        "holding_month": holding_month,
                        "security_id": security_id,
                        "action": action,
                        "trade_value_sek": trade_value,
                        "price_sek": price_sek,
                        "shares_delta": shares_delta,
                        "cost_bps": cost_bps,
                        "cost_sek": cost_sek,
                        "execution_date": str(prices[security_id].execution_date.date()),
                    }
                )

            return holdings_local, cash_local, trades_local, total_cost_local

        holdings_after, cash_after, trades, total_cost_sek = simulate_rebalance(
            holdings_in=holdings,
            cash_in=cash_sek,
            target_total_value=target_total,
        )
        if cash_after < 0.0 and total_cost_sek > 0.0:
            adjusted_target = max(0.0, target_total - total_cost_sek)
            holdings_after, cash_after, trades, total_cost_sek = simulate_rebalance(
                holdings_in=holdings,
                cash_in=cash_sek,
                target_total_value=adjusted_target,
            )

        value_after = portfolio_value(holdings_after, prices, cash_after)

        ensure_history_header(args.history_path)
        with args.history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{execution_date.date()},{holding_month},{value_before:.2f},{value_after:.2f},"
                f"{period_return:.6f},{total_cost_sek:.2f},{cash_after:.2f},\n"
            )

        ensure_ledger_header(args.ledger_path)
        with args.ledger_path.open("a", encoding="utf-8") as handle:
            for trade in trades:
                handle.write(
                    f"{trade['trade_date']},{trade['holding_month']},{trade['security_id']},{trade['action']},"
                    f"{trade['trade_value_sek']:.2f},{trade['price_sek']:.4f},{trade['shares_delta']:.6f},"
                    f"{trade['cost_bps']:.2f},{trade['cost_sek']:.2f},{trade['execution_date']}\n"
                )

        holdings = holdings_after
        cash_sek = cash_after
        last_value = value_after

        state.update(
            {
                "last_trade_date": str(execution_date.date()),
                "last_holding_month": holding_month,
                "last_portfolio_value_sek": float(value_after),
                "cash_sek": float(cash_after),
                "holdings": holdings_after,
            }
        )
        write_state(args.state_path, state)

        print(f"Paper-trade update complete for {holding_month} (trade date {execution_date.date()}).")
        print(f"Portfolio value before: {value_before:.2f} SEK")
        print(f"Portfolio value after:  {value_after:.2f} SEK")
        print(f"Total costs this rebalance: {total_cost_sek:.2f} SEK")

    processed_months: set[str] = set()

    if args.backfill_history:
        history_picks = role_picks[role_picks["record_type"] == "history"].copy()
        if args.start_holding_month:
            history_picks = history_picks[history_picks["holding_month"] >= args.start_holding_month]
        for holding_month, group in history_picks.groupby("holding_month"):
            exec_dates = pd.to_datetime(group["next_execution_date"].dropna().unique())
            if len(exec_dates) != 1:
                raise ValueError(f"Expected single next_execution_date for {holding_month}.")
            process_month(group, exec_dates[0], str(holding_month))
            processed_months.add(str(holding_month))

    current_picks = role_picks[role_picks["record_type"] == "current"].copy()
    if current_picks.empty:
        return 0
    holding_month = str(current_picks["holding_month"].iloc[0])
    if holding_month not in processed_months:
        exec_dates = pd.to_datetime(current_picks["next_execution_date"].dropna().unique())
        if len(exec_dates) != 1:
            raise ValueError("Expected a single next_execution_date for current picks.")
        execution_date = pd.Timestamp(args.trade_date) if args.trade_date else exec_dates[0]
        process_month(current_picks, execution_date, holding_month)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
