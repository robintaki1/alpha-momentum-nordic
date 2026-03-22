from __future__ import annotations

import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import strategy_variants
from phase1_lib import enrich_with_fx, read_parquet

PRIMARY_TRACK = {
    "universe_variant": "Full Nordics",
    "execution_model": "next_open",
    "fx_scenario": "base",
    "cost_model_name": config.PRIMARY_SELECTION_COST_MODEL,
}

EXECUTION_MODELS = ("next_open", "next_close")
FX_SCENARIOS = ("low", "base", "high")
CASHOUT_ACTION_TYPES = {"cash_merger", "delisting_cashout"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def serialize_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json.loads(json.dumps(payload, default=_json_default))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def annualized_sharpe(returns: Sequence[float], *, periods_per_year: int = 12) -> float:
    if len(returns) < 2:
        return 0.0
    std = statistics.stdev(returns)
    if std == 0:
        return 0.0
    return statistics.fmean(returns) / std * math.sqrt(float(periods_per_year))


def max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = 1.0 - (equity / peak if peak else 1.0)
        max_dd = max(max_dd, drawdown)
    return max_dd


def total_return(returns: Sequence[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def month_labels_between(start: str, end: str) -> list[str]:
    months: list[str] = []
    current = pd.Period(start, freq="M")
    stop = pd.Period(end, freq="M")
    while current <= stop:
        months.append(str(current))
        current += 1
    return months


def thesis_settings(thesis_name: str | None) -> dict[str, Any]:
    resolved_name = thesis_name or config.DEFAULT_RESEARCH_THESIS
    thesis = config.RESEARCH_THESIS_SETTINGS[resolved_name]
    return {
        "name": resolved_name,
        "label": str(thesis["label"]),
        "excluded_countries": tuple(str(value) for value in thesis.get("excluded_countries", ())),
        "scope_note": str(thesis["scope_note"]),
    }


@dataclass(frozen=True)
class ResearchThesis:
    name: str
    label: str
    excluded_countries: tuple[str, ...]
    scope_note: str

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "excluded_countries": list(self.excluded_countries),
            "scope_note": self.scope_note,
        }


@dataclass
class WindowSimulation:
    months: list[str]
    monthly_returns: list[float]
    primary_benchmark_returns: list[float] | None
    secondary_benchmark_returns: list[float] | None
    tertiary_benchmark_returns: list[float] | None
    details: list[dict[str, Any]] | None = None


def build_thesis(thesis_name: str | None = None) -> ResearchThesis:
    settings = thesis_settings(thesis_name)
    return ResearchThesis(
        name=settings["name"],
        label=settings["label"],
        excluded_countries=tuple(settings["excluded_countries"]),
        scope_note=settings["scope_note"],
    )


class ResearchDataset:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.universe = read_parquet(data_dir / "universe_pti.parquet", "universe_pti").copy()
        self.raw_prices = read_parquet(data_dir / "prices_raw_daily.parquet", "prices_raw_daily")
        self.adjusted_prices = read_parquet(data_dir / "prices_adjusted_daily.parquet", "prices_adjusted_daily")
        self.benchmark_prices = read_parquet(data_dir / "benchmark_prices.parquet", "benchmark_prices")
        self.fx_frame = read_parquet(data_dir / "riksbank_fx_daily.parquet", "riksbank_fx_daily")
        self.corporate_actions = read_parquet(data_dir / "corporate_actions.parquet", "corporate_actions")

        self.signal_months = sorted(self.universe["rebalance_month"].unique())
        self.security_ids = sorted(self.universe["security_id"].unique())
        self.security_index = {security_id: index for index, security_id in enumerate(self.security_ids)}
        self.month_index = {month: index for index, month in enumerate(self.signal_months)}
        self.holding_months = self._build_holding_months()

        self.security_country = self._security_metadata("country_code")
        self.security_currency = self._security_metadata("currency")
        self.security_is_non_sek = self.security_currency != "SEK"

        panel = self._build_monthly_panel()
        self.anchor_adj_close_local = panel["anchor_adj_close_local"]
        self.base_variant_masks = {
            "Full Nordics": panel["is_eligible_full_nordics"],
            "SE-only": panel["is_eligible_se_only"],
        }
        self.asof_matches_anchor = panel["asof_matches_anchor"]
        self.close_raw_sek = panel["close_raw_sek"]
        self.median_daily_value_60d_sek = panel["median_daily_value_60d_sek"]
        self.market_cap_sek = panel["market_cap_sek"]
        self.monthly_return_matrix = self._build_monthly_return_matrix()
        self.entry_available = {
            "next_open": panel["entry_available_next_open"],
            "next_close": panel["entry_available_next_close"],
        }
        self.holding_returns = {
            "next_open": panel["gross_return_next_open"],
            "next_close": panel["gross_return_next_close"],
        }

        self.scores = self._precompute_scores()
        self.rank_orders = self._precompute_rank_orders()
        self.capacity_masks = self._precompute_capacity_masks()
        self.cost_fractions = self._precompute_cost_fractions()
        self.flat_cost_fractions = self._precompute_flat_cost_fractions()
        self.benchmark_monthly_returns = self._build_benchmark_monthly_returns()
        self.benchmark_monthly_prices = self._build_benchmark_monthly_prices()

    def _security_metadata(self, column: str) -> np.ndarray:
        rows = (
            self.universe[["security_id", column]]
            .drop_duplicates(subset=["security_id"])
            .set_index("security_id")
            .reindex(self.security_ids)
        )
        return rows[column].to_numpy()

    def _pivot(self, frame: pd.DataFrame, value_column: str) -> np.ndarray:
        pivoted = (
            frame.pivot(index="rebalance_month", columns="security_id", values=value_column)
            .reindex(index=self.signal_months, columns=self.security_ids)
        )
        return pivoted.to_numpy()

    def _build_holding_months(self) -> list[str | None]:
        month_frame = (
            self.universe[["rebalance_month", "next_execution_date"]]
            .drop_duplicates(subset=["rebalance_month"])
            .sort_values("rebalance_month")
        )
        month_frame["holding_month"] = month_frame["next_execution_date"].dt.to_period("M").astype(str)
        return month_frame.set_index("rebalance_month").reindex(self.signal_months)["holding_month"].tolist()

    def _build_monthly_panel(self) -> dict[str, np.ndarray]:
        monthly = self.universe.copy()
        monthly["asof_matches_anchor"] = monthly["asof_trade_date"] == monthly["anchor_trade_date"]
        monthly = monthly.merge(
            self.adjusted_prices[["security_id", "date", "adj_close"]].rename(
                columns={"date": "anchor_trade_date", "adj_close": "anchor_adj_close_local"}
            ),
            on=["security_id", "anchor_trade_date"],
            how="left",
        )

        calendar = (
            monthly[["rebalance_month", "exchange_group", "next_execution_date"]]
            .drop_duplicates()
            .sort_values(["exchange_group", "rebalance_month"])
        )
        calendar["scheduled_exit_date"] = calendar.groupby("exchange_group")["next_execution_date"].shift(-1)
        monthly = monthly.merge(
            calendar[["rebalance_month", "exchange_group", "scheduled_exit_date"]],
            on=["rebalance_month", "exchange_group"],
            how="left",
        )

        daily = enrich_with_fx(
            self.raw_prices[["security_id", "date", "open_raw", "close_raw", "currency"]].copy(),
            self.fx_frame,
            date_col="date",
            currency_col="currency",
        )
        daily = daily.merge(
            self.adjusted_prices[["security_id", "date", "adj_factor"]],
            on=["security_id", "date"],
            how="left",
        )
        daily["adj_open_sek"] = daily["open_raw"] * daily["adj_factor"] * daily["sek_per_ccy"]
        daily["adj_close_sek"] = daily["close_raw"] * daily["adj_factor"] * daily["sek_per_ccy"]
        daily = daily.rename(columns={"date": "trade_date"}).sort_values(["trade_date", "security_id"])

        request_columns = [
            "rebalance_month",
            "security_id",
            "next_execution_date",
            "scheduled_exit_date",
        ]
        requests = monthly[request_columns].sort_values(["next_execution_date", "security_id"]).reset_index(drop=True)
        entry = pd.merge_asof(
            requests,
            daily,
            left_on="next_execution_date",
            right_on="trade_date",
            by="security_id",
            direction="forward",
            allow_exact_matches=True,
        ).rename(
            columns={
                "trade_date": "entry_trade_date",
                "adj_open_sek": "entry_adj_open_sek",
                "adj_close_sek": "entry_adj_close_sek",
            }
        )
        entry = entry[
            [
                "rebalance_month",
                "security_id",
                "scheduled_exit_date",
                "entry_trade_date",
                "entry_adj_open_sek",
                "entry_adj_close_sek",
            ]
        ]
        monthly = monthly.merge(entry, on=["rebalance_month", "security_id", "scheduled_exit_date"], how="left")

        exit_requests = monthly.loc[
            monthly["scheduled_exit_date"].notna(),
            ["rebalance_month", "security_id", "scheduled_exit_date", "next_execution_date"],
        ].copy()
        exit_requests = exit_requests.sort_values(["scheduled_exit_date", "security_id"]).reset_index(drop=True)
        exit_forward = pd.merge_asof(
            exit_requests,
            daily,
            left_on="scheduled_exit_date",
            right_on="trade_date",
            by="security_id",
            direction="forward",
            allow_exact_matches=True,
        ).rename(
            columns={
                "trade_date": "exit_trade_date_forward",
                "adj_open_sek": "exit_adj_open_sek_forward",
                "adj_close_sek": "exit_adj_close_sek_forward",
            }
        )
        exit_backward = pd.merge_asof(
            exit_requests,
            daily,
            left_on="scheduled_exit_date",
            right_on="trade_date",
            by="security_id",
            direction="backward",
            allow_exact_matches=True,
        ).rename(
            columns={
                "trade_date": "exit_trade_date_backward",
                "adj_close_sek": "exit_adj_close_sek_backward",
            }
        )
        exit_frame = exit_forward[
            [
                "rebalance_month",
                "security_id",
                "scheduled_exit_date",
                "next_execution_date",
                "exit_trade_date_forward",
                "exit_adj_open_sek_forward",
                "exit_adj_close_sek_forward",
            ]
        ].merge(
            exit_backward[
                [
                    "rebalance_month",
                    "security_id",
                    "scheduled_exit_date",
                    "next_execution_date",
                    "exit_trade_date_backward",
                    "exit_adj_close_sek_backward",
                ]
            ],
            on=["rebalance_month", "security_id", "scheduled_exit_date", "next_execution_date"],
            how="left",
        )
        monthly = monthly.merge(
            exit_frame,
            on=["rebalance_month", "security_id", "scheduled_exit_date", "next_execution_date"],
            how="left",
        )

        cashout_dates = {
            security_id: np.sort(frame["event_date"].to_numpy(dtype="datetime64[ns]"))
            for security_id, frame in self.corporate_actions.loc[
                self.corporate_actions["action_type"].isin(CASHOUT_ACTION_TYPES)
            ].groupby("security_id")
        }
        monthly["has_cashout_during_holding"] = [
            self._has_cashout(
                cashout_dates,
                security_id=row.security_id,
                start_date=row.next_execution_date,
                end_date=row.scheduled_exit_date,
            )
            for row in monthly.itertuples(index=False)
        ]

        monthly["entry_available_next_open"] = (
            monthly["entry_trade_date"].notna()
            & monthly["entry_adj_open_sek"].notna()
            & (
                monthly["scheduled_exit_date"].isna()
                | (monthly["entry_trade_date"] <= monthly["scheduled_exit_date"])
            )
        )
        monthly["entry_available_next_close"] = (
            monthly["entry_trade_date"].notna()
            & monthly["entry_adj_close_sek"].notna()
            & (
                monthly["scheduled_exit_date"].isna()
                | (monthly["entry_trade_date"] <= monthly["scheduled_exit_date"])
            )
        )

        monthly["gross_return_next_open"] = self._resolve_holding_return(monthly, execution_model="next_open")
        monthly["gross_return_next_close"] = self._resolve_holding_return(monthly, execution_model="next_close")

        monthly = monthly.sort_values(["rebalance_month", "security_id"]).reset_index(drop=True)
        return {
            "anchor_adj_close_local": self._pivot(monthly, "anchor_adj_close_local"),
            "asof_matches_anchor": self._pivot(monthly, "asof_matches_anchor").astype(bool),
            "is_eligible_full_nordics": self._pivot(monthly, "is_eligible_full_nordics").astype(bool),
            "is_eligible_se_only": self._pivot(monthly, "is_eligible_se_only").astype(bool),
            "close_raw_sek": self._pivot(monthly, "close_raw_sek"),
            "median_daily_value_60d_sek": self._pivot(monthly, "median_daily_value_60d_sek"),
            "market_cap_sek": self._pivot(monthly, "market_cap_sek"),
            "entry_available_next_open": self._pivot(monthly, "entry_available_next_open").astype(bool),
            "entry_available_next_close": self._pivot(monthly, "entry_available_next_close").astype(bool),
            "gross_return_next_open": self._pivot(monthly, "gross_return_next_open"),
            "gross_return_next_close": self._pivot(monthly, "gross_return_next_close"),
        }

    def _build_monthly_return_matrix(self) -> np.ndarray:
        returns = np.full_like(self.anchor_adj_close_local, np.nan, dtype=float)
        if self.anchor_adj_close_local.shape[0] <= 1:
            return returns
        prior = self.anchor_adj_close_local[:-1]
        current = self.anchor_adj_close_local[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = current / prior - 1.0
        returns[1:] = delta
        return returns

    def _build_benchmark_monthly_prices(self) -> dict[str, list[float]]:
        benchmark_prices: dict[str, list[float]] = {}
        for benchmark_id, return_map in self.benchmark_monthly_returns.items():
            series: list[float] = []
            price = 1.0
            for month in self.holding_months:
                if month is None:
                    series.append(float("nan"))
                    continue
                monthly_return = return_map.get(month)
                if monthly_return is None or not np.isfinite(monthly_return):
                    series.append(float("nan"))
                    continue
                if not np.isfinite(price):
                    price = 1.0
                price *= 1.0 + float(monthly_return)
                series.append(price)
            benchmark_prices[benchmark_id] = series
        return benchmark_prices

    def volatility_for_index(self, signal_index: int, vol_window: int, vol_skip: int) -> np.ndarray | None:
        end_index = signal_index - int(vol_skip)
        start_index = end_index - int(vol_window) + 1
        if start_index < 0 or end_index < 0:
            return None
        window = self.monthly_return_matrix[start_index : end_index + 1]
        if window.size == 0:
            return None
        return np.nanstd(window, axis=0, ddof=1)

    def trend_filter_on(self, signal_index: int, ma_window: int) -> bool:
        series = self.benchmark_monthly_prices.get(config.PRIMARY_PASSIVE_BENCHMARK_ID)
        if not series:
            return True
        price_index = signal_index - 1  # use last fully known month to avoid look-ahead
        if price_index < 0:
            return True
        window_start = price_index - int(ma_window) + 1
        if window_start < 0:
            return True
        if price_index >= len(series):
            return True
        window = series[window_start : price_index + 1]
        if any(not np.isfinite(value) for value in window):
            return True
        current_price = series[price_index]
        if not np.isfinite(current_price):
            return True
        average = float(sum(window)) / float(len(window))
        return current_price >= average

    def _weight_by_country(self, selected_indices: Sequence[int], weights: dict[int, float] | None) -> dict[str, float]:
        if not selected_indices:
            return {}
        if not weights:
            denom = float(max(1, len(selected_indices)))
            return {
                country: float(count) / denom
                for country, count in self._counts_by_country(selected_indices, denominator=None).items()
            }
        totals: dict[str, float] = {}
        for index in selected_indices:
            country = self.security_country[index]
            totals[country] = totals.get(country, 0.0) + float(weights.get(index, 0.0))
        return totals

    @staticmethod
    def _has_cashout(
        cashout_dates: dict[str, np.ndarray],
        *,
        security_id: str,
        start_date: pd.Timestamp | float,
        end_date: pd.Timestamp | float,
    ) -> bool:
        if pd.isna(start_date) or pd.isna(end_date):
            return False
        dates = cashout_dates.get(security_id)
        if dates is None or len(dates) == 0:
            return False
        start_value = np.datetime64(pd.Timestamp(start_date))
        end_value = np.datetime64(pd.Timestamp(end_date))
        left = np.searchsorted(dates, start_value, side="left")
        right = np.searchsorted(dates, end_value, side="right")
        return bool(right > left)

    @staticmethod
    def _resolve_holding_return(monthly: pd.DataFrame, *, execution_model: str) -> pd.Series:
        if execution_model == "next_open":
            entry_column = "entry_adj_open_sek"
            exit_forward_column = "exit_adj_open_sek_forward"
            entry_valid_column = "entry_available_next_open"
        else:
            entry_column = "entry_adj_close_sek"
            exit_forward_column = "exit_adj_close_sek_forward"
            entry_valid_column = "entry_available_next_close"

        fallback_multiplier = np.where(
            monthly["has_cashout_during_holding"],
            1.0,
            1.0 + config.DELIST_FALLBACK_HAIRCUT,
        )
        fallback_price = monthly["exit_adj_close_sek_backward"] * fallback_multiplier
        usable_backward = (
            monthly["exit_trade_date_backward"].notna()
            & monthly["entry_trade_date"].notna()
            & (monthly["exit_trade_date_backward"] >= monthly["entry_trade_date"])
        )
        exit_price = np.where(
            monthly[exit_forward_column].notna(),
            monthly[exit_forward_column],
            np.where(usable_backward, fallback_price, np.nan),
        )
        returns = np.full(len(monthly), np.nan, dtype=float)
        valid_entry = monthly[entry_valid_column] & monthly[entry_column].gt(0.0)
        fallback_only = valid_entry & pd.isna(exit_price)
        valid_exit = valid_entry & np.isfinite(exit_price) & (exit_price > 0.0)
        valid_exit_mask = valid_exit.to_numpy(dtype=bool)
        returns[valid_exit_mask] = (
            exit_price[valid_exit_mask].astype(float)
            / monthly.loc[valid_exit, entry_column].to_numpy(dtype=float)
            - 1.0
        )
        returns[fallback_only.to_numpy()] = config.DELIST_FALLBACK_HAIRCUT
        return pd.Series(returns, index=monthly.index)

    def _precompute_scores(self) -> dict[tuple[int, int], np.ndarray]:
        scores: dict[tuple[int, int], np.ndarray] = {}
        profile_sets = config.RESEARCH_PROFILE_SETS.values()
        all_lookbacks = sorted(
            {
                value
                for profile_set in profile_sets
                for profile in profile_set.values()
                for value in profile["lookbacks"]
            }
        )
        all_skips = sorted(
            {value for profile_set in profile_sets for profile in profile_set.values() for value in profile["skips"]}
        )
        for lookback in all_lookbacks:
            for skip in all_skips:
                current = np.roll(self.anchor_adj_close_local, shift=skip, axis=0)
                previous = np.roll(self.anchor_adj_close_local, shift=skip + lookback, axis=0)
                current[:skip, :] = np.nan
                previous[: skip + lookback, :] = np.nan
                score = current / previous - 1.0
                invalid = (~np.isfinite(current)) | (~np.isfinite(previous)) | (current <= 0.0) | (previous <= 0.0)
                score[invalid] = np.nan
                scores[(lookback, skip)] = score
        return scores

    def _precompute_rank_orders(self) -> dict[tuple[int, int], list[np.ndarray]]:
        rank_orders: dict[tuple[int, int], list[np.ndarray]] = {}
        security_order = np.arange(len(self.security_ids))
        for key, matrix in self.scores.items():
            month_orders: list[np.ndarray] = []
            for month_index in range(matrix.shape[0]):
                month_scores = matrix[month_index]
                sort_key = np.where(np.isfinite(month_scores), -month_scores, np.inf)
                month_orders.append(np.lexsort((security_order, sort_key)))
            rank_orders[key] = month_orders
        return rank_orders

    def _precompute_capacity_masks(self) -> dict[int, np.ndarray]:
        masks: dict[int, np.ndarray] = {}
        median_value = self.median_daily_value_60d_sek
        all_top_ns = sorted(
            {
                value
                for profile_set in config.RESEARCH_PROFILE_SETS.values()
                for profile in profile_set.values()
                for value in profile["top_ns"]
            }
        )
        for top_n in all_top_ns:
            order_notional = config.SIM_CAPITAL_SEK / float(top_n)
            ratio = np.full_like(median_value, np.inf, dtype=float)
            np.divide(order_notional, median_value, out=ratio, where=median_value > 0.0)
            masks[top_n] = np.isfinite(ratio) & (ratio <= config.MAX_ORDER_FRACTION_OF_60D_MEDIAN_DAILY_VALUE)
        return masks

    def _precompute_cost_fractions(self) -> dict[tuple[int, str, str], np.ndarray]:
        costs: dict[tuple[int, str, str], np.ndarray] = {}
        median_value = self.median_daily_value_60d_sek
        close_raw_sek = self.close_raw_sek
        currency_fx = np.where(self.security_is_non_sek[None, :], 1.0, 0.0)
        spread = np.select(
            [
                median_value >= 50_000_000.0,
                median_value >= 20_000_000.0,
                median_value >= 10_000_000.0,
                median_value >= 5_000_000.0,
            ],
            [15.0, 25.0, 40.0, 60.0],
            default=100.0,
        )
        all_top_ns = sorted(
            {
                value
                for profile_set in config.RESEARCH_PROFILE_SETS.values()
                for profile in profile_set.values()
                for value in profile["top_ns"]
            }
        )
        for top_n in all_top_ns:
            order_notional = config.SIM_CAPITAL_SEK / float(top_n)
            brokerage_sek = max(
                config.BROKERAGE_MIN_SEK,
                order_notional * (config.BROKERAGE_BPS / 10_000.0),
            )
            brokerage_bps = (brokerage_sek / order_notional) * 10_000.0
            ratio = np.full_like(median_value, np.inf, dtype=float)
            np.divide(order_notional, median_value, out=ratio, where=median_value > 0.0)
            participation = np.select(
                [
                    ratio <= 0.0025,
                    ratio <= 0.0050,
                    ratio <= 0.0075,
                    ratio <= 0.0100,
                ],
                [0.0, 10.0, 20.0, 40.0],
                default=np.nan,
            )
            capacity_mask = self.capacity_masks[top_n]
            low_price = np.where(close_raw_sek < config.LOW_PRICE_THRESHOLD_SEK, config.LOW_PRICE_ADDON_BPS, 0.0)
            for execution_model in EXECUTION_MODELS:
                multiplier = (
                    config.NEXT_OPEN_IMPACT_MULTIPLIER
                    if execution_model == "next_open"
                    else config.NEXT_CLOSE_IMPACT_MULTIPLIER
                )
                impact = (spread + participation) * multiplier
                for fx_scenario in FX_SCENARIOS:
                    fx_bps = config.FX_FRICTION_SCENARIOS_BPS[fx_scenario] * currency_fx
                    total_bps = brokerage_bps + impact + low_price + fx_bps
                    cost_fraction = np.where(capacity_mask, total_bps / 10_000.0 / float(top_n), np.nan)
                    costs[(top_n, execution_model, fx_scenario)] = cost_fraction
        return costs

    def _precompute_flat_cost_fractions(self) -> dict[tuple[int, int, int], np.ndarray]:
        flat_costs: dict[tuple[int, int, int], np.ndarray] = {}
        currency_fx = np.where(self.security_is_non_sek[None, :], 1.0, 0.0)
        all_top_ns = sorted(
            {
                value
                for profile_set in config.RESEARCH_PROFILE_SETS.values()
                for profile in profile_set.values()
                for value in profile["top_ns"]
            }
        )
        for top_n in all_top_ns:
            base_weight = 1.0 / float(top_n)
            for sek_bps in config.COST_SENSITIVITY_BPS_SEK:
                for extra_fx_bps in config.FX_COST_SENSITIVITY_BPS_NON_SEK:
                    total_bps = float(sek_bps) + float(extra_fx_bps) * currency_fx
                    flat_costs[(top_n, int(sek_bps), int(extra_fx_bps))] = (
                        self.capacity_masks[top_n] * (base_weight * total_bps / 10_000.0)
                    )
        return flat_costs

    def _build_benchmark_monthly_returns(self) -> dict[str, dict[str, float]]:
        benchmark_prices = enrich_with_fx(
            self.benchmark_prices[["benchmark_id", "date", "currency", "adj_close"]].copy(),
            self.fx_frame,
            date_col="date",
            currency_col="currency",
        )
        benchmark_prices["adj_close_sek"] = benchmark_prices["adj_close"] * benchmark_prices["sek_per_ccy"]
        benchmark_returns: dict[str, dict[str, float]] = {}
        holding_months = sorted({month for month in self.holding_months if month is not None})
        for benchmark_id, frame in benchmark_prices.groupby("benchmark_id"):
            sorted_frame = frame.sort_values("date").reset_index(drop=True)
            dates = sorted_frame["date"].to_numpy(dtype="datetime64[ns]")
            prices = sorted_frame["adj_close_sek"].to_numpy(dtype=float)
            month_map: dict[str, float] = {}
            for month in holding_months:
                start = pd.Period(month, freq="M").to_timestamp()
                end = (pd.Period(month, freq="M") + 1).to_timestamp()
                entry_index = int(np.searchsorted(dates, np.datetime64(start), side="left"))
                exit_index = int(np.searchsorted(dates, np.datetime64(end), side="left"))
                if entry_index >= len(prices) or exit_index >= len(prices):
                    month_map[month] = float("nan")
                    continue
                entry_price = prices[entry_index]
                exit_price = prices[exit_index]
                month_map[month] = (exit_price / entry_price) - 1.0 if entry_price > 0.0 else float("nan")
            benchmark_returns[benchmark_id] = month_map
        return benchmark_returns

    def variant_mask(
        self,
        variant_name: str,
        *,
        excluded_country: str | None = None,
        excluded_countries: Sequence[str] | None = None,
    ) -> np.ndarray:
        if variant_name == "Full Nordics":
            mask = self.base_variant_masks["Full Nordics"].copy()
        elif variant_name == "SE-only":
            mask = self.base_variant_masks["SE-only"].copy()
        elif variant_name == "largest-third-by-market-cap":
            return np.zeros_like(self.base_variant_masks["Full Nordics"], dtype=bool)
        else:
            raise ValueError(f"Unsupported universe variant '{variant_name}'.")
        countries_to_exclude = set(excluded_countries or ())
        if excluded_country is not None:
            countries_to_exclude.add(excluded_country)
        for country_code in countries_to_exclude:
            mask = mask & (self.security_country != country_code)[None, :]
        return mask

    def _window_signal_indices(self, start_month: str, end_month: str) -> list[int]:
        allowed = set(month_labels_between(start_month, end_month))
        return [
            index
            for index, month in enumerate(self.holding_months)
            if month is not None and month in allowed
        ]

    def _benchmark_returns_for_window(self, benchmark_id: str | None, months: Sequence[str]) -> list[float] | None:
        if not benchmark_id or benchmark_id not in self.benchmark_monthly_returns:
            return None
        values = [self.benchmark_monthly_returns[benchmark_id].get(month, float("nan")) for month in months]
        if any(not np.isfinite(value) for value in values):
            return None
        return [float(value) for value in values]

    def simulate_window(
        self,
        *,
        params: dict[str, int],
        universe_variant: str,
        execution_model: str,
        fx_scenario: str,
        start_month: str,
        end_month: str,
        excluded_country: str | None = None,
        excluded_countries: Sequence[str] | None = None,
        collect_details: bool = False,
        shuffled_selection_seed: int | None = None,
        flat_cost_bps: int | None = None,
        extra_fx_cost_bps: int | None = None,
    ) -> WindowSimulation:
        lookback = int(params["l"])
        skip = int(params["skip"])
        top_n = int(params["top_n"])
        signal_indices = self._window_signal_indices(start_month, end_month)
        base_mask = self.variant_mask(
            universe_variant,
            excluded_country=excluded_country,
            excluded_countries=excluded_countries,
        )
        rankable = (
            base_mask
            & self.asof_matches_anchor
            & self.capacity_masks[top_n]
            & self.entry_available[execution_model]
            & np.isfinite(self.holding_returns[execution_model])
            & np.isfinite(self.scores[(lookback, skip)])
        )
        rng = random.Random(shuffled_selection_seed) if shuffled_selection_seed is not None else None
        months: list[str] = []
        monthly_returns: list[float] = []
        details: list[dict[str, Any]] = [] if collect_details else []
        previous_selection: set[int] = set()
        previous_weights: dict[int, float] = {}
        holding_ages: dict[int, int] = {}
        strategy = strategy_variants.resolve_strategy(params)
        weighting = strategy.get("weighting", "equal")
        trend_filter = bool(strategy.get("trend_filter", False))
        ma_window = int(strategy.get("ma_window", 10) or 10)
        liquidity_min_mdv = strategy.get("liquidity_min_mdv")
        vol_window = int(strategy.get("vol_window", 12) or 12)
        vol_skip = int(strategy.get("vol_skip", 1) or 1)

        for signal_index in signal_indices:
            holding_month = self.holding_months[signal_index]
            if holding_month is None:
                continue
            current_rankable = rankable[signal_index]
            if liquidity_min_mdv is not None:
                current_rankable = current_rankable & (
                    self.median_daily_value_60d_sek[signal_index] >= float(liquidity_min_mdv)
                )
            filter_on = True
            if trend_filter:
                filter_on = self.trend_filter_on(signal_index, ma_window)
            if not filter_on:
                selected_indices = []
                current_selection = set()
                weights: dict[int, float] = {}
                entries: set[int] = set()
                exits = previous_selection
                holding_ages = {}
            else:
                if rng is None:
                    selected_indices, holding_ages = strategy_variants.select_indices(
                        rank_order=self.rank_orders[(lookback, skip)][signal_index],
                        rankable_mask=current_rankable,
                        top_n=top_n,
                        previous_selection=previous_selection,
                        strategy=strategy,
                        holding_ages=holding_ages,
                    )
                else:
                    eligible = np.flatnonzero(current_rankable).tolist()
                    take = min(top_n, len(eligible))
                    selected_indices = rng.sample(eligible, k=take) if take else []
                current_selection = set(selected_indices)
                weight_values: np.ndarray | None = None
                if weighting == "cap":
                    weight_values = self.market_cap_sek[signal_index]
                elif weighting == "inv_vol":
                    weight_values = self.volatility_for_index(signal_index, vol_window, vol_skip)
                weights = strategy_variants.compute_weights(
                    selected_indices=selected_indices,
                    top_n=top_n,
                    weighting=weighting,
                    weight_values=weight_values,
                )
                entries = current_selection - previous_selection
                exits = previous_selection - current_selection

            if weights:
                gross_return = sum(
                    float(self.holding_returns[execution_model][signal_index, index]) * weights.get(index, 0.0)
                    for index in selected_indices
                )
            else:
                gross_return = 0.0
            if flat_cost_bps is None:
                one_way_cost = self.cost_fractions[(top_n, execution_model, fx_scenario)]
            else:
                one_way_cost = self.flat_cost_fractions[(top_n, int(flat_cost_bps), int(extra_fx_cost_bps or 0))]
            trade_cost = 0.0
            if entries:
                per_security_cost = one_way_cost[signal_index] * float(top_n)
                for index in entries:
                    cost = per_security_cost[index]
                    if not np.isfinite(cost):
                        continue
                    trade_cost += float(cost) * float(weights.get(index, 0.0))
            if exits:
                per_security_cost = one_way_cost[signal_index] * float(top_n)
                for index in exits:
                    cost = per_security_cost[index]
                    if not np.isfinite(cost):
                        continue
                    trade_cost += float(cost) * float(previous_weights.get(index, 0.0))
            net_return = gross_return - trade_cost

            months.append(holding_month)
            monthly_returns.append(net_return)
            if collect_details:
                details.append(
                    self._window_detail(
                        signal_index=signal_index,
                        selected_indices=selected_indices,
                        rankable_mask=current_rankable,
                        entries=entries,
                        exits=exits,
                        gross_return=gross_return,
                        net_return=net_return,
                        trade_cost=trade_cost,
                        weights=weights,
                    )
                )
            previous_selection = current_selection
            previous_weights = weights if filter_on else {}

        tertiary_id = getattr(config, "TERTIARY_OPPORTUNITY_COST_BENCHMARK_ID", None)
        return WindowSimulation(
            months=months,
            monthly_returns=monthly_returns,
            primary_benchmark_returns=self._benchmark_returns_for_window(config.PRIMARY_PASSIVE_BENCHMARK_ID, months),
            secondary_benchmark_returns=self._benchmark_returns_for_window(
                config.SECONDARY_OPPORTUNITY_COST_BENCHMARK_ID,
                months,
            ),
            tertiary_benchmark_returns=self._benchmark_returns_for_window(tertiary_id, months),
            details=details if collect_details else None,
        )

    def selection_snapshot(
        self,
        *,
        params: dict[str, int],
        signal_month: str,
        universe_variant: str,
        execution_model: str,
        fx_scenario: str,
        excluded_country: str | None = None,
        excluded_countries: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        strategy = strategy_variants.resolve_strategy(params)
        lookback = int(params["l"])
        skip = int(params["skip"])
        top_n = int(params["top_n"])
        weighting = strategy.get("weighting", "equal")
        trend_filter = bool(strategy.get("trend_filter", False))
        ma_window = int(strategy.get("ma_window", 10) or 10)
        liquidity_min_mdv = strategy.get("liquidity_min_mdv")
        vol_window = int(strategy.get("vol_window", 12) or 12)
        vol_skip = int(strategy.get("vol_skip", 1) or 1)
        if signal_month not in self.month_index:
            raise ValueError(f"Signal month '{signal_month}' is not available in the research dataset.")
        signal_index = self.month_index[signal_month]
        base_mask = self.variant_mask(
            universe_variant,
            excluded_country=excluded_country,
            excluded_countries=excluded_countries,
        )
        selectable = (
            base_mask
            & self.asof_matches_anchor
            & self.capacity_masks[top_n]
            & self.entry_available[execution_model]
            & np.isfinite(self.scores[(lookback, skip)])
        )
        if liquidity_min_mdv is not None:
            selectable = selectable & (self.median_daily_value_60d_sek >= float(liquidity_min_mdv))
        filter_on = True
        if trend_filter:
            filter_on = self.trend_filter_on(signal_index, ma_window)
        if filter_on:
            selected_indices, _ = strategy_variants.select_indices(
                rank_order=self.rank_orders[(lookback, skip)][signal_index],
                rankable_mask=selectable[signal_index],
                top_n=top_n,
                previous_selection=set(),
                strategy=strategy,
                holding_ages={},
            )
        else:
            selected_indices = []
        weight_values: np.ndarray | None = None
        if weighting == "cap":
            weight_values = self.market_cap_sek[signal_index]
        elif weighting == "inv_vol":
            weight_values = self.volatility_for_index(signal_index, vol_window, vol_skip)
        weights = strategy_variants.compute_weights(
            selected_indices=selected_indices,
            top_n=top_n,
            weighting=weighting,
            weight_values=weight_values,
        )
        month_rows = self.universe.loc[self.universe["rebalance_month"] == signal_month]
        anchor_trade_date = month_rows["anchor_trade_date"].dropna().iloc[0] if month_rows["anchor_trade_date"].notna().any() else None
        next_execution_date = month_rows["next_execution_date"].dropna().iloc[0] if month_rows["next_execution_date"].notna().any() else None
        selected_security_ids = [self.security_ids[index] for index in selected_indices]
        selected_scores = [float(self.scores[(lookback, skip)][signal_index, index]) for index in selected_indices]
        return {
            "signal_month": signal_month,
            "holding_month": self.holding_months[signal_index],
            "anchor_trade_date": anchor_trade_date,
            "next_execution_date": next_execution_date,
            "selected_security_ids": selected_security_ids,
            "selected_scores": selected_scores,
            "selected_weights": [weights.get(index, 0.0) for index in selected_indices] if weights else [],
            "eligible_count": int(np.count_nonzero(selectable[signal_index])),
            "selected_count": len(selected_security_ids),
            "eligible_by_country": self._counts_by_country(np.flatnonzero(selectable[signal_index]).tolist(), denominator=None),
            "selected_by_country": self._counts_by_country(selected_indices, denominator=None),
            "weight_by_country": self._weight_by_country(selected_indices, weights),
            "universe_variant": universe_variant,
            "execution_model": execution_model,
            "fx_scenario": fx_scenario,
            "filter_on": filter_on,
        }

    def _window_detail(
        self,
        *,
        signal_index: int,
        selected_indices: list[int],
        rankable_mask: np.ndarray,
        entries: set[int],
        exits: set[int],
        gross_return: float,
        net_return: float,
        trade_cost: float,
        weights: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "holding_month": self.holding_months[signal_index],
            "eligible_by_country": self._counts_by_country(np.flatnonzero(rankable_mask).tolist(), denominator=None),
            "selected_by_country": self._counts_by_country(selected_indices, denominator=None),
            "weight_by_country": self._weight_by_country(selected_indices, weights),
            "turnover_by_country": self._counts_by_country(list(entries) + list(exits), denominator=float(max(1, len(entries) + len(exits)))),
            "return_contribution_by_country": self._contribution_by_country(signal_index, selected_indices, max(1, len(selected_indices))),
            "gross_return": gross_return,
            "trade_cost": trade_cost,
            "net_return": net_return,
            "selected_security_ids": [self.security_ids[index] for index in selected_indices],
            "selected_weights": [weights.get(index, 0.0) for index in selected_indices] if weights else [],
        }

    def _counts_by_country(self, security_indices: list[int], *, denominator: float | None) -> dict[str, float]:
        if not security_indices:
            return {}
        counts: dict[str, float] = {}
        for security_index in security_indices:
            country_code = str(self.security_country[security_index])
            counts[country_code] = counts.get(country_code, 0.0) + 1.0
        if denominator is None or denominator <= 0.0:
            return counts
        return {country_code: count / denominator for country_code, count in counts.items()}

    def _contribution_by_country(self, signal_index: int, selected_indices: list[int], selected_count: int) -> dict[str, float]:
        if not selected_indices:
            return {}
        contributions: dict[str, float] = {}
        weight = 1.0 / float(selected_count)
        security_returns = self.holding_returns["next_open"][signal_index, selected_indices]
        for security_index, security_return in zip(selected_indices, security_returns, strict=True):
            country_code = str(self.security_country[security_index])
            contributions[country_code] = contributions.get(country_code, 0.0) + float(security_return) * weight
        return contributions

