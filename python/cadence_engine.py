from __future__ import annotations

import math
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
from paper_trading_engine import ResearchDataset, WindowSimulation, month_labels_between
from phase1_lib import enrich_with_fx


CASHOUT_ACTION_TYPES = {"cash_merger", "delisting_cashout"}


@dataclass(frozen=True)
class CadenceSpec:
    cadence_id: str
    cadence_label: str
    schedule_type: str
    periods_per_year: int
    canonical_offset_id: int
    offset_ids: tuple[int, ...]
    offset_label_prefix: str


def load_cadence_spec(cadence_id: str) -> CadenceSpec:
    settings = config.CADENCE_COMPARE_SETTINGS[cadence_id]
    return CadenceSpec(
        cadence_id=cadence_id,
        cadence_label=settings["label"],
        schedule_type=settings["schedule_type"],
        periods_per_year=int(settings["periods_per_year"]),
        canonical_offset_id=int(settings["canonical_offset_id"]),
        offset_ids=tuple(int(value) for value in settings["offset_ids"]),
        offset_label_prefix=str(settings.get("offset_label_prefix", "offset")),
    )


def cadence_period_label(periods_per_year: int) -> str:
    return "months" if periods_per_year == 12 else "periods"


def _month_end_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    frame = pd.DataFrame({"date": dates})
    frame["period"] = frame["date"].dt.to_period("M")
    return list(frame.groupby("period")["date"].max().sort_values())


def _iso_week_end_dates(dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({"date": dates})
    iso = frame["date"].dt.isocalendar()
    frame["iso_year"] = iso.year
    frame["iso_week"] = iso.week
    week_end = frame.groupby(["iso_year", "iso_week"])["date"].max().reset_index()
    week_end = week_end.sort_values("date").reset_index(drop=True)
    return week_end


def build_rebalance_dates(
    trade_calendar: pd.DatetimeIndex,
    *,
    schedule_type: str,
    offset_id: int,
) -> list[pd.Timestamp]:
    if schedule_type == "month_end":
        return _month_end_dates(trade_calendar)
    if schedule_type == "quarter_end":
        month_end = _month_end_dates(trade_calendar)
        period = 3
        remainder = (period - offset_id) % period
        return [date for date in month_end if (date.month % period) == remainder]
    if schedule_type == "half_year_end":
        month_end = _month_end_dates(trade_calendar)
        period = 6
        remainder = (period - offset_id) % period
        return [date for date in month_end if (date.month % period) == remainder]
    if schedule_type == "iso_weekly":
        week_end = _iso_week_end_dates(trade_calendar)
        return list(week_end["date"])
    if schedule_type == "iso_biweekly":
        week_end = _iso_week_end_dates(trade_calendar)
        week_end["week_index"] = week_end["iso_year"] * 53 + week_end["iso_week"]
        week_end["parity"] = week_end["week_index"] % 2
        return list(week_end.loc[week_end["parity"] == offset_id, "date"])
    raise ValueError(f"Unsupported schedule_type '{schedule_type}'.")


class CadenceDataset:
    def __init__(
        self,
        data_dir: Path,
        *,
        cadence_id: str,
        offset_id: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.base = ResearchDataset(data_dir)
        self.cadence = load_cadence_spec(cadence_id)
        self.offset_id = offset_id if offset_id is not None else self.cadence.canonical_offset_id
        if self.offset_id not in self.cadence.offset_ids:
            raise ValueError(f"offset_id {self.offset_id} is not valid for cadence {cadence_id}.")

        self.trade_calendar = pd.DatetimeIndex(sorted(self.base.raw_prices["date"].unique()))
        self.rebalance_dates = build_rebalance_dates(
            self.trade_calendar,
            schedule_type=self.cadence.schedule_type,
            offset_id=self.offset_id,
        )
        self._prepare_daily_prices()
        self.entry_indices, self.exit_indices = self._build_entry_exit_indices()
        self.holding_months, self.period_month_indices = self._build_holding_months()
        self.scores = self._precompute_scores()
        self.rank_orders = self._precompute_rank_orders()
        self.entry_available, self.holding_returns = self._precompute_holding_returns()
        self.benchmark_period_returns = self._build_benchmark_period_returns()
        self.benchmark_period_prices = self._build_benchmark_period_prices()

    def variant_mask(
        self,
        variant_name: str,
        *,
        excluded_country: str | None = None,
        excluded_countries: Sequence[str] | None = None,
    ) -> np.ndarray:
        return self.base.variant_mask(
            variant_name,
            excluded_country=excluded_country,
            excluded_countries=excluded_countries,
        )

    def _prepare_daily_prices(self) -> None:
        daily = self.base.raw_prices[["security_id", "date", "open_raw", "close_raw", "currency"]].copy()
        daily = enrich_with_fx(
            daily,
            self.base.fx_frame,
            date_col="date",
            currency_col="currency",
        )
        daily = daily.merge(
            self.base.adjusted_prices[["security_id", "date", "adj_factor"]],
            on=["security_id", "date"],
            how="left",
        )
        daily["adj_open_sek"] = daily["open_raw"] * daily["adj_factor"] * daily["sek_per_ccy"]
        daily["adj_close_sek"] = daily["close_raw"] * daily["adj_factor"] * daily["sek_per_ccy"]

        pivot_open = (
            daily.pivot(index="date", columns="security_id", values="adj_open_sek")
            .reindex(columns=self.base.security_ids)
            .reindex(index=self.trade_calendar)
        )
        pivot_close = (
            daily.pivot(index="date", columns="security_id", values="adj_close_sek")
            .reindex(columns=self.base.security_ids)
            .reindex(index=self.trade_calendar)
        )

        self.open_raw = pivot_open.to_numpy()
        self.close_raw = pivot_close.to_numpy()
        # Forward-fill only to avoid look-ahead when data is missing.
        self.close_ffill = pivot_close.ffill().to_numpy()

    def _build_entry_exit_indices(self) -> tuple[list[int], list[int]]:
        trade_dates = self.trade_calendar.to_numpy(dtype="datetime64[ns]")
        entry_indices: list[int] = []
        for date in self.rebalance_dates:
            idx = int(np.searchsorted(trade_dates, np.datetime64(date), side="right"))
            entry_indices.append(idx if idx < len(trade_dates) else -1)
        exit_indices = entry_indices[1:] + [-1]
        return entry_indices, exit_indices

    def _build_holding_months(self) -> tuple[list[str | None], list[int]]:
        trade_dates = self.trade_calendar.to_numpy(dtype="datetime64[ns]")
        holding_months: list[str | None] = []
        month_indices: list[int] = []
        for entry_index in self.entry_indices:
            if entry_index < 0 or entry_index >= len(trade_dates):
                holding_months.append(None)
                month_indices.append(-1)
                continue
            entry_date = pd.Timestamp(trade_dates[entry_index])
            month_label = str(entry_date.to_period("M"))
            holding_months.append(month_label)
            month_indices.append(self.base.month_index.get(month_label, -1))
        return holding_months, month_indices

    def _price_index_for_date(self, date: pd.Timestamp) -> int:
        trade_dates = self.trade_calendar.to_numpy(dtype="datetime64[ns]")
        idx = int(np.searchsorted(trade_dates, np.datetime64(date), side="right")) - 1
        return idx

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
                matrix = np.full((len(self.rebalance_dates), len(self.base.security_ids)), np.nan, dtype=float)
                for period_index, rebalance_date in enumerate(self.rebalance_dates):
                    current_date = rebalance_date - pd.DateOffset(months=int(skip))
                    previous_date = rebalance_date - pd.DateOffset(months=int(skip + lookback))
                    current_index = self._price_index_for_date(pd.Timestamp(current_date))
                    previous_index = self._price_index_for_date(pd.Timestamp(previous_date))
                    if current_index < 0 or previous_index < 0:
                        continue
                    current = self.close_ffill[current_index]
                    previous = self.close_ffill[previous_index]
                    score = current / previous - 1.0
                    invalid = (~np.isfinite(current)) | (~np.isfinite(previous)) | (current <= 0.0) | (previous <= 0.0)
                    score[invalid] = np.nan
                    matrix[period_index] = score
                scores[(int(lookback), int(skip))] = matrix
        return scores

    def _precompute_rank_orders(self) -> dict[tuple[int, int], list[np.ndarray]]:
        rank_orders: dict[tuple[int, int], list[np.ndarray]] = {}
        security_order = np.arange(len(self.base.security_ids))
        for key, matrix in self.scores.items():
            month_orders: list[np.ndarray] = []
            for period_index in range(matrix.shape[0]):
                period_scores = matrix[period_index]
                sort_key = np.where(np.isfinite(period_scores), -period_scores, np.inf)
                month_orders.append(np.lexsort((security_order, sort_key)))
            rank_orders[key] = month_orders
        return rank_orders

    def _precompute_holding_returns(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        n_periods = len(self.rebalance_dates)
        n_securities = len(self.base.security_ids)
        returns_open = np.full((n_periods, n_securities), np.nan, dtype=float)
        returns_close = np.full((n_periods, n_securities), np.nan, dtype=float)
        entry_open_ok = np.zeros((n_periods, n_securities), dtype=bool)
        entry_close_ok = np.zeros((n_periods, n_securities), dtype=bool)

        for period_index, entry_index in enumerate(self.entry_indices):
            exit_index = self.exit_indices[period_index]
            if entry_index < 0 or exit_index < 0:
                continue
            if exit_index <= entry_index:
                continue
            entry_open = self.open_raw[entry_index]
            entry_close = self.close_raw[entry_index]
            exit_open_forward = self.open_raw[exit_index]
            exit_close_forward = self.close_raw[exit_index]
            exit_close_backward = self.close_ffill[exit_index]
            fallback_multiplier = 1.0 + config.DELIST_FALLBACK_HAIRCUT
            exit_open = np.where(np.isfinite(exit_open_forward), exit_open_forward, exit_close_backward * fallback_multiplier)
            exit_close = np.where(np.isfinite(exit_close_forward), exit_close_forward, exit_close_backward * fallback_multiplier)

            entry_open_ok[period_index] = np.isfinite(entry_open) & (entry_open > 0.0)
            entry_close_ok[period_index] = np.isfinite(entry_close) & (entry_close > 0.0)
            valid_open = entry_open_ok[period_index] & np.isfinite(exit_open) & (exit_open > 0.0)
            valid_close = entry_close_ok[period_index] & np.isfinite(exit_close) & (exit_close > 0.0)
            returns_open[period_index, valid_open] = exit_open[valid_open] / entry_open[valid_open] - 1.0
            returns_close[period_index, valid_close] = exit_close[valid_close] / entry_close[valid_close] - 1.0

        entry_available = {"next_open": entry_open_ok, "next_close": entry_close_ok}
        holding_returns = {"next_open": returns_open, "next_close": returns_close}
        return entry_available, holding_returns

    def _build_benchmark_period_returns(self) -> dict[str, list[float]]:
        benchmark_prices = enrich_with_fx(
            self.base.benchmark_prices[["benchmark_id", "date", "currency", "adj_close"]].copy(),
            self.base.fx_frame,
            date_col="date",
            currency_col="currency",
        )
        benchmark_prices["adj_close_sek"] = benchmark_prices["adj_close"] * benchmark_prices["sek_per_ccy"]
        trade_calendar = self.trade_calendar
        benchmark_returns: dict[str, list[float]] = {}
        for benchmark_id, frame in benchmark_prices.groupby("benchmark_id"):
            pivot = (
                frame.pivot(index="date", columns="benchmark_id", values="adj_close_sek")
                .reindex(index=trade_calendar)
                .ffill()
            )
            series = pivot[benchmark_id].to_numpy(dtype=float)
            period_returns: list[float] = []
            for entry_index, exit_index in zip(self.entry_indices, self.exit_indices, strict=True):
                if entry_index < 0 or exit_index < 0 or exit_index <= entry_index:
                    period_returns.append(float("nan"))
                    continue
                entry_price = series[entry_index]
                exit_price = series[exit_index]
                if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0.0:
                    period_returns.append(float("nan"))
                else:
                    period_returns.append(float(exit_price / entry_price - 1.0))
            benchmark_returns[benchmark_id] = period_returns
        return benchmark_returns

    def _build_benchmark_period_prices(self) -> dict[str, list[float]]:
        benchmark_prices: dict[str, list[float]] = {}
        for benchmark_id, period_returns in self.benchmark_period_returns.items():
            series: list[float] = []
            price = 1.0
            for value in period_returns:
                if value is None or not np.isfinite(value):
                    series.append(float("nan"))
                    continue
                if not np.isfinite(price):
                    price = 1.0
                price *= 1.0 + float(value)
                series.append(price)
            benchmark_prices[benchmark_id] = series
        return benchmark_prices

    def volatility_for_period(self, period_index: int, vol_window: int, vol_skip: int) -> np.ndarray | None:
        month_index = self.period_month_indices[period_index]
        if month_index < 0:
            return None
        return self.base.volatility_for_index(month_index, vol_window, vol_skip)

    def trend_filter_on(self, period_index: int, ma_window: int) -> bool:
        series = self.benchmark_period_prices.get(config.PRIMARY_PASSIVE_BENCHMARK_ID)
        if not series:
            return True
        window_periods = max(1, int(round(float(ma_window) * self.cadence.periods_per_year / 12.0)))
        price_index = period_index - 1  # use last fully known period to avoid look-ahead
        if price_index < 0:
            return True
        window_start = price_index - window_periods + 1
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

    def _window_signal_indices(self, start_month: str, end_month: str) -> list[int]:
        allowed = set(month_labels_between(start_month, end_month))
        return [
            index
            for index, month in enumerate(self.holding_months)
            if month is not None and month in allowed
        ]

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

        months: list[str] = []
        monthly_returns: list[float] = []
        used_indices: list[int] = []
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
        rng = np.random.default_rng(shuffled_selection_seed) if shuffled_selection_seed is not None else None

        for period_index in signal_indices:
            month_index = self.period_month_indices[period_index]
            if month_index < 0:
                continue
            rankable = (
                base_mask[month_index]
                & self.base.asof_matches_anchor[month_index]
                & self.base.capacity_masks[top_n][month_index]
                & self.entry_available[execution_model][period_index]
                & np.isfinite(self.holding_returns[execution_model][period_index])
                & np.isfinite(self.scores[(lookback, skip)][period_index])
            )
            if liquidity_min_mdv is not None:
                rankable = rankable & (
                    self.base.median_daily_value_60d_sek[month_index] >= float(liquidity_min_mdv)
                )
            filter_on = True
            if trend_filter:
                filter_on = self.trend_filter_on(period_index, ma_window)
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
                        rank_order=self.rank_orders[(lookback, skip)][period_index],
                        rankable_mask=rankable,
                        top_n=top_n,
                        previous_selection=previous_selection,
                        strategy=strategy,
                        holding_ages=holding_ages,
                    )
                else:
                    eligible = np.flatnonzero(rankable).tolist()
                    take = min(top_n, len(eligible))
                    selected_indices = rng.choice(eligible, size=take, replace=False).tolist() if take else []
                current_selection = set(selected_indices)
                weight_values: np.ndarray | None = None
                if weighting == "cap":
                    weight_values = self.base.market_cap_sek[month_index]
                elif weighting == "inv_vol":
                    weight_values = self.volatility_for_period(period_index, vol_window, vol_skip)
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
                    float(self.holding_returns[execution_model][period_index, index]) * weights.get(index, 0.0)
                    for index in selected_indices
                )
            else:
                gross_return = 0.0
            if flat_cost_bps is None:
                one_way_cost = self.base.cost_fractions[(top_n, execution_model, fx_scenario)][month_index]
            else:
                one_way_cost = self.base.flat_cost_fractions[(top_n, int(flat_cost_bps), int(extra_fx_cost_bps or 0))][month_index]
            trade_cost = 0.0
            if entries:
                per_security_cost = one_way_cost * float(top_n)
                for index in entries:
                    cost = per_security_cost[index]
                    if not np.isfinite(cost):
                        continue
                    trade_cost += float(cost) * float(weights.get(index, 0.0))
            if exits:
                per_security_cost = one_way_cost * float(top_n)
                for index in exits:
                    cost = per_security_cost[index]
                    if not np.isfinite(cost):
                        continue
                    trade_cost += float(cost) * float(previous_weights.get(index, 0.0))
            net_return = gross_return - trade_cost

            holding_month = self.holding_months[period_index]
            if holding_month is None:
                continue
            months.append(holding_month)
            monthly_returns.append(net_return)
            used_indices.append(period_index)
            if collect_details:
                details.append(
                    {
                        "holding_month": holding_month,
                        "gross_return": gross_return,
                        "trade_cost": trade_cost,
                        "net_return": net_return,
                        "selected_security_ids": [self.base.security_ids[index] for index in selected_indices],
                        "selected_weights": [weights.get(index, 0.0) for index in selected_indices] if weights else [],
                    }
                )
            previous_selection = current_selection
            previous_weights = weights if filter_on else {}

        primary_benchmark = self.benchmark_period_returns.get(config.PRIMARY_PASSIVE_BENCHMARK_ID)
        secondary_benchmark = self.benchmark_period_returns.get(config.SECONDARY_OPPORTUNITY_COST_BENCHMARK_ID)
        tertiary_benchmark_id = getattr(config, "TERTIARY_OPPORTUNITY_COST_BENCHMARK_ID", None)
        tertiary_benchmark = (
            self.benchmark_period_returns.get(tertiary_benchmark_id) if tertiary_benchmark_id else None
        )
        primary_series = [primary_benchmark[index] for index in used_indices] if primary_benchmark else None
        secondary_series = [secondary_benchmark[index] for index in used_indices] if secondary_benchmark else None
        tertiary_series = [tertiary_benchmark[index] for index in used_indices] if tertiary_benchmark else None

        return WindowSimulation(
            months=months,
            monthly_returns=monthly_returns,
            primary_benchmark_returns=primary_series,
            secondary_benchmark_returns=secondary_series,
            tertiary_benchmark_returns=tertiary_series,
            details=details if collect_details else None,
        )

    def negative_control_months(
        self,
        *,
        params: dict[str, int],
        universe_variant: str,
        execution_model: str,
        fx_scenario: str,
        start_month: str,
        end_month: str,
        excluded_countries: Sequence[str],
    ) -> list[dict[str, Any]]:
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
        signal_indices = self._window_signal_indices(start_month, end_month)
        base_mask = self.variant_mask(
            universe_variant,
            excluded_countries=excluded_countries,
        )
        months: list[dict[str, Any]] = []
        for period_index in signal_indices:
            month_index = self.period_month_indices[period_index]
            if month_index < 0:
                continue
            rankable = (
                base_mask[month_index]
                & self.base.asof_matches_anchor[month_index]
                & self.base.capacity_masks[top_n][month_index]
                & self.entry_available[execution_model][period_index]
                & np.isfinite(self.holding_returns[execution_model][period_index])
                & np.isfinite(self.scores[(lookback, skip)][period_index])
            )
            if liquidity_min_mdv is not None:
                rankable = rankable & (self.base.median_daily_value_60d_sek[month_index] >= float(liquidity_min_mdv))
            filter_on = True
            if trend_filter:
                filter_on = self.trend_filter_on(period_index, ma_window)
            positions: list[dict[str, float]] = []
            scores = self.scores[(lookback, skip)][period_index]
            returns = self.holding_returns[execution_model][period_index]
            weight_values: np.ndarray | None = None
            if weighting == "cap":
                weight_values = self.base.market_cap_sek[month_index]
            elif weighting == "inv_vol":
                weight_values = self.volatility_for_period(period_index, vol_window, vol_skip)
            for security_index in np.flatnonzero(rankable):
                score = scores[security_index]
                next_return = returns[security_index]
                if not np.isfinite(score) or not np.isfinite(next_return):
                    continue
                payload = {"score": float(score), "next_return": float(next_return)}
                if weighting in ("cap", "inv_vol") and weight_values is not None:
                    payload["weight_value"] = float(weight_values[security_index])
                positions.append(payload)
            if positions:
                baseline_return = float(np.mean([item["next_return"] for item in positions]))
            else:
                baseline_return = 0.0
            months.append(
                {
                    "positions": positions,
                    "baseline_return": baseline_return,
                    "weighting": weighting,
                    "filter_on": filter_on,
                }
            )
        return months
