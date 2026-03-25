from __future__ import annotations

import argparse
from collections import Counter
import os
import html
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from paper_trading_engine import (
    PRIMARY_TRACK,
    ResearchDataset,
    annualized_sharpe,
    build_thesis,
    load_json,
    max_drawdown,
    serialize_json,
    total_return,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a side-by-side forward-monitoring view for the frozen lead and shadow theses."
    )
    parser.add_argument("--data-dir", default=Path("data"), type=Path)
    parser.add_argument("--results-root", default=Path("results"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/forward_monitor"), type=Path)
    parser.add_argument(
        "--selection-summaries",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit selection_summary.json paths; order determines lead then shadow and can point to multiple candidates from the same thesis.",
    )
    parser.add_argument(
        "--theses",
        nargs="+",
        choices=tuple(config.RESEARCH_THESIS_SETTINGS.keys()),
        default=None,
        help="Optional override for thesis ordering; when omitted, the rebuild winner is treated as lead.",
    )
    parser.add_argument("--history-months", default=6, type=int)
    parser.add_argument(
        "--skip-package-export",
        action="store_true",
        help="Skip portfolio package side effects; useful for manual forward-trial monitors.",
    )
    return parser.parse_args()


def month_after(month: str) -> str:
    return str(pd.Period(month, freq="M") + 1)


def month_before(month: str) -> str:
    return str(pd.Period(month, freq="M") - 1)


def month_plus(month: str, months: int) -> str:
    return str(pd.Period(month, freq="M") + months)


def month_label(month: str) -> str:
    try:
        return pd.Period(month, freq="M").to_timestamp().strftime("%b %Y")
    except Exception:
        return str(month)


def forward_trial_decision_windows(start_month: str) -> dict[str, str]:
    earliest_review = month_plus(start_month, 6)
    preferred_review = month_plus(start_month, 12)
    return {
        "earliest_review_month": earliest_review,
        "earliest_review_label": month_label(earliest_review),
        "preferred_review_month": preferred_review,
        "preferred_review_label": month_label(preferred_review),
    }


def selected_strategy_summary(thesis: dict[str, Any]) -> str:
    params = thesis.get("params") or {}
    universe = (thesis.get("label") or thesis.get("name") or "Lead thesis").split("[", 1)[0].strip()
    parts = [universe]
    lookback = params.get("l")
    skip = params.get("skip")
    if lookback is not None and skip is not None:
        try:
            parts.append(f"{int(lookback)}-{int(skip)} price momentum")
        except (TypeError, ValueError):
            parts.append(f"{lookback}-{skip} price momentum")
    else:
        parts.append("price momentum")
    top_n = params.get("top_n")
    if top_n is not None:
        try:
            parts.append(f"Top {int(top_n)}")
        except (TypeError, ValueError):
            parts.append(f"Top {top_n}")
    if params.get("trend_filter") and params.get("ma_window") is not None:
        try:
            parts.append(f"trend filter (MA{int(params['ma_window'])})")
        except (TypeError, ValueError):
            parts.append(f"trend filter (MA{params['ma_window']})")
    else:
        strategy_id = params.get("strategy_id")
        if strategy_id and strategy_id != "baseline":
            parts.append(str(strategy_id).replace("_", " "))
    return " | ".join(parts)


def thesis_results_dir(results_root: Path, thesis_name: str) -> Path:
    if thesis_name == config.DEFAULT_RESEARCH_THESIS:
        return results_root
    return results_root / thesis_name


def resolve_authoritative_selection_path(results_root: Path, thesis_name: str) -> Path:
    override = config.AUTHORITATIVE_SELECTION_SUMMARY_OVERRIDES.get(thesis_name)
    if override and _path_within_root(results_root):
        override_path = Path(override)
        if not override_path.is_absolute():
            override_path = (ROOT / override_path).resolve()
        if override_path.is_dir():
            override_path = override_path / "selection_summary.json"
        if override_path.exists():
            return override_path
    return thesis_results_dir(results_root, thesis_name) / "selection_summary.json"


def _path_within_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def candidate_short_label(candidate_id: str | None, params: dict[str, Any]) -> str:
    parts: list[str] = []
    top_n = params.get("top_n")
    if top_n is not None:
        try:
            parts.append(f"n{int(top_n)}")
        except (TypeError, ValueError):
            parts.append(f"n{top_n}")
    strategy_id = params.get("strategy_id")
    if strategy_id and strategy_id != "baseline":
        parts.append(str(strategy_id))
    if params.get("trend_filter") and params.get("ma_window"):
        try:
            parts.append(f"ma{int(params['ma_window'])}")
        except (TypeError, ValueError):
            parts.append(f"ma{params['ma_window']}")
    if parts:
        return " / ".join(parts)
    return str(candidate_id or "locked_candidate")


def load_frozen_selection_from_path(selection_path: Path) -> dict[str, Any]:
    resolved = Path(selection_path)
    if resolved.is_dir():
        resolved = resolved / "selection_summary.json"
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing frozen selection summary: {resolved}")
    selection = load_json(resolved)
    if selection.get("selection_status") != "selected":
        raise ValueError(f"Selection summary does not have a selected locked candidate: {resolved}")
    locked = selection.get("locked_candidate")
    if not isinstance(locked, dict) or not isinstance(locked.get("params"), dict):
        raise ValueError(f"Selection summary is missing locked_candidate.params: {resolved}")
    thesis_payload = selection.get("thesis")
    if not isinstance(thesis_payload, dict):
        thesis_name = resolved.parent.name
        thesis_payload = build_thesis(thesis_name).manifest_metadata()
    thesis_name = str(thesis_payload.get("name") or resolved.parent.name)
    params = dict(locked["params"])
    candidate_id = locked.get("candidate_id")
    return {
        "name": thesis_name,
        "results_dir": str(resolved.parent),
        "selection_path": str(resolved),
        "thesis": thesis_payload,
        "params": params,
        "selection_mode": selection.get("mode"),
        "candidate_id": candidate_id,
        "candidate_label": candidate_short_label(candidate_id, params),
        "selection_source_kind": "explicit_path",
    }


def load_frozen_selection(results_root: Path, thesis_name: str) -> dict[str, Any]:
    selection_path = resolve_authoritative_selection_path(results_root, thesis_name)
    loaded = load_frozen_selection_from_path(selection_path)
    loaded["name"] = thesis_name
    loaded["results_dir"] = str(thesis_results_dir(results_root, thesis_name))
    loaded["selection_source_kind"] = "authoritative_default"
    return loaded


def build_security_lookup(data_dir: Path) -> pd.DataFrame:
    security_master_path = data_dir / "security_master.parquet"
    if security_master_path.exists():
        master = pd.read_parquet(
            security_master_path,
            columns=[
                "security_id",
                "eodhd_symbol",
                "ticker_local",
                "company_name",
                "country_code",
                "exchange_group",
                "currency",
            ],
        ).drop_duplicates(subset=["security_id"])
    else:
        universe = pd.read_parquet(
            data_dir / "universe_pti.parquet",
            columns=["security_id", "country_code", "exchange_group", "currency"],
        ).drop_duplicates(subset=["security_id"])
        universe["eodhd_symbol"] = universe["security_id"]
        universe["ticker_local"] = universe["security_id"].str.split(".").str[0]
        universe["company_name"] = universe["security_id"]
        master = universe[
            [
                "security_id",
                "eodhd_symbol",
                "ticker_local",
                "company_name",
                "country_code",
                "exchange_group",
                "currency",
            ]
        ]
    return master.set_index("security_id")


def build_universe_lookup(dataset: ResearchDataset) -> pd.DataFrame:
    frame = dataset.universe[
        [
            "rebalance_month",
            "security_id",
            "anchor_trade_date",
            "asof_trade_date",
            "next_execution_date",
            "close_raw_sek",
            "median_daily_value_60d_sek",
            "market_cap_sek",
        ]
    ].drop_duplicates(subset=["rebalance_month", "security_id"])
    return frame.set_index(["rebalance_month", "security_id"])


def holding_to_signal_map(dataset: ResearchDataset) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for signal_month, holding_month in zip(dataset.signal_months, dataset.holding_months, strict=True):
        if holding_month is not None:
            mapping[holding_month] = signal_month
    return mapping


def scalar_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def params_label(params: dict[str, int]) -> str:
    strategy_id = params.get("strategy_id", "baseline")
    extras = []
    if params.get("band_buffer") is not None:
        extras.append(f"band={params.get('band_buffer')}")
    if params.get("min_hold_months") is not None:
        extras.append(f"min_hold={params.get('min_hold_months')}")
    if params.get("weighting") and params.get("weighting") != "equal":
        extras.append(f"weight={params.get('weighting')}")
    if params.get("trend_filter"):
        extras.append(f"trend_ma={params.get('ma_window', 'n/a')}")
    if params.get("liquidity_min_mdv"):
        extras.append(f"liq_mdv>={params.get('liquidity_min_mdv')}")
    extra_text = f" / {', '.join(extras)}" if extras else ""
    return (
        f"L={params.get('l')} / skip={params.get('skip')} / top_n={params.get('top_n')} / strat={strategy_id}"
        f"{extra_text}"
    )


def score_for_position(dataset: ResearchDataset, params: dict[str, int], signal_month: str, security_id: str) -> float | None:
    signal_index = dataset.month_index[signal_month]
    security_index = dataset.security_index[security_id]
    value = dataset.scores[(int(params["l"]), int(params["skip"]))][signal_index, security_index]
    return float(value) if pd.notna(value) else None


def one_way_cost_bps(
    dataset: ResearchDataset,
    params: dict[str, int],
    signal_month: str,
    security_id: str,
    execution_model: str,
    fx_scenario: str,
) -> float | None:
    signal_index = dataset.month_index[signal_month]
    security_index = dataset.security_index[security_id]
    top_n = int(params["top_n"])
    cost_fraction = dataset.cost_fractions[(top_n, execution_model, fx_scenario)][signal_index, security_index]
    if pd.isna(cost_fraction):
        return None
    return float(cost_fraction) * 10_000.0 * float(top_n)


def build_position_rows(
    *,
    dataset: ResearchDataset,
    params: dict[str, int],
    signal_month: str,
    security_ids: Sequence[str],
    selected_weights: Sequence[float] | None,
    security_lookup: pd.DataFrame,
    universe_lookup: pd.DataFrame,
    execution_model: str,
    fx_scenario: str,
    current_entries: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, security_id in enumerate(security_ids, start=1):
        weight = (
            float(selected_weights[position - 1])
            if selected_weights is not None and position - 1 < len(selected_weights)
            else 1.0 / float(max(1, int(params["top_n"])))
        )
        order_notional_sek = float(config.SIM_CAPITAL_SEK) * weight
        security_meta = security_lookup.loc[security_id] if security_id in security_lookup.index else None
        universe_key = (signal_month, security_id)
        universe_meta = universe_lookup.loc[universe_key] if universe_key in universe_lookup.index else None
        anchor_trade_date = scalar_or_none(universe_meta["anchor_trade_date"]) if universe_meta is not None else None
        asof_trade_date = scalar_or_none(universe_meta["asof_trade_date"]) if universe_meta is not None else None
        adv_value = scalar_or_none(universe_meta["median_daily_value_60d_sek"]) if universe_meta is not None else None
        adv_participation_ratio = (
            float(order_notional_sek) / float(adv_value)
            if adv_value is not None and float(adv_value) > 0.0
            else None
        )
        if asof_trade_date is None:
            data_status = "missing_asof"
        elif anchor_trade_date is None or asof_trade_date == anchor_trade_date:
            data_status = "fresh"
        else:
            data_status = "stale_asof"
        rows.append(
            {
                "position": position,
                "security_id": security_id,
                "ticker_local": scalar_or_none(security_meta["ticker_local"]) if security_meta is not None else None,
                "eodhd_symbol": scalar_or_none(security_meta["eodhd_symbol"]) if security_meta is not None else None,
                "company_name": scalar_or_none(security_meta["company_name"]) if security_meta is not None else None,
                "country_code": scalar_or_none(security_meta["country_code"]) if security_meta is not None else None,
                "exchange_group": scalar_or_none(security_meta["exchange_group"]) if security_meta is not None else None,
                "currency": scalar_or_none(security_meta["currency"]) if security_meta is not None else None,
                "signal_score": score_for_position(dataset, params, signal_month, security_id),
                "anchor_trade_date": anchor_trade_date,
                "asof_trade_date": asof_trade_date,
                "anchor_close_sek": scalar_or_none(universe_meta["close_raw_sek"]) if universe_meta is not None else None,
                "median_daily_value_60d_sek": adv_value,
                "market_cap_sek": scalar_or_none(universe_meta["market_cap_sek"]) if universe_meta is not None else None,
                "order_notional_sek": order_notional_sek,
                "target_weight": weight,
                "adv_participation_ratio": adv_participation_ratio,
                "data_status": data_status,
                "one_way_cost_bps": one_way_cost_bps(
                    dataset,
                    params,
                    signal_month,
                    security_id,
                    execution_model,
                    fx_scenario,
                ),
                "action": "enter" if security_id in current_entries else "hold",
            }
        )
    return rows


def overlap_summary(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    left_set = set(left)
    right_set = set(right)
    overlap = sorted(left_set & right_set)
    union = left_set | right_set
    denominator = min(len(left_set), len(right_set))
    return {
        "overlap_security_ids": overlap,
        "overlap_count": len(overlap),
        "overlap_share_of_smaller_book": (len(overlap) / float(denominator)) if denominator else 0.0,
        "jaccard": (len(overlap) / float(len(union))) if union else 0.0,
    }


def summarize_pre_trade_audit(
    current_pick: dict[str, Any],
    params: dict[str, int],
    *,
    previous_record: dict[str, Any] | None,
) -> dict[str, Any]:
    positions = current_pick["positions"]
    top_n = int(params["top_n"])
    top_country_code = None
    top_country_weight = 0.0
    if current_pick.get("weight_by_country"):
        top_country_code, top_country_weight = max(
            current_pick["weight_by_country"].items(),
            key=lambda item: float(item[1]),
        )
    participation_values = [float(position["adv_participation_ratio"]) for position in positions if position.get("adv_participation_ratio") is not None]
    max_participation_ratio = max(participation_values) if participation_values else None
    avg_participation_ratio = (sum(participation_values) / float(len(participation_values))) if participation_values else None
    one_way_cost_values = [float(position["one_way_cost_bps"]) for position in positions if position.get("one_way_cost_bps") is not None]
    avg_one_way_cost_bps = (sum(one_way_cost_values) / float(len(one_way_cost_values))) if one_way_cost_values else None
    entry_portfolio_cost_bps = (
        sum(float(position["one_way_cost_bps"]) for position in positions if position.get("action") == "enter" and position.get("one_way_cost_bps") is not None)
        / float(max(1, top_n))
    )
    previous_cost_map = {}
    if previous_record is not None:
        previous_cost_map = {
            position["security_id"]: float(position["one_way_cost_bps"])
            for position in previous_record.get("positions", [])
            if position.get("one_way_cost_bps") is not None
        }
    exit_portfolio_cost_bps = sum(previous_cost_map.get(security_id, 0.0) for security_id in current_pick["exit_security_ids"]) / float(max(1, top_n))
    stale_names = [
        position["ticker_local"] or position["security_id"]
        for position in positions
        if position.get("data_status") == "stale_asof"
    ]
    missing_asof_names = [
        position["ticker_local"] or position["security_id"]
        for position in positions
        if position.get("data_status") == "missing_asof"
    ]
    missing_adv_names = [
        position["ticker_local"] or position["security_id"]
        for position in positions
        if position.get("adv_participation_ratio") is None
    ]
    turnover_name_fraction = len(current_pick["entry_security_ids"]) / float(max(1, top_n))
    flags: list[str] = []
    if top_country_code is not None and top_country_weight >= 0.60:
        flags.append(f"Country concentration is high in {top_country_code} at {top_country_weight * 100.0:.1f}% weight.")
    if max_participation_ratio is not None and max_participation_ratio >= 0.005:
        flags.append(
            f"Max single-name ADV usage is {max_participation_ratio * 100.0:.2f}% of 60-day median value, which is worth watching."
        )
    if stale_names:
        flags.append(f"Stale as-of data appears on {', '.join(stale_names)}.")
    if missing_asof_names:
        flags.append(f"Missing as-of timestamps appear on {', '.join(missing_asof_names)}.")
    if missing_adv_names:
        flags.append(f"Missing ADV values appear on {', '.join(missing_adv_names)}.")
    if turnover_name_fraction >= 0.40:
        flags.append(f"This rebalance changes {turnover_name_fraction * 100.0:.1f}% of the book by name count.")
    if not flags:
        flags.append("No immediate pre-trade audit red flags were triggered by the frozen current book.")
    return {
        "top_country_code": top_country_code,
        "top_country_weight": top_country_weight,
        "max_adv_participation_ratio": max_participation_ratio,
        "avg_adv_participation_ratio": avg_participation_ratio,
        "avg_one_way_cost_bps": avg_one_way_cost_bps,
        "entry_portfolio_cost_bps": entry_portfolio_cost_bps,
        "exit_portfolio_cost_bps": exit_portfolio_cost_bps,
        "estimated_rebalance_cost_bps": entry_portfolio_cost_bps + exit_portfolio_cost_bps,
        "entry_name_count": len(current_pick["entry_security_ids"]),
        "exit_name_count": len(current_pick["exit_security_ids"]),
        "turnover_name_fraction": turnover_name_fraction,
        "stale_name_count": len(stale_names),
        "missing_asof_count": len(missing_asof_names),
        "missing_adv_count": len(missing_adv_names),
        "stale_names": stale_names,
        "missing_asof_names": missing_asof_names,
        "missing_adv_names": missing_adv_names,
        "flags": flags,
    }


def build_history_records(
    *,
    dataset: ResearchDataset,
    selection: dict[str, Any],
    start_month: str,
    end_month: str,
    security_lookup: pd.DataFrame,
    universe_lookup: pd.DataFrame,
    holding_signal_map: dict[str, str],
    history_months: int,
    exclude_holding_month: str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    thesis = build_thesis(selection["name"])
    simulation = dataset.simulate_window(
        params=selection["params"],
        universe_variant=PRIMARY_TRACK["universe_variant"],
        execution_model=PRIMARY_TRACK["execution_model"],
        fx_scenario=PRIMARY_TRACK["fx_scenario"],
        start_month=start_month,
        end_month=end_month,
        excluded_countries=thesis.excluded_countries,
        collect_details=True,
    )
    records: list[dict[str, Any]] = []
    previous_selection: set[str] = set()
    for detail in simulation.details or []:
        if exclude_holding_month and detail["holding_month"] == exclude_holding_month:
            continue
        signal_month = holding_signal_map.get(detail["holding_month"])
        if signal_month is None:
            continue
        selected_security_ids = list(detail["selected_security_ids"])
        current_selection = set(selected_security_ids)
        entry_security_ids = {security_id for security_id in selected_security_ids if security_id not in previous_selection}
        exit_security_ids = sorted(previous_selection - current_selection)
        positions = build_position_rows(
            dataset=dataset,
            params=selection["params"],
            signal_month=signal_month,
            security_ids=selected_security_ids,
            selected_weights=detail.get("selected_weights"),
            security_lookup=security_lookup,
            universe_lookup=universe_lookup,
            execution_model=PRIMARY_TRACK["execution_model"],
            fx_scenario=PRIMARY_TRACK["fx_scenario"],
            current_entries=entry_security_ids,
        )
        anchor_trade_date = None
        next_execution_date = None
        if selected_security_ids:
            anchor_trade_date = scalar_or_none(universe_lookup.loc[(signal_month, selected_security_ids[0]), "anchor_trade_date"])
            next_execution_date = scalar_or_none(universe_lookup.loc[(signal_month, selected_security_ids[0]), "next_execution_date"])
        records.append(
            {
                "holding_month": detail["holding_month"],
                "signal_month": signal_month,
                "anchor_trade_date": anchor_trade_date,
                "next_execution_date": next_execution_date,
                "gross_return": float(detail["gross_return"]),
                "trade_cost": float(detail["trade_cost"]),
                "net_return": float(detail["net_return"]),
                "selected_security_ids": selected_security_ids,
                "entry_security_ids": sorted(entry_security_ids),
                "exit_security_ids": exit_security_ids,
                "selected_by_country": detail["selected_by_country"],
                "weight_by_country": detail["weight_by_country"],
                "positions": positions,
            }
        )
        previous_selection = current_selection
    if history_months > 0:
        records = records[-history_months:]
    latest_selection = set(records[-1]["selected_security_ids"]) if records else set()
    return records, latest_selection


def build_current_pick(
    *,
    dataset: ResearchDataset,
    selection: dict[str, Any],
    security_lookup: pd.DataFrame,
    universe_lookup: pd.DataFrame,
    previous_selection: set[str],
) -> dict[str, Any]:
    thesis = build_thesis(selection["name"])
    signal_month = dataset.signal_months[-1]
    snapshot = dataset.selection_snapshot(
        params=selection["params"],
        signal_month=signal_month,
        universe_variant=PRIMARY_TRACK["universe_variant"],
        execution_model=PRIMARY_TRACK["execution_model"],
        fx_scenario=PRIMARY_TRACK["fx_scenario"],
        excluded_countries=thesis.excluded_countries,
    )
    selected_security_ids = list(snapshot["selected_security_ids"])
    current_selection = set(selected_security_ids)
    entry_security_ids = {security_id for security_id in selected_security_ids if security_id not in previous_selection}
    positions = build_position_rows(
        dataset=dataset,
        params=selection["params"],
        signal_month=signal_month,
        security_ids=selected_security_ids,
        selected_weights=snapshot.get("selected_weights"),
        security_lookup=security_lookup,
        universe_lookup=universe_lookup,
        execution_model=PRIMARY_TRACK["execution_model"],
        fx_scenario=PRIMARY_TRACK["fx_scenario"],
        current_entries=entry_security_ids,
    )
    return {
        "signal_month": snapshot["signal_month"],
        "holding_month": snapshot["holding_month"],
        "anchor_trade_date": scalar_or_none(snapshot["anchor_trade_date"]),
        "next_execution_date": scalar_or_none(snapshot["next_execution_date"]),
        "eligible_count": int(snapshot["eligible_count"]),
        "selected_count": int(snapshot["selected_count"]),
        "eligible_by_country": snapshot["eligible_by_country"],
        "selected_by_country": snapshot["selected_by_country"],
        "weight_by_country": snapshot["weight_by_country"],
        "selected_security_ids": selected_security_ids,
        "entry_security_ids": sorted(entry_security_ids),
        "exit_security_ids": sorted(previous_selection - current_selection),
        "positions": positions,
        "execution_model": snapshot["execution_model"],
        "fx_scenario": snapshot["fx_scenario"],
        "universe_variant": snapshot["universe_variant"],
    }


def _safe_cost_fraction(value: float | np.floating[Any]) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _performance_summary(monthly_returns: Sequence[float]) -> dict[str, Any]:
    return {
        "months": len(monthly_returns),
        "net_sharpe": annualized_sharpe(monthly_returns),
        "total_return": total_return(monthly_returns),
        "max_drawdown": max_drawdown(monthly_returns),
    }


def simulate_primary_track_with_contributions(
    *,
    dataset: ResearchDataset,
    selection: dict[str, Any],
    start_month: str,
    end_month: str,
    security_lookup: pd.DataFrame,
    excluded_security_ids: Sequence[str] = (),
) -> dict[str, Any]:
    thesis = build_thesis(selection["name"])
    params = selection["params"]
    lookback = int(params["l"])
    skip = int(params["skip"])
    top_n = int(params["top_n"])
    execution_model = PRIMARY_TRACK["execution_model"]
    fx_scenario = PRIMARY_TRACK["fx_scenario"]
    signal_indices = dataset._window_signal_indices(start_month, end_month)
    base_mask = dataset.variant_mask(
        PRIMARY_TRACK["universe_variant"],
        excluded_countries=thesis.excluded_countries,
    )
    if excluded_security_ids:
        allowed_security_mask = np.ones(len(dataset.security_ids), dtype=bool)
        for security_id in excluded_security_ids:
            security_index = dataset.security_index.get(security_id)
            if security_index is not None:
                allowed_security_mask[security_index] = False
        base_mask = base_mask & allowed_security_mask[None, :]
    rankable = (
        base_mask
        & dataset.asof_matches_anchor
        & dataset.capacity_masks[top_n]
        & dataset.entry_available[execution_model]
        & np.isfinite(dataset.holding_returns[execution_model])
        & np.isfinite(dataset.scores[(lookback, skip)])
    )
    one_way_cost = dataset.cost_fractions[(top_n, execution_model, fx_scenario)]
    security_totals: dict[str, dict[str, Any]] = {}
    monthly_breakdown: list[dict[str, Any]] = []
    previous_selection: set[int] = set()

    for signal_index in signal_indices:
        holding_month = dataset.holding_months[signal_index]
        if holding_month is None:
            continue
        selected_indices: list[int] = []
        for security_index in dataset.rank_orders[(lookback, skip)][signal_index]:
            if rankable[signal_index, security_index]:
                selected_indices.append(int(security_index))
                if len(selected_indices) >= top_n:
                    break
        current_selection = set(selected_indices)
        entries = current_selection - previous_selection
        exits = previous_selection - current_selection
        month_contributions: dict[str, float] = {}
        month_selected_security_ids = [dataset.security_ids[index] for index in selected_indices]

        for security_index in selected_indices:
            security_id = dataset.security_ids[security_index]
            security_meta = security_lookup.loc[security_id] if security_id in security_lookup.index else None
            gross_contribution = float(dataset.holding_returns[execution_model][signal_index, security_index]) / float(top_n)
            entry_cost = _safe_cost_fraction(one_way_cost[signal_index, security_index]) if security_index in entries else 0.0
            net_contribution = gross_contribution - entry_cost
            row = security_totals.setdefault(
                security_id,
                {
                    "security_id": security_id,
                    "ticker_local": scalar_or_none(security_meta["ticker_local"]) if security_meta is not None else None,
                    "company_name": scalar_or_none(security_meta["company_name"]) if security_meta is not None else None,
                    "country_code": scalar_or_none(security_meta["country_code"]) if security_meta is not None else None,
                    "months_selected": 0,
                    "entry_count": 0,
                    "exit_count": 0,
                    "cumulative_gross_contribution": 0.0,
                    "cumulative_net_contribution": 0.0,
                    "cumulative_trade_cost": 0.0,
                },
            )
            row["months_selected"] += 1
            if security_index in entries:
                row["entry_count"] += 1
            row["cumulative_gross_contribution"] += gross_contribution
            row["cumulative_net_contribution"] += net_contribution
            row["cumulative_trade_cost"] += entry_cost
            month_contributions[security_id] = month_contributions.get(security_id, 0.0) + net_contribution

        for security_index in exits:
            security_id = dataset.security_ids[security_index]
            security_meta = security_lookup.loc[security_id] if security_id in security_lookup.index else None
            exit_cost = _safe_cost_fraction(one_way_cost[signal_index, security_index])
            row = security_totals.setdefault(
                security_id,
                {
                    "security_id": security_id,
                    "ticker_local": scalar_or_none(security_meta["ticker_local"]) if security_meta is not None else None,
                    "company_name": scalar_or_none(security_meta["company_name"]) if security_meta is not None else None,
                    "country_code": scalar_or_none(security_meta["country_code"]) if security_meta is not None else None,
                    "months_selected": 0,
                    "entry_count": 0,
                    "exit_count": 0,
                    "cumulative_gross_contribution": 0.0,
                    "cumulative_net_contribution": 0.0,
                    "cumulative_trade_cost": 0.0,
                },
            )
            row["exit_count"] += 1
            row["cumulative_net_contribution"] -= exit_cost
            row["cumulative_trade_cost"] += exit_cost
            month_contributions[security_id] = month_contributions.get(security_id, 0.0) - exit_cost

        monthly_breakdown.append(
            {
                "holding_month": holding_month,
                "selected_security_ids": month_selected_security_ids,
                "entry_security_ids": sorted(dataset.security_ids[index] for index in entries),
                "exit_security_ids": sorted(dataset.security_ids[index] for index in exits),
                "net_contribution_by_security": month_contributions,
                "net_return": sum(month_contributions.values()),
            }
        )
        previous_selection = current_selection

    monthly_returns = [float(row["net_return"]) for row in monthly_breakdown]
    return {
        "months": [row["holding_month"] for row in monthly_breakdown],
        "monthly_returns": monthly_returns,
        "performance": _performance_summary(monthly_returns),
        "security_totals": security_totals,
    }


def summarize_name_dependence(
    *,
    dataset: ResearchDataset,
    selection: dict[str, Any],
    security_lookup: pd.DataFrame,
) -> dict[str, Any]:
    base = simulate_primary_track_with_contributions(
        dataset=dataset,
        selection=selection,
        start_month=config.OOS_START,
        end_month=config.OOS_END,
        security_lookup=security_lookup,
    )
    contributions = sorted(
        base["security_totals"].values(),
        key=lambda row: float(row["cumulative_net_contribution"]),
        reverse=True,
    )
    positive_total = sum(max(float(row["cumulative_net_contribution"]), 0.0) for row in contributions)
    absolute_total = sum(abs(float(row["cumulative_net_contribution"])) for row in contributions)
    positive_names = sum(1 for row in contributions if float(row["cumulative_net_contribution"]) > 0.0)
    negative_names = sum(1 for row in contributions if float(row["cumulative_net_contribution"]) < 0.0)

    contributor_rows: list[dict[str, Any]] = []
    for row in contributions:
        net_contribution = float(row["cumulative_net_contribution"])
        contributor_rows.append(
            {
                **row,
                "share_of_positive_contribution": (
                    max(net_contribution, 0.0) / positive_total if positive_total > 0.0 else None
                ),
                "share_of_absolute_contribution": (
                    abs(net_contribution) / absolute_total if absolute_total > 0.0 else None
                ),
                "avg_net_contribution_when_selected": (
                    net_contribution / float(max(1, int(row["months_selected"])))
                ),
            }
        )

    top_one_ids = [contributor_rows[0]["security_id"]] if contributor_rows else []
    top_three_ids = [row["security_id"] for row in contributor_rows[:3]]
    without_top_one = (
        simulate_primary_track_with_contributions(
            dataset=dataset,
            selection=selection,
            start_month=config.OOS_START,
            end_month=config.OOS_END,
            security_lookup=security_lookup,
            excluded_security_ids=top_one_ids,
        )["performance"]
        if top_one_ids
        else None
    )
    without_top_three = (
        simulate_primary_track_with_contributions(
            dataset=dataset,
            selection=selection,
            start_month=config.OOS_START,
            end_month=config.OOS_END,
            security_lookup=security_lookup,
            excluded_security_ids=top_three_ids,
        )["performance"]
        if top_three_ids
        else None
    )
    base_performance = base["performance"]
    flags: list[str] = []
    top_name_positive_share = contributor_rows[0]["share_of_positive_contribution"] if contributor_rows else None
    top_three_positive_share = (
        sum(row["share_of_positive_contribution"] or 0.0 for row in contributor_rows[:3]) if contributor_rows else None
    )
    if top_name_positive_share is not None and float(top_name_positive_share) >= 0.35:
        flags.append(
            f"The top single name supplied {top_name_positive_share * 100.0:.1f}% of positive holdout contribution."
        )
    if top_three_positive_share is not None and float(top_three_positive_share) >= 0.65:
        flags.append(
            f"The top three names supplied {top_three_positive_share * 100.0:.1f}% of positive holdout contribution."
        )
    if without_top_one is not None and base_performance["net_sharpe"] > 0.0:
        retained = float(without_top_one["net_sharpe"]) / float(base_performance["net_sharpe"])
        if retained < 0.70:
            flags.append(
                f"Removing the top contributor would cut holdout Sharpe retention to {retained * 100.0:.1f}% of base."
            )
    if without_top_three is not None and base_performance["net_sharpe"] > 0.0:
        retained = float(without_top_three["net_sharpe"]) / float(base_performance["net_sharpe"])
        if retained < 0.50:
            flags.append(
                f"Removing the top three contributors would cut holdout Sharpe retention to {retained * 100.0:.1f}% of base."
            )
    if not flags:
        flags.append("The holdout contribution profile looks reasonably broad for this equal-weight top-N implementation.")

    return {
        "window_start": config.OOS_START,
        "window_end": config.OOS_END,
        "performance": base_performance,
        "unique_names_held": len(contributor_rows),
        "positive_name_count": positive_names,
        "negative_name_count": negative_names,
        "top_name_share_of_positive_contribution": top_name_positive_share,
        "top_three_share_of_positive_contribution": top_three_positive_share,
        "top_name_share_of_absolute_contribution": (
            contributor_rows[0]["share_of_absolute_contribution"] if contributor_rows else None
        ),
        "top_three_share_of_absolute_contribution": (
            sum(row["share_of_absolute_contribution"] or 0.0 for row in contributor_rows[:3]) if contributor_rows else None
        ),
        "without_top_name": {
            "excluded_security_ids": top_one_ids,
            "performance": without_top_one,
            "sharpe_retention_vs_base": (
                float(without_top_one["net_sharpe"]) / float(base_performance["net_sharpe"])
                if without_top_one is not None and float(base_performance["net_sharpe"]) != 0.0
                else None
            ),
            "total_return_retention_vs_base": (
                float(without_top_one["total_return"]) / float(base_performance["total_return"])
                if without_top_one is not None and float(base_performance["total_return"]) != 0.0
                else None
            ),
        },
        "without_top_three": {
            "excluded_security_ids": top_three_ids,
            "performance": without_top_three,
            "sharpe_retention_vs_base": (
                float(without_top_three["net_sharpe"]) / float(base_performance["net_sharpe"])
                if without_top_three is not None and float(base_performance["net_sharpe"]) != 0.0
                else None
            ),
            "total_return_retention_vs_base": (
                float(without_top_three["total_return"]) / float(base_performance["total_return"])
                if without_top_three is not None and float(base_performance["total_return"]) != 0.0
                else None
            ),
        },
        "top_contributors": contributor_rows[:10],
        "flags": flags,
    }


def build_monitor_payload(
    *,
    data_dir: Path,
    results_root: Path,
    theses: Sequence[str],
    history_months: int,
    selection_summaries: Sequence[Path] | None = None,
) -> dict[str, Any]:
    dataset = ResearchDataset(data_dir)
    security_lookup = build_security_lookup(data_dir)
    universe_lookup = build_universe_lookup(dataset)
    holding_signal = holding_to_signal_map(dataset)
    monitor_start = month_after(config.OOS_END)
    latest_available_holding_month = next(
        (holding_month for holding_month in reversed(dataset.holding_months) if holding_month is not None),
        None,
    )
    if latest_available_holding_month is None:
        raise ValueError("The research dataset does not expose any holding months.")

    pending_holding_month = dataset.holding_months[-1]
    history_end_month = latest_available_holding_month
    if pending_holding_month and pd.Period(pending_holding_month, freq="M") > pd.Period(monitor_start, freq="M"):
        history_end_month = month_before(pending_holding_month)

    explicit_selection_paths = list(selection_summaries or [])
    loaded_selections = (
        [load_frozen_selection_from_path(selection_path) for selection_path in explicit_selection_paths]
        if explicit_selection_paths
        else [load_frozen_selection(results_root, thesis_name) for thesis_name in theses]
    )
    if not loaded_selections:
        raise ValueError("Need at least one frozen selection to build the forward monitor.")
    duplicate_name_counts = Counter(selection["name"] for selection in loaded_selections)
    monitor_mode = "manual_forward_trial" if explicit_selection_paths else "authoritative_reference_monitor"

    thesis_payloads: list[dict[str, Any]] = []
    for index, selection in enumerate(loaded_selections):
        selection["role"] = "lead" if index == 0 else "shadow"
        selection["role_display"] = "Lead" if index == 0 else "Shadow"
        selection["label"] = str(selection["thesis"]["label"])
        if explicit_selection_paths or duplicate_name_counts[selection["name"]] > 1:
            selection["label"] = f"{selection['label']} [{selection['candidate_label']}]"
        history, previous_selection = build_history_records(
            dataset=dataset,
            selection=selection,
            start_month=monitor_start,
            end_month=history_end_month,
            security_lookup=security_lookup,
            universe_lookup=universe_lookup,
            holding_signal_map=holding_signal,
            history_months=history_months,
            exclude_holding_month=pending_holding_month,
        )
        current_previous_selection = previous_selection
        current_previous_record = history[-1] if history else None
        current_pick = build_current_pick(
            dataset=dataset,
            selection=selection,
            security_lookup=security_lookup,
            universe_lookup=universe_lookup,
            previous_selection=current_previous_selection,
        )
        pre_trade_audit = summarize_pre_trade_audit(
            current_pick,
            selection["params"],
            previous_record=current_previous_record,
        )
        name_dependence = summarize_name_dependence(
            dataset=dataset,
            selection=selection,
            security_lookup=security_lookup,
        )
        thesis_payloads.append(
            {
                "name": selection["name"],
                "label": selection["label"],
                "base_label": selection["thesis"]["label"],
                "role": selection["role"],
                "role_display": selection["role_display"],
                "scope_note": selection["thesis"]["scope_note"],
                "results_dir": selection["results_dir"],
                "selection_path": selection["selection_path"],
                "selection_mode": selection["selection_mode"],
                "params": selection["params"],
                "params_label": params_label(selection["params"]),
                "candidate_id": selection.get("candidate_id"),
                "candidate_label": selection.get("candidate_label"),
                "selection_source_kind": selection.get("selection_source_kind"),
                "history": history,
                "current_pick": current_pick,
                "pre_trade_audit": pre_trade_audit,
                "name_dependence": name_dependence,
            }
        )

    lead = thesis_payloads[0]
    comparison = thesis_payloads[1] if len(thesis_payloads) > 1 else None
    current_overlap = overlap_summary(
        lead["current_pick"]["selected_security_ids"],
        comparison["current_pick"]["selected_security_ids"] if comparison else [],
    )
    history_overlap: list[dict[str, Any]] = []
    if comparison is not None:
        comparison_history_by_month = {row["holding_month"]: row for row in comparison["history"]}
        for row in lead["history"]:
            peer = comparison_history_by_month.get(row["holding_month"])
            if peer is None:
                continue
            overlap = overlap_summary(row["selected_security_ids"], peer["selected_security_ids"])
            history_overlap.append(
                {
                    "holding_month": row["holding_month"],
                    "lead_net_return": row["net_return"],
                    "comparison_net_return": peer["net_return"],
                    **overlap,
                }
            )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "monitor_mode": monitor_mode,
        "results_root": str(results_root),
        "monitoring_window": {
            "start_after_holdout": monitor_start,
            "latest_available_holding_month": latest_available_holding_month,
            "latest_completed_holding_month": history_end_month,
            "current_pick_month": lead["current_pick"]["holding_month"],
        },
        "lead_role": lead["role"],
        "lead_thesis": lead["name"],
        "authoritative_status": summarize_authoritative_status(results_root, lead["name"]),
        "theses": thesis_payloads,
        "current_overlap": current_overlap,
        "history_overlap": history_overlap,
        "artifacts": {
            "lead_results_dir": lead["results_dir"],
            "comparison_results_dir": comparison["results_dir"] if comparison else None,
        },
    }


def frozen_strategy_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    theses = payload["theses"]
    lead = theses[0]
    shadow = theses[1] if len(theses) > 1 else None
    authoritative = payload.get("authoritative_status", {})
    is_active = bool(authoritative.get("active_validated_strategy"))
    manual_trial = payload.get("monitor_mode") == "manual_forward_trial"
    status = "manual_forward_trial" if manual_trial else ("paper_trading_active" if is_active else "legacy_monthly_candidate_preserved_for_reference")
    governance_notes = [
        "This manifest freezes the active paper-trading lead/shadow definitions under the authoritative package.",
        "No parameter, thesis, or universe edits should be made without explicitly starting a new research cycle and replacing this manifest.",
    ]
    if manual_trial:
        governance_notes = [
            "This manifest freezes a manual forward-trial lead/shadow pair from explicit selection sources.",
            "It does not override the repo-level authoritative verdict; use it only for unchanged paper-trading observation.",
        ]
    elif not is_active:
        governance_notes = [
            "This manifest preserves the old monthly lead/shadow definitions for reference after cadence revalidation became authoritative.",
            "No parameter, thesis, or universe edits should be made without explicitly starting a new research cycle and replacing this manifest.",
        ]
    manifest: dict[str, Any] = {
        "generated_at_utc": payload["generated_at_utc"],
        "monitor_mode": payload.get("monitor_mode"),
        "status": status,
        "authoritative_validation_model": authoritative.get("authoritative_validation_model", "current_validation_model"),
        "authoritative_active_validated_strategy": is_active,
        "authoritative_verdict": authoritative.get("verdict"),
        "lead_strategy": {
            "thesis_name": lead["name"],
            "thesis_label": lead["label"],
            "candidate_id": lead.get("candidate_id"),
            "candidate_label": lead.get("candidate_label"),
            "selection_mode": lead["selection_mode"],
            "params": lead["params"],
            "params_label": lead["params_label"],
            "selection_path": lead["selection_path"],
            "results_dir": lead["results_dir"],
            "current_pick_month": lead["current_pick"]["holding_month"],
            "current_pick_security_ids": lead["current_pick"]["selected_security_ids"],
        },
        "governance": {
            "search_reopened": False,
            "paper_trading_only": manual_trial or is_active,
            "legacy_reference_only": (not manual_trial) and (not is_active),
            "notes": governance_notes,
        },
    }
    if shadow is not None:
        manifest["shadow_control"] = {
            "thesis_name": shadow["name"],
            "thesis_label": shadow["label"],
            "candidate_id": shadow.get("candidate_id"),
            "candidate_label": shadow.get("candidate_label"),
            "selection_mode": shadow["selection_mode"],
            "params": shadow["params"],
            "params_label": shadow["params_label"],
            "selection_path": shadow["selection_path"],
            "results_dir": shadow["results_dir"],
            "current_pick_month": shadow["current_pick"]["holding_month"],
            "current_pick_security_ids": shadow["current_pick"]["selected_security_ids"],
        }
    return manifest


def flatten_pick_rows(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for thesis in payload["theses"]:
        current = thesis["current_pick"]
        for position in current["positions"]:
            rows.append(
                {
                    "record_type": "current",
                    "holding_month": current["holding_month"],
                    "signal_month": current["signal_month"],
                    "thesis_name": thesis["name"],
                    "thesis_label": thesis["label"],
                    "role": thesis["role"],
                    "candidate_id": thesis.get("candidate_id"),
                    "candidate_label": thesis.get("candidate_label"),
                    "params_label": thesis["params_label"],
                    "anchor_trade_date": current.get("anchor_trade_date"),
                    "next_execution_date": current.get("next_execution_date"),
                    **position,
                }
            )
        for record in thesis["history"]:
            for position in record["positions"]:
                rows.append(
                    {
                        "record_type": "history",
                        "holding_month": record["holding_month"],
                        "signal_month": record["signal_month"],
                        "thesis_name": thesis["name"],
                        "thesis_label": thesis["label"],
                        "role": thesis["role"],
                        "candidate_id": thesis.get("candidate_id"),
                        "candidate_label": thesis.get("candidate_label"),
                        "params_label": thesis["params_label"],
                        "anchor_trade_date": record.get("anchor_trade_date"),
                        "next_execution_date": record.get("next_execution_date"),
                        "gross_return": record["gross_return"],
                        "trade_cost": record["trade_cost"],
                        "net_return": record["net_return"],
                        **position,
                    }
                )
    return pd.DataFrame(rows)


def flatten_name_dependence_rows(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for thesis in payload["theses"]:
        audit = thesis["name_dependence"]
        for rank, contributor in enumerate(audit["top_contributors"], start=1):
            rows.append(
                {
                    "thesis_name": thesis["name"],
                    "thesis_label": thesis["label"],
                    "role": thesis["role"],
                    "params_label": thesis["params_label"],
                    "holdout_window_start": audit["window_start"],
                    "holdout_window_end": audit["window_end"],
                    "rank": rank,
                    **contributor,
                }
            )
    return pd.DataFrame(rows)


def thesis_results_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def resolve_default_theses(results_root: Path) -> list[str]:
    candidate_theses = list(config.RESEARCH_THESIS_SETTINGS)
    rebuild_summary_path = results_root / "cadence_compare_rebuild" / "summary" / "cadence_comparison.json"
    rebuild_summary = load_json_if_exists(rebuild_summary_path)
    lead = None
    if rebuild_summary is not None:
        winner = rebuild_summary.get("winner") or {}
        lead = (winner.get("thesis") or {}).get("name")
    if not lead or lead not in candidate_theses:
        lead = config.DEFAULT_RESEARCH_THESIS
    shadow = next((name for name in candidate_theses if name != lead), None)
    ordered = [lead]
    if shadow:
        ordered.append(shadow)
    return ordered


def load_pair_selection_summary(pair: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pair:
        return None
    output_dir = pair.get("output_dir")
    if not output_dir:
        return None
    output_path = thesis_results_path(str(output_dir))
    if output_path is None:
        return None
    selection_path = output_path / "selection_summary.json"
    return load_json_if_exists(selection_path)


def load_cadence_comparison_summary(results_root: Path) -> dict[str, Any] | None:
    summary_paths = (
        results_root / "cadence_compare_legacy_entry_exit" / "summary" / "cadence_comparison.json",
        results_root / "cadence_compare" / "summary" / "cadence_comparison.json",
    )
    for summary_path in summary_paths:
        payload = load_json_if_exists(summary_path)
        if payload is not None:
            payload = dict(payload)
            payload["_summary_path"] = str(summary_path)
            return payload
    return None


def _relative_href(from_dir: Path, to_path: Path) -> str:
    try:
        return to_path.relative_to(from_dir).as_posix()
    except ValueError:
        return Path(os.path.relpath(to_path, from_dir)).as_posix()


def _pair_selection_summary_path(results_root: Path, thesis_name: str, cadence_id: str) -> Path | None:
    legacy_path = results_root / "_timing_legacy_cert" / thesis_name / cadence_id / "selection_summary.json"
    if legacy_path.exists():
        return legacy_path
    if thesis_name == config.DEFAULT_RESEARCH_THESIS:
        fallback = results_root / "selection_summary.json"
        if fallback.exists():
            return fallback
    return None


def build_pair_dashboard_html(pair: dict[str, Any], results_root: Path) -> str:
    thesis = pair.get("thesis", {})
    cadence = pair.get("cadence", {})
    certification = pair.get("certification", {})
    holdout = pair.get("holdout", {})
    phase4_gate = holdout.get("phase4_gate", {})

    thesis_name = thesis.get("name") or "n/a"
    thesis_label = thesis.get("label") or "n/a"
    cadence_label = cadence.get("cadence_label") or "n/a"
    cadence_id = cadence.get("cadence_id") or "n/a"
    selection_status = certification.get("selection_status", "n/a")
    pbo = certification.get("backtest_overfitting", {}).get("pbo")
    pbo_text = f"{float(pbo) * 100.0:.1f}%" if pbo is not None else "n/a"
    holdout_sharpe = phase4_gate.get("base_main_net_sharpe")
    holdout_sharpe_text = format_float(holdout_sharpe)

    pair_dir = Path(pair.get("output_dir", results_root))
    summary_path = results_root / "cadence_compare" / "summary" / "dashboard.html"
    summary_report_path = results_root / "cadence_compare" / "summary" / "cadence_comparison_report.html"
    package_path = results_root.parent / "portfolio" / "alpha_momentum_validation_package" / "dashboard.html"
    dossier_path = results_root.parent / "portfolio" / "alpha_momentum_validation_package" / "research_audit_dossier.html"
    full_report_path = (
        results_root.parent
        / "portfolio"
        / "alpha_momentum_validation_package"
        / "full_research_report.html"
    )
    selection_path = _pair_selection_summary_path(results_root, thesis_name, cadence_id)

    summary_href = _relative_href(pair_dir, summary_path)
    summary_report_href = _relative_href(pair_dir, summary_report_path)
    package_href = _relative_href(pair_dir, package_path)
    dossier_href = _relative_href(pair_dir, dossier_path)
    full_report_href = _relative_href(pair_dir, full_report_path)
    selection_href = _relative_href(pair_dir, selection_path) if selection_path else None

    selection_card = (
        f'<a class="card" href="{html.escape(selection_href)}">'
        f'<div class="label">Selection Summary</div>'
        f'<div class="value">Open selection summary</div>'
        f"</a>"
        if selection_href
        else '<div class="card"><div class="label">Selection Summary</div><div class="value">not available</div></div>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pair Dashboard</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:980px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin-top:16px; }}
    .card {{ display:block; text-decoration:none; color:inherit; background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:20px; padding:18px; }}
    .label {{ color:#6c5842; text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; }}
    .value {{ font-size:1.4rem; margin-top:6px; }}
    a {{ color:#b06b1d; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>{html.escape(thesis_label)} - {html.escape(cadence_label)}</h1>
      <p>Thesis: <strong>{html.escape(thesis_name)}</strong> / Cadence: <strong>{html.escape(cadence_id)}</strong></p>
      <div class="grid">
        <div class="card">
          <div class="label">Selection Status</div>
          <div class="value">{html.escape(str(selection_status))}</div>
        </div>
        <div class="card">
          <div class="label">PBO</div>
          <div class="value">{html.escape(pbo_text)}</div>
        </div>
        <div class="card">
          <div class="label">Holdout Sharpe</div>
          <div class="value">{html.escape(holdout_sharpe_text)}</div>
        </div>
      </div>
    </section>
    <section>
      <h2>Open Next</h2>
      <div class="grid">
        <a class="card" href="{html.escape(summary_href)}">
          <div class="label">Cadence Summary</div>
          <div class="value">Back to summary</div>
        </a>
        <a class="card" href="{html.escape(summary_report_href)}">
          <div class="label">Cadence Report</div>
          <div class="value">Open full comparison report</div>
        </a>
        <a class="card" href="{html.escape(package_href)}">
          <div class="label">Validation Package</div>
          <div class="value">Open package dashboard</div>
        </a>
        <a class="card" href="{html.escape(full_report_href)}">
          <div class="label">Full Research Results</div>
          <div class="value">Open consolidated report</div>
        </a>
        <a class="card" href="{html.escape(dossier_href)}">
          <div class="label">Research Dossier</div>
          <div class="value">Open audit narrative</div>
        </a>
        {selection_card}
      </div>
    </section>
  </main>
</body>
</html>
"""


def export_pair_dashboards_from_cadence_summary(results_root: Path) -> list[Path]:
    summary_path = results_root / "cadence_compare" / "summary" / "cadence_comparison.json"
    summary = load_json_if_exists(summary_path)
    if summary is None:
        return []
    written: list[Path] = []
    for pair in summary.get("pairs", []):
        output_dir = pair.get("output_dir")
        if not output_dir:
            continue
        pair_dir = Path(output_dir)
        pair_dir.mkdir(parents=True, exist_ok=True)
        html_text = build_pair_dashboard_html(pair, results_root)
        output_path = pair_dir / "dashboard.html"
        output_path.write_text(html_text, encoding="utf-8")
        written.append(output_path)
    return written


def build_full_research_report_html(results_root: Path, package_dir: Path) -> str:
    summary = load_cadence_comparison_summary(results_root)
    winner = summary.get("winner") if summary else None

    thesis = winner.get("thesis", {}) if winner else {}
    cadence = winner.get("cadence", {}) if winner else {}
    certification = winner.get("certification", {}) if winner else {}
    holdout = winner.get("holdout", {}) if winner else {}
    phase4_gate = holdout.get("phase4_gate", {})

    thesis_label = thesis.get("label") or "n/a"
    cadence_label = cadence.get("cadence_label") or "n/a"
    thesis_name = thesis.get("name") or "n/a"
    cadence_id = cadence.get("cadence_id") or "n/a"

    selection_status = certification.get("selection_status", "n/a")
    pbo = certification.get("backtest_overfitting", {}).get("pbo")
    pbo_text = f"{float(pbo) * 100.0:.1f}%" if pbo is not None else "n/a"
    holdout_sharpe = format_float(phase4_gate.get("base_main_net_sharpe"))

    summary_dashboard = package_dir / "cadence_compare_summary" / "dashboard.html"
    summary_report = package_dir / "cadence_compare_summary" / "cadence_comparison_report.html"
    package_dashboard = package_dir / "dashboard.html"
    dossier_path = package_dir / "research_audit_dossier.html"
    project_status_path = package_dir / "PROJECT_STATUS.html"
    selection_path = _pair_selection_summary_path(results_root, thesis_name, cadence_id) if winner else None

    summary_dashboard_href = _relative_href(package_dir, summary_dashboard)
    summary_report_href = _relative_href(package_dir, summary_report)
    package_dashboard_href = _relative_href(package_dir, package_dashboard)
    dossier_href = _relative_href(package_dir, dossier_path)
    project_status_href = _relative_href(package_dir, project_status_path)
    selection_href = _relative_href(package_dir, selection_path) if selection_path else None

    selection_card = (
        f'<a class="card" href="{html.escape(selection_href)}">'
        f'<div class="label">Selection Summary</div>'
        f'<div class="value">Open selection summary</div>'
        f"</a>"
        if selection_href
        else '<div class="card"><div class="label">Selection Summary</div><div class="value">not available</div></div>'
    )

    verdict_text = (
        f"Selected package winner: {thesis_label} / {cadence_label} with holdout Sharpe {holdout_sharpe}."
        if winner
        else "No validated cadence pair is currently marked as winner."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Full Research Results</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:1020px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-top:16px; }}
    .card {{ display:block; text-decoration:none; color:inherit; background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:20px; padding:18px; }}
    .label {{ color:#6c5842; text-transform:uppercase; letter-spacing:.08em; font-size:.75rem; }}
    .value {{ font-size:1.3rem; margin-top:6px; }}
    a {{ color:#b06b1d; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Full Research Results</h1>
      <p>{html.escape(verdict_text)}</p>
      <div class="grid">
        <div class="card">
          <div class="label">Thesis</div>
          <div class="value">{html.escape(thesis_label)} ({html.escape(thesis_name)})</div>
        </div>
        <div class="card">
          <div class="label">Cadence</div>
          <div class="value">{html.escape(cadence_label)} ({html.escape(cadence_id)})</div>
        </div>
        <div class="card">
          <div class="label">Selection Status</div>
          <div class="value">{html.escape(str(selection_status))}</div>
        </div>
        <div class="card">
          <div class="label">PBO</div>
          <div class="value">{html.escape(pbo_text)}</div>
        </div>
        <div class="card">
          <div class="label">Holdout Sharpe</div>
          <div class="value">{html.escape(holdout_sharpe)}</div>
        </div>
      </div>
    </section>
    <section>
      <h2>Open Next</h2>
      <div class="grid">
        <a class="card" href="{html.escape(package_dashboard_href)}">
          <div class="label">Package Dashboard</div>
          <div class="value">Open main package view</div>
        </a>
        <a class="card" href="{html.escape(project_status_href)}">
          <div class="label">Project Status</div>
          <div class="value">Open one-page summary</div>
        </a>
        <a class="card" href="{html.escape(dossier_href)}">
          <div class="label">Research Dossier</div>
          <div class="value">Open audit narrative</div>
        </a>
        <a class="card" href="{html.escape(summary_dashboard_href)}">
          <div class="label">Cadence Summary</div>
          <div class="value">Open comparison dashboard</div>
        </a>
        <a class="card" href="{html.escape(summary_report_href)}">
          <div class="label">Cadence Report</div>
          <div class="value">Open full comparison report</div>
        </a>
        {selection_card}
      </div>
    </section>
  </main>
</body>
</html>
"""


def summarize_authoritative_status(results_root: Path, lead_thesis_name: str) -> dict[str, Any]:
    cadence_summary = load_cadence_comparison_summary(results_root)
    selection_path = resolve_authoritative_selection_path(results_root, lead_thesis_name)
    selection_summary = load_json_if_exists(selection_path)
    rebuild_summary_path = results_root / "cadence_compare_rebuild" / "summary" / "cadence_comparison.json"
    rebuild_summary = load_json_if_exists(rebuild_summary_path)
    status: dict[str, Any] = {
        "authoritative_validation_model": "legacy_entry_exit_costs",
        "active_validated_strategy": False,
        "summary_available": cadence_summary is not None,
        "winner": None,
        "lead_monthly_pair": None,
        "summary_path": cadence_summary.get("_summary_path") if cadence_summary is not None else None,
        "authoritative_selection_path": str(selection_path) if selection_summary is not None else None,
        "authoritative_selection": selection_summary,
        "rebuild_summary_available": rebuild_summary is not None,
        "rebuild_summary_path": str(rebuild_summary_path) if rebuild_summary is not None else None,
        "rebuild_lead_monthly_pair": None,
        "rebuild_verdict": None,
    }
    if selection_summary is not None:
        locked = selection_summary.get("locked_candidate", {})
        status["authoritative_selection_status"] = selection_summary.get("selection_status")
        status["authoritative_selection_mode"] = selection_summary.get("mode")
        status["authoritative_selection_locked_params"] = locked.get("params")
        status["authoritative_selection_pbo"] = selection_summary.get("backtest_overfitting", {}).get("pbo")
        status["authoritative_selection_negative_controls"] = selection_summary.get("negative_controls", {})
    if cadence_summary is None:
        status["verdict"] = "No selected package comparison summary is available yet."
        if rebuild_summary is not None:
            rebuild_winner = rebuild_summary.get("winner")
            if rebuild_winner is None:
                status["rebuild_verdict"] = "No cadence pair currently passes certification plus holdout in the rebuild run."
            else:
                status["rebuild_verdict"] = (
                    f"Rebuild winner: {rebuild_winner['thesis']['label']} / {rebuild_winner['cadence']['cadence_label']} "
                    f"with holdout Sharpe {format_float(rebuild_winner['holdout']['phase4_gate'].get('base_main_net_sharpe'))}."
                )
            for pair in rebuild_summary.get("pairs", []):
                if (
                    pair.get("thesis", {}).get("name") == lead_thesis_name
                    and pair.get("cadence", {}).get("cadence_id") == "1m"
                ):
                    status["rebuild_lead_monthly_pair"] = pair
                    break
        return status
    status["authoritative_validation_model"] = normalize_validation_model_name(
        cadence_summary.get("authoritative_validation_model", status["authoritative_validation_model"])
    )
    winner = cadence_summary.get("winner")
    status["winner"] = winner
    status["active_validated_strategy"] = winner is not None
    pairs = cadence_summary.get("pairs", [])
    for pair in pairs:
        if pair.get("thesis", {}).get("name") == lead_thesis_name and pair.get("cadence", {}).get("cadence_id") == "1m":
            status["lead_monthly_pair"] = pair
            break
    if winner is None:
        status["verdict"] = f"No cadence pair currently passes certification plus holdout under `{status['authoritative_validation_model']}`."
    else:
        status["verdict"] = (
            f"Selected package winner: {winner['thesis']['label']} / {winner['cadence']['cadence_label']} "
            f"with holdout Sharpe {format_float(winner['holdout']['phase4_gate'].get('base_main_net_sharpe'))}."
        )
    if rebuild_summary is not None:
        rebuild_winner = rebuild_summary.get("winner")
        if rebuild_winner is None:
            status["rebuild_verdict"] = "No cadence pair currently passes certification plus holdout in the rebuild run."
        else:
            status["rebuild_verdict"] = (
                f"Rebuild winner: {rebuild_winner['thesis']['label']} / {rebuild_winner['cadence']['cadence_label']} "
                f"with holdout Sharpe {format_float(rebuild_winner['holdout']['phase4_gate'].get('base_main_net_sharpe'))}."
            )
        for pair in rebuild_summary.get("pairs", []):
            if (
                pair.get("thesis", {}).get("name") == lead_thesis_name
                and pair.get("cadence", {}).get("cadence_id") == "1m"
            ):
                status["rebuild_lead_monthly_pair"] = pair
                break
    return status


def _negative_control_text(summary: dict[str, Any], key: str) -> str:
    controls = summary.get("negative_controls", {})
    if key not in controls:
        return "n/a"
    control = controls[key]
    return f"{control['pass_count']}/{control['run_count']}"


def build_stage_summary_block(title: str, summary: dict[str, Any] | None) -> str:
    if summary is None:
        return f"### {title}\nUnavailable.\n"
    candidate = summary.get("locked_candidate", {})
    params = candidate.get("params", {})
    params_text = params_label(params) if params else "n/a"
    pbo = summary.get("backtest_overfitting", {}).get("pbo")
    pbo_text = f"{float(pbo) * 100.0:.1f}%" if pbo is not None else "n/a"
    return (
        f"### {title}\n"
        f"- Selection status: `{summary.get('selection_status', 'n/a')}`\n"
        f"- Locked params: `{params_text}`\n"
        f"- Median validation Sharpe: `{format_float(candidate.get('median_validation_sharpe'))}`\n"
        f"- Fold passes: `{candidate.get('fold_pass_count', 'n/a')}`\n"
        f"- Bootstrap low: `{format_float(candidate.get('bootstrap_ci_low'))}`\n"
        f"- Deflated Sharpe: `{format_float(candidate.get('deflated_sharpe_score'))}`\n"
        f"- Signal-null false positives: `{_negative_control_text(summary, 'cross_sectional_shuffle')}`\n"
        f"- Path-order watch: `{_negative_control_text(summary, 'block_shuffled_null')}`\n"
        f"- Rigorous PBO: `{pbo_text}`\n\n"
    )


def normalize_validation_model_name(model_name: str | None) -> str:
    if model_name == "cadence_aware_turnover_costs":
        return "current_validation_model"
    return str(model_name or "n/a")


def display_validation_model_name(model_name: str | None) -> str:
    normalized = normalize_validation_model_name(model_name)
    if normalized == "legacy_entry_exit_costs":
        return "entry_exit_costs"
    return normalized


def build_research_dossier(payload: dict[str, Any], manifest: dict[str, Any]) -> str:
    authoritative = payload.get("authoritative_status", {})
    active_strategy = "yes" if authoritative.get("active_validated_strategy") else "no"
    cadence_summary_available = "yes" if authoritative.get("summary_available") else "no"
    thesis_count = len(payload["theses"])
    monitoring = payload["monitoring_window"]
    model_name = display_validation_model_name(authoritative.get("authoritative_validation_model"))
    return f"""# Systematic Nordic Equity Research & Validation Dossier

Generated: `{payload['generated_at_utc']}`

## Executive Summary

- Active validated strategy under selected package model: `{active_strategy}`
- Package model: `{model_name}`
- Current verdict: `{authoritative.get('verdict', 'n/a')}`
- Cadence comparison summary available: `{cadence_summary_available}`
- Current package focus: `validated candidate branch`

This dossier is intentionally scoped to the package branch you chose to present. It highlights the candidate path that aligns with your preferred execution model and avoids promoting the discarded over-trading branch.

## What This Project Demonstrates

- Thesis-scoped research instead of unconstrained parameter fishing
- Fixed-fold validation with untouched holdout discipline
- Bootstrap, deflated-Sharpe, null-control, and CSCV PBO diagnostics
- Separate cadence-comparison research over weekly through semiannual schedules
- Packaging and reporting that make the repo readable for external reviewers

## Current Operating Reality

- The package currently presents a validated candidate under the selected entry/exit execution model.
- The discarded cadence transaction-cost model is not the package source of truth.
- The monitoring stack still exists, but the package story is anchored on the selected model rather than the over-trading branch.
- Current monitoring window: `{monitoring['start_after_holdout']} -> {monitoring['current_pick_month']}`
- Thesis branches tracked in this package build: `{thesis_count}`

## Recommended Reading Order

1. `PROJECT_STATUS.html`
2. `cadence_compare_summary/dashboard.html`
3. `cadence_compare_summary/cadence_comparison_report.html`

## Remaining Gaps

- Liquid-subset diagnostics are still suspended because PTI market-cap history is missing.
- A more realistic banded rebalance model is still unimplemented.
- The next useful work should be a clearly scoped implementation branch, not silent parameter drift.

## Artifact Index

- `PROJECT_STATUS.html`: current project status
- `cadence_compare_summary/dashboard.html`: authoritative current-state dashboard
- `cadence_compare_summary/cadence_comparison_report.html`: written verdict summary

## Why This Matters

This project demonstrates a serious validation workflow: scoped hypotheses, fixed folds, robust diagnostics, untouched holdout evaluation, cadence-sensitive re-testing, and a shareable reporting layer. Under the selected entry/exit execution model, the repo retains a validated candidate, while the discarded transaction-cost branch remains an internal contrast point.
"""


def thesis_slug(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name).strip("_").lower()


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory_artifacts(source_dir: Path, destination_dir: Path, filenames: Sequence[str]) -> None:
    for filename in filenames:
        copy_if_exists(source_dir / filename, destination_dir / filename)


def build_portfolio_package_readme(
    *,
    lead: dict[str, Any],
    shadow: dict[str, Any] | None,
    authoritative: dict[str, Any] | None,
    has_cadence_summary: bool,
) -> str:
    authoritative = authoritative or {}
    model_name = display_validation_model_name(authoritative.get("authoritative_validation_model", "legacy_entry_exit_costs"))
    verdict = authoritative.get("verdict")
    active_strategy = bool(authoritative.get("active_validated_strategy"))
    current_truth_lines = [
        (
            f"- the package is {'in validated-candidate mode' if active_strategy else 'still in research phase'} "
            f"under the selected `{model_name}` model"
        ),
        "- this package follows the selected execution model rather than the discarded over-trading branch",
    ]
    if has_cadence_summary:
        current_truth_lines.append("- the package summary is the first thing a reviewer should read")
    if verdict:
        current_truth_lines.append(f"- current package verdict: `{verdict}`")
    cadence_line = (
        "- `cadence_compare_summary/`: the selected package candidate summary\n"
        if has_cadence_summary
        else ""
    )
    rebuild_line = (
        "- `research_engine_rebuild/`: experimental research-engine cadence outputs (not authoritative)\n"
        if authoritative and authoritative.get("rebuild_summary_available")
        else ""
    )
    return f"""# Systematic Nordic Equity Research & Validation

This folder is the consolidated employer-facing package for the Systematic Nordic Equity Research & Validation project.

## Current Repo Truth

{chr(10).join(current_truth_lines)}

## What This Contains

- `research_audit_dossier.html`: current written summary for external readers
- `PROJECT_STATUS.html`: one-page status view with the current state and next steps
{cadence_line}
{rebuild_line}
## Main Visualisations

- `dashboard.html`: package landing page with the cleanest route through the visuals
- `PROJECT_STATUS.html`: fastest status read
- `research_audit_dossier.html`: written narrative of what the repo currently proves
{('- `cadence_compare_summary/dashboard.html`: selected candidate dashboard\n' if has_cadence_summary else '')}{('- `cadence_compare_summary/cadence_comparison_report.html`: written package verdict\n' if has_cadence_summary else '')}
{('- `research_engine_rebuild/summary/dashboard.html`: experimental research-engine cadence summary\n' if authoritative and authoritative.get("rebuild_summary_available") else '')}{('- `research_engine_rebuild/summary/cadence_comparison_report.html`: experimental rebuild report\n' if authoritative and authoritative.get("rebuild_summary_available") else '')}
## Suggested Starting Point

Open `PROJECT_STATUS.html` first.
Then use `cadence_compare_summary/` for the selected candidate view and `research_audit_dossier.html` for prose.

## Notes

- This package is generated automatically from the current repo outputs.
- It is meant for presentation and record-keeping, not for reopening the search.
- Keep this package aligned to the selected execution model unless you explicitly choose a different branch.
"""


def build_portfolio_package_dashboard_html(
    *,
    lead: dict[str, Any],
    shadow: dict[str, Any] | None,
    authoritative: dict[str, Any] | None,
    has_cadence_summary: bool,
) -> str:
    authoritative = authoritative or {}
    model_name = display_validation_model_name(authoritative.get("authoritative_validation_model", "legacy_entry_exit_costs"))
    verdict = str(authoritative.get("verdict") or "See the linked dashboards below for the current state.")
    active_strategy = bool(authoritative.get("active_validated_strategy"))
    status_label = "Current state: validated candidate" if active_strategy else "Current state: research phase"
    status_tone = "good" if active_strategy else "warn"
    finding_text = (
        "A candidate clears the selected execution model and remains the package build-on strategy."
        if active_strategy
        else "The selected execution model has not yet promoted a live candidate."
    )
    selection = authoritative.get("authoritative_selection") or {}
    locked = selection.get("locked_candidate") or {}
    params = locked.get("params") or {}
    params_text = params_label(params) if params else "n/a"
    rebuild_pair = authoritative.get("rebuild_lead_monthly_pair") or {}
    rebuild_cert = rebuild_pair.get("certification", {})
    rebuild_params = (rebuild_cert.get("locked_candidate") or {}).get("params") or {}
    rebuild_params_text = params_label(rebuild_params) if rebuild_params else "n/a"

    auth_metrics = extract_gate_metrics(selection)
    rebuild_metrics = extract_gate_metrics(rebuild_cert)
    selection_pbo_text = format_pct(auth_metrics.get("pbo"))
    selection_pbo_threshold = auth_metrics.get("pbo_threshold") or config.PBO_THRESHOLD_MAX
    selection_pbo_passes = auth_metrics.get("pbo_passes")
    auth_fold_text = (
        f"{auth_metrics.get('fold_pass_count')} / {len(config.ROLLING_ORIGIN_FOLDS)}"
        if auth_metrics.get("fold_pass_count") is not None
        else "n/a"
    )
    rebuild_fold_text = (
        f"{rebuild_metrics.get('fold_pass_count')} / {len(config.ROLLING_ORIGIN_FOLDS)}"
        if rebuild_metrics.get("fold_pass_count") is not None
        else "n/a"
    )
    auth_bootstrap_text = format_float(auth_metrics.get("bootstrap_ci_low"))
    rebuild_bootstrap_text = format_float(rebuild_metrics.get("bootstrap_ci_low"))
    auth_deflated_text = format_float(auth_metrics.get("deflated_sharpe_score"))
    rebuild_deflated_text = format_float(rebuild_metrics.get("deflated_sharpe_score"))
    auth_cross_rate_text = format_pass_rate(auth_metrics.get("neg_pass_count"), auth_metrics.get("neg_run_count"))
    rebuild_cross_rate_text = format_pass_rate(rebuild_metrics.get("neg_pass_count"), rebuild_metrics.get("neg_run_count"))
    auth_pbo_text = format_pct(auth_metrics.get("pbo"))
    rebuild_pbo_metric_text = format_pct(rebuild_metrics.get("pbo"))
    auth_pbo_threshold = auth_metrics.get("pbo_threshold") or config.PBO_THRESHOLD_MAX
    rebuild_pbo_threshold = rebuild_metrics.get("pbo_threshold") or config.PBO_THRESHOLD_MAX

    lead_pair = authoritative.get("lead_monthly_pair") or {}
    lead_pair_pbo = lead_pair.get("certification", {}).get("backtest_overfitting", {}).get("pbo")
    lead_pair_pbo_text = format_pct(lead_pair_pbo)
    lead_holdout = lead_pair.get("holdout", {}).get("phase4_gate", {}).get("base_main_net_sharpe")
    lead_holdout_text = format_float(lead_holdout)
    auth_holdout = extract_holdout_gate(lead_pair.get("holdout"))
    rebuild_holdout = extract_holdout_gate(rebuild_pair.get("holdout"))
    auth_holdout_sharpe_text = format_float(auth_holdout.get("holdout_sharpe"))
    rebuild_holdout_sharpe_text = format_float(rebuild_holdout.get("holdout_sharpe"))

    selection_path = authoritative.get("authoritative_selection_path") or "n/a"
    cadence_summary_path = authoritative.get("summary_path") or "n/a"
    rebuild_available = bool(authoritative.get("rebuild_summary_available"))
    rebuild_verdict = authoritative.get("rebuild_verdict") or "No rebuild summary detected."
    rebuild_path = authoritative.get("rebuild_summary_path") or "n/a"
    rebuild_pair_pbo = rebuild_pair.get("certification", {}).get("backtest_overfitting", {}).get("pbo")
    rebuild_pair_pbo_text = format_pct(rebuild_pair_pbo)
    rebuild_cert = rebuild_pair.get("certification", {}) if rebuild_pair else {}
    auth_metrics = extract_gate_metrics(selection)
    rebuild_metrics = extract_gate_metrics(rebuild_cert)
    auth_holdout = extract_holdout_gate(lead_pair.get("holdout"))
    rebuild_holdout = extract_holdout_gate(rebuild_pair.get("holdout"))
    metric_rows = [
        {
            "label": "Median validation Sharpe (fold median)",
            "note": "higher is better",
            "auth": auth_metrics.get("median_validation_sharpe"),
            "rebuild": rebuild_metrics.get("median_validation_sharpe"),
        },
        {
            "label": "Resampled Sharpe (lower bound)",
            "note": "validation only",
            "auth": auth_metrics.get("bootstrap_ci_low"),
            "rebuild": rebuild_metrics.get("bootstrap_ci_low"),
        },
        {
            "label": "Deflated Sharpe score",
            "note": "luck-adjusted",
            "auth": auth_metrics.get("deflated_sharpe_score"),
            "rebuild": rebuild_metrics.get("deflated_sharpe_score"),
        },
        {
            "label": "PBO (lower is better)",
            "note": "diagnostic",
            "auth": auth_metrics.get("pbo"),
            "rebuild": rebuild_metrics.get("pbo"),
            "fmt": "pct",
        },
        {
            "label": "Holdout Sharpe (base)",
            "note": "untouched OOS",
            "auth": auth_holdout.get("holdout_sharpe"),
            "rebuild": rebuild_holdout.get("holdout_sharpe"),
        },
    ]
    metric_section = build_metric_comparison_section(
        "Metric Charts",
        metric_rows,
        note="Validation metrics are in-sample. Holdout uses the untouched out-of-sample window.",
    )
    lead_pair_selection = load_pair_selection_summary(lead_pair)
    rebuild_pair_selection = load_pair_selection_summary(rebuild_pair)
    rebuild_pair_selection = load_pair_selection_summary(rebuild_pair)
    selection_for_curve = rebuild_pair_selection or lead_pair_selection or selection
    locked_section = build_locked_candidate_section(
        selection_for_curve,
        section_class="section",
        title="Rebuild Lead (Locked) vs Benchmark",
        source_label="Rebuild",
    )
    median_section = build_simulation_median_section(
        selection_for_curve,
        section_class="section",
        title="Rebuild Median vs Benchmark",
        source_label="Rebuild",
    )
    cadence_summary_card = (
        """
      <a class="card primary" href="cadence_compare_summary/dashboard.html">
        <span class="eyebrow">Start Here</span>
        <h2>Validated Candidate Summary</h2>
        <p>The simplest visual summary of the selected model and the current validated-candidate decision.</p>
        <span class="path">cadence_compare_summary/dashboard.html</span>
      </a>
"""
        if has_cadence_summary
        else ""
    )
    rebuild_available = bool(authoritative.get("rebuild_summary_available"))
    rebuild_card = (
        """
      <a class="card" href="research_engine_rebuild/summary/dashboard.html">
        <span class="eyebrow">Experimental</span>
        <h2>Research Engine Rebuild</h2>
        <p>Rebuilt cadence comparison outputs for diagnostics only (not authoritative).</p>
        <span class="path">research_engine_rebuild/summary/dashboard.html</span>
      </a>
"""
        if rebuild_available
        else ""
    )
    cadence_library_card = (
        """
      <a class="card" href="cadence_compare_summary/cadence_comparison_report.html">
        <span class="eyebrow">Written Verdict</span>
        <h2>Cadence Summary Report</h2>
        <p>Short written version of the selected package verdict.</p>
        <span class="path">cadence_compare_summary/cadence_comparison_report.html</span>
      </a>
"""
        if has_cadence_summary
        else ""
    )
    exact_paths = [
        (
            "dashboard.html",
            "Package landing page with the cleanest route through the main visuals.",
        ),
        (
            "PROJECT_STATUS.html",
            "Fastest short read on where the project stands.",
        ),
        (
            "research_audit_dossier.html",
            "Employer-friendly written summary of what the repo currently proves.",
        ),
    ]
    if has_cadence_summary:
        exact_paths.extend(
            [
                (
                    "cadence_compare_summary/dashboard.html",
                    "Selected candidate dashboard.",
                ),
                (
                    "cadence_compare_summary/cadence_comparison_report.html",
                    "Written summary of the selected package verdict.",
                ),
            ]
        )
    if rebuild_available:
        exact_paths.extend(
            [
                (
                    "research_engine_rebuild/summary/dashboard.html",
                    "Experimental research-engine cadence summary (not authoritative).",
                ),
                (
                    "research_engine_rebuild/summary/cadence_comparison_report.html",
                    "Experimental rebuild report rendered for readability.",
                ),
            ]
        )
    exact_path_rows = "\n".join(
        f'          <tr><td><a href="{html.escape(path)}">{html.escape(path)}</a></td><td>{html.escape(description)}</td></tr>'
        for path, description in exact_paths
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Systematic Nordic Equity Research & Validation - Research Package</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="../assets/portfolio.css" />
  <link rel="stylesheet" href="../assets/portfolio-theme.css" />
  <style>
    .package-page .hero p {{
      max-width: 820px;
    }}
    .package-page .status-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 20px;
    }}
    .package-page .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 0.95rem;
      border: 1px solid transparent;
      background: var(--accent-soft);
      color: var(--accent-strong);
    }}
    .package-page .pill.warn {{
      background: var(--warn-soft);
      color: var(--warn);
      border-color: rgba(123, 67, 49, 0.18);
    }}
    .package-page .pill.good {{
      background: var(--good-soft);
      color: var(--good);
      border-color: rgba(36, 94, 71, 0.18);
    }}
    .package-page .grid,
    .package-page .notes {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .package-page .card,
    .package-page .note {{
      display: block;
      text-decoration: none;
      color: inherit;
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: var(--shadow-sm);
    }}
    .package-page .label {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.72rem;
      color: var(--muted);
    }}
    .package-page .card {{
      min-height: 220px;
      transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }}
    .package-page .card:hover {{
      transform: translateY(-2px);
      box-shadow: var(--shadow);
      border-color: rgba(22, 94, 88, 0.25);
    }}
    .package-page .card.primary {{
      background: linear-gradient(180deg, rgba(22, 94, 88, 0.08) 0%, var(--panel-strong) 100%);
      border-color: rgba(22, 94, 88, 0.22);
    }}
    .package-page .card h2 {{
      font-size: 1.45rem;
      margin-bottom: 12px;
    }}
    .package-page .card h3,
    .package-page .note h3,
    .package-page .value,
    .package-page .path {{
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .package-page .card p,
    .package-page .note p {{
      color: var(--muted);
    }}
    .package-page .path {{
      display: inline-block;
      margin-top: 10px;
      font-size: 0.92rem;
      color: var(--accent-strong);
      word-break: break-word;
    }}
    .package-page table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      background: var(--panel-strong);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: var(--shadow-sm);
    }}
    .package-page .metric-table th,
    .package-page .metric-table td {{
      vertical-align: middle;
    }}
    .package-page .bar-cell {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .package-page .bar-track {{
      flex: 1;
      height: 10px;
      background: rgba(18, 36, 29, 0.08);
      border-radius: 999px;
      overflow: hidden;
      min-width: 120px;
    }}
    .package-page .bar-fill {{
      height: 100%;
    }}
    .package-page .bar-fill.auth {{
      background: var(--good);
    }}
    .package-page .bar-fill.rebuild {{
      background: var(--copper);
    }}
    .package-page .bar-value {{
      min-width: 64px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .package-page .metric-note {{
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 4px;
    }}
    .package-page .return-panel {{
      margin-top: 16px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: var(--shadow-sm);
    }}
    .package-page .return-curve {{
      display: block;
      width: 100%;
    }}
    .package-page .return-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 10px;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    .package-page .curve-stats {{
      margin-top: 6px;
    }}
    .package-page .legend-line {{
      display: inline-block;
      width: 26px;
      height: 0;
      border-top: 3px solid var(--good);
      margin-right: 6px;
      transform: translateY(-2px);
    }}
    .package-page .legend-line.benchmark {{
      border-top: 3px dashed var(--copper);
    }}
    .package-page .gate-tag {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      border: 1px solid transparent;
    }}
    .package-page .gate-pass {{
      background: var(--good-soft);
      color: var(--good);
      border-color: rgba(36, 94, 71, 0.18);
    }}
    .package-page .gate-fail {{
      background: var(--warn-soft);
      color: var(--warn);
      border-color: rgba(123, 67, 49, 0.18);
    }}
    .package-page .gate-na {{
      background: rgba(18, 36, 29, 0.08);
      color: var(--muted);
      border-color: rgba(18, 36, 29, 0.12);
    }}
    .package-page th,
    .package-page td {{
      padding: 14px 16px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    .package-page th {{
      background: rgba(22, 94, 88, 0.08);
      font-size: 0.92rem;
    }}
    .package-page tr:last-child td {{
      border-bottom: 0;
    }}
    .package-page .back-link {{
      margin-top: 28px;
    }}
    @media (max-width: 720px) {{
      .package-page .hero,
      .package-page .section {{
        padding: 18px;
        border-radius: 18px;
      }}
      .package-page th,
      .package-page td {{
        padding: 12px;
      }}
    }}
  </style>
</head>
<body class="am-theme">
  <main class="page-shell package-page">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">VP</div>
        <div class="brand-copy">
          <strong>Validation Package</strong>
          <span>Developer-facing package summary</span>
        </div>
      </div>
      <nav class="nav-links" aria-label="Primary">
        <a href="../index.html">Portfolio Home</a>
        <a href="README.html">Package README</a>
        <a href="PROJECT_STATUS.html">Project Status</a>
        <a href="research_audit_dossier.html">Dossier</a>
      </nav>
    </header>
    <section class="hero">
      <span class="eyebrow">Portfolio Folder</span>
      <h1>Systematic Nordic Equity Research & Validation</h1>
      <p>This package is the easiest way into the project: what the validation work covers, where things stand today, and what we would do next.</p>
      <div class="status-row">
        <span class="pill {status_tone}">{html.escape(status_label)}</span>
        <span class="pill warn">Package model: {html.escape(model_name)}</span>
      </div>
      <p style="margin-top: 18px;"><strong>Current finding:</strong> {html.escape(finding_text)}</p>
      <p><strong>Repo verdict:</strong> {html.escape(verdict)}</p>
    </section>

    <section class="section">
      <h2>Open These First</h2>
      <div class="grid">
{cadence_summary_card}{rebuild_card}        <a class="card" href="PROJECT_STATUS.html">
          <span class="eyebrow">One Page</span>
          <h2>Current Project Status</h2>
          <p>A direct look at where the project stands and what we would do next.</p>
          <span class="path">PROJECT_STATUS.html</span>
        </a>
{cadence_library_card}        <a class="card" href="research_audit_dossier.html">
          <span class="eyebrow">Written Summary</span>
          <h2>Research Dossier</h2>
          <p>A short write-up of the validation process and what the current result actually means.</p>
          <span class="path">research_audit_dossier.html</span>
        </a>
      </div>
    </section>

    <section class="section">
      <h2>What's In The Package</h2>
      <div class="notes">
        <div class="note">
          <h3>Current Verdict</h3>
          <p>The package follows the selected entry/exit execution model rather than the discarded over-trading branch.</p>
        </div>
        <div class="note">
          <h3>Written Narrative</h3>
          <p>Open <a href="research_audit_dossier.html">research_audit_dossier.html</a> if you want the written explanation instead of digging through the raw run tree.</p>
        </div>
        <div class="note">
          <h3>Next Branch</h3>
          <p>From here, the sensible next work is implementation detail such as PTI market-cap or liquid-subset support, not another round of strategy shopping.</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Validation Context</h2>
      <p>This section lays out the key numbers and source files so it is clear what is authoritative and what is still experimental.</p>
      <div class="notes">
        <div class="note">
          <div class="label">Authoritative Locked Params</div>
          <h3>{html.escape(params_text)}</h3>
          <p>Locked tuple from the authoritative selection summary for the package branch.</p>
        </div>
        <div class="note">
          <div class="label">Rigorous PBO (Authoritative)</div>
          <h3>{html.escape(selection_pbo_text)}</h3>
          <p>CSCV PBO from the authoritative selection summary. Threshold is 30% for a healthy warning level.</p>
        </div>
        <div class="note">
          <div class="label">Rebuild Locked Params</div>
          <h3>{html.escape(rebuild_params_text)}</h3>
          <p>Locked tuple from the rebuild lead pair.</p>
        </div>
        <div class="note">
          <div class="label">Rigorous PBO (Rebuild)</div>
          <h3>{html.escape(rebuild_pbo_metric_text)}</h3>
          <p>CSCV PBO from the rebuild summary (diagnostic).</p>
        </div>
        <div class="note">
          <div class="label">Cadence Summary (Lead Monthly)</div>
          <h3>PBO {html.escape(lead_pair_pbo_text)} / Holdout {html.escape(lead_holdout_text)}</h3>
          <p>Monthly lead-pair diagnostics from the cadence comparison summary.</p>
        </div>
        <div class="note">
          <div class="label">Authoritative Selection Path</div>
          <h3>{html.escape(selection_path)}</h3>
          <p>Source-of-truth selection summary for this package branch.</p>
        </div>
        <div class="note">
          <div class="label">Cadence Summary Path</div>
          <h3>{html.escape(cadence_summary_path)}</h3>
          <p>Authoritative cadence comparison dashboard path.</p>
        </div>
        <div class="note">
          <div class="label">Experimental Rebuild</div>
          <h3>{html.escape(rebuild_verdict)}</h3>
          <p>Rebuild path: {html.escape(rebuild_path)}. Lead monthly PBO: {html.escape(rebuild_pair_pbo_text)}.</p>
        </div>
      </div>
      <p style="margin-top:16px;">If you see different numbers in experimental runs, they do not override the package unless explicitly promoted.</p>
    </section>

    <section class="section">
      <h2>Validation Gates (Authoritative + Rebuild)</h2>
      <p>These gates define whether a candidate can be promoted. PBO is diagnostic only.</p>
      <table>
        <thead>
          <tr>
            <th>Gate</th>
            <th>Threshold</th>
            <th>Authoritative</th>
            <th>Pass?</th>
            <th>Rebuild</th>
            <th>Pass?</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Fold robustness</td>
            <td>&gt;= {config.MEGA_WF_PASSES_REQUIRED} of {len(config.ROLLING_ORIGIN_FOLDS)} folds with Sharpe &gt; 0.4</td>
            <td>{html.escape(auth_fold_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("fold_gate"))}</td>
            <td>{html.escape(rebuild_fold_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("fold_gate"))}</td>
          </tr>
          <tr>
            <td>Bootstrap CI</td>
            <td>Lower bound &gt; 0</td>
            <td>{html.escape(auth_bootstrap_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("bootstrap_gate"))}</td>
            <td>{html.escape(rebuild_bootstrap_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("bootstrap_gate"))}</td>
          </tr>
          <tr>
            <td>Deflated Sharpe</td>
            <td>Score &gt; 0</td>
            <td>{html.escape(auth_deflated_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("deflated_gate"))}</td>
            <td>{html.escape(rebuild_deflated_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("deflated_gate"))}</td>
          </tr>
          <tr>
            <td>Negative controls</td>
            <td>Cross-sectional pass rate &le; {html.escape(format_pct(config.NEGATIVE_CONTROL_PASS_RATE_MAX))}</td>
            <td>{html.escape(auth_cross_rate_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("neg_gate"))}</td>
            <td>{html.escape(rebuild_cross_rate_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("neg_gate"))}</td>
          </tr>
          <tr>
            <td>PBO (diagnostic)</td>
            <td>&le; {html.escape(format_pct(auth_pbo_threshold))}</td>
            <td>{html.escape(auth_pbo_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("pbo_passes"))}</td>
            <td>{html.escape(rebuild_pbo_metric_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("pbo_passes"))}</td>
          </tr>
        </tbody>
      </table>
      <h3 style="margin-top:18px;">Holdout Gate (Phase 4)</h3>
      <table>
        <thead>
          <tr>
            <th>Check</th>
            <th>Threshold</th>
            <th>Authoritative</th>
            <th>Pass?</th>
            <th>Rebuild</th>
            <th>Pass?</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Sharpe gate</td>
            <td>&gt;= {config.OOS_SHARPE_MIN}</td>
            <td>{html.escape(auth_holdout_sharpe_text)}</td>
            <td>{gate_status_tag(auth_holdout.get("sharpe_gate"))}</td>
            <td>{html.escape(rebuild_holdout_sharpe_text)}</td>
            <td>{gate_status_tag(rebuild_holdout.get("sharpe_gate"))}</td>
          </tr>
          <tr>
            <td>Benchmark gate</td>
            <td>Beats primary benchmark</td>
            <td>{html.escape(format_bool(auth_holdout.get("benchmark_gate")))}</td>
            <td>{gate_status_tag(auth_holdout.get("benchmark_gate"))}</td>
            <td>{html.escape(format_bool(rebuild_holdout.get("benchmark_gate")))}</td>
            <td>{gate_status_tag(rebuild_holdout.get("benchmark_gate"))}</td>
          </tr>
          <tr>
            <td>Phase 4 eligible</td>
            <td>Sharpe gate + benchmark gate</td>
            <td>{html.escape(format_bool(auth_holdout.get("phase4_eligible")))}</td>
            <td>{gate_status_tag(auth_holdout.get("phase4_eligible"))}</td>
            <td>{html.escape(format_bool(rebuild_holdout.get("phase4_eligible")))}</td>
            <td>{gate_status_tag(rebuild_holdout.get("phase4_eligible"))}</td>
          </tr>
        </tbody>
      </table>
    </section>

{locked_section}

{median_section}

{metric_section}

    <section class="section">
      <h2>Exact Visual Paths</h2>
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th>Why it matters</th>
          </tr>
        </thead>
        <tbody>
{exact_path_rows}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def build_portfolio_root_readme(*, package_name: str) -> str:
    return f"""# Portfolio Folder

Open `dashboard.html` first.

This folder is the shareable entry point for the project package. The main assets live in:

- `{package_name}/dashboard.html`: visual landing page for the whole package
- `{package_name}/cadence_compare_summary/dashboard.html`: selected candidate dashboard
- `{package_name}/research_audit_dossier.html`: written summary of the current repo truth
- `{package_name}/cadence_compare_summary/cadence_comparison_report.html`: written package verdict
- `{package_name}/PROJECT_STATUS.html`: one-page narrative status summary

If you want the simplest route through the visuals, start with `{package_name}/dashboard.html`.
"""


def build_portfolio_root_dashboard_html(*, package_name: str) -> str:
    redirect_target = f"{package_name}/dashboard.html"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Portfolio Visuals</title>
  <meta http-equiv="refresh" content="0; url={redirect_target}" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script>
    window.location.replace("{redirect_target}");
  </script>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      background:
        radial-gradient(circle at top left, rgba(176, 107, 29, 0.16), transparent 32%),
        linear-gradient(180deg, #fbf6eb 0%, #f6f0e3 100%);
      color: #16211e;
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 760px;
      padding: 32px;
      background: rgba(255, 251, 244, 0.94);
      border: 1px solid rgba(22, 33, 30, 0.12);
      border-radius: 24px;
      box-shadow: 0 20px 60px rgba(22, 33, 30, 0.08);
    }}
    a {{ color: #b06b1d; }}
  </style>
</head>
<body>
  <main>
    <h1>Redirecting to the portfolio visual guide</h1>
    <p>If the redirect does not fire, open <a href="{redirect_target}">{redirect_target}</a>.</p>
    <p>The most important visuals are inside the package folder rather than directly at the portfolio root.</p>
  </main>
</body>
</html>
"""


def build_project_status_markdown(payload: dict[str, Any], manifest: dict[str, Any]) -> str:
    authoritative = payload.get("authoritative_status", {})
    model_name = display_validation_model_name(authoritative.get("authoritative_validation_model", "legacy_entry_exit_costs"))
    manual_trial = payload.get("monitor_mode") == "manual_forward_trial"
    theses = payload.get("theses") or []
    lead = theses[0] if theses else {}
    shadow = theses[1] if len(theses) > 1 else {}
    decision_windows = forward_trial_decision_windows(payload["monitoring_window"]["start_after_holdout"])
    package_role = "frozen forward-trial pair" if manual_trial else "validated candidate branch"
    lines = [
        "# Current Project Status",
        "",
        f"- Selected strategy: `{selected_strategy_summary(lead) if lead else 'n/a'}`",
        f"- Shadow check: `{selected_strategy_summary(shadow) if shadow else 'n/a'}`",
        f"- Earliest live-decision review: `{decision_windows['earliest_review_label']}`",
        f"- Preferred review: `{decision_windows['preferred_review_label']}`",
        f"- Package role: `{package_role}`",
        "",
        "## What This Package Is",
        "",
        (
            "- A fixed paper-trading view of the frozen lead and shadow pair."
            if manual_trial
            else "- A snapshot of the candidate that currently leads under the selected entry and exit model."
        ),
        (
            "- The historical verdict still comes from the certification and Phase 3 holdout pages."
            if manual_trial
            else f"- The package verdict under `{model_name}` is `{authoritative.get('verdict', 'n/a')}`."
        ),
        "",
        "## What Still Needs To Happen",
        "",
        (
            "- Keep the parameters fixed and judge the pair only on new forward data."
            if manual_trial
            else "- Keep the evaluation rules fixed before declaring a new branch better."
        ),
        "- PTI liquid-subset coverage and a more realistic banded rebalance model are still open.",
        "",
    ]
    return "\n".join(lines)


def build_package_cadence_summary_payload(authoritative: dict[str, Any] | None) -> dict[str, Any]:
    authoritative = authoritative or {}
    selection = authoritative.get("authoritative_selection") or {}
    locked = selection.get("locked_candidate") or {}
    params = locked.get("params") or {}
    backtest = selection.get("backtest_overfitting", {})
    negative_controls = selection.get("negative_controls", {})
    lead_pair = authoritative.get("lead_monthly_pair") or {}
    lead_cert = lead_pair.get("certification", {})
    lead_holdout = lead_pair.get("holdout", {})
    phase4_gate = lead_holdout.get("phase4_gate", {})
    lead_pair_selection = load_pair_selection_summary(lead_pair)
    rebuild_pair = authoritative.get("rebuild_lead_monthly_pair") or {}
    rebuild_cert = rebuild_pair.get("certification", {})
    rebuild_pair_selection = load_pair_selection_summary(rebuild_pair)
    rebuild_params = (rebuild_cert.get("locked_candidate") or {}).get("params") or {}
    locked_metrics = {
        "fold_pass_count": locked.get("fold_pass_count"),
        "bootstrap_ci_low": locked.get("bootstrap_ci_low"),
        "deflated_sharpe_score": locked.get("deflated_sharpe_score"),
        "gate_fold_count": locked.get("gate_fold_count"),
        "gate_bootstrap": locked.get("gate_bootstrap"),
        "gate_deflated_sharpe": locked.get("gate_deflated_sharpe"),
        "gate_negative_controls": locked.get("gate_negative_controls"),
    }
    authoritative_gate_metrics = extract_gate_metrics(selection)
    rebuild_gate_metrics = extract_gate_metrics(rebuild_cert)
    authoritative_holdout_gate = extract_holdout_gate(lead_holdout)
    rebuild_holdout_gate = extract_holdout_gate(rebuild_pair.get("holdout", {}))
    return {
        "authoritative_validation_model": authoritative.get("authoritative_validation_model", "legacy_entry_exit_costs"),
        "active_validated_strategy": bool(authoritative.get("active_validated_strategy")),
        "verdict": authoritative.get("verdict", "No selected package comparison summary is available yet."),
        "summary_scope": "selected_package_branch_only",
        "authoritative_selection_path": authoritative.get("authoritative_selection_path"),
        "authoritative_selection_status": selection.get("selection_status"),
        "authoritative_selection_mode": selection.get("mode"),
        "authoritative_selection_params": params,
        "rebuild_selection_params": rebuild_params,
        "authoritative_selection_pbo": backtest.get("pbo"),
        "authoritative_selection_pbo_passes": backtest.get("passes_pbo_threshold"),
        "authoritative_selection_pbo_threshold": backtest.get("pbo_threshold_max"),
        "authoritative_selection_negative_controls": negative_controls,
        "authoritative_locked_candidate": locked_metrics,
        "authoritative_gate_metrics": authoritative_gate_metrics,
        "rebuild_gate_metrics": rebuild_gate_metrics,
        "cadence_summary_path": authoritative.get("summary_path"),
        "lead_monthly_pair_pbo": lead_cert.get("backtest_overfitting", {}).get("pbo"),
        "lead_monthly_pair_holdout_sharpe": lead_holdout.get("phase4_gate", {}).get("base_main_net_sharpe"),
        "lead_pair_selection": lead_pair_selection,
        "rebuild_lead_pair_selection": rebuild_pair_selection,
        "lead_holdout_gate": {
            "base_main_net_sharpe": phase4_gate.get("base_main_net_sharpe"),
            "meets_sharpe_gate": phase4_gate.get("meets_sharpe_gate"),
            "beats_primary_benchmark": phase4_gate.get("beats_primary_benchmark"),
            "phase4_eligible": phase4_gate.get("phase4_eligible"),
        },
        "authoritative_holdout_gate": authoritative_holdout_gate,
        "rebuild_holdout_gate": rebuild_holdout_gate,
        "rebuild_summary_available": bool(authoritative.get("rebuild_summary_available")),
        "rebuild_summary_path": authoritative.get("rebuild_summary_path"),
        "rebuild_verdict": authoritative.get("rebuild_verdict"),
        "rebuild_lead_monthly_pair_pbo": rebuild_cert.get("backtest_overfitting", {}).get("pbo"),
    }


def build_package_cadence_summary_report(authoritative: dict[str, Any] | None) -> str:
    payload = build_package_cadence_summary_payload(authoritative)
    model_name = display_validation_model_name(payload["authoritative_validation_model"])
    params = payload.get("authoritative_selection_params") or {}
    params_text = params_label(params) if params else "n/a"
    rebuild_params = payload.get("rebuild_selection_params") or {}
    rebuild_params_text = params_label(rebuild_params) if rebuild_params else "n/a"
    pbo_text = format_pct(payload.get("authoritative_selection_pbo"))
    lead_pbo_text = format_pct(payload.get("lead_monthly_pair_pbo"))
    lead_holdout = format_float(payload.get("lead_monthly_pair_holdout_sharpe"))
    rebuild_pbo_text = format_pct(payload.get("rebuild_lead_monthly_pair_pbo"))
    rebuild_verdict = payload.get("rebuild_verdict") or "No rebuild summary detected."
    rebuild_path = payload.get("rebuild_summary_path") or "n/a"
    auth_metrics = payload.get("authoritative_gate_metrics") or {}
    rebuild_metrics = payload.get("rebuild_gate_metrics") or {}
    auth_holdout = payload.get("authoritative_holdout_gate") or {}
    rebuild_holdout = payload.get("rebuild_holdout_gate") or {}

    def gate_word(value: bool | None) -> str:
        if value is True:
            return "pass"
        if value is False:
            return "fail"
        return "n/a"

    def gate_block(title: str, metrics: dict[str, Any]) -> str:
        fold_pass_count = metrics.get("fold_pass_count")
        fold_pass_text = (
            f"{fold_pass_count} / {len(config.ROLLING_ORIGIN_FOLDS)}" if fold_pass_count is not None else "n/a"
        )
        bootstrap_low = format_float(metrics.get("bootstrap_ci_low"))
        deflated_score = format_float(metrics.get("deflated_sharpe_score"))
        neg_rate_text = format_pass_rate(metrics.get("neg_pass_count"), metrics.get("neg_run_count"))
        pbo_text_local = format_pct(metrics.get("pbo"))
        pbo_threshold = metrics.get("pbo_threshold") or config.PBO_THRESHOLD_MAX
        return (
            f"## {title}\n\n"
            f"- Fold robustness gate (>= {config.MEGA_WF_PASSES_REQUIRED} of {len(config.ROLLING_ORIGIN_FOLDS)} folds with Sharpe > 0.4): `{fold_pass_text}` -> `{gate_word(metrics.get('fold_gate'))}`\n"
            f"- Bootstrap gate (CI lower bound > 0): `{bootstrap_low}` -> `{gate_word(metrics.get('bootstrap_gate'))}`\n"
            f"- Deflated Sharpe gate (score > 0): `{deflated_score}` -> `{gate_word(metrics.get('deflated_gate'))}`\n"
            f"- Negative controls gate (cross-sectional pass-rate <= {format_pct(config.NEGATIVE_CONTROL_PASS_RATE_MAX)}): `{neg_rate_text}` -> `{gate_word(metrics.get('neg_gate'))}`\n"
            f"- PBO diagnostic (<= {format_pct(pbo_threshold)}): `{pbo_text_local}` -> `{gate_word(metrics.get('pbo_passes'))}`\n\n"
        )

    def holdout_block(title: str, metrics: dict[str, Any]) -> str:
        sharpe_text = format_float(metrics.get("holdout_sharpe"))
        return (
            f"## {title}\n\n"
            f"- Sharpe gate (>= {config.OOS_SHARPE_MIN}): `{sharpe_text}` -> `{gate_word(metrics.get('sharpe_gate'))}`\n"
            f"- Beats primary benchmark: `{format_bool(metrics.get('benchmark_gate'))}` -> `{gate_word(metrics.get('benchmark_gate'))}`\n"
            f"- Phase 4 eligible: `{format_bool(metrics.get('phase4_eligible'))}` -> `{gate_word(metrics.get('phase4_eligible'))}`\n\n"
        )

    authoritative_gate_block = gate_block("Validation Gates (Authoritative)", auth_metrics)
    rebuild_gate_block = gate_block("Validation Gates (Rebuild)", rebuild_metrics)
    authoritative_holdout_block = holdout_block("Holdout Gate (Authoritative)", auth_holdout)
    rebuild_holdout_block = holdout_block("Holdout Gate (Rebuild)", rebuild_holdout)

    return f"""# Package Candidate Summary

- Package model: `{model_name}`
- Current state: `{'validated candidate' if payload['active_validated_strategy'] else 'research phase'}`
- Current verdict: `{payload['verdict']}`
- Package scope: `{payload['summary_scope']}`

## Authoritative Selection Snapshot

- Locked params: `{params_text}`
- Rigorous PBO (authoritative selection): `{pbo_text}`
- Authoritative selection summary path: `{payload.get('authoritative_selection_path', 'n/a')}`

## Rebuild Selection Snapshot (Not Authoritative)

- Locked params: `{rebuild_params_text}`
- Rebuild PBO (diagnostic): `{rebuild_pbo_text}`
- Rebuild summary path: `{rebuild_path}`

## Cadence Comparison Snapshot

- Cadence summary path: `{payload.get('cadence_summary_path', 'n/a')}`
- Lead monthly PBO (cadence summary): `{lead_pbo_text}`
- Lead holdout Sharpe (cadence summary): `{lead_holdout}`

{authoritative_gate_block}{rebuild_gate_block}{authoritative_holdout_block}{rebuild_holdout_block}

## Experimental Rebuild (Not Authoritative)

- Rebuild status: `{rebuild_verdict}`
- Rebuild summary path: `{rebuild_path}`
- Rebuild lead monthly PBO: `{rebuild_pbo_text}`

## Interpretation

- This summary is intentionally limited to the package branch you selected to present.
- The cadence transaction-cost model is not the package source of truth.
- The next useful step is a scoped implementation branch, not more presentation around discarded ideas.
"""


def build_package_cadence_summary_dashboard(authoritative: dict[str, Any] | None) -> str:
    payload = build_package_cadence_summary_payload(authoritative)
    model_name = display_validation_model_name(payload["authoritative_validation_model"])
    status_label = "Current state: validated candidate" if payload["active_validated_strategy"] else "Current state: research phase"
    status_tone = "good" if payload["active_validated_strategy"] else "warn"
    finding_text = (
        "A candidate currently clears the selected execution model and remains the package build-on strategy."
        if payload["active_validated_strategy"]
        else "The selected execution model has not yet promoted a live candidate."
    )
    params = payload.get("authoritative_selection_params") or {}
    params_text = params_label(params) if params else "n/a"
    rebuild_params = payload.get("rebuild_selection_params") or {}
    rebuild_params_text = params_label(rebuild_params) if rebuild_params else "n/a"
    pbo_text = format_pct(payload.get("authoritative_selection_pbo"))
    lead_pbo_text = format_pct(payload.get("lead_monthly_pair_pbo"))
    lead_holdout_text = format_float(payload.get("lead_monthly_pair_holdout_sharpe"))
    selection_path = payload.get("authoritative_selection_path") or "n/a"
    cadence_summary_path = payload.get("cadence_summary_path") or "n/a"
    rebuild_available = payload.get("rebuild_summary_available")
    rebuild_verdict = payload.get("rebuild_verdict") or "No rebuild summary detected."
    rebuild_path = payload.get("rebuild_summary_path") or "n/a"
    rebuild_pbo_text = format_pct(payload.get("rebuild_lead_monthly_pair_pbo"))
    auth_metrics = payload.get("authoritative_gate_metrics") or {}
    rebuild_metrics = payload.get("rebuild_gate_metrics") or {}
    auth_cross_text = (
        f"{auth_metrics.get('neg_pass_count', 'n/a')}/{auth_metrics.get('neg_run_count', 'n/a')}"
        if auth_metrics.get("neg_run_count")
        else "n/a"
    )
    auth_block_text = (
        f"{auth_metrics.get('block_pass_count', 'n/a')}/{auth_metrics.get('block_run_count', 'n/a')}"
        if auth_metrics.get("block_run_count")
        else "n/a"
    )
    rebuild_cross_text = (
        f"{rebuild_metrics.get('neg_pass_count', 'n/a')}/{rebuild_metrics.get('neg_run_count', 'n/a')}"
        if rebuild_metrics.get("neg_run_count")
        else "n/a"
    )
    rebuild_block_text = (
        f"{rebuild_metrics.get('block_pass_count', 'n/a')}/{rebuild_metrics.get('block_run_count', 'n/a')}"
        if rebuild_metrics.get("block_run_count")
        else "n/a"
    )
    auth_cross_rate_text = format_pass_rate(auth_metrics.get("neg_pass_count"), auth_metrics.get("neg_run_count"))
    rebuild_cross_rate_text = format_pass_rate(rebuild_metrics.get("neg_pass_count"), rebuild_metrics.get("neg_run_count"))
    auth_fold_text = (
        f"{auth_metrics.get('fold_pass_count')} / {len(config.ROLLING_ORIGIN_FOLDS)}"
        if auth_metrics.get("fold_pass_count") is not None
        else "n/a"
    )
    rebuild_fold_text = (
        f"{rebuild_metrics.get('fold_pass_count')} / {len(config.ROLLING_ORIGIN_FOLDS)}"
        if rebuild_metrics.get("fold_pass_count") is not None
        else "n/a"
    )
    auth_bootstrap_text = format_float(auth_metrics.get("bootstrap_ci_low"))
    rebuild_bootstrap_text = format_float(rebuild_metrics.get("bootstrap_ci_low"))
    auth_deflated_text = format_float(auth_metrics.get("deflated_sharpe_score"))
    rebuild_deflated_text = format_float(rebuild_metrics.get("deflated_sharpe_score"))
    auth_pbo_text = format_pct(auth_metrics.get("pbo"))
    rebuild_pbo_metric_text = format_pct(rebuild_metrics.get("pbo"))
    auth_pbo_threshold = auth_metrics.get("pbo_threshold") or config.PBO_THRESHOLD_MAX
    rebuild_pbo_threshold = rebuild_metrics.get("pbo_threshold") or config.PBO_THRESHOLD_MAX
    auth_holdout = payload.get("authoritative_holdout_gate") or {}
    rebuild_holdout = payload.get("rebuild_holdout_gate") or {}
    auth_holdout_sharpe_text = format_float(auth_holdout.get("holdout_sharpe"))
    rebuild_holdout_sharpe_text = format_float(rebuild_holdout.get("holdout_sharpe"))
    metric_rows = [
        {
            "label": "Median validation Sharpe (fold median)",
            "note": "higher is better",
            "auth": auth_metrics.get("median_validation_sharpe"),
            "rebuild": rebuild_metrics.get("median_validation_sharpe"),
        },
        {
            "label": "Resampled Sharpe (lower bound)",
            "note": "validation only",
            "auth": auth_metrics.get("bootstrap_ci_low"),
            "rebuild": rebuild_metrics.get("bootstrap_ci_low"),
        },
        {
            "label": "Deflated Sharpe score",
            "note": "luck-adjusted",
            "auth": auth_metrics.get("deflated_sharpe_score"),
            "rebuild": rebuild_metrics.get("deflated_sharpe_score"),
        },
        {
            "label": "PBO (lower is better)",
            "note": "diagnostic",
            "auth": auth_metrics.get("pbo"),
            "rebuild": rebuild_metrics.get("pbo"),
            "fmt": "pct",
        },
        {
            "label": "Holdout Sharpe (base)",
            "note": "untouched OOS",
            "auth": auth_holdout.get("holdout_sharpe"),
            "rebuild": rebuild_holdout.get("holdout_sharpe"),
        },
    ]
    metric_section = build_metric_comparison_section(
        "Metric Charts",
        metric_rows,
        note="Validation metrics are in-sample. Holdout uses the untouched out-of-sample window.",
    )
    selection_for_curve = payload.get("rebuild_lead_pair_selection") or payload.get("lead_pair_selection")
    if selection_for_curve is None:
        selection_for_curve = (authoritative or {}).get("authoritative_selection")
    locked_section = build_locked_candidate_section(
        selection_for_curve,
        title="Rebuild Lead (Locked) vs Benchmark",
        source_label="Rebuild",
    )
    median_section = build_simulation_median_section(
        selection_for_curve,
        title="Rebuild Median vs Benchmark",
        source_label="Rebuild",
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Package Candidate Summary</title>
  <style>
    body {{ margin:0; font-family:Georgia, serif; background:#f7f2e7; color:#17211a; }}
    main {{ max-width:980px; margin:0 auto; padding:28px 20px 48px; }}
    section {{ background:rgba(255,250,241,.92); border:1px solid rgba(23,33,26,.12); border-radius:24px; padding:24px; box-shadow:0 18px 50px rgba(33,41,36,.08); margin-top:18px; }}
    .status {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:14px; }}
    .pill {{ display:inline-flex; padding:10px 14px; border-radius:999px; font-size:.95rem; border:1px solid transparent; }}
    .pill.warn {{ background:#f7ddd3; color:#7d3d2a; border-color:rgba(125,61,42,.18); }}
    .pill.good {{ background:#dcebdd; color:#295741; border-color:rgba(41,87,65,.18); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-top:16px; }}
    .card {{ display:block; text-decoration:none; color:inherit; background:#fffdf8; border:1px solid rgba(23,33,26,.12); border-radius:20px; padding:20px; }}
    .label {{ text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; color: #6c5842; }}
    .card strong {{ display:block; margin-top:6px; font-size:1.2rem; overflow-wrap:anywhere; word-break: break-word; }}
    .card p {{ margin:8px 0 0; color:#5b6762; }}
    .muted {{ color:#5b6762; }}
    table {{ width:100%; border-collapse:collapse; margin-top:14px; background:#fffdf8; border-radius:18px; overflow:hidden; }}
    .metric-table th, .metric-table td {{ vertical-align: middle; }}
    .bar-cell {{ display: flex; align-items: center; gap: 10px; }}
    .bar-track {{
      flex: 1;
      height: 10px;
      background: rgba(23,33,26,.08);
      border-radius: 999px;
      overflow: hidden;
      min-width: 120px;
    }}
    .bar-fill {{ height: 100%; }}
    .bar-fill.auth {{ background: #295741; }}
    .bar-fill.rebuild {{ background: #b06b1d; }}
    .bar-value {{ min-width: 64px; text-align: right; font-variant-numeric: tabular-nums; }}
    .metric-note {{ color: #5b6762; font-size: 0.85rem; margin-top: 4px; }}
    .return-panel {{
      margin-top: 16px;
      background: #fffdf8;
      border: 1px solid rgba(23,33,26,.12);
      border-radius: 18px;
      padding: 14px;
    }}
    .return-curve {{
      display: block;
      width: 100%;
    }}
    .return-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 10px;
      font-size: 0.85rem;
      color: #5b6762;
    }}
    .curve-stats {{
      margin-top: 6px;
    }}
    .legend-line {{
      display: inline-block;
      width: 26px;
      height: 0;
      border-top: 3px solid #295741;
      margin-right: 6px;
      transform: translateY(-2px);
    }}
    .legend-line.benchmark {{
      border-top: 3px dashed #b06b1d;
    }}
    th, td {{ padding:12px 14px; text-align:left; border-bottom:1px solid rgba(23,33,26,.12); vertical-align:top; }}
    th {{ background:rgba(176,107,29,.08); font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; }}
    .gate-tag {{ display:inline-flex; align-items:center; padding:4px 10px; border-radius:999px; font-size:.75rem; border:1px solid transparent; }}
    .gate-pass {{ background:#dcebdd; color:#295741; border-color:rgba(41,87,65,.18); }}
    .gate-fail {{ background:#f7ddd3; color:#7d3d2a; border-color:rgba(125,61,42,.18); }}
    .gate-na {{ background:#eee8dc; color:#6a645c; border-color:rgba(22,33,26,.12); }}
    .path {{ display:inline-block; margin-top:10px; color:#b06b1d; }}
    a {{ color:#b06b1d; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Current Candidate Decision</h1>
      <p>This view is intentionally limited to the package branch you selected to present.</p>
      <div class="status">
        <span class="pill {status_tone}">{html.escape(status_label)}</span>
        <span class="pill warn">Trading model: {html.escape(model_name)}</span>
      </div>
      <p style="margin-top:16px;"><strong>Current finding:</strong> {html.escape(finding_text)}</p>
      <p><strong>Repo verdict:</strong> {html.escape(str(payload['verdict']))}</p>
    </section>
    <section>
      <h2>Decision Details (Employer-Readable)</h2>
      <div class="grid">
        <div class="card">
          <div class="label">Authoritative Locked Params</div>
          <strong>{html.escape(params_text)}</strong>
          <p>Locked tuple from the authoritative selection summary.</p>
        </div>
        <div class="card">
          <div class="label">Rebuild Locked Params</div>
          <strong>{html.escape(rebuild_params_text)}</strong>
          <p>Locked tuple from the rebuild lead pair.</p>
        </div>
        <div class="card">
          <div class="label">Rigorous PBO (Authoritative)</div>
          <strong>{html.escape(pbo_text)}</strong>
          <p>CSCV PBO threshold is 30% for a healthy warning level.</p>
        </div>
        <div class="card">
          <div class="label">Rebuild PBO (Diagnostic)</div>
          <strong>{html.escape(rebuild_pbo_text)}</strong>
          <p>CSCV PBO from the rebuild summary.</p>
        </div>
        <div class="card">
          <div class="label">Negative Controls</div>
          <strong>{html.escape(auth_cross_text)} / {html.escape(auth_block_text)}</strong>
          <p>Cross-sectional shuffle and block-shuffled null pass counts.</p>
        </div>
        <div class="card">
          <div class="label">Rebuild Negative Controls</div>
          <strong>{html.escape(rebuild_cross_text)} / {html.escape(rebuild_block_text)}</strong>
          <p>Cross-sectional shuffle and block-shuffled null pass counts.</p>
        </div>
        <div class="card">
          <div class="label">Cadence Summary Check</div>
          <strong>PBO {html.escape(lead_pbo_text)} / Holdout {html.escape(lead_holdout_text)}</strong>
          <p>Lead monthly pair from the cadence comparison summary.</p>
        </div>
        <div class="card">
          <div class="label">Authoritative Selection Path</div>
          <strong>{html.escape(selection_path)}</strong>
          <p>Source-of-truth selection summary for the package branch.</p>
        </div>
        <div class="card">
          <div class="label">Cadence Summary Path</div>
          <strong>{html.escape(cadence_summary_path)}</strong>
          <p>Authoritative cadence comparison dashboard path.</p>
        </div>
      </div>
      <p style="margin-top:16px;">If you see different PBO numbers in experimental runs, they are not authoritative unless explicitly promoted to the package branch.</p>
    </section>
    <section>
      <h2>Validation Gates (Certification)</h2>
      <table>
        <thead>
          <tr>
            <th>Gate</th>
            <th>Threshold</th>
            <th>Authoritative</th>
            <th>Pass?</th>
            <th>Rebuild</th>
            <th>Pass?</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Fold robustness</td>
            <td>&gt;= {config.MEGA_WF_PASSES_REQUIRED} of {len(config.ROLLING_ORIGIN_FOLDS)} folds with Sharpe &gt; 0.4</td>
            <td>{html.escape(auth_fold_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("fold_gate"))}</td>
            <td>{html.escape(rebuild_fold_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("fold_gate"))}</td>
          </tr>
          <tr>
            <td>Bootstrap CI</td>
            <td>Lower bound &gt; 0</td>
            <td>{html.escape(auth_bootstrap_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("bootstrap_gate"))}</td>
            <td>{html.escape(rebuild_bootstrap_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("bootstrap_gate"))}</td>
          </tr>
          <tr>
            <td>Deflated Sharpe</td>
            <td>Score &gt; 0</td>
            <td>{html.escape(auth_deflated_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("deflated_gate"))}</td>
            <td>{html.escape(rebuild_deflated_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("deflated_gate"))}</td>
          </tr>
          <tr>
            <td>Negative controls</td>
            <td>Cross-sectional pass rate &le; {html.escape(format_pct(config.NEGATIVE_CONTROL_PASS_RATE_MAX))}</td>
            <td>{html.escape(auth_cross_rate_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("neg_gate"))}</td>
            <td>{html.escape(rebuild_cross_rate_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("neg_gate"))}</td>
          </tr>
          <tr>
            <td>PBO (diagnostic)</td>
            <td>&le; {html.escape(format_pct(auth_pbo_threshold))}</td>
            <td>{html.escape(auth_pbo_text)}</td>
            <td>{gate_status_tag(auth_metrics.get("pbo_passes"))}</td>
            <td>{html.escape(rebuild_pbo_metric_text)}</td>
            <td>{gate_status_tag(rebuild_metrics.get("pbo_passes"))}</td>
          </tr>
        </tbody>
      </table>
      <p class="muted" style="margin-top:12px;">Hard gates are fold/boot/deflated/negative controls. PBO is diagnostic only.</p>
    </section>
    <section>
      <h2>Holdout Gate (Phase 4)</h2>
      <table>
        <thead>
          <tr>
            <th>Check</th>
            <th>Threshold</th>
            <th>Authoritative</th>
            <th>Pass?</th>
            <th>Rebuild</th>
            <th>Pass?</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Sharpe gate</td>
            <td>&gt;= {config.OOS_SHARPE_MIN}</td>
            <td>{html.escape(auth_holdout_sharpe_text)}</td>
            <td>{gate_status_tag(auth_holdout.get("sharpe_gate"))}</td>
            <td>{html.escape(rebuild_holdout_sharpe_text)}</td>
            <td>{gate_status_tag(rebuild_holdout.get("sharpe_gate"))}</td>
          </tr>
          <tr>
            <td>Benchmark gate</td>
            <td>Beats primary benchmark</td>
            <td>{html.escape(format_bool(auth_holdout.get("benchmark_gate")))}</td>
            <td>{gate_status_tag(auth_holdout.get("benchmark_gate"))}</td>
            <td>{html.escape(format_bool(rebuild_holdout.get("benchmark_gate")))}</td>
            <td>{gate_status_tag(rebuild_holdout.get("benchmark_gate"))}</td>
          </tr>
          <tr>
            <td>Phase 4 eligible</td>
            <td>Sharpe gate + benchmark gate</td>
            <td>{html.escape(format_bool(auth_holdout.get("phase4_eligible")))}</td>
            <td>{gate_status_tag(auth_holdout.get("phase4_eligible"))}</td>
            <td>{html.escape(format_bool(rebuild_holdout.get("phase4_eligible")))}</td>
            <td>{gate_status_tag(rebuild_holdout.get("phase4_eligible"))}</td>
          </tr>
        </tbody>
      </table>
      <p class="muted" style="margin-top:12px;">Holdout checks use the untouched out-of-sample window.</p>
    </section>

{locked_section}

{median_section}

{metric_section}

    <section>
      <h2>Experimental Rebuild (Not Authoritative)</h2>
      <div class="grid">
        <div class="card">
          <div class="label">Rebuild Status</div>
          <strong>{html.escape(rebuild_verdict)}</strong>
          <p>Tracked for research context only. Does not override the package decision.</p>
        </div>
        <div class="card">
          <div class="label">Rebuild Lead Monthly PBO</div>
          <strong>{html.escape(rebuild_pbo_text)}</strong>
          <p>Shown for transparency; not part of the package decision.</p>
        </div>
        <div class="card">
          <div class="label">Rebuild Summary Path</div>
          <strong>{html.escape(rebuild_path)}</strong>
          <p>Optional research cycle output (separate results root).</p>
        </div>
      </div>
    </section>
    <section>
      <h2>Open Next</h2>
      <div class="grid">
        <a class="card" href="../PROJECT_STATUS.html">
          <h3>Project Status</h3>
          <p>A straightforward look at what is working, what is still blocked, and what we would do next.</p>
          <span class="path">../PROJECT_STATUS.html</span>
        </a>
        <a class="card" href="../research_audit_dossier.html">
          <h3>Research Dossier</h3>
          <p>Short narrative of what the repo currently proves and why that matters.</p>
          <span class="path">../research_audit_dossier.html</span>
        </a>
      </div>
    </section>
  </main>
</body>
</html>
"""


def export_portfolio_package(
    *,
    payload: dict[str, Any],
    project_status: str,
    research_dossier: str,
    output_dir: Path,
    results_root: Path,
) -> Path:
    portfolio_dir = results_root.parent / "portfolio"
    package_dir = portfolio_dir / "alpha_momentum_validation_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    for stale_dir in (package_dir / "forward_monitor",):
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)
    for stale_dir in (package_dir / "cadence_compare", package_dir / "cadence_compare_summary"):
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)
    for stale_pattern in ("lead_thesis_*", "shadow_control_*"):
        for stale_dir in package_dir.glob(stale_pattern):
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir, ignore_errors=True)

    cadence_package_dir = package_dir / "cadence_compare_summary"
    has_cadence_summary = bool(payload.get("authoritative_status", {}).get("summary_available"))
    if has_cadence_summary:
        cadence_package_dir.mkdir(parents=True, exist_ok=True)
        cadence_summary_payload = build_package_cadence_summary_payload(payload.get("authoritative_status"))
        serialize_json(cadence_package_dir / "cadence_comparison.json", cadence_summary_payload)
        report_text = build_package_cadence_summary_report(payload.get("authoritative_status"))
        (cadence_package_dir / "dashboard.html").write_text(
            build_package_cadence_summary_dashboard(payload.get("authoritative_status")),
            encoding="utf-8",
        )
        (cadence_package_dir / "cadence_comparison_report.html").write_text(
            build_markdown_dashboard_html(
            title="Cadence Summary Report",
            subtitle="Short written version of the selected package verdict.",
            markdown_text=report_text,
        ),
            encoding="utf-8",
        )

    rebuild_summary_path = payload.get("authoritative_status", {}).get("rebuild_summary_path")
    if rebuild_summary_path:
        rebuild_src_dir = Path(rebuild_summary_path).parent
        rebuild_root = rebuild_src_dir.parent
        rebuild_pkg_dir = package_dir / "research_engine_rebuild"
        rebuild_pkg_summary = rebuild_pkg_dir / "summary"
        rebuild_pkg_summary.mkdir(parents=True, exist_ok=True)
        copy_if_exists(rebuild_src_dir / "dashboard.html", rebuild_pkg_summary / "dashboard.html")
        copy_if_exists(rebuild_src_dir / "cadence_comparison.json", rebuild_pkg_summary / "cadence_comparison.json")
        copy_if_exists(
            rebuild_src_dir / "cadence_comparison_report.html",
            rebuild_pkg_summary / "cadence_comparison_report.html",
        )
        summary_json = rebuild_src_dir / "cadence_comparison.json"
        if summary_json.exists():
            rebuild_payload = load_json(summary_json)
            for pair in rebuild_payload.get("pairs", []):
                pair_output_dir = pair.get("output_dir")
                if not pair_output_dir:
                    continue
                output_path = Path(pair_output_dir)
                if not output_path.exists():
                    continue
                try:
                    rel_path = output_path.resolve().relative_to(rebuild_root.resolve())
                except ValueError:
                    continue
                destination = rebuild_pkg_dir / rel_path
                if destination.exists():
                    shutil.rmtree(destination, ignore_errors=True)
                shutil.copytree(output_path, destination, dirs_exist_ok=True)

    lead = payload["theses"][0]
    shadow = payload["theses"][1] if len(payload["theses"]) > 1 else None

    readme = build_portfolio_package_readme(
        lead=lead,
        shadow=shadow,
        authoritative=payload.get("authoritative_status"),
        has_cadence_summary=has_cadence_summary,
    )
    (package_dir / "README.html").write_text(
        build_markdown_dashboard_html(
            title="Validation Package README",
            subtitle="What is in the package and where to start.",
            markdown_text=readme,
        ),
        encoding="utf-8",
    )
    (package_dir / "PROJECT_STATUS.html").write_text(
        build_markdown_dashboard_html(
            title="Current Project Status",
            subtitle="A clear look at where the project stands and what we would do next.",
            markdown_text=project_status,
        ),
        encoding="utf-8",
    )
    (package_dir / "research_audit_dossier.html").write_text(
        build_markdown_dashboard_html(
            title="Research Dossier",
            subtitle="A short write-up of the validation process and what the current result actually means.",
            markdown_text=research_dossier,
        ),
        encoding="utf-8",
    )
    (package_dir / "full_research_report.html").write_text(
        build_full_research_report_html(results_root, package_dir),
        encoding="utf-8",
    )
    (package_dir / "dashboard.html").write_text(
        build_portfolio_package_dashboard_html(
            lead=lead,
            shadow=shadow,
            authoritative=payload.get("authoritative_status"),
            has_cadence_summary=has_cadence_summary,
        ),
        encoding="utf-8",
    )
    (portfolio_dir / "README.html").write_text(
        build_markdown_dashboard_html(
            title="Portfolio Folder",
            subtitle="Starting point for the validation package visuals.",
            markdown_text=build_portfolio_root_readme(package_name=package_dir.name),
            back_href=f"{package_dir.name}/dashboard.html",
            back_label="Open package dashboard",
        ),
        encoding="utf-8",
    )
    (portfolio_dir / "dashboard.html").write_text(
        build_portfolio_root_dashboard_html(package_name=package_dir.name),
        encoding="utf-8",
    )
    return package_dir


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_bool(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def format_pass_rate(pass_count: int | None, run_count: int | None) -> str:
    if not run_count:
        return "n/a"
    rate = pass_count / run_count if pass_count is not None else 0.0
    return f"{pass_count}/{run_count} ({rate * 100.0:.1f}%)"


def gate_status_tag(passed: bool | None) -> str:
    if passed is True:
        label = "PASS"
        cls = "gate-pass"
    elif passed is False:
        label = "FAIL"
        cls = "gate-fail"
    else:
        label = "n/a"
        cls = "gate-na"
    return f'<span class="gate-tag {cls}">{label}</span>'


def extract_gate_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    summary = summary or {}
    locked = summary.get("locked_candidate") or {}
    backtest = summary.get("backtest_overfitting", {})
    negative_controls = summary.get("negative_controls", {})
    cross = negative_controls.get("cross_sectional_shuffle") or {}
    block = negative_controls.get("block_shuffled_null") or {}
    pass_count = cross.get("pass_count")
    run_count = cross.get("run_count")
    block_pass = block.get("pass_count")
    block_run = block.get("run_count")

    fold_pass_count = locked.get("fold_pass_count")
    fold_gate = locked.get("gate_fold_count")
    if fold_gate is None and fold_pass_count is not None:
        fold_gate = fold_pass_count >= config.MEGA_WF_PASSES_REQUIRED

    bootstrap_low = locked.get("bootstrap_ci_low")
    bootstrap_high = locked.get("bootstrap_ci_high")
    median_validation_sharpe = locked.get("median_validation_sharpe")
    overall_sharpe = locked.get("overall_sharpe")
    bootstrap_gate = locked.get("gate_bootstrap")
    if bootstrap_gate is None and bootstrap_low is not None:
        bootstrap_gate = bootstrap_low > 0.0

    deflated_score = locked.get("deflated_sharpe_score")
    deflated_gate = locked.get("gate_deflated_sharpe")
    if deflated_gate is None and deflated_score is not None:
        deflated_gate = deflated_score > 0.0

    neg_gate = locked.get("gate_negative_controls")
    if neg_gate is None and run_count:
        neg_gate = (pass_count or 0) / run_count <= config.NEGATIVE_CONTROL_PASS_RATE_MAX

    pbo = backtest.get("pbo")
    pbo_threshold = backtest.get("pbo_threshold_max") or config.PBO_THRESHOLD_MAX
    pbo_passes = backtest.get("passes_pbo_threshold")
    if pbo_passes is None and pbo is not None:
        pbo_passes = pbo <= pbo_threshold

    return {
        "median_validation_sharpe": median_validation_sharpe,
        "overall_sharpe": overall_sharpe,
        "fold_pass_count": fold_pass_count,
        "fold_gate": fold_gate,
        "bootstrap_ci_low": bootstrap_low,
        "bootstrap_ci_high": bootstrap_high,
        "bootstrap_gate": bootstrap_gate,
        "deflated_sharpe_score": deflated_score,
        "deflated_gate": deflated_gate,
        "neg_pass_count": pass_count,
        "neg_run_count": run_count,
        "block_pass_count": block_pass,
        "block_run_count": block_run,
        "neg_gate": neg_gate,
        "pbo": pbo,
        "pbo_threshold": pbo_threshold,
        "pbo_passes": pbo_passes,
    }


def extract_holdout_gate(holdout: dict[str, Any] | None) -> dict[str, Any]:
    holdout = holdout or {}
    phase4 = holdout.get("phase4_gate", {})
    sharpe = phase4.get("base_main_net_sharpe")
    sharpe_gate = phase4.get("meets_sharpe_gate")
    if sharpe_gate is None and sharpe is not None:
        sharpe_gate = sharpe >= config.OOS_SHARPE_MIN
    benchmark_gate = phase4.get("beats_primary_benchmark")
    phase4_eligible = phase4.get("phase4_eligible")
    return {
        "holdout_sharpe": sharpe,
        "sharpe_gate": sharpe_gate,
        "benchmark_gate": benchmark_gate,
        "phase4_eligible": phase4_eligible,
    }


def format_currency_msek(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1_000_000.0:,.1f}m SEK"


def format_bps(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f} bps"


def status_badge(label: str, tone: str) -> str:
    return f'<span class="badge {tone}">{html.escape(label)}</span>'


def format_metric_value(value: float | None, *, fmt: str = "float", digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if fmt == "pct":
        return f"{value * 100.0:.1f}%"
    if fmt == "int":
        return f"{int(value)}"
    return f"{value:.{digits}f}"


def metric_value_for_width(value: float | None, *, fmt: str = "float") -> float | None:
    if value is None:
        return None
    if fmt == "pct":
        return value * 100.0
    return float(value)


def build_metric_comparison_section(title: str, rows: list[dict[str, Any]], note: str | None = None) -> str:
    def bar_cell(value: float | None, max_value: float, fmt: str, tone: str) -> str:
        display = format_metric_value(value, fmt=fmt)
        if value is None or max_value <= 0:
            pct = 0.0
        else:
            width_value = metric_value_for_width(value, fmt=fmt) or 0.0
            pct = min(100.0, max(0.0, width_value / max_value * 100.0))
        return (
            '<div class="bar-cell">'
            f'<div class="bar-track"><div class="bar-fill {tone}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="bar-value">{html.escape(display)}</div>'
            "</div>"
        )

    row_html = []
    for row in rows:
        label = row["label"]
        note_text = row.get("note")
        fmt = row.get("fmt", "float")
        auth_value = row.get("auth")
        rebuild_value = row.get("rebuild")
        values_for_width = [
            metric_value_for_width(auth_value, fmt=fmt),
            metric_value_for_width(rebuild_value, fmt=fmt),
        ]
        max_value = max([value for value in values_for_width if value is not None] or [1.0])
        label_html = html.escape(label)
        if note_text:
            label_html = f'{label_html}<div class="metric-note">{html.escape(note_text)}</div>'
        row_html.append(
            "<tr>"
            f"<td>{label_html}</td>"
            f"<td>{bar_cell(auth_value, max_value, fmt, 'auth')}</td>"
            f"<td>{bar_cell(rebuild_value, max_value, fmt, 'rebuild')}</td>"
            "</tr>"
        )
    note_html = f'<p class="muted" style="margin-top:10px;">{html.escape(note)}</p>' if note else ""
    return (
        f'<section class="section">'
        f"<h2>{html.escape(title)}</h2>"
        '<table class="metric-table">'
        "<thead><tr><th>Metric</th><th>Authoritative</th><th>Rebuild</th></tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody></table>{note_html}</section>"
    )


def _median_simulation_series(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    if not selection:
        return None
    candidates = selection.get("ranked_candidates") or []
    if not candidates:
        return None
    candidate_returns: list[list[float]] = []
    for candidate in candidates:
        series = candidate.get("concatenated_returns")
        if isinstance(series, list) and series:
            candidate_returns.append([float(value) for value in series])
    if not candidate_returns:
        return None
    min_len = min(len(series) for series in candidate_returns)
    if min_len < 2:
        return None
    candidate_returns = [series[:min_len] for series in candidate_returns]
    median_returns = np.nanmedian(np.array(candidate_returns, dtype=float), axis=0).tolist()
    benchmark_series = None
    for candidate in candidates:
        series = candidate.get("primary_benchmark_returns")
        if isinstance(series, list) and series:
            benchmark_series = [float(value) for value in series]
            break
    if benchmark_series:
        min_len = min(min_len, len(benchmark_series))
        median_returns = median_returns[:min_len]
        benchmark_series = benchmark_series[:min_len]
    return {
        "median_returns": median_returns,
        "benchmark_returns": benchmark_series or [],
        "candidate_count": len(candidate_returns),
        "period_count": min_len,
        "period_label": selection.get("period_label") or "periods",
        "periods_per_year": selection.get("periods_per_year"),
    }


def _locked_candidate_series(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    if not selection:
        return None
    candidate = selection.get("locked_candidate") or {}
    if not candidate:
        candidates = selection.get("ranked_candidates") or []
        selected_id = selection.get("selected_candidate_id")
        if selected_id:
            candidate = next(
                (item for item in candidates if item.get("candidate_id") == selected_id),
                {},
            )
        if not candidate:
            candidate = next((item for item in candidates if item.get("selected")), {})
    returns = candidate.get("concatenated_returns")
    if not isinstance(returns, list) or not returns:
        return None
    return_series = [float(value) for value in returns]
    benchmark_raw = (
        candidate.get("primary_benchmark_returns")
        or selection.get("primary_benchmark_returns")
        or []
    )
    benchmark_series: list[float] = []
    if isinstance(benchmark_raw, list) and benchmark_raw:
        benchmark_series = [float(value) for value in benchmark_raw]
        min_len = min(len(return_series), len(benchmark_series))
        return_series = return_series[:min_len]
        benchmark_series = benchmark_series[:min_len]
    period_label = candidate.get("period_label") or selection.get("period_label") or "periods"
    periods_per_year = candidate.get("periods_per_year") or selection.get("periods_per_year")
    return {
        "returns": return_series,
        "benchmark_returns": benchmark_series,
        "period_count": len(return_series),
        "period_label": period_label,
        "periods_per_year": periods_per_year,
        "candidate_id": candidate.get("candidate_id"),
    }


def render_return_curve(
    returns: Sequence[float] | None,
    benchmark: Sequence[float] | None = None,
    *,
    line_color: str = "#295741",
    benchmark_color: str = "#b06b1d",
    period_label: str | None = None,
) -> str:
    if not returns:
        return '<div class="muted">No simulation curve available.</div>'
    equity = [1.0]
    for value in returns:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not np.isfinite(numeric):
            numeric = 0.0
        equity.append(equity[-1] * (1.0 + numeric))
    bench_equity: list[float] = []
    if benchmark:
        bench_equity = [1.0]
        for value in benchmark:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 0.0
            if not np.isfinite(numeric):
                numeric = 0.0
            bench_equity.append(bench_equity[-1] * (1.0 + numeric))
    if len(equity) < 2:
        return '<div class="muted">No simulation curve available.</div>'
    min_val = min(equity + (bench_equity or []))
    max_val = max(equity + (bench_equity or []))
    span = max(max_val - min_val, 1e-9)

    width = 640.0
    height = 200.0
    margin_left = 52.0
    margin_right = 16.0
    margin_top = 12.0
    margin_bottom = 34.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def _points(series: list[float]) -> str:
        if len(series) < 2:
            return ""
        items = []
        last_index = len(series) - 1
        for idx, value in enumerate(series):
            x = margin_left + idx / last_index * plot_width
            y = margin_top + (1.0 - (value - min_val) / span) * plot_height
            items.append(f"{x:.2f},{y:.2f}")
        return " ".join(items)

    def _tick_values(min_value: float, max_value: float, ticks: int = 4) -> list[float]:
        if ticks < 2:
            return [min_value, max_value]
        raw = np.linspace(min_value, max_value, ticks)
        return [float(value) for value in raw]

    candidate_points = _points(equity)
    benchmark_points = _points(bench_equity) if bench_equity else ""
    benchmark_poly = (
        f'<polyline points="{benchmark_points}" fill="none" stroke="{benchmark_color}" stroke-width="2" '
        'stroke-dasharray="5 4" />'
        if benchmark_points
        else ""
    )

    y_ticks = _tick_values(min_val, max_val, ticks=4)
    y_tick_lines = []
    for tick in y_ticks:
        y = margin_top + (1.0 - (tick - min_val) / span) * plot_height
        y_tick_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="rgba(22,33,30,0.08)" />'
        )
        y_tick_lines.append(
            f'<line x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="rgba(22,33,30,0.35)" />'
        )
        y_tick_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 3:.2f}" text-anchor="end" font-size="10" fill="#6c5842">{tick:.2f}</text>'
        )

    period_count = len(returns)
    last_index = max(period_count, 1)
    x_ticks = [0, last_index // 2, last_index]
    x_tick_lines = []
    for tick in x_ticks:
        x = margin_left + (tick / last_index) * plot_width
        x_tick_lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 6}" stroke="rgba(22,33,30,0.35)" />'
        )
        x_tick_lines.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 20}" text-anchor="middle" font-size="10" fill="#6c5842">{tick}</text>'
        )

    period_label = (period_label or "periods").strip()
    x_axis_label = period_label.capitalize()
    return (
        f'<svg class="return-curve" viewBox="0 0 {width:.0f} {height:.0f}" width="100%" height="200" preserveAspectRatio="none">'
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="rgba(22,33,30,0.18)" />'
        f'{"".join(y_tick_lines)}'
        f'{"".join(x_tick_lines)}'
        f'<polyline points="{candidate_points}" fill="none" stroke="{line_color}" stroke-width="2.6" />'
        f"{benchmark_poly}"
        f'<text x="{margin_left}" y="{margin_top - 2}" text-anchor="start" font-size="10" fill="#6c5842">Equity multiple</text>'
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 6}" text-anchor="middle" font-size="10" fill="#6c5842">{html.escape(x_axis_label)}</text>'
        "</svg>"
    )


def build_simulation_median_section(
    selection: dict[str, Any] | None,
    *,
    section_class: str | None = None,
    title: str = "Simulation Median vs Benchmark",
    source_label: str | None = None,
) -> str:
    series = _median_simulation_series(selection)
    class_attr = f' class="{section_class}"' if section_class else ""
    if series is None:
        label_prefix = f"{source_label} " if source_label else ""
        return (
            f"<section{class_attr}><h2>{html.escape(title)}</h2>"
            f"<p class=\"muted\">{html.escape(label_prefix)}median simulation curve not available.</p></section>"
        )
    curve_html = render_return_curve(
        series["median_returns"],
        series.get("benchmark_returns"),
        period_label=series.get("period_label"),
    )
    candidate_count = series.get("candidate_count", 0)
    period_label = series.get("period_label", "periods")
    period_count = series.get("period_count", 0)
    periods_per_year = series.get("periods_per_year")
    if periods_per_year is None:
        if str(period_label).lower().startswith("week"):
            periods_per_year = 52
        elif str(period_label).lower().startswith("month"):
            periods_per_year = 12
        else:
            periods_per_year = 12
    label_prefix = f"{source_label} " if source_label else ""
    note = (
        f"{label_prefix}median across {candidate_count} simulations over {period_count} {period_label} in the testing window."
        if candidate_count
        else f"{label_prefix}median across all simulations in the testing window."
    )

    def _equity_multiple(returns: Sequence[float]) -> float:
        value = 1.0
        for item in returns:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                numeric = 0.0
            if not np.isfinite(numeric):
                numeric = 0.0
            value *= 1.0 + numeric
        return float(value)

    def _cagr(equity_multiple: float, periods: int) -> float | None:
        if not periods or not periods_per_year:
            return None
        years = periods / float(periods_per_year)
        if years <= 0:
            return None
        return equity_multiple ** (1.0 / years) - 1.0

    median_eq = _equity_multiple(series["median_returns"]) if series.get("median_returns") else None
    bench_eq = _equity_multiple(series.get("benchmark_returns") or []) if series.get("benchmark_returns") else None
    median_cagr = _cagr(median_eq, period_count) if median_eq is not None else None
    bench_cagr = _cagr(bench_eq, period_count) if bench_eq is not None else None
    stats_parts = []
    if median_eq is not None and median_cagr is not None:
        stats_parts.append(f"Median total multiple: {median_eq:.2f}x (CAGR {median_cagr * 100.0:.1f}%)")
    if bench_eq is not None and bench_cagr is not None:
        stats_parts.append(f"Benchmark total multiple: {bench_eq:.2f}x (CAGR {bench_cagr * 100.0:.1f}%)")
    stats_text = " · ".join(stats_parts)
    return (
        f"<section{class_attr}>"
        f"<h2>{html.escape(title)}</h2>"
        f'<p class="muted">{html.escape(note)} Primary benchmark shown for comparison.</p>'
        f'{f"<p class=\"muted curve-stats\">{html.escape(stats_text)}</p>" if stats_text else ""}'
        '<div class="return-panel">'
        f"{curve_html}"
        '<div class="return-legend">'
        '<span><span class="legend-line"></span>Median simulation</span>'
        '<span><span class="legend-line benchmark"></span>Primary benchmark</span>'
        "</div></div></section>"
    )


def build_locked_candidate_section(
    selection: dict[str, Any] | None,
    *,
    section_class: str | None = None,
    title: str = "Lead (Locked) vs Benchmark",
    source_label: str | None = None,
) -> str:
    series = _locked_candidate_series(selection)
    class_attr = f' class="{section_class}"' if section_class else ""
    if series is None:
        label_prefix = f"{source_label} " if source_label else ""
        return (
            f"<section{class_attr}><h2>{html.escape(title)}</h2>"
            f"<p class=\"muted\">{html.escape(label_prefix)}locked candidate curve not available.</p></section>"
        )
    curve_html = render_return_curve(
        series["returns"],
        series.get("benchmark_returns"),
        period_label=series.get("period_label"),
    )
    period_label = series.get("period_label", "periods")
    period_count = series.get("period_count", 0)
    periods_per_year = series.get("periods_per_year")
    if periods_per_year is None:
        if str(period_label).lower().startswith("week"):
            periods_per_year = 52
        elif str(period_label).lower().startswith("month"):
            periods_per_year = 12
        else:
            periods_per_year = 12
    label_prefix = f"{source_label} " if source_label else ""
    candidate_id = series.get("candidate_id")
    candidate_note = f" Candidate: {candidate_id}." if candidate_id else ""
    flat_count = sum(
        1
        for value in series["returns"]
        if np.isfinite(value) and abs(float(value)) <= 1e-12
    )
    flat_note = f" Flat months (cash): {flat_count}." if period_count else ""
    note = (
        f"{label_prefix}locked candidate curve over {period_count} {period_label} in the testing window."
        f"{candidate_note}{flat_note} Flat months occur when the trend filter is off."
    )

    def _equity_multiple(returns: Sequence[float]) -> float:
        value = 1.0
        for item in returns:
            try:
                numeric = float(item)
            except (TypeError, ValueError):
                numeric = 0.0
            if not np.isfinite(numeric):
                numeric = 0.0
            value *= 1.0 + numeric
        return float(value)

    def _cagr(equity_multiple: float, periods: int) -> float | None:
        if not periods or not periods_per_year:
            return None
        years = periods / float(periods_per_year)
        if years <= 0:
            return None
        return equity_multiple ** (1.0 / years) - 1.0

    eq_multiple = _equity_multiple(series["returns"]) if series.get("returns") else None
    bench_eq = _equity_multiple(series.get("benchmark_returns") or []) if series.get("benchmark_returns") else None
    eq_cagr = _cagr(eq_multiple, period_count) if eq_multiple is not None else None
    bench_cagr = _cagr(bench_eq, period_count) if bench_eq is not None else None
    stats_parts = []
    if eq_multiple is not None and eq_cagr is not None:
        stats_parts.append(f"Lead total multiple: {eq_multiple:.2f}x (CAGR {eq_cagr * 100.0:.1f}%)")
    if bench_eq is not None and bench_cagr is not None:
        stats_parts.append(f"Benchmark total multiple: {bench_eq:.2f}x (CAGR {bench_cagr * 100.0:.1f}%)")
    stats_text = " Â· ".join(stats_parts)
    return (
        f"<section{class_attr}>"
        f"<h2>{html.escape(title)}</h2>"
        f'<p class="muted">{html.escape(note)} Primary benchmark shown for comparison.</p>'
        f'{f"<p class=\"muted curve-stats\">{html.escape(stats_text)}</p>" if stats_text else ""}'
        '<div class="return-panel">'
        f"{curve_html}"
        '<div class="return-legend">'
        '<span><span class="legend-line"></span>Lead (locked)</span>'
        '<span><span class="legend-line benchmark"></span>Primary benchmark</span>'
        "</div></div></section>"
    )


def _markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    html_lines: list[str] = []
    in_list = False
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if line.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
            continue
        if line.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
            continue
        if line.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
            continue
        if line.lstrip().startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{html.escape(line.lstrip()[2:])}</li>")
            continue
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        html_lines.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def build_markdown_dashboard_html(
    *,
    title: str,
    subtitle: str,
    markdown_text: str,
    eyebrow: str = "Package Dashboard",
    back_href: str = "dashboard.html",
    back_label: str = "Back to package dashboard",
) -> str:
    body_html = _markdown_to_html(markdown_text)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --hero: rgba(255, 253, 249, 0.98);
      --panel: rgba(251, 248, 241, 0.96);
      --ink: #1b2420;
      --muted: #5f685f;
      --muted-strong: #37443d;
      --line: rgba(27, 36, 32, 0.12);
      --accent: #1f6159;
      --accent-strong: #154943;
      --shadow: 0 22px 54px rgba(34, 41, 37, 0.08);
      --blur: 14px;
      --saturate: 120%;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Palatino Linotype", Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31, 97, 89, 0.12), transparent 32%),
        radial-gradient(circle at top right, rgba(166, 109, 69, 0.11), transparent 26%),
        radial-gradient(circle at 18% 78%, rgba(120, 144, 125, 0.08), transparent 24%),
        linear-gradient(180deg, #fbf8f2 0%, #f5f1e8 48%, #e3dbcd 100%);
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.06) 25%, transparent 25%) 0 0 / 38px 38px,
        linear-gradient(315deg, rgba(255, 255, 255, 0.04) 25%, transparent 25%) 0 0 / 38px 38px;
      opacity: 0.24;
    }}
    main {{
      position: relative;
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      flex-wrap: wrap;
      margin-bottom: 18px;
      padding: 12px 14px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: rgba(250, 247, 240, 0.82);
      backdrop-filter: blur(var(--blur)) saturate(var(--saturate));
      -webkit-backdrop-filter: blur(var(--blur)) saturate(var(--saturate));
      box-shadow:
        var(--shadow),
        inset 0 1px 0 rgba(255, 255, 255, 0.72);
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-mark {{
      width: 42px;
      height: 42px;
      border-radius: 14px;
      background: linear-gradient(160deg, #1f6159, #154943);
      color: #f7f2e8;
      display: grid;
      place-items: center;
      font: 700 0.84rem Consolas, "Courier New", monospace;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border: 1px solid rgba(21, 73, 67, 0.24);
      box-shadow:
        0 10px 22px rgba(21, 73, 67, 0.14),
        inset 0 1px 0 rgba(255, 255, 255, 0.18);
    }}
    .brand-copy strong,
    .brand-copy span {{
      display: block;
    }}
    .brand-copy strong {{
      font-size: 1rem;
      color: var(--ink);
    }}
    .brand-copy span {{
      margin-top: 3px;
      color: var(--muted-strong);
      font-size: 0.88rem;
    }}
    .topbar a,
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 11px 16px;
      border-radius: 999px;
      border: 1px solid rgba(27, 36, 32, 0.12);
      color: var(--ink);
      background: rgba(255, 253, 249, 0.92);
      text-decoration: none;
      backdrop-filter: blur(calc(var(--blur) * 0.7));
      -webkit-backdrop-filter: blur(calc(var(--blur) * 0.7));
      box-shadow:
        0 10px 22px rgba(34, 41, 37, 0.06),
        inset 0 1px 0 rgba(255, 255, 255, 0.68);
      transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }}
    .topbar a:hover,
    .button:hover {{
      transform: translateY(-1px);
      border-color: rgba(31, 97, 89, 0.2);
      box-shadow:
        0 12px 24px rgba(34, 41, 37, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.72);
    }}
    .hero,
    .section {{
      border: 1px solid var(--line);
      border-radius: 30px;
      padding: 34px 36px;
      box-shadow:
        var(--shadow),
        inset 0 1px 0 rgba(255, 255, 255, 0.62);
    }}
    .hero {{ background: var(--hero); }}
    .section {{ margin-top: 24px; background: var(--panel); }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      font: 0.78rem Consolas, "Courier New", monospace;
      color: var(--accent-strong);
      background: rgba(31, 97, 89, 0.08);
      border: 1px solid rgba(31, 97, 89, 0.15);
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2.6rem, 5vw, 4.2rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
      color: var(--ink);
    }}
    h2,
    h3 {{
      margin-top: 28px;
      color: var(--ink);
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.24);
    }}
    h2 {{
      font-size: clamp(1.7rem, 3vw, 2.45rem);
      letter-spacing: -0.02em;
    }}
    h3 {{
      font-size: 1.2rem;
    }}
    p,
    li {{
      color: var(--muted-strong);
      line-height: 1.72;
      font-size: 1.02rem;
    }}
    .hero p {{
      max-width: 72ch;
      margin-top: 18px;
      color: var(--muted-strong);
    }}
    .hero-actions {{
      margin-top: 20px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .markdown-body > :first-child {{
      margin-top: 0;
    }}
    .markdown-body h1 {{
      font-size: clamp(2.2rem, 4vw, 3.4rem);
      margin-bottom: 8px;
    }}
    .markdown-body h2,
    .markdown-body h3 {{
      margin-bottom: 10px;
    }}
    ul,
    ol {{
      margin: 14px 0 18px 1.35rem;
      padding: 0;
    }}
    li {{
      margin-bottom: 8px;
    }}
    a {{
      color: var(--accent-strong);
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.95em;
      color: var(--ink);
      background: rgba(27, 36, 32, 0.05);
      padding: 0.15em 0.4em;
      border-radius: 8px;
      border: 1px solid rgba(27, 36, 32, 0.1);
    }}
    pre {{
      margin: 18px 0;
      padding: 14px 16px;
      border-radius: 18px;
      overflow: auto;
      border: 1px solid rgba(27, 36, 32, 0.1);
      background: rgba(248, 244, 236, 0.92);
    }}
    pre code {{
      padding: 0;
      background: transparent;
      border: 0;
    }}
    blockquote {{
      margin: 18px 0;
      padding-left: 16px;
      border-left: 3px solid rgba(31, 97, 89, 0.22);
    }}
    hr {{
      border: 0;
      border-top: 1px solid var(--line);
      margin: 28px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 18px;
    }}
    th,
    td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(27, 36, 32, 0.1);
      vertical-align: top;
    }}
    th {{
      color: var(--accent-strong);
      font: 0.78rem Consolas, "Courier New", monospace;
      text-transform: uppercase;
      letter-spacing: 0.10em;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 20px 14px 36px; }}
      .hero,
      .section {{ padding: 24px 20px; border-radius: 24px; }}
    }}
  </style>
</head>
<body class="am-theme">
  <main>
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">FT</div>
        <div class="brand-copy">
          <strong>{html.escape(title)}</strong>
          <span>{html.escape(eyebrow)}</span>
        </div>
      </div>
      <a href="{html.escape(back_href)}">{html.escape(back_label)}</a>
    </header>
    <section class="hero">
      <span class="eyebrow">{html.escape(eyebrow)}</span>
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(subtitle)}</p>
      <div class="hero-actions">
        <a class="button" href="{html.escape(back_href)}">{html.escape(back_label)}</a>
      </div>
    </section>
    <section class="section markdown-body">
      {body_html}
    </section>
  </main>
</body>
</html>
"""


def render_positions_table(record: dict[str, Any]) -> str:
    rows = []
    for position in record["positions"]:
        rows.append(
            "<tr>"
            f"<td>{position['position']}</td>"
            f"<td>{html.escape(position['ticker_local'] or position['security_id'])}</td>"
            f"<td>{html.escape(position['company_name'] or '')}</td>"
            f"<td>{html.escape(position['country_code'] or '')}</td>"
            f"<td>{html.escape(position['action'])}</td>"
            f"<td>{format_float(position.get('signal_score'))}</td>"
            f"<td>{format_currency_msek(position.get('median_daily_value_60d_sek'))}</td>"
            f"<td>{format_pct(position.get('adv_participation_ratio'))}</td>"
            f"<td>{format_float(position.get('one_way_cost_bps'), 1)}</td>"
            f"<td>{html.escape(position.get('data_status') or 'n/a')}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="10" class="empty">No picks available.</td></tr>'


def thesis_card(thesis: dict[str, Any]) -> str:
    current = thesis["current_pick"]
    entered = ", ".join(current["entry_security_ids"]) or "none"
    exited = ", ".join(current["exit_security_ids"]) or "none"
    candidate_line = html.escape(thesis.get("candidate_label") or "locked candidate")
    return f"""
<article class="card">
  <div class="head">
    <div>
      <h3>{html.escape(thesis['label'])}</h3>
      <p><strong>Candidate:</strong> {candidate_line}</p>
      <p>{html.escape(thesis['params_label'])}</p>
    </div>
    {status_badge(thesis.get("role_display") or ("Lead" if thesis["role"] == "lead" else "Shadow"), "good" if thesis["role"] == "lead" else "muted")}
  </div>
  <p>{html.escape(thesis['scope_note'])}</p>
  <div class="metrics">
    <div><span>Signal month</span><strong>{html.escape(current['signal_month'])}</strong></div>
    <div><span>Holding month</span><strong>{html.escape(current['holding_month'] or 'n/a')}</strong></div>
    <div><span>Eligible names</span><strong>{current['eligible_count']}</strong></div>
    <div><span>Execution date</span><strong>{html.escape(str(current['next_execution_date']) if current['next_execution_date'] else 'n/a')}</strong></div>
    <div><span>New entries</span><strong>{html.escape(entered)}</strong></div>
    <div><span>Exits vs last realized month</span><strong>{html.escape(exited)}</strong></div>
  </div>
  <table>
    <thead>
      <tr><th>Pos</th><th>Ticker</th><th>Company</th><th>Country</th><th>Action</th><th>Score</th><th>ADV 60d</th><th>ADV use</th><th>1w cost bps</th><th>Data</th></tr>
    </thead>
    <tbody>{render_positions_table(current)}</tbody>
  </table>
</article>
"""


def render_history_rows(payload: dict[str, Any]) -> str:
    theses = payload["theses"]
    if len(theses) < 2:
        return '<tr><td colspan="6" class="empty">Need two theses to compare overlap history.</td></tr>'
    lead = theses[0]
    comparison = theses[1]
    comparison_by_month = {row["holding_month"]: row for row in comparison["history"]}
    rows: list[str] = []
    for row in lead["history"]:
        peer = comparison_by_month.get(row["holding_month"])
        overlap = overlap_summary(row["selected_security_ids"], peer["selected_security_ids"] if peer else [])
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['holding_month'])}</td>"
            f"<td>{format_float(row.get('net_return'))}</td>"
            f"<td>{format_float(peer.get('net_return')) if peer else 'n/a'}</td>"
            f"<td>{overlap['overlap_count']}</td>"
            f"<td>{format_pct(overlap['overlap_share_of_smaller_book'])}</td>"
            f"<td>{html.escape(', '.join(overlap['overlap_security_ids']) or 'none')}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="6" class="empty">No post-holdout realized months are available yet.</td></tr>'


def audit_tone(audit: dict[str, Any]) -> str:
    if audit["missing_asof_count"] or audit["stale_name_count"] or audit["missing_adv_count"]:
        return "warn"
    if audit["top_country_weight"] and float(audit["top_country_weight"]) >= 0.60:
        return "warn"
    if audit["max_adv_participation_ratio"] and float(audit["max_adv_participation_ratio"]) >= 0.005:
        return "warn"
    return "good"


def render_audit_card(thesis: dict[str, Any]) -> str:
    audit = thesis["pre_trade_audit"]
    top_country = (
        f"{audit['top_country_code']} {format_pct(audit['top_country_weight'])}"
        if audit["top_country_code"] is not None
        else "n/a"
    )
    flags = "".join(f"<li>{html.escape(flag)}</li>" for flag in audit["flags"])
    return f"""
<article class="card">
  <div class="head">
    <div>
      <h3>{html.escape(thesis['label'])}</h3>
      <p>{html.escape(thesis['params_label'])}</p>
    </div>
    {status_badge('Watch' if audit_tone(audit) == 'warn' else 'Ready', audit_tone(audit))}
  </div>
  <div class="metrics">
    <div><span>Top country</span><strong>{html.escape(top_country)}</strong></div>
    <div><span>Max ADV usage</span><strong>{format_pct(audit['max_adv_participation_ratio'])}</strong></div>
    <div><span>Average ADV usage</span><strong>{format_pct(audit['avg_adv_participation_ratio'])}</strong></div>
    <div><span>Book avg 1w cost</span><strong>{format_bps(audit['avg_one_way_cost_bps'])}</strong></div>
    <div><span>Estimated rebalance drag</span><strong>{format_bps(audit['estimated_rebalance_cost_bps'])}</strong></div>
    <div><span>Name turnover</span><strong>{format_pct(audit['turnover_name_fraction'])}</strong></div>
  </div>
  <div class="metrics">
    <div><span>Stale as-of names</span><strong>{audit['stale_name_count']}</strong></div>
    <div><span>Missing as-of names</span><strong>{audit['missing_asof_count']}</strong></div>
    <div><span>Missing ADV names</span><strong>{audit['missing_adv_count']}</strong></div>
  </div>
  <ul>{flags}</ul>
</article>
"""


def pre_trade_audit_section(payload: dict[str, Any]) -> str:
    cards = "".join(render_audit_card(thesis) for thesis in payload["theses"])
    return (
        '<section class="section">'
        '<div class="head"><div><h2>Pre-Trade Audit</h2>'
        '<p>These checks are allowed because they stress the frozen live books without reopening the search space.</p>'
        f'</div>{status_badge("Watch", "warn")}</div>'
        f'<div class="grid">{cards}</div>'
        '</section>'
    )


def name_dependence_tone(audit: dict[str, Any]) -> str:
    if audit["without_top_name"]["sharpe_retention_vs_base"] is not None and float(audit["without_top_name"]["sharpe_retention_vs_base"]) < 0.70:
        return "warn"
    if audit["top_name_share_of_positive_contribution"] is not None and float(audit["top_name_share_of_positive_contribution"]) >= 0.35:
        return "warn"
    if audit["top_three_share_of_positive_contribution"] is not None and float(audit["top_three_share_of_positive_contribution"]) >= 0.65:
        return "warn"
    return "good"


def render_name_dependence_table(rows: Sequence[dict[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['ticker_local'] or row['security_id'])}</td>"
            f"<td>{html.escape(row['company_name'] or '')}</td>"
            f"<td>{html.escape(row['country_code'] or '')}</td>"
            f"<td>{int(row['months_selected'])}</td>"
            f"<td>{format_float(row['cumulative_net_contribution'])}</td>"
            f"<td>{format_pct(row.get('share_of_positive_contribution'))}</td>"
            f"<td>{format_pct(row.get('share_of_absolute_contribution'))}</td>"
            "</tr>"
        )
    return "".join(body) or '<tr><td colspan="7" class="empty">No holdout contribution rows are available.</td></tr>'


def render_name_dependence_card(thesis: dict[str, Any]) -> str:
    audit = thesis["name_dependence"]
    without_top_name = audit["without_top_name"]["performance"]
    without_top_three = audit["without_top_three"]["performance"]
    flags = "".join(f"<li>{html.escape(flag)}</li>" for flag in audit["flags"])
    return f"""
<article class="card">
  <div class="head">
    <div>
      <h3>{html.escape(thesis['label'])}</h3>
      <p>{html.escape(thesis['params_label'])}</p>
    </div>
    {status_badge('Watch' if name_dependence_tone(audit) == 'warn' else 'Ready', name_dependence_tone(audit))}
  </div>
  <p>Untouched holdout breadth check for {html.escape(audit['window_start'])} to {html.escape(audit['window_end'])}. This asks whether the preserved monthly result stayed broad, not whether we can find a better model.</p>
  <div class="metrics">
    <div><span>Unique names held</span><strong>{audit['unique_names_held']}</strong></div>
    <div><span>Positive contributors</span><strong>{audit['positive_name_count']}</strong></div>
    <div><span>Negative contributors</span><strong>{audit['negative_name_count']}</strong></div>
    <div><span>Top name share of positive gains</span><strong>{format_pct(audit['top_name_share_of_positive_contribution'])}</strong></div>
    <div><span>Top 3 share of positive gains</span><strong>{format_pct(audit['top_three_share_of_positive_contribution'])}</strong></div>
    <div><span>Top name share of abs contribution</span><strong>{format_pct(audit['top_name_share_of_absolute_contribution'])}</strong></div>
  </div>
  <div class="metrics">
    <div><span>Base holdout Sharpe</span><strong>{format_float(audit['performance']['net_sharpe'])}</strong></div>
    <div><span>Without top name Sharpe</span><strong>{format_float(without_top_name['net_sharpe'] if without_top_name else None)}</strong></div>
    <div><span>Sharpe retained</span><strong>{format_pct(audit['without_top_name']['sharpe_retention_vs_base'])}</strong></div>
    <div><span>Without top 3 Sharpe</span><strong>{format_float(without_top_three['net_sharpe'] if without_top_three else None)}</strong></div>
    <div><span>Top 3 Sharpe retained</span><strong>{format_pct(audit['without_top_three']['sharpe_retention_vs_base'])}</strong></div>
    <div><span>Without top name return</span><strong>{format_pct(without_top_name['total_return'] if without_top_name else None)}</strong></div>
  </div>
  <table>
    <thead>
      <tr><th>Ticker</th><th>Company</th><th>Country</th><th>Months held</th><th>Cumulative net contribution</th><th>Share of positive gains</th><th>Share of abs contribution</th></tr>
    </thead>
    <tbody>{render_name_dependence_table(audit['top_contributors'][:5])}</tbody>
  </table>
  <ul>{flags}</ul>
</article>
"""


def name_dependence_section(payload: dict[str, Any]) -> str:
    cards = "".join(render_name_dependence_card(thesis) for thesis in payload["theses"])
    return (
        '<section class="section">'
        '<div class="head"><div><h2>Holdout Breadth And Name Dependence</h2>'
        '<p>This uses the untouched holdout window, not the tiny live-monitor window, to check whether one or two names drove the preserved monthly result.</p>'
        f'</div>{status_badge("Watch", "warn")}</div>'
        f'<div class="grid">{cards}</div>'
        '</section>'
    )


def render_forward_dashboard(payload: dict[str, Any]) -> str:
    theses = payload["theses"]
    lead = theses[0]
    current_overlap = payload["current_overlap"]
    authoritative = payload.get("authoritative_status", {})
    active_validated = bool(authoritative.get("active_validated_strategy"))
    manual_trial = payload.get("monitor_mode") == "manual_forward_trial"
    monitoring_window = payload["monitoring_window"]
    decision_windows = forward_trial_decision_windows(monitoring_window["start_after_holdout"])
    selected_strategy = selected_strategy_summary(lead)
    eyebrow = (
        "Forward Trial / Manual Paper Trading"
        if manual_trial
        else ("Forward Monitor / Paper Trading" if active_validated else "Legacy Monitor / Reference Package")
    )
    dashboard_title = (
        "Systematic Nordic Equity Research & Validation<br>Frozen Forward Trial"
        if manual_trial
        else (
            "Systematic Nordic Equity Research & Validation<br>Forward Monitor"
            if active_validated
            else "Systematic Nordic Equity Research & Validation<br>Legacy Monthly Monitor"
        )
    )
    hero_summary = (
        "This page freezes an explicit lead/shadow pair for forward observation without reopening the search or overriding the repo-level verdict."
        if manual_trial
        else (
            "This page keeps the validated strategy definitions fixed and shows the current monthly picks side by side."
            if active_validated
            else "This page preserves the old monthly candidate definitions for reference. The stricter cadence review is now authoritative, and there is no active validated lead yet."
        )
    )
    purpose_copy = (
        "The job here is straightforward: paper trade the frozen lead and shadow books side by side, without retuning, and see how they behave on genuinely new months."
        if manual_trial
        else (
            "This is where the lead candidate is paper traded against the broader baseline, with the rules left unchanged so we can see whether the live picks behave the way validation suggested."
            if active_validated
            else "This page is for inspecting the preserved monthly books, their audit trail, and the shadow comparison without treating any of it as an active mandate."
        )
    )
    picks_heading = "Frozen Forward-Trial Picks" if manual_trial else ("Current Monthly Picks" if active_validated else "Legacy Monthly Picks")
    picks_copy = (
        "These are the frozen current-book names formed from explicit locked candidates under the repo's production assumptions: "
        if manual_trial
        else (
            "These are the frozen current-book names formed with the repo's production assumptions: "
            if active_validated
            else "These are the preserved monthly names formed with the older monthly selection assumptions: "
        )
    )
    cards = "".join(thesis_card(thesis) for thesis in theses)
    comparison_name = html.escape(theses[1].get("candidate_label") or theses[1]["name"]) if len(theses) > 1 else "comparison"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{"Systematic Nordic Equity Research & Validation Forward Trial" if manual_trial else ("Systematic Nordic Equity Research & Validation Forward Monitor" if active_validated else "Systematic Nordic Equity Research & Validation Legacy Monthly Monitor")}</title>
  <style>
    :root {{
      --ink:#17211a; --muted:#5d685f; --paper:#f6f1e7; --card:rgba(255,250,241,.92);
      --line:rgba(23,33,26,.12); --accent:#1d5b45; --good:#2b603a; --warn:#8d5a18;
      --good-soft:#d7e9d9; --warn-soft:#f3e0bc; --muted-soft:rgba(255,255,255,.6);
      --shadow:0 18px 50px rgba(33,41,36,.08);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--ink); font-family:"Palatino Linotype", Georgia, serif;
      background:
        radial-gradient(circle at top left, rgba(29,91,69,.10), transparent 34%),
        radial-gradient(circle at top right, rgba(141,90,24,.10), transparent 28%),
        linear-gradient(180deg, #fbf8f2 0%, var(--paper) 100%);
    }}
    main {{ max-width:1360px; margin:0 auto; padding:28px 20px 48px; }}
    .hero, .section, .card {{ border:1px solid var(--line); border-radius:26px; background:var(--card); box-shadow:var(--shadow); }}
    .hero, .section {{ padding:28px; }}
    .section {{ margin-top:20px; }}
    .hero-grid, .grid {{ display:grid; gap:18px; }}
    .hero-grid {{ grid-template-columns:minmax(0,1.3fr) minmax(320px,.9fr); }}
    .grid {{ grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); }}
    h1 {{ margin:12px 0 0; font-size:clamp(2.3rem,4vw,3.8rem); line-height:.95; letter-spacing:-.04em; }}
    h2, h3, p {{ margin:0; }}
    p {{ color:var(--muted); line-height:1.6; }}
    .eyebrow, .badge {{
      display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px;
      font:.78rem Consolas, "Courier New", monospace; letter-spacing:.06em; text-transform:uppercase;
    }}
    .eyebrow {{ color:var(--accent); background:rgba(29,91,69,.08); border:1px solid rgba(29,91,69,.14); }}
    .badge.good {{ background:var(--good-soft); color:var(--good); }}
    .badge.warn {{ background:var(--warn-soft); color:var(--warn); }}
    .badge.muted {{ background:var(--muted-soft); color:var(--muted); }}
    .meta, .metrics {{ display:grid; gap:12px; }}
    .meta {{ grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); margin-top:18px; }}
    .metrics {{ grid-template-columns:repeat(3,minmax(0,1fr)); margin-top:16px; }}
    .meta div, .metrics div {{ padding:14px; border-radius:16px; border:1px solid rgba(23,33,26,.08); background:rgba(255,255,255,.55); }}
    .meta span, .metrics span {{ display:block; color:var(--muted); font-size:.84rem; }}
    .meta strong, .metrics strong {{ display:block; margin-top:4px; font-size:1.04rem; }}
    .head {{ display:flex; justify-content:space-between; gap:12px; align-items:start; margin-bottom:14px; flex-wrap:wrap; }}
    .card {{ padding:18px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:.92rem; }}
    th, td {{ text-align:left; padding:10px 8px; border-bottom:1px solid rgba(23,33,26,.08); vertical-align:top; }}
    th {{ color:var(--muted); font:.78rem Consolas, "Courier New", monospace; text-transform:uppercase; letter-spacing:.05em; }}
    .artifact-list {{ display:grid; gap:10px; }}
    .artifact-list a {{ text-decoration:none; color:var(--ink); padding:12px 14px; border-radius:16px; border:1px solid rgba(23,33,26,.08); background:rgba(255,255,255,.55); display:flex; justify-content:space-between; }}
    .empty {{ color:var(--muted); font-style:italic; }}
    @media (max-width:960px) {{ .hero-grid {{ grid-template-columns:1fr; }} }}
    @media (max-width:780px) {{ .metrics {{ grid-template-columns:1fr; }} main {{ padding:18px 14px 38px; }} }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-grid">
        <div>
          <span class="eyebrow">{html.escape(eyebrow)}</span>
          <h1>{dashboard_title}</h1>
          <p>{html.escape(hero_summary)}</p>
          <div class="meta">
            <div><span>Selected strategy</span><strong>{html.escape(selected_strategy)}</strong></div>
            <div><span>{'Lead candidate' if manual_trial else 'Lead thesis'}</span><strong>{html.escape(lead['label'])}</strong></div>
            <div><span>Authoritative verdict</span><strong>{html.escape(authoritative.get('verdict', 'n/a'))}</strong></div>
            <div><span>Monitor start</span><strong>{html.escape(monitoring_window['start_after_holdout'])}</strong></div>
            <div><span>Earliest live-decision review</span><strong>{html.escape(decision_windows['earliest_review_label'])} (after 6 paper months)</strong></div>
            <div><span>Preferred review</span><strong>{html.escape(decision_windows['preferred_review_label'])} (after 12 paper months)</strong></div>
            <div><span>Latest completed month</span><strong>{html.escape(monitoring_window.get('latest_completed_holding_month') or 'n/a')}</strong></div>
            <div><span>Current pick month</span><strong>{html.escape(monitoring_window['current_pick_month'] or 'n/a')}</strong></div>
            <div><span>Generated</span><strong>{html.escape(payload['generated_at_utc'])}</strong></div>
          </div>
        </div>
        <div class="card">
          <div class="head"><h3>What This Is For</h3>{status_badge('Watch', 'warn')}</div>
          <p>{html.escape(purpose_copy)}</p>
          <p style="margin-top:12px;"><strong>Decision clock:</strong> if the frozen lead stays accepted, the earliest live-trading review window opens in {html.escape(decision_windows['earliest_review_label'])}; the more conservative 12-month review lands in {html.escape(decision_windows['preferred_review_label'])}.</p>
          <div class="metrics">
            <div><span>Current overlap</span><strong>{current_overlap['overlap_count']} names</strong></div>
            <div><span>Share of smaller book</span><strong>{format_pct(current_overlap['overlap_share_of_smaller_book'])}</strong></div>
            <div><span>Overlap names</span><strong>{html.escape(', '.join(current_overlap['overlap_security_ids']) or 'none')}</strong></div>
          </div>
        </div>
      </div>
    </section>
    <section class="section">
      <div class="head"><div><h2>{html.escape(picks_heading)}</h2><p>{html.escape(picks_copy)}{html.escape(PRIMARY_TRACK['universe_variant'])}, {html.escape(PRIMARY_TRACK['execution_model'])}, {html.escape(PRIMARY_TRACK['fx_scenario'])} FX, and {html.escape(PRIMARY_TRACK['cost_model_name'])} costs.</p></div>{status_badge('Ready', 'good')}</div>
      <div class="grid">{cards}</div>
    </section>
    {pre_trade_audit_section(payload)}
    {name_dependence_section(payload)}
    <section class="section">
      <div class="head"><div><h2>Recent Realized Monitoring Months</h2><p>These are the realized post-holdout months so far. They are for monitoring drift and overlap, not for reopening the model search.</p></div>{status_badge('Watch', 'warn')}</div>
      <table>
        <thead>
          <tr><th>Holding month</th><th>{html.escape(theses[0]['name'])} net return</th><th>{comparison_name} net return</th><th>Overlap count</th><th>Overlap share</th><th>Overlap names</th></tr>
        </thead>
        <tbody>{render_history_rows(payload)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def build_forward_monitor(
    *,
    data_dir: Path,
    results_root: Path,
    output_dir: Path,
    theses: Sequence[str],
    history_months: int,
    selection_summaries: Sequence[Path] | None = None,
    export_package: bool = True,
) -> dict[str, Any]:
    payload = build_monitor_payload(
        data_dir=data_dir,
        results_root=results_root,
        theses=theses,
        history_months=history_months,
        selection_summaries=selection_summaries,
    )
    manifest = frozen_strategy_manifest(payload)
    dossier = build_research_dossier(payload, manifest)
    project_status = build_project_status_markdown(payload, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    serialize_json(output_dir / "forward_monitor_summary.json", payload)
    serialize_json(output_dir / "frozen_strategy_manifest.json", manifest)
    flatten_pick_rows(payload).to_csv(output_dir / "forward_monitor_picks.csv", index=False)
    flatten_name_dependence_rows(payload).to_csv(output_dir / "holdout_name_dependence.csv", index=False)
    (output_dir / "research_audit_dossier.html").write_text(
        build_markdown_dashboard_html(
            title="Research Dossier",
            subtitle="A short write-up of the validation process and what the current result actually means.",
            markdown_text=dossier,
        ),
        encoding="utf-8",
    )
    (output_dir / "project_status.html").write_text(
        build_markdown_dashboard_html(
            title="Current Project Status",
            subtitle="A clear look at where the project stands and what we would do next.",
            markdown_text=project_status,
        ),
        encoding="utf-8",
    )
    if export_package and output_dir != results_root:
        (results_root / "project_status.html").write_text(
            build_markdown_dashboard_html(
                title="Current Project Status",
                subtitle="A clear look at where the project stands and what we would do next.",
                markdown_text=project_status,
            ),
            encoding="utf-8",
        )
    (output_dir / "dashboard.html").write_text(render_forward_dashboard(payload), encoding="utf-8")
    if export_package:
        export_portfolio_package(
            payload=payload,
            project_status=project_status,
            research_dossier=dossier,
            output_dir=output_dir,
            results_root=results_root,
        )
        export_pair_dashboards_from_cadence_summary(results_root)
    return payload


def main() -> int:
    args = parse_args()
    theses = args.theses or resolve_default_theses(args.results_root)
    payload = build_forward_monitor(
        data_dir=args.data_dir,
        results_root=args.results_root,
        output_dir=args.output_dir,
        theses=theses,
        history_months=int(args.history_months),
        selection_summaries=args.selection_summaries,
        export_package=not bool(args.skip_package_export),
    )
    print(
        f"Forward monitor ready for {payload['monitoring_window']['current_pick_month']}. "
        f"Open {args.output_dir / 'dashboard.html'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
