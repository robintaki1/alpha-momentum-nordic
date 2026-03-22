from __future__ import annotations

import math
import os
import re
import shutil
import sys
import time
import importlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import engine

DATE_COLUMNS_BY_ARTIFACT = {
    "security_master": ["listing_date", "delisting_date", "last_metadata_refresh_ts"],
    "security_status_pti_monthly": ["anchor_trade_date", "next_execution_date"],
    "prices_raw_daily": ["date"],
    "prices_adjusted_daily": ["date"],
    "corporate_actions": ["event_date"],
    "delisted_metadata": ["listing_date", "delisting_date", "last_trade_date"],
    "benchmark_prices": ["date"],
    "riksbank_fx_daily": ["date"],
    "universe_pti": ["anchor_trade_date", "next_execution_date", "asof_trade_date"],
}
PRIMARY_KEYS = {
    "security_master": ["security_id"],
    "security_status_pti_monthly": ["rebalance_month", "security_id"],
    "prices_raw_daily": ["security_id", "date"],
    "prices_adjusted_daily": ["security_id", "date"],
    "benchmark_prices": ["benchmark_id", "date"],
    "riksbank_fx_daily": ["currency", "date"],
    "universe_pti": ["rebalance_month", "security_id"],
}
REQUIRED_COLUMNS = {
    "security_master": [
        "security_id", "eodhd_symbol", "ticker_local", "isin", "company_name", "country_code",
        "exchange_name", "exchange_group", "issuer_group_id", "vendor_roster_status", "currency",
        "security_type_raw", "security_type_normalized", "share_class_raw", "share_class_normalized",
        "listing_date", "delisting_date", "is_delisted", "shares_outstanding", "last_metadata_refresh_ts",
    ],
    "security_status_pti_monthly": [
        "rebalance_month", "anchor_trade_date", "next_execution_date", "security_id", "country_code",
        "exchange_group", "is_listed_asof_anchor", "delisted_before_next_execution",
        "is_primary_common_share_asof_anchor", "eligible_for_main_universe_asof_anchor",
        "listing_age_months", "market_cap_local_asof_anchor", "market_cap_sek_asof_anchor", "status_source",
    ],
    "prices_raw_daily": [
        "security_id", "date", "open_raw", "high_raw", "low_raw", "close_raw", "volume",
        "currency", "trade_value_local", "trade_value_sek", "source",
    ],
    "prices_adjusted_daily": ["security_id", "date", "adj_close", "adj_factor", "currency", "source"],
    "corporate_actions": [
        "security_id", "event_date", "action_type", "action_value", "action_ratio", "currency", "source",
    ],
    "delisted_metadata": [
        "security_id", "eodhd_symbol", "company_name", "country_code", "exchange_name", "currency",
        "listing_date", "delisting_date", "delisting_reason", "last_trade_date", "cashout_value", "source",
    ],
    "benchmark_prices": [
        "benchmark_id", "benchmark_name", "benchmark_type", "date", "currency",
        "close_raw", "adj_close", "close_sek", "source",
    ],
    "riksbank_fx_daily": ["currency", "date", "sek_per_ccy", "source"],
    "universe_pti": [
        "rebalance_month", "anchor_trade_date", "next_execution_date", "security_id", "country_code",
        "exchange_group", "currency", "asof_trade_date", "listing_age_months", "close_raw_local",
        "close_raw_sek", "trade_value_sek", "median_daily_value_60d_sek", "market_cap_sek",
        "is_eligible_full_nordics", "is_eligible_se_only", "is_eligible_liquid_subset",
        "exclusion_reason_full_nordics", "exclusion_reason_se_only", "exclusion_reason_liquid_subset",
    ],
}
MARKET_CAP_MONTHLY_COLUMNS = ["security_id", "rebalance_month", "market_cap_local", "market_cap_sek", "source"]
NORDIC_EXCHANGES = {
    "ST": {"exchange_name": "Nasdaq Stockholm", "exchange_group": "Nasdaq Stockholm", "country_code": "SE", "currency": "SEK"},
    "CO": {"exchange_name": "Nasdaq Copenhagen", "exchange_group": "Nasdaq Copenhagen", "country_code": "DK", "currency": "DKK"},
    "OL": {"exchange_name": "Oslo Bors", "exchange_group": "Oslo Bors", "country_code": "NO", "currency": "NOK"},
}
RIKSBANK_SERIES = {"NOK": "SEKNOKPMI", "DKK": "SEKDKKPMI", "EUR": "SEKEURPMI", "USD": "SEKUSDPMI"}
ALLOWED_BENCHMARK_TYPES = {"total_return", "proxy"}
ALLOWED_VENDOR_ROSTER_STATUS = {"active_only", "delisted_only", "both"}
ALLOWED_ACTION_TYPES = {"split", "reverse_split", "cash_dividend", "special_dividend", "cash_merger", "delisting_cashout"}
BOOLEAN_TRUE = {"1", "true", "yes", "y"}
EXCLUDED_CODE_SUFFIXES = ("-TR", "-UR", "-SR", "-IL", "-BTA", "-BTU", "-TO", "-WRT", "-UNIT", "-SPAC", "-ME", "-PREF")
EXCLUDED_NAME_PATTERNS = (
    " fund", " etf", "warrant", "certifikat", "certificate", "mini future", "subscription right",
    "teckningsratt", "teckningsrätt", "preferens", "preferred", "depository receipt", "adr", "gdr",
    "spac", "bond", "obligation", "rights", " bevis",
)
LEGAL_ENTITY_SUFFIXES = {"ab", "abp", "asa", "oy", "oyj", "as", "aps", "plc", "nv", "sa", "ag", "ltd", "limited"}
SHARE_CLASS_SUFFIXES = {"a", "b", "c", "d", "ord", "ordinary", "pref", "preference", "series", "ser", "share", "shares"}
FULL_HISTORY_FLOOR = pd.Timestamp("1900-01-01")
MAX_ACTIVE_COMMON_SHARE_COUNTS = {"ST": 500, "CO": 220, "OL": 260}
BENCHMARKS = [
    {
        "benchmark_id": config.PRIMARY_PASSIVE_BENCHMARK_ID,
        "benchmark_name": config.PRIMARY_PASSIVE_BENCHMARK,
        "benchmark_type": config.PRIMARY_PASSIVE_BENCHMARK_TYPE,
        "eodhd_symbol": config.PRIMARY_PASSIVE_BENCHMARK_EODHD_SYMBOL,
    },
    {
        "benchmark_id": config.SECONDARY_OPPORTUNITY_COST_BENCHMARK_ID,
        "benchmark_name": config.SECONDARY_OPPORTUNITY_COST_BENCHMARK,
        "benchmark_type": config.SECONDARY_OPPORTUNITY_COST_BENCHMARK_TYPE,
        "eodhd_symbol": config.SECONDARY_OPPORTUNITY_COST_BENCHMARK_EODHD_SYMBOL,
    },
    {
        "benchmark_id": config.TERTIARY_OPPORTUNITY_COST_BENCHMARK_ID,
        "benchmark_name": config.TERTIARY_OPPORTUNITY_COST_BENCHMARK,
        "benchmark_type": config.TERTIARY_OPPORTUNITY_COST_BENCHMARK_TYPE,
        "eodhd_symbol": config.TERTIARY_OPPORTUNITY_COST_BENCHMARK_EODHD_SYMBOL,
    },
]
BENCHMARK_EXCHANGE_CURRENCIES = {
    "ST": "SEK",
    "AS": "EUR",
    "US": "USD",
}


class Phase1Error(RuntimeError):
    pass


@dataclass
class ValidationSection:
    name: str
    status: str
    messages: list[str]


@dataclass
class MainMarketAllowlist:
    path: Path
    entries: pd.DataFrame
    by_security_id: dict[str, set[int]]
    by_exchange_isin: dict[tuple[str, str], set[int]]
    by_isin_only: dict[str, set[int]]

    def match_row_indices(self, *, security_id: str, isin: str | None, exchange_group: str | None = None) -> set[int]:
        matched = set(self.by_security_id.get(normalize_text(security_id).upper(), set()))
        if matched:
            return matched
        exchange_key = normalize_text(exchange_group).upper()
        isin_key = normalize_text(isin).upper()
        if exchange_key and isin_key:
            matched.update(self.by_exchange_isin.get((exchange_key, isin_key), set()))
        if not matched and isin_key:
            matched.update(self.by_isin_only.get(isin_key, set()))
        return matched

    def unmatched_entries(self, matched_row_indices: set[int]) -> pd.DataFrame:
        missing_indices = sorted(set(self.entries.index) - matched_row_indices)
        return self.entries.loc[missing_indices].copy()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in BOOLEAN_TRUE:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value '{value}'.")


def configured_nordic_exchange_codes() -> tuple[str, ...]:
    raw_codes = getattr(config, "ACTIVE_NORDIC_EXCHANGE_CODES", tuple(NORDIC_EXCHANGES))
    codes = tuple(str(code).strip().upper() for code in raw_codes)
    if not codes:
        raise Phase1Error("ACTIVE_NORDIC_EXCHANGE_CODES must include at least one exchange code.")
    invalid = [code for code in codes if code not in NORDIC_EXCHANGES]
    if invalid:
        raise Phase1Error(f"Unsupported ACTIVE_NORDIC_EXCHANGE_CODES values: {invalid}")
    return codes


def active_nordic_exchanges() -> dict[str, dict[str, str]]:
    return {code: NORDIC_EXCHANGES[code] for code in configured_nordic_exchange_codes()}


def active_nordic_exchange_groups() -> set[str]:
    return {info["exchange_group"] for info in active_nordic_exchanges().values()}


def today_stockholm() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ZoneInfo("Europe/Stockholm")).normalize()


def resolve_end_date(end_value: str | None) -> pd.Timestamp:
    if not end_value or end_value == "auto":
        current = today_stockholm() - pd.Timedelta(days=1)
        while current.dayofweek >= 5:
            current -= pd.Timedelta(days=1)
        return current.tz_localize(None)
    return pd.Timestamp(end_value).normalize()


def normalize_date_column(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def atomic_write_parquet(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_path.parent / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / output_path.name
    frame.to_parquet(staging_path, index=False)
    shutil.move(str(staging_path), output_path)


def read_parquet(path: Path, artifact_name: str) -> pd.DataFrame:
    if not path.exists():
        raise Phase1Error(f"Missing required artifact: {artifact_name} at {path}")
    frame = pd.read_parquet(path)
    return normalize_date_column(frame, DATE_COLUMNS_BY_ARTIFACT.get(artifact_name, []))


def load_dotenv_value(key: str) -> str | None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        cleaned = value.strip().strip('"').strip("'")
        return cleaned or None
    return None


def load_eodhd_api_key() -> str:
    api_key = config.EODHD_API_KEY or os.environ.get("EODHD_API_KEY") or load_dotenv_value("EODHD_API_KEY")
    if not api_key:
        raise Phase1Error("EODHD_API_KEY is required.")
    return api_key


def load_main_market_allowlist(path: Path | None) -> MainMarketAllowlist | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    renamed = {column: normalize_text(column).lower() for column in frame.columns}
    frame = frame.rename(columns=renamed)
    if "eodhd_symbol" in frame.columns and "security_id" not in frame.columns:
        frame["security_id"] = frame["eodhd_symbol"]
    if "security_id" not in frame.columns:
        frame["security_id"] = ""
    if "isin" not in frame.columns:
        frame["isin"] = ""
    if "exchange_group" not in frame.columns:
        frame["exchange_group"] = ""
    if "include" in frame.columns:
        include_mask = []
        for value in frame["include"]:
            normalized = normalize_text(value)
            include_mask.append(True if not normalized else parse_bool(normalized))
        frame = frame.loc[include_mask].copy()
    frame["security_id"] = frame["security_id"].map(lambda value: normalize_text(value).upper())
    frame["isin"] = frame["isin"].map(lambda value: normalize_text(value).upper())
    frame["exchange_group"] = frame["exchange_group"].map(lambda value: normalize_text(value).upper())
    active_exchange_groups = {group.upper() for group in active_nordic_exchange_groups()}
    active_exchange_codes = set(configured_nordic_exchange_codes())
    if not frame.empty:
        in_scope_mask: list[bool] = []
        for row in frame.itertuples(index=False):
            exchange_group = normalize_text(getattr(row, "exchange_group", "")).upper()
            security_id = normalize_text(getattr(row, "security_id", "")).upper()
            if exchange_group:
                in_scope_mask.append(exchange_group in active_exchange_groups)
                continue
            exchange_code = exchange_code_from_security_id(security_id)
            in_scope_mask.append(not exchange_code or exchange_code in active_exchange_codes)
        frame = frame.loc[in_scope_mask].copy()
    frame = frame.loc[(frame["security_id"] != "") | (frame["isin"] != "")].copy()
    if frame.empty:
        raise Phase1Error(
            f"Main-market allowlist at {path} is empty. Populate at least one security_id or ISIN row before rerunning."
        )
    frame = frame.drop_duplicates(subset=["security_id", "isin"]).reset_index(drop=True)
    by_security_id: dict[str, set[int]] = {}
    by_exchange_isin: dict[tuple[str, str], set[int]] = {}
    by_isin_only: dict[str, set[int]] = {}
    for index, row in frame.iterrows():
        if row["security_id"]:
            by_security_id.setdefault(row["security_id"], set()).add(index)
        if row["exchange_group"] and row["isin"]:
            by_exchange_isin.setdefault((row["exchange_group"], row["isin"]), set()).add(index)
        elif row["isin"] and not row["security_id"]:
            by_isin_only.setdefault(row["isin"], set()).add(index)
    return MainMarketAllowlist(
        path=path,
        entries=frame,
        by_security_id=by_security_id,
        by_exchange_isin=by_exchange_isin,
        by_isin_only=by_isin_only,
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def exchange_code_from_security_id(security_id: str) -> str:
    normalized = normalize_text(security_id).upper()
    if "." not in normalized:
        return ""
    return normalized.rsplit(".", 1)[-1]


def build_eodhd_symbol(code: str, exchange_code: str) -> str:
    return f"{normalize_text(code)}.{exchange_code}"


def share_class_from_name(code: str, name: str) -> str:
    lowered = f"{code} {name}".lower()
    if any(token in lowered for token in (" preferred", " preferens", " pref")):
        return "preferred"
    if any(token in lowered for token in (" right", " rights", " teckningsr", "-tr", "-ur", "-sr")):
        return "rights"
    if any(token in lowered for token in (" adr", " gdr", "depository receipt")):
        return "depositary"
    return "ordinary"


def security_type_from_row(raw_type: str, code: str, name: str) -> str:
    lowered_type = normalize_text(raw_type).lower()
    lowered_name = f" {normalize_text(name).lower()} "
    lowered_code = normalize_text(code).lower()
    if "fund" in lowered_type:
        return "fund"
    if "etf" in lowered_type:
        return "etf"
    if "common stock" not in lowered_type:
        return "other"
    if any(pattern in lowered_name for pattern in EXCLUDED_NAME_PATTERNS):
        return "other"
    if any(lowered_code.endswith(suffix.lower()) for suffix in EXCLUDED_CODE_SUFFIXES):
        return "other"
    return "common_share"


def issuer_group_id(company_name: str) -> str:
    normalized = normalize_text(company_name)
    ascii_name = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii").lower()
    ascii_name = re.sub(r"\([^)]*\)", " ", ascii_name)
    ascii_name = ascii_name.replace("&", " and ").replace("/", " ")
    tokens = [token for token in re.split(r"[^a-z0-9]+", ascii_name) if token]
    while tokens and tokens[-1] in SHARE_CLASS_SUFFIXES:
        tokens.pop()
    while tokens and tokens[-1] in LEGAL_ENTITY_SUFFIXES:
        tokens.pop()
    compact = "-".join(tokens).strip("-")
    return compact or re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-") or normalized


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    retries: int = 3,
    timeout: int = 30,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt == retries:
                break
            time.sleep(0.5 * attempt)
    raise Phase1Error(f"Request failed for {url}: {last_error}") from last_error


def load_exchange_roster(
    session: requests.Session,
    api_key: str,
    exchange_code: str,
    *,
    delisted: bool,
) -> list[dict[str, Any]]:
    params = {"api_token": api_key, "fmt": "json"}
    if delisted:
        params["delisted"] = 1
    payload = fetch_json(session, f"https://eodhd.com/api/exchange-symbol-list/{exchange_code}", params)
    if not isinstance(payload, list):
        raise Phase1Error(f"Unexpected roster payload for exchange {exchange_code}.")
    return payload


def normalize_security_row(raw_row: dict[str, Any], exchange_code: str) -> dict[str, Any] | None:
    info = NORDIC_EXCHANGES[exchange_code]
    code = normalize_text(raw_row.get("Code"))
    name = normalize_text(raw_row.get("Name"))
    raw_type = normalize_text(raw_row.get("Type"))
    normalized_type = security_type_from_row(raw_type, code, name)
    if normalized_type != "common_share":
        return None
    share_class = share_class_from_name(code, name)
    if share_class != "ordinary":
        return None
    symbol = build_eodhd_symbol(code, exchange_code)
    return {
        "security_id": symbol,
        "eodhd_symbol": symbol,
        "ticker_local": code,
        "isin": normalize_text(raw_row.get("Isin")) or None,
        "company_name": name,
        "country_code": info["country_code"],
        "exchange_name": info["exchange_name"],
        "exchange_group": info["exchange_group"],
        "issuer_group_id": issuer_group_id(name),
        "currency": normalize_text(raw_row.get("Currency")) or info["currency"],
        "security_type_raw": raw_type,
        "security_type_normalized": normalized_type,
        "share_class_raw": raw_type,
        "share_class_normalized": share_class,
        "listing_date": pd.NaT,
        "delisting_date": pd.NaT,
        "is_delisted": False,
        "shares_outstanding": np.nan,
        "last_metadata_refresh_ts": pd.Timestamp.utcnow(),
        "_active": False,
        "_delisted": False,
    }


def load_security_universe(
    session: requests.Session,
    api_key: str,
    include_delisted: bool,
    *,
    main_market_allowlist_path: Path | None = None,
) -> pd.DataFrame:
    combined: dict[str, dict[str, Any]] = {}
    active_exchange_map = active_nordic_exchanges()
    active_counts: dict[str, int] = {exchange_code: 0 for exchange_code in active_exchange_map}
    allowlist = load_main_market_allowlist(main_market_allowlist_path)
    matched_allowlist_rows: set[int] = set()
    for exchange_code in active_exchange_map:
        active_rows = load_exchange_roster(session, api_key, exchange_code, delisted=False)
        delisted_rows = load_exchange_roster(session, api_key, exchange_code, delisted=True) if include_delisted else []
        if not active_rows:
            raise Phase1Error(f"Exchange roster for {exchange_code} is empty.")

        for row in active_rows:
            normalized = normalize_security_row(row, exchange_code)
            if normalized is None:
                continue
            if allowlist is not None:
                matched_rows = allowlist.match_row_indices(
                    security_id=normalized["security_id"],
                    isin=normalized["isin"],
                    exchange_group=normalized["exchange_group"],
                )
                if not matched_rows:
                    continue
                matched_allowlist_rows.update(matched_rows)
            active_counts[exchange_code] += 1
            combined.setdefault(normalized["security_id"], normalized)["_active"] = True
        for row in delisted_rows:
            normalized = normalize_security_row(row, exchange_code)
            if normalized is None:
                continue
            if allowlist is not None:
                matched_rows = allowlist.match_row_indices(
                    security_id=normalized["security_id"],
                    isin=normalized["isin"],
                    exchange_group=normalized["exchange_group"],
                )
                if not matched_rows:
                    continue
                matched_allowlist_rows.update(matched_rows)
            combined.setdefault(normalized["security_id"], normalized)["_delisted"] = True

    if allowlist is None:
        oversized_active_counts = {
            exchange_code: active_counts[exchange_code]
            for exchange_code in active_exchange_map
            if active_counts[exchange_code] > MAX_ACTIVE_COMMON_SHARE_COUNTS[exchange_code]
        }
        if oversized_active_counts:
            details = ", ".join(
                f"{exchange_code}={count} active common shares"
                for exchange_code, count in oversized_active_counts.items()
            )
            raise Phase1Error(
                "EODHD active exchange rosters are broader than the strict Nordic main-market universe "
                f"({details}). The current payload does not expose enough venue-segment metadata to separate "
                "main-market listings from smaller Nordic growth markets, so Phase 1 stops here instead of "
                f"downloading an out-of-scope universe. Add a curated allowlist at {config.MAIN_MARKET_ALLOWLIST_PATH} "
                "or pass --main-market-allowlist with a populated CSV."
            )
    else:
        unmatched_entries = allowlist.unmatched_entries(matched_allowlist_rows)
        if not unmatched_entries.empty:
            preview_values = []
            for row in unmatched_entries.head(10).itertuples(index=False):
                preview_values.append(row.security_id or row.isin)
            suffix = " ..." if len(unmatched_entries) > 10 else ""
            raise Phase1Error(
                f"Main-market allowlist entries were not found in EODHD Nordic rosters: {', '.join(preview_values)}{suffix}"
            )

    if not combined:
        raise Phase1Error("No in-scope common-share securities were found in EODHD rosters.")

    rows = []
    for record in combined.values():
        active = record.pop("_active")
        delisted = record.pop("_delisted")
        if active and delisted:
            record["vendor_roster_status"] = "both"
        elif active:
            record["vendor_roster_status"] = "active_only"
        else:
            record["vendor_roster_status"] = "delisted_only"
        rows.append(record)

    frame = pd.DataFrame(rows).sort_values(["exchange_group", "ticker_local"]).reset_index(drop=True)
    return normalize_date_column(frame, ["listing_date", "delisting_date", "last_metadata_refresh_ts"])


def fetch_eod_history(
    session: requests.Session,
    api_key: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    payload = fetch_json(
        session,
        f"https://eodhd.com/api/eod/{symbol}",
        {
            "api_token": api_key,
            "fmt": "json",
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
        },
    )
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    frame = frame.rename(columns={"open": "open_raw", "high": "high_raw", "low": "low_raw", "close": "close_raw"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["volume"] = frame["volume"].astype(float)
    for column in ("open_raw", "high_raw", "low_raw", "close_raw"):
        frame[column] = frame[column].astype(float)
    return frame[["date", "open_raw", "high_raw", "low_raw", "close_raw", "volume"]]


def fetch_dividends(session: requests.Session, api_key: str, symbol: str) -> pd.DataFrame:
    payload = fetch_json(session, f"https://eodhd.com/api/div/{symbol}", {"api_token": api_key, "fmt": "json"})
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = frame["value"].astype(float)
    if "currency" not in frame.columns:
        frame["currency"] = None
    return frame[["date", "value", "currency"]]


def fetch_splits(session: requests.Session, api_key: str, symbol: str) -> pd.DataFrame:
    payload = fetch_json(session, f"https://eodhd.com/api/splits/{symbol}", {"api_token": api_key, "fmt": "json"})
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[["date", "split"]]


def fetch_benchmark_history(
    session: requests.Session,
    api_key: str,
    benchmark: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    payload = fetch_json(
        session,
        f"https://eodhd.com/api/eod/{benchmark['eodhd_symbol']}",
        {
            "api_token": api_key,
            "fmt": "json",
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
        },
    )
    frame = pd.DataFrame(payload)
    if frame.empty:
        raise Phase1Error(f"Benchmark series {benchmark['benchmark_id']} returned no data.")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close_raw"] = frame["close"].astype(float)
    frame["adj_close"] = frame["adjusted_close"].astype(float)
    frame["close_sek"] = np.nan
    frame["benchmark_id"] = benchmark["benchmark_id"]
    frame["benchmark_name"] = benchmark["benchmark_name"]
    frame["benchmark_type"] = benchmark["benchmark_type"]
    exchange = benchmark["eodhd_symbol"].split(".")[-1]
    currency = BENCHMARK_EXCHANGE_CURRENCIES.get(exchange)
    if currency is None:
        raise Phase1Error(
            f"Unsupported benchmark exchange suffix '{exchange}' for {benchmark['benchmark_id']}."
        )
    frame["currency"] = currency
    frame["source"] = "eodhd"
    return frame[REQUIRED_COLUMNS["benchmark_prices"]]


def merge_corporate_actions(security_id: str, dividends: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not dividends.empty:
        for dividend in dividends.itertuples(index=False):
            rows.append(
                {
                    "security_id": security_id,
                    "event_date": dividend.date,
                    "action_type": "cash_dividend",
                    "action_value": float(dividend.value),
                    "action_ratio": np.nan,
                    "currency": normalize_text(dividend.currency) or None,
                    "source": "eodhd",
                }
            )
    if not splits.empty:
        for split in splits.itertuples(index=False):
            numerator, denominator = normalize_text(split.split).split("/")
            ratio = float(numerator) / float(denominator)
            rows.append(
                {
                    "security_id": security_id,
                    "event_date": split.date,
                    "action_type": "split" if ratio >= 1.0 else "reverse_split",
                    "action_value": ratio,
                    "action_ratio": ratio,
                    "currency": None,
                    "source": "eodhd",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS["corporate_actions"])
    frame = frame.sort_values(["security_id", "event_date", "action_type"]).reset_index(drop=True)
    return normalize_date_column(frame, ["event_date"])


def download_eodhd_artifacts(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_delisted: bool,
    out_dir: Path,
    main_market_allowlist_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    api_key = load_eodhd_api_key()
    session = requests.Session()
    security_master = load_security_universe(
        session,
        api_key,
        include_delisted,
        main_market_allowlist_path=main_market_allowlist_path,
    )
    total_symbols = len(security_master)
    raw_rows: list[pd.DataFrame] = []
    action_rows: list[pd.DataFrame] = []
    benchmark_rows: list[pd.DataFrame] = []
    failures: list[str] = []
    window_failures: list[str] = []
    listing_dates: dict[str, pd.Timestamp] = {}
    last_trade_dates: dict[str, pd.Timestamp] = {}

    if progress_callback is not None:
        progress_callback(
            f"Downloading EODHD history for {total_symbols} securities "
            f"from {start.date()} to {end.date()}."
        )

    for index, security in enumerate(security_master.itertuples(index=False), start=1):
        if progress_callback is not None and (index == 1 or index == total_symbols or index % 10 == 0):
            progress_callback(f"[{index}/{total_symbols}] {security.eodhd_symbol}")
        full_history = fetch_eod_history(session, api_key, security.eodhd_symbol, FULL_HISTORY_FLOOR, end)
        if full_history.empty:
            failures.append(security.eodhd_symbol)
            continue
        listing_dates[security.security_id] = full_history["date"].min()
        last_trade_dates[security.security_id] = full_history["date"].max()
        price_frame = full_history.loc[(full_history["date"] >= start) & (full_history["date"] <= end)].copy()
        if price_frame.empty:
            overlaps_window = listing_dates[security.security_id] <= end and last_trade_dates[security.security_id] >= start
            if overlaps_window:
                window_failures.append(security.eodhd_symbol)
            continue
        price_frame["security_id"] = security.security_id
        price_frame["currency"] = security.currency
        price_frame["trade_value_local"] = price_frame["close_raw"] * price_frame["volume"]
        price_frame["trade_value_sek"] = np.nan
        price_frame["source"] = "eodhd"
        raw_rows.append(price_frame[REQUIRED_COLUMNS["prices_raw_daily"]])
        action_rows.append(merge_corporate_actions(security.security_id, fetch_dividends(session, api_key, security.eodhd_symbol), fetch_splits(session, api_key, security.eodhd_symbol)))

    if failures:
        raise Phase1Error(f"EOD history fetch failed for {len(failures)} symbols; strict Phase 1 mode does not allow partial success.")
    if window_failures:
        raise Phase1Error(
            f"EOD history did not return requested-window rows for {len(window_failures)} overlapping symbols; "
            "strict Phase 1 mode does not allow partial success."
        )
    if not raw_rows:
        raise Phase1Error("No requested-window price history was returned for the selected date range.")

    for benchmark in BENCHMARKS:
        if progress_callback is not None:
            progress_callback(f"Downloading benchmark {benchmark['benchmark_id']} ({benchmark['eodhd_symbol']}).")
        benchmark_rows.append(fetch_benchmark_history(session, api_key, benchmark, start, end))

    security_master = security_master.copy()
    security_master["listing_date"] = security_master["security_id"].map(listing_dates)
    security_master["delisting_date"] = security_master["security_id"].map(last_trade_dates)
    security_master.loc[security_master["vendor_roster_status"] == "active_only", "delisting_date"] = pd.NaT
    security_master["is_delisted"] = security_master["delisting_date"].notna()
    security_master.loc[security_master["vendor_roster_status"] == "active_only", "is_delisted"] = False
    security_master = security_master[REQUIRED_COLUMNS["security_master"]].sort_values("security_id").reset_index(drop=True)

    prices_raw = pd.concat(raw_rows, ignore_index=True).sort_values(["security_id", "date"]).reset_index(drop=True)
    non_empty_action_rows = [frame for frame in action_rows if not frame.empty]
    corporate_actions = (
        pd.concat(non_empty_action_rows, ignore_index=True)
        if non_empty_action_rows
        else pd.DataFrame(columns=REQUIRED_COLUMNS["corporate_actions"])
    )
    if not corporate_actions.empty:
        corporate_actions = corporate_actions.sort_values(["security_id", "event_date", "action_type"]).reset_index(drop=True)
    benchmark_prices = pd.concat(benchmark_rows, ignore_index=True).sort_values(["benchmark_id", "date"]).reset_index(drop=True)

    delisted_metadata = security_master.loc[security_master["vendor_roster_status"] != "active_only", [
        "security_id", "eodhd_symbol", "company_name", "country_code", "exchange_name", "currency", "listing_date", "delisting_date",
    ]].copy()
    delisted_metadata["delisting_reason"] = None
    delisted_metadata["last_trade_date"] = delisted_metadata["security_id"].map(last_trade_dates)
    delisted_metadata["cashout_value"] = np.nan
    delisted_metadata["source"] = "eodhd"
    delisted_metadata = delisted_metadata[REQUIRED_COLUMNS["delisted_metadata"]].sort_values("security_id").reset_index(drop=True)

    output = {
        "security_master": security_master,
        "prices_raw_daily": prices_raw,
        "corporate_actions": corporate_actions,
        "delisted_metadata": delisted_metadata,
        "benchmark_prices": benchmark_prices,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name, frame in output.items():
        atomic_write_parquet(frame, out_dir / f"{artifact_name}.parquet")
        if progress_callback is not None:
            progress_callback(f"Wrote {artifact_name}.parquet ({len(frame)} rows).")
    return output


def validate_fx_freshness(fx_frame: pd.DataFrame, end: pd.Timestamp) -> None:
    fx_frame = normalize_date_column(fx_frame, ["date"])
    for currency in RIKSBANK_SERIES:
        subset = fx_frame.loc[(fx_frame["currency"] == currency) & (fx_frame["date"] <= end)]
        if subset.empty:
            raise Phase1Error(f"FX series for {currency} is missing entirely.")
        latest = subset["date"].max()
        if (end - latest).days > 7:
            raise Phase1Error(f"FX series for {currency} is stale relative to {end.date()}.")


def download_riksbank_fx_artifact(*, start: pd.Timestamp, end: pd.Timestamp, out_path: Path) -> pd.DataFrame:
    session = requests.Session()
    frames = []
    for currency, series_id in RIKSBANK_SERIES.items():
        payload = fetch_json(
            session,
            f"https://api.riksbank.se/swea/v1/Observations/{series_id}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
            {},
        )
        frame = pd.DataFrame(payload)
        if frame.empty:
            raise Phase1Error(f"Riksbank series for {currency} returned no data.")
        frame["date"] = pd.to_datetime(frame["date"])
        frame["sek_per_ccy"] = frame["value"].astype(float)
        frame["currency"] = currency
        frame["source"] = "riksbank"
        frames.append(frame[REQUIRED_COLUMNS["riksbank_fx_daily"]])

    result = pd.concat(frames, ignore_index=True).sort_values(["currency", "date"]).reset_index(drop=True)
    if (result["sek_per_ccy"] <= 0).any():
        raise Phase1Error("Riksbank FX series contains non-positive values.")
    validate_fx_freshness(result, end)
    atomic_write_parquet(result, out_path)
    return result


def enrich_with_fx(frame: pd.DataFrame, fx_frame: pd.DataFrame, *, date_col: str, currency_col: str) -> pd.DataFrame:
    frame = frame.copy()
    fx_frame = fx_frame.copy().sort_values(["currency", "date"])
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame["sek_per_ccy"] = np.nan
    sek_mask = frame[currency_col].eq("SEK")
    frame.loc[sek_mask, "sek_per_ccy"] = 1.0
    for currency in sorted(set(frame[currency_col].dropna()) - {"SEK"}):
        subset = frame.loc[frame[currency_col] == currency].sort_values(date_col).copy()
        subset["_row_id"] = subset.index
        currency_rates = fx_frame.loc[fx_frame["currency"] == currency, ["date", "sek_per_ccy"]].sort_values("date")
        if currency_rates.empty:
            raise Phase1Error(f"Missing FX rate history for {currency}.")
        merged = pd.merge_asof(
            subset,
            currency_rates,
            left_on=date_col,
            right_on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        if merged["sek_per_ccy_y"].isna().any():
            raise Phase1Error(f"FX asof join failed for {currency}.")
        frame.loc[merged["_row_id"].to_numpy(), "sek_per_ccy"] = merged["sek_per_ccy_y"].to_numpy()
    if frame["sek_per_ccy"].isna().any():
        raise Phase1Error("FX enrichment left missing rates in the output.")
    return frame


def apply_fx_to_raw_prices(raw_prices: pd.DataFrame, fx_frame: pd.DataFrame) -> pd.DataFrame:
    frame = enrich_with_fx(raw_prices, fx_frame, date_col="date", currency_col="currency")
    frame["trade_value_local"] = frame["close_raw"] * frame["volume"]
    frame["trade_value_sek"] = frame["trade_value_local"] * frame["sek_per_ccy"]
    return frame.sort_values(["security_id", "date"]).reset_index(drop=True)


def apply_fx_to_benchmarks(benchmark_prices: pd.DataFrame, fx_frame: pd.DataFrame) -> pd.DataFrame:
    frame = enrich_with_fx(benchmark_prices, fx_frame, date_col="date", currency_col="currency")
    frame["close_sek"] = frame["close_raw"] * frame["sek_per_ccy"]
    return frame[REQUIRED_COLUMNS["benchmark_prices"]].sort_values(["benchmark_id", "date"]).reset_index(drop=True)


def price_outlier_flags(raw_prices: pd.DataFrame, corporate_actions: pd.DataFrame) -> pd.DataFrame:
    action_dates = (
        corporate_actions.assign(event_date=pd.to_datetime(corporate_actions["event_date"]))
        .groupby(["security_id", "event_date"])
        .size()
        .rename("has_action")
        .reset_index()
    )
    frame = raw_prices.copy().sort_values(["security_id", "date"])
    frame["pct_move"] = frame.groupby("security_id")["close_raw"].pct_change().abs()
    frame = frame.merge(action_dates, how="left", left_on=["security_id", "date"], right_on=["security_id", "event_date"])
    frame["has_action"] = frame["has_action"].fillna(0).astype(int)
    frame["bad_outlier_data"] = (
        (frame["close_raw"] < 1.0)
        | (frame["close_raw"] > 100000.0)
        | ((frame["pct_move"] > 0.70) & (frame["has_action"] == 0))
    )
    return frame[["security_id", "date", "bad_outlier_data"]]


def reconstruct_adjusted_prices(raw_prices: pd.DataFrame, corporate_actions: pd.DataFrame) -> pd.DataFrame:
    if raw_prices.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS["prices_adjusted_daily"])

    action_frame = corporate_actions.copy()
    if not action_frame.empty:
        action_frame["event_date"] = pd.to_datetime(action_frame["event_date"])

    rows: list[pd.DataFrame] = []
    for security_id, security_prices in raw_prices.sort_values(["security_id", "date"]).groupby("security_id"):
        security_prices = security_prices.sort_values("date")
        security_actions = action_frame.loc[action_frame["security_id"] == security_id] if not action_frame.empty else pd.DataFrame()
        events_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        if not security_actions.empty:
            for event_date, event_rows in security_actions.groupby("event_date"):
                events_by_date[event_date] = event_rows

        adj_close_values = [float(security_prices.iloc[0]["close_raw"])]
        previous_adj = adj_close_values[0]
        previous_close = float(security_prices.iloc[0]["close_raw"])

        for row in security_prices.iloc[1:].itertuples(index=False):
            split_ratio = 1.0
            cash_value = 0.0
            for event in events_by_date.get(row.date, pd.DataFrame()).itertuples(index=False):
                if event.action_type in {"split", "reverse_split"} and not pd.isna(event.action_ratio):
                    split_ratio *= float(event.action_ratio)
                elif event.action_type in {"cash_dividend", "special_dividend", "cash_merger", "delisting_cashout"}:
                    cash_value += float(event.action_value)
            total_return = ((float(row.close_raw) + cash_value) * split_ratio) / previous_close
            previous_adj *= total_return
            previous_close = float(row.close_raw)
            adj_close_values.append(previous_adj)

        security_adjusted = security_prices[["security_id", "date", "currency"]].copy()
        security_adjusted["adj_close"] = adj_close_values
        security_adjusted["adj_factor"] = security_adjusted["adj_close"] / security_prices["close_raw"].to_numpy()
        security_adjusted["source"] = "phase1_local_reconstruction"
        rows.append(security_adjusted[REQUIRED_COLUMNS["prices_adjusted_daily"]])
    return pd.concat(rows, ignore_index=True).sort_values(["security_id", "date"]).reset_index(drop=True)


def month_difference(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
    return max(int(end_date.to_period("M").ordinal - start_date.to_period("M").ordinal), 0)


def build_exchange_calendar(raw_prices: pd.DataFrame) -> pd.DataFrame:
    price_frame = raw_prices[["date", "exchange_group"]].copy()
    price_frame["rebalance_month"] = price_frame["date"].dt.to_period("M").astype(str)
    monthly = price_frame.groupby(["exchange_group", "rebalance_month"])["date"].agg(anchor_trade_date="max", month_start="min").reset_index()
    monthly["next_month"] = (pd.PeriodIndex(monthly["rebalance_month"], freq="M") + 1).astype(str)
    next_exec = monthly[["exchange_group", "rebalance_month", "month_start"]].rename(columns={"rebalance_month": "next_month", "month_start": "next_execution_date"})
    monthly = monthly.merge(next_exec, on=["exchange_group", "next_month"], how="left")
    monthly = monthly.dropna(subset=["next_execution_date"]).drop(columns=["month_start", "next_month"])
    return monthly.sort_values(["exchange_group", "rebalance_month"]).reset_index(drop=True)


def load_market_cap_monthly(input_dir: Path) -> pd.DataFrame | None:
    path = input_dir / "market_cap_monthly.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    missing = [column for column in MARKET_CAP_MONTHLY_COLUMNS if column not in frame.columns]
    if missing:
        raise Phase1Error(f"market_cap_monthly.parquet is missing required columns: {', '.join(missing)}")
    frame = frame[MARKET_CAP_MONTHLY_COLUMNS].copy()
    duplicate_mask = frame.duplicated(["security_id", "rebalance_month"], keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask, ["security_id", "rebalance_month"]].head(5).to_dict("records")
        raise Phase1Error(f"market_cap_monthly.parquet has duplicate keys: {duplicates}")
    return frame.sort_values(["rebalance_month", "security_id"]).reset_index(drop=True)


def build_phase1_universe(*, input_dir: Path, emit_se_only: bool, emit_liquid_subset: bool, out_path: Path) -> dict[str, pd.DataFrame]:
    security_master = read_parquet(input_dir / "security_master.parquet", "security_master")
    raw_prices = read_parquet(input_dir / "prices_raw_daily.parquet", "prices_raw_daily")
    corporate_actions = read_parquet(input_dir / "corporate_actions.parquet", "corporate_actions")
    benchmark_prices = read_parquet(input_dir / "benchmark_prices.parquet", "benchmark_prices")
    fx_frame = read_parquet(input_dir / "riksbank_fx_daily.parquet", "riksbank_fx_daily")
    delisted_metadata_path = input_dir / "delisted_metadata.parquet"
    delisted_metadata = (
        read_parquet(delisted_metadata_path, "delisted_metadata")
        if delisted_metadata_path.exists()
        else None
    )
    in_scope_exchange_groups = active_nordic_exchange_groups()
    security_master = security_master.loc[security_master["exchange_group"].isin(in_scope_exchange_groups)].copy()
    if security_master.empty:
        raise Phase1Error("security_master contains no in-scope exchanges after applying ACTIVE_NORDIC_EXCHANGE_CODES.")
    in_scope_security_ids = set(security_master["security_id"])
    raw_prices = raw_prices.loc[raw_prices["security_id"].isin(in_scope_security_ids)].copy()
    corporate_actions = corporate_actions.loc[corporate_actions["security_id"].isin(in_scope_security_ids)].copy()
    if delisted_metadata is not None:
        delisted_metadata = delisted_metadata.loc[delisted_metadata["security_id"].isin(in_scope_security_ids)].copy()
    if raw_prices.empty:
        raise Phase1Error("prices_raw_daily contains no in-scope securities after applying ACTIVE_NORDIC_EXCHANGE_CODES.")

    validate_fx_freshness(fx_frame, raw_prices["date"].max())

    raw_prices = apply_fx_to_raw_prices(raw_prices, fx_frame)
    benchmark_prices = apply_fx_to_benchmarks(benchmark_prices, fx_frame)
    outlier_flags = price_outlier_flags(raw_prices, corporate_actions)
    raw_prices = raw_prices.merge(outlier_flags, on=["security_id", "date"], how="left")
    raw_prices["bad_outlier_data"] = raw_prices["bad_outlier_data"].fillna(False)
    raw_prices["close_raw_sek"] = raw_prices["close_raw"] * raw_prices["sek_per_ccy"]
    raw_prices["median_trade_value_local_60d"] = raw_prices.groupby("security_id")["trade_value_local"].transform(lambda values: values.rolling(window=60, min_periods=1).median())
    raw_prices["median_daily_value_60d_sek"] = raw_prices.groupby("security_id")["trade_value_sek"].transform(lambda values: values.rolling(window=60, min_periods=1).median())
    adjusted_prices = reconstruct_adjusted_prices(raw_prices, corporate_actions)

    joined_prices = raw_prices.merge(security_master[["security_id", "exchange_group"]], on="security_id", how="left")
    calendar = build_exchange_calendar(joined_prices)
    if security_master["issuer_group_id"].isna().any():
        raise Phase1Error("issuer_group_id is required for all in-scope common-share candidates.")
    if calendar.empty:
        raise Phase1Error("No exchange calendar could be built from the requested raw prices.")

    earliest_anchor = calendar["anchor_trade_date"].min()
    latest_anchor = calendar["anchor_trade_date"].max()
    relevant_security_master = security_master.loc[
        security_master["listing_date"].notna()
        & (security_master["listing_date"] <= latest_anchor)
        & (security_master["delisting_date"].isna() | (security_master["delisting_date"] >= earliest_anchor))
    ].copy()
    if relevant_security_master.empty:
        raise Phase1Error("No securities overlap the requested build window after applying static listing metadata.")

    skeleton = relevant_security_master[[
        "security_id", "country_code", "exchange_group", "currency", "listing_date", "delisting_date",
        "issuer_group_id", "security_type_normalized", "share_class_normalized",
    ]].merge(calendar, on="exchange_group", how="left")
    skeleton = skeleton.sort_values(["anchor_trade_date", "security_id"]).reset_index(drop=True)
    monthly_source = raw_prices[[
        "security_id", "date", "close_raw", "close_raw_sek", "trade_value_sek", "median_daily_value_60d_sek",
        "median_trade_value_local_60d", "bad_outlier_data",
    ]].sort_values(["date", "security_id"])
    monthly_rows = pd.merge_asof(
        skeleton,
        monthly_source,
        left_on="anchor_trade_date",
        right_on="date",
        by="security_id",
        direction="backward",
        allow_exact_matches=True,
    ).rename(columns={"date": "asof_trade_date", "close_raw": "close_raw_local"})

    monthly_rows["listing_age_months"] = monthly_rows.apply(lambda row: month_difference(row["listing_date"], row["anchor_trade_date"]), axis=1)
    monthly_rows["is_listed_asof_anchor"] = monthly_rows["listing_date"].notna() & (monthly_rows["listing_date"] <= monthly_rows["anchor_trade_date"]) & monthly_rows["asof_trade_date"].notna()
    monthly_rows["delisted_before_next_execution"] = monthly_rows["delisting_date"].notna() & (monthly_rows["delisting_date"] < monthly_rows["next_execution_date"])
    monthly_rows["stale_days"] = (monthly_rows["anchor_trade_date"] - monthly_rows["asof_trade_date"]).dt.days.fillna(9999)
    market_cap_monthly = load_market_cap_monthly(input_dir)
    if market_cap_monthly is not None:
        monthly_rows = monthly_rows.merge(market_cap_monthly, on=["security_id", "rebalance_month"], how="left")
        monthly_rows["market_cap_local_asof_anchor"] = monthly_rows["market_cap_local"]
        monthly_rows["market_cap_sek_asof_anchor"] = monthly_rows["market_cap_sek"]
        monthly_rows["status_source"] = monthly_rows["source"].fillna("missing_market_cap")
        monthly_rows = monthly_rows.drop(columns=["market_cap_local", "market_cap_sek", "source"])
    elif relevant_security_master["shares_outstanding"].notna().any():
        shares = relevant_security_master.set_index("security_id")["shares_outstanding"]
        monthly_rows["shares_outstanding"] = monthly_rows["security_id"].map(shares)
        monthly_rows["market_cap_local_asof_anchor"] = monthly_rows["shares_outstanding"] * monthly_rows["close_raw_local"]
        monthly_rows["market_cap_sek_asof_anchor"] = monthly_rows["shares_outstanding"] * monthly_rows["close_raw_sek"]
        monthly_rows["status_source"] = np.where(
            monthly_rows["shares_outstanding"].notna(),
            "shares_outstanding_times_anchor_close",
            "missing_market_cap",
        )
        monthly_rows = monthly_rows.drop(columns=["shares_outstanding"])
    else:
        monthly_rows["market_cap_local_asof_anchor"] = np.nan
        monthly_rows["market_cap_sek_asof_anchor"] = np.nan
        monthly_rows["status_source"] = "missing_market_cap"

    primary_candidates = monthly_rows.loc[
        monthly_rows["is_listed_asof_anchor"]
        & monthly_rows["security_type_normalized"].eq("common_share")
        & monthly_rows["share_class_normalized"].eq("ordinary")
    ].copy()
    primary_candidates["metric_trade"] = primary_candidates["median_daily_value_60d_sek"].fillna(-1.0)
    primary_candidates["metric_cap"] = primary_candidates["market_cap_sek_asof_anchor"].fillna(-1.0)
    primary_candidates["listing_sort"] = primary_candidates["listing_date"].fillna(pd.Timestamp.max)
    primary_candidates = primary_candidates.sort_values(
        ["rebalance_month", "issuer_group_id", "metric_trade", "metric_cap", "listing_sort", "security_id"],
        ascending=[True, True, False, False, True, True],
    )
    primary_keys = set(zip(primary_candidates.groupby(["rebalance_month", "issuer_group_id"]).head(1)["rebalance_month"], primary_candidates.groupby(["rebalance_month", "issuer_group_id"]).head(1)["security_id"]))
    monthly_rows["is_primary_common_share_asof_anchor"] = [
        (month, security_id) in primary_keys for month, security_id in zip(monthly_rows["rebalance_month"], monthly_rows["security_id"])
    ]
    monthly_rows["eligible_for_main_universe_asof_anchor"] = monthly_rows["is_primary_common_share_asof_anchor"] & monthly_rows["security_type_normalized"].eq("common_share") & monthly_rows["share_class_normalized"].eq("ordinary")

    status = monthly_rows[REQUIRED_COLUMNS["security_status_pti_monthly"]].copy()

    def exclusion_reason(row: pd.Series, *, se_only: bool) -> str:
        if row["security_type_normalized"] != "common_share" or row["share_class_normalized"] != "ordinary":
            return "non_common_share"
        if row["exchange_group"] not in in_scope_exchange_groups:
            return "excluded_venue"
        if se_only and row["exchange_group"] != NORDIC_EXCHANGES["ST"]["exchange_group"]:
            return "outside_se_only_scope"
        if not row["is_primary_common_share_asof_anchor"]:
            return "non_primary_listing"
        if row["listing_age_months"] < config.MIN_LISTING_MONTHS:
            return "too_short_listing_history"
        if pd.isna(row["asof_trade_date"]) or pd.isna(row["close_raw_local"]) or pd.isna(row["trade_value_sek"]):
            return "missing_price_or_volume"
        if row["stale_days"] > 0:
            return "stale_last_trade"
        if row["delisted_before_next_execution"]:
            return "delisted_before_execution"
        if pd.isna(row["close_raw_sek"]) or row["close_raw_sek"] < config.MIN_PRICE_SEK:
            return "price_below_min_sek"
        if bool(row["bad_outlier_data"]):
            return "bad_outlier_data"
        return ""

    universe = monthly_rows[[
        "rebalance_month", "anchor_trade_date", "next_execution_date", "security_id", "country_code", "exchange_group",
        "currency", "asof_trade_date", "listing_age_months", "close_raw_local", "close_raw_sek", "trade_value_sek",
        "median_daily_value_60d_sek", "market_cap_sek_asof_anchor", "bad_outlier_data", "security_type_normalized",
        "share_class_normalized", "is_primary_common_share_asof_anchor", "stale_days", "delisted_before_next_execution",
    ]].copy()
    universe["market_cap_sek"] = universe["market_cap_sek_asof_anchor"]
    universe["exclusion_reason_full_nordics"] = universe.apply(lambda row: exclusion_reason(row, se_only=False), axis=1)
    universe["is_eligible_full_nordics"] = universe["exclusion_reason_full_nordics"].eq("")
    if emit_se_only:
        universe["exclusion_reason_se_only"] = universe.apply(lambda row: exclusion_reason(row, se_only=True), axis=1)
        universe["is_eligible_se_only"] = universe["exclusion_reason_se_only"].eq("")
    else:
        universe["exclusion_reason_se_only"] = "outside_se_only_scope"
        universe["is_eligible_se_only"] = False

    universe["exclusion_reason_liquid_subset"] = universe["exclusion_reason_full_nordics"]
    universe["is_eligible_liquid_subset"] = False
    if emit_liquid_subset:
        eligible_rows = universe.loc[universe["is_eligible_full_nordics"]].copy()
        missing_market_cap = eligible_rows["market_cap_sek"].isna()
        missing_share = float(missing_market_cap.mean()) if not eligible_rows.empty else 1.0
        if eligible_rows.empty or missing_share > 0.05:
            universe.loc[universe["is_eligible_full_nordics"], "exclusion_reason_liquid_subset"] = "market_cap_data_unavailable"
        else:
            eligible_rows = eligible_rows.loc[~missing_market_cap].copy()
            eligible_rows["market_cap_rank"] = eligible_rows.groupby("rebalance_month")["market_cap_sek"].rank(method="first", ascending=False)
            month_counts = eligible_rows.groupby("rebalance_month")["security_id"].transform("count")
            eligible_rows["liquid_cut"] = np.ceil(month_counts / 3.0)
            liquid_month_keys = set(zip(
                eligible_rows.loc[eligible_rows["market_cap_rank"] <= eligible_rows["liquid_cut"], "rebalance_month"],
                eligible_rows.loc[eligible_rows["market_cap_rank"] <= eligible_rows["liquid_cut"], "security_id"],
            ))
            universe["is_eligible_liquid_subset"] = [
                (month, security_id) in liquid_month_keys for month, security_id in zip(universe["rebalance_month"], universe["security_id"])
            ]
            universe.loc[universe["is_eligible_full_nordics"] & ~universe["is_eligible_liquid_subset"] & universe["market_cap_sek"].notna(), "exclusion_reason_liquid_subset"] = "outside_liquid_subset_cut"
            universe.loc[universe["is_eligible_full_nordics"] & universe["market_cap_sek"].isna(), "exclusion_reason_liquid_subset"] = "market_cap_data_unavailable"
    else:
        universe.loc[universe["is_eligible_full_nordics"], "exclusion_reason_liquid_subset"] = "liquid_subset_disabled"

    universe = universe[REQUIRED_COLUMNS["universe_pti"]].sort_values(["rebalance_month", "security_id"]).reset_index(drop=True)
    atomic_write_parquet(security_master[REQUIRED_COLUMNS["security_master"]], input_dir / "security_master.parquet")
    atomic_write_parquet(raw_prices[REQUIRED_COLUMNS["prices_raw_daily"]], input_dir / "prices_raw_daily.parquet")
    atomic_write_parquet(corporate_actions[REQUIRED_COLUMNS["corporate_actions"]], input_dir / "corporate_actions.parquet")
    if delisted_metadata is not None:
        atomic_write_parquet(delisted_metadata[REQUIRED_COLUMNS["delisted_metadata"]], delisted_metadata_path)
    atomic_write_parquet(benchmark_prices, input_dir / "benchmark_prices.parquet")
    atomic_write_parquet(status, input_dir / "security_status_pti_monthly.parquet")
    atomic_write_parquet(adjusted_prices, input_dir / "prices_adjusted_daily.parquet")
    atomic_write_parquet(universe, out_path)
    return {
        "security_status_pti_monthly": status,
        "prices_adjusted_daily": adjusted_prices,
        "universe_pti": universe,
        "prices_raw_daily": raw_prices[REQUIRED_COLUMNS["prices_raw_daily"]],
        "benchmark_prices": benchmark_prices,
    }


def _check_required_columns(frame: pd.DataFrame, artifact_name: str) -> None:
    required = REQUIRED_COLUMNS.get(artifact_name, [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise Phase1Error(f"{artifact_name} is missing required columns: {', '.join(missing)}")


def _check_primary_key(frame: pd.DataFrame, artifact_name: str) -> None:
    keys = PRIMARY_KEYS.get(artifact_name)
    if not keys or frame.empty:
        return
    duplicate_mask = frame.duplicated(keys, keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask, keys].head(5).to_dict("records")
        raise Phase1Error(f"{artifact_name} violates primary key {keys}; duplicates include {duplicates}")


def _check_date_sorting(frame: pd.DataFrame, artifact_name: str) -> None:
    if frame.empty:
        return
    if artifact_name in {"prices_raw_daily", "prices_adjusted_daily"}:
        grouped = frame.groupby("security_id")["date"]
    elif artifact_name == "benchmark_prices":
        grouped = frame.groupby("benchmark_id")["date"]
    elif artifact_name == "riksbank_fx_daily":
        grouped = frame.groupby("currency")["date"]
    else:
        return

    for key, series in grouped:
        normalized = pd.to_datetime(series)
        if not normalized.is_monotonic_increasing:
            raise Phase1Error(f"{artifact_name} dates are not ascending within {key}.")


def _expected_fx_rate(fx_frame: pd.DataFrame, currency: str, on_date: pd.Timestamp) -> float:
    if currency == "SEK":
        return 1.0
    subset = fx_frame.loc[(fx_frame["currency"] == currency) & (fx_frame["date"] <= on_date), ["date", "sek_per_ccy"]]
    if subset.empty:
        raise Phase1Error(f"Missing FX observation for {currency} on or before {on_date.date()}.")
    return float(subset.sort_values("date").iloc[-1]["sek_per_ccy"])


def _artifact_paths(input_dir: Path) -> dict[str, Path]:
    return {
        "security_master": input_dir / "security_master.parquet",
        "security_status_pti_monthly": input_dir / "security_status_pti_monthly.parquet",
        "prices_raw_daily": input_dir / "prices_raw_daily.parquet",
        "prices_adjusted_daily": input_dir / "prices_adjusted_daily.parquet",
        "corporate_actions": input_dir / "corporate_actions.parquet",
        "delisted_metadata": input_dir / "delisted_metadata.parquet",
        "benchmark_prices": input_dir / "benchmark_prices.parquet",
        "riksbank_fx_daily": input_dir / "riksbank_fx_daily.parquet",
        "universe_pti": input_dir / "universe_pti.parquet",
    }


def load_cpp_module():
    module_name = "alpha_momentum_cpp"
    search_dirs = [
        ROOT,
        ROOT / "build",
        ROOT / "build" / "Release",
        ROOT / "build" / "Debug",
        ROOT / "build" / "RelWithDebInfo",
        ROOT / "build" / "MinSizeRel",
    ]
    for candidate in ROOT.glob("build/**/alpha_momentum_cpp*.pyd"):
        search_dirs.append(candidate.parent)

    last_error: Exception | None = None
    for directory in search_dirs:
        if directory.exists() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        try:
            importlib.invalidate_caches()
            return importlib.import_module(module_name)
        except ImportError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError(f"Could not import {module_name}.")


def validate_phase1(
    *,
    input_dir: Path,
    require_cpp: bool,
    require_benchmarks: bool,
) -> tuple[bool, list[ValidationSection]]:
    sections: list[ValidationSection] = []
    artifact_paths = _artifact_paths(input_dir)
    artifacts: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for artifact_name, path in artifact_paths.items():
        try:
            frame = read_parquet(path, artifact_name)
            _check_required_columns(frame, artifact_name)
            _check_primary_key(frame, artifact_name)
            _check_date_sorting(frame, artifact_name)
            artifacts[artifact_name] = frame
        except Exception as exc:
            errors.append(str(exc))

    if errors:
        sections.append(ValidationSection("Artifacts", "FAIL", errors))
        return False, sections

    sections.append(
        ValidationSection(
            "Artifacts",
            "PASS",
            [f"Loaded {name}.parquet" for name in artifact_paths],
        )
    )

    security_master = artifacts["security_master"]
    status = artifacts["security_status_pti_monthly"]
    raw_prices = artifacts["prices_raw_daily"]
    adjusted_prices = artifacts["prices_adjusted_daily"]
    corporate_actions = artifacts["corporate_actions"]
    delisted_metadata = artifacts["delisted_metadata"]
    benchmark_prices = artifacts["benchmark_prices"]
    fx_frame = artifacts["riksbank_fx_daily"]
    universe = artifacts["universe_pti"]

    security_messages: list[str] = []
    try:
        invalid_status = sorted(set(security_master["vendor_roster_status"]) - ALLOWED_VENDOR_ROSTER_STATUS)
        if invalid_status:
            raise Phase1Error(f"security_master has unsupported vendor_roster_status values: {invalid_status}")
        if security_master["issuer_group_id"].isna().any():
            raise Phase1Error("security_master has missing issuer_group_id values.")
        if security_master["security_id"].ne(security_master["eodhd_symbol"]).any():
            raise Phase1Error("security_master security_id must equal eodhd_symbol for all rows.")
        if security_master["security_type_normalized"].ne("common_share").any():
            raise Phase1Error("security_master contains out-of-scope security_type_normalized values.")
        if security_master["share_class_normalized"].ne("ordinary").any():
            raise Phase1Error("security_master contains out-of-scope share_class_normalized values.")
        invalid_exchange = sorted(set(security_master["exchange_group"]) - active_nordic_exchange_groups())
        if invalid_exchange:
            raise Phase1Error(f"security_master has unexpected exchange_group values: {invalid_exchange}")
        security_messages.append(f"{len(security_master)} in-scope securities loaded.")
        sections.append(ValidationSection("Security Master", "PASS", security_messages))
    except Exception as exc:
        sections.append(ValidationSection("Security Master", "FAIL", [str(exc)]))

    price_messages: list[str] = []
    try:
        if raw_prices.empty:
            raise Phase1Error("prices_raw_daily is empty.")
        if (raw_prices[["open_raw", "high_raw", "low_raw", "close_raw"]] <= 0).any().any():
            raise Phase1Error("prices_raw_daily contains non-positive OHLC values.")
        if (raw_prices["volume"] < 0).any():
            raise Phase1Error("prices_raw_daily contains negative volume.")
        raw_prices = raw_prices.copy()
        raw_prices["expected_trade_value_local"] = raw_prices["close_raw"] * raw_prices["volume"]
        if not np.allclose(raw_prices["trade_value_local"], raw_prices["expected_trade_value_local"], rtol=0.0, atol=1e-9):
            raise Phase1Error("prices_raw_daily trade_value_local does not match close_raw * volume.")
        if not np.isfinite(raw_prices["trade_value_sek"]).all() or (raw_prices["trade_value_sek"] < 0).any():
            raise Phase1Error("prices_raw_daily contains invalid trade_value_sek values.")
        recomputed_adjusted = reconstruct_adjusted_prices(raw_prices[REQUIRED_COLUMNS["prices_raw_daily"]], corporate_actions)
        merged_adjusted = adjusted_prices.merge(
            recomputed_adjusted,
            on=["security_id", "date"],
            how="outer",
            suffixes=("_stored", "_recomputed"),
            indicator=True,
        )
        if not merged_adjusted["_merge"].eq("both").all():
            raise Phase1Error("prices_adjusted_daily does not align with the local reconstruction key set.")
        if not np.allclose(merged_adjusted["adj_close_stored"], merged_adjusted["adj_close_recomputed"], rtol=0.0, atol=1e-9):
            raise Phase1Error("prices_adjusted_daily adj_close differs from the local reconstruction.")
        if not np.allclose(merged_adjusted["adj_factor_stored"], merged_adjusted["adj_factor_recomputed"], rtol=0.0, atol=1e-9):
            raise Phase1Error("prices_adjusted_daily adj_factor differs from the local reconstruction.")
        invalid_action_types = sorted(set(corporate_actions["action_type"]) - ALLOWED_ACTION_TYPES)
        if invalid_action_types:
            raise Phase1Error(f"corporate_actions contains unsupported action_type values: {invalid_action_types}")
        if not corporate_actions.empty:
            split_mask = corporate_actions["action_type"].isin({"split", "reverse_split"})
            if corporate_actions.loc[split_mask, "action_ratio"].le(0).any():
                raise Phase1Error("corporate_actions split ratios must be positive.")
        price_messages.append(f"{len(raw_prices)} raw rows and {len(adjusted_prices)} adjusted rows validated.")
        sections.append(ValidationSection("Prices And Actions", "PASS", price_messages))
    except Exception as exc:
        sections.append(ValidationSection("Prices And Actions", "FAIL", [str(exc)]))

    fx_messages: list[str] = []
    try:
        observed_currencies = set(fx_frame["currency"])
        if observed_currencies != set(RIKSBANK_SERIES):
            raise Phase1Error(f"riksbank_fx_daily currencies must equal {sorted(RIKSBANK_SERIES)}; found {sorted(observed_currencies)}")
        if (fx_frame["sek_per_ccy"] <= 0).any():
            raise Phase1Error("riksbank_fx_daily contains non-positive sek_per_ccy values.")
        validate_fx_freshness(fx_frame, raw_prices["date"].max())
        sample_prices = raw_prices.head(50)
        for row in sample_prices.itertuples(index=False):
            expected_sek = float(row.trade_value_local) * _expected_fx_rate(fx_frame, row.currency, row.date)
            if not np.isclose(float(row.trade_value_sek), expected_sek, rtol=0.0, atol=1e-6):
                raise Phase1Error("prices_raw_daily trade_value_sek does not match same-day-or-earlier FX conversion.")
        fx_messages.append("FX positivity, freshness, and as-of conversion checks passed.")
        sections.append(ValidationSection("FX", "PASS", fx_messages))
    except Exception as exc:
        sections.append(ValidationSection("FX", "FAIL", [str(exc)]))

    delisting_messages: list[str] = []
    try:
        if not delisted_metadata.empty:
            missing_security_rows = set(delisted_metadata["security_id"]) - set(security_master["security_id"])
            if missing_security_rows:
                raise Phase1Error(f"delisted_metadata references unknown securities: {sorted(missing_security_rows)[:5]}")
            active_delisted = security_master.set_index("security_id").loc[delisted_metadata["security_id"], "vendor_roster_status"].eq("active_only")
            if active_delisted.any():
                raise Phase1Error("delisted_metadata includes securities tagged active_only in security_master.")
        last_trade = raw_prices.groupby("security_id")["date"].max().rename("last_price_date")
        latest_date = raw_prices["date"].max()
        stale_histories = last_trade.loc[(latest_date - last_trade).dt.days > 30]
        if not stale_histories.empty:
            delisted_ids = set(delisted_metadata["security_id"])
            unresolved = [security_id for security_id in stale_histories.index if security_id not in delisted_ids]
            if unresolved:
                raise Phase1Error(
                    "Found in-scope price histories ending more than 30 calendar days before end_date without delisted coverage: "
                    f"{unresolved[:5]}"
                )
        delisting_messages.append(f"{len(delisted_metadata)} delisted rows reconciled against price histories.")
        sections.append(ValidationSection("Delistings", "PASS", delisting_messages))
    except Exception as exc:
        sections.append(ValidationSection("Delistings", "FAIL", [str(exc)]))

    benchmark_messages: list[str] = []
    try:
        expected_ids = {benchmark["benchmark_id"] for benchmark in BENCHMARKS}
        observed_ids = set(benchmark_prices["benchmark_id"])
        if require_benchmarks and observed_ids != expected_ids:
            raise Phase1Error(f"benchmark_prices benchmark_id set must equal {sorted(expected_ids)}; found {sorted(observed_ids)}")
        invalid_types = sorted(set(benchmark_prices["benchmark_type"]) - ALLOWED_BENCHMARK_TYPES)
        if invalid_types:
            raise Phase1Error(f"benchmark_prices contains unsupported benchmark_type values: {invalid_types}")
        grouped = benchmark_prices.groupby("benchmark_id")
        for benchmark_id, frame in grouped:
            if frame["benchmark_name"].nunique() != 1:
                raise Phase1Error(f"benchmark_prices has inconsistent benchmark_name values for {benchmark_id}.")
            if frame["benchmark_type"].nunique() != 1:
                raise Phase1Error(f"benchmark_prices has inconsistent benchmark_type values for {benchmark_id}.")
            if (frame[["close_raw", "adj_close", "close_sek"]] <= 0).any().any():
                raise Phase1Error(f"benchmark_prices contains non-positive price data for {benchmark_id}.")
            if not frame["date"].is_monotonic_increasing:
                raise Phase1Error(f"benchmark_prices dates are not ascending for {benchmark_id}.")
        benchmark_messages.append(f"{len(grouped)} benchmark series validated.")
        sections.append(ValidationSection("Benchmarks", "PASS", benchmark_messages))
    except Exception as exc:
        sections.append(ValidationSection("Benchmarks", "FAIL", [str(exc)]))

    universe_messages: list[str] = []
    try:
        if len(universe) != len(status):
            raise Phase1Error("universe_pti row count must match security_status_pti_monthly row count.")
        if status["status_source"].eq("trade_value_proxy_market_cap").any():
            raise Phase1Error("security_status_pti_monthly must not use trade_value_proxy_market_cap.")
        merged = universe.merge(
            status,
            on=["rebalance_month", "security_id", "anchor_trade_date", "next_execution_date", "country_code", "exchange_group"],
            how="inner",
            suffixes=("_universe", "_status"),
        )
        if len(merged) != len(universe):
            raise Phase1Error("universe_pti rows must all match security_status_pti_monthly on key fields.")
        if not merged["listing_age_months_universe"].equals(merged["listing_age_months_status"]):
            raise Phase1Error("universe_pti listing_age_months must match security_status_pti_monthly.")
        comparable_market_cap = merged["market_cap_sek"].notna() & merged["market_cap_sek_asof_anchor"].notna()
        if comparable_market_cap.any() and not np.allclose(
            merged.loc[comparable_market_cap, "market_cap_sek"],
            merged.loc[comparable_market_cap, "market_cap_sek_asof_anchor"],
            rtol=0.0,
            atol=1e-9,
        ):
            raise Phase1Error("universe_pti market_cap_sek must match security_status_pti_monthly market_cap_sek_asof_anchor.")
        if merged["is_eligible_se_only"].any() and not merged.loc[merged["is_eligible_se_only"], "exchange_group"].eq(NORDIC_EXCHANGES["ST"]["exchange_group"]).all():
            raise Phase1Error("SE-only eligibility must only occur on Nasdaq Stockholm.")
        if not merged.loc[merged["is_eligible_liquid_subset"], "is_eligible_full_nordics"].all():
            raise Phase1Error("largest-third-by-market-cap eligibility must be a subset of Full Nordics.")
        if not merged.loc[merged["is_eligible_full_nordics"], "eligible_for_main_universe_asof_anchor"].all():
            raise Phase1Error("Full Nordics eligibility requires eligible_for_main_universe_asof_anchor in security_status_pti_monthly.")
        if not merged.loc[merged["is_eligible_full_nordics"], "exclusion_reason_full_nordics"].eq("").all():
            raise Phase1Error("Eligible Full Nordics rows must have blank exclusion_reason_full_nordics.")
        if not merged.loc[~merged["is_eligible_full_nordics"], "exclusion_reason_full_nordics"].ne("").all():
            raise Phase1Error("Ineligible Full Nordics rows must have explicit exclusion_reason_full_nordics.")
        if not merged.loc[merged["is_eligible_full_nordics"], "listing_age_months_universe"].ge(config.MIN_LISTING_MONTHS).all():
            raise Phase1Error("Eligible Full Nordics rows must satisfy the minimum listing age.")
        if not merged.loc[merged["is_eligible_full_nordics"], "close_raw_sek"].ge(config.MIN_PRICE_SEK).all():
            raise Phase1Error("Eligible Full Nordics rows must satisfy MIN_PRICE_SEK.")
        if (merged["next_execution_date"] <= merged["anchor_trade_date"]).any():
            raise Phase1Error("next_execution_date must be after anchor_trade_date for every universe row.")
        if merged.loc[merged["is_eligible_full_nordics"], "delisted_before_next_execution"].any():
            raise Phase1Error("Eligible Full Nordics rows cannot be delisted before next execution.")
        full_eligible = merged.loc[merged["is_eligible_full_nordics"]].copy()
        observed_liquid = set(zip(
            merged.loc[merged["is_eligible_liquid_subset"], "rebalance_month"],
            merged.loc[merged["is_eligible_liquid_subset"], "security_id"],
        ))
        if observed_liquid:
            if merged.loc[merged["is_eligible_liquid_subset"], "market_cap_sek"].isna().any():
                raise Phase1Error("largest-third-by-market-cap eligibility requires non-missing market_cap_sek.")
            expected_liquid = set()
            eligible = full_eligible.loc[full_eligible["market_cap_sek"].notna()].copy()
            eligible["market_cap_rank"] = eligible.groupby("rebalance_month")["market_cap_sek"].rank(method="first", ascending=False)
            eligible["liquid_cut"] = np.ceil(eligible.groupby("rebalance_month")["security_id"].transform("count") / 3.0)
            expected_liquid = set(zip(
                eligible.loc[eligible["market_cap_rank"] <= eligible["liquid_cut"], "rebalance_month"],
                eligible.loc[eligible["market_cap_rank"] <= eligible["liquid_cut"], "security_id"],
            ))
            if observed_liquid != expected_liquid:
                raise Phase1Error("largest-third-by-market-cap eligibility does not match the point-in-time market-cap ranking.")
        else:
            allowed_disabled_reasons = {"liquid_subset_disabled", "market_cap_data_unavailable"}
            invalid_disabled = full_eligible.loc[~full_eligible["exclusion_reason_liquid_subset"].isin(allowed_disabled_reasons | {"outside_liquid_subset_cut"})]
            if not invalid_disabled.empty:
                raise Phase1Error("Full-Nordics rows have invalid exclusion_reason_liquid_subset values.")
        universe_messages.append(f"{len(universe)} universe rows validated across all variants.")
        sections.append(ValidationSection("Universe", "PASS", universe_messages))
    except Exception as exc:
        sections.append(ValidationSection("Universe", "FAIL", [str(exc)]))

    parity_messages: list[str] = []
    try:
        if not require_cpp:
            sections.append(ValidationSection("C++ Parity", "SKIP", ["Skipped because --require-cpp was false."]))
        else:
            alpha_momentum_cpp = load_cpp_module()

            eligible_rows = universe.loc[universe["is_eligible_full_nordics"]].copy()
            if eligible_rows.empty:
                raise Phase1Error("No eligible Full Nordics rows are available for the parity fixture.")
            latest_month = eligible_rows["rebalance_month"].max()
            sample = eligible_rows.loc[eligible_rows["rebalance_month"] == latest_month, ["security_id", "anchor_trade_date"]].copy()
            fixture_length = config.L + config.SKIP + 1
            price_matrix: list[np.ndarray] = []
            security_ids: list[str] = []
            for row in sample.itertuples(index=False):
                history = adjusted_prices.loc[
                    (adjusted_prices["security_id"] == row.security_id) & (adjusted_prices["date"] <= row.anchor_trade_date),
                    "adj_close",
                ].tail(fixture_length)
                if len(history) != fixture_length or (history <= 0).any():
                    continue
                price_matrix.append(history.to_numpy(dtype=float))
                security_ids.append(row.security_id)
            if not price_matrix:
                raise Phase1Error("No eligible securities had enough adjusted-price history for the parity fixture.")
            prices = np.vstack(price_matrix)
            eligible_mask = np.ones(len(security_ids), dtype=bool)
            python_scores = engine.compute_momentum_scores(prices, lookback=config.L, skip=config.SKIP)
            cpp_scores = np.asarray(alpha_momentum_cpp.compute_momentum_scores(prices, lookback=config.L, skip=config.SKIP), dtype=float)
            if not np.allclose(python_scores, cpp_scores, rtol=0.0, atol=1e-12):
                raise Phase1Error("C++ momentum scores do not match the Python reference.")
            top_n = min(config.TOP_N, len(security_ids))
            python_selected = engine.select_top_n(python_scores, eligible_mask, top_n=top_n)
            cpp_selected = list(alpha_momentum_cpp.select_top_n(cpp_scores, eligible_mask, top_n=top_n))
            if python_selected != cpp_selected:
                raise Phase1Error("C++ top-N selection does not match the Python reference.")
            python_weights = engine.equal_weight_positions(python_selected, len(security_ids))
            cpp_weights = np.asarray(alpha_momentum_cpp.equal_weight_positions(cpp_selected, len(security_ids)), dtype=float)
            if not np.allclose(python_weights, cpp_weights, rtol=0.0, atol=1e-12):
                raise Phase1Error("C++ equal-weight positions do not match the Python reference.")
            parity_messages.append(f"C++ parity passed on {len(security_ids)} securities for {latest_month}.")
            sections.append(ValidationSection("C++ Parity", "PASS", parity_messages))
    except Exception as exc:
        sections.append(ValidationSection("C++ Parity", "FAIL", [str(exc)]))

    success = all(section.status != "FAIL" for section in sections)
    return success, sections


def format_validation_report(sections: list[ValidationSection]) -> str:
    lines: list[str] = []
    for section in sections:
        lines.append(f"[{section.status}] {section.name}")
        for message in section.messages:
            lines.append(f"- {message}")
    success = all(section.status != "FAIL" for section in sections)
    lines.append("PASS: Phase 1 green" if success else "FAIL: Phase 1 not complete")
    return "\n".join(lines)
