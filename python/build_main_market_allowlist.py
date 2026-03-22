from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from phase1_lib import (
    MainMarketAllowlist,
    NORDIC_EXCHANGES,
    Phase1Error,
    active_nordic_exchange_groups,
    active_nordic_exchanges,
    load_eodhd_api_key,
    load_exchange_roster,
    load_main_market_allowlist,
    normalize_security_row,
    normalize_text,
    parse_bool,
)

REVIEW_COLUMNS = [
    "include",
    "recommendation",
    "recommendation_reason",
    "review_status",
    "seed_source",
    "security_id",
    "isin",
    "company_name",
    "exchange_group",
    "country_code",
    "currency",
    "vendor_roster_status",
    "ticker_local",
    "notes",
]
FINAL_COLUMNS = ["security_id", "isin", "company_name", "exchange_group", "notes", "include"]
LIKELY_EXCLUDE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(etf|etn)\b", re.IGNORECASE), "fund_or_exchange_traded_product"),
    (re.compile(r"\b(fund|fond|fonden)\b", re.IGNORECASE), "fund_or_exchange_traded_product"),
    (re.compile(r"\binvesteringsforeningen\b", re.IGNORECASE), "fund_or_exchange_traded_product"),
    (re.compile(r"\b(oblig|bond)\b", re.IGNORECASE), "bond_or_obligation_product"),
    (re.compile(r"\b(certificate|certifikat|warrant|rights|unit|trust)\b", re.IGNORECASE), "non_common_share_instrument"),
    (re.compile(r"\b(pref|preference|pfd)\b", re.IGNORECASE), "preferred_share_like"),
    (re.compile(r"\b(clo|pensionplanner)\b", re.IGNORECASE), "fund_or_credit_vehicle"),
    (re.compile(r"\bakk\b", re.IGNORECASE), "accumulating_fund_share_class"),
    (re.compile(r"\bkl\b", re.IGNORECASE), "fund_share_class_kl"),
]
NASDAQ_OFFICIAL_REFERER = "https://www.nasdaqomxnordic.com/shares/listed-companies/stockholm"
NASDAQ_OFFICIAL_MAIN_MARKET_URL = "https://api.nasdaq.com/api/nordic/screener/shares"
NASDAQ_OFFICIAL_INSTRUMENT_INFO_URL = "https://api.nasdaq.com/api/nordic/instruments/{orderbook_id}/info"
NASDAQ_EXCHANGE_GROUPS = {
    exchange_group
    for exchange_group in active_nordic_exchange_groups()
    if exchange_group.startswith("Nasdaq ")
}
NASDAQ_MAIN_MARKET_SEGMENTS = {"Large Cap", "Mid Cap", "Small Cap"}
EURONEXT_OSLO_REFERER = "https://live.euronext.com/en/markets/oslo/equities/list"
EURONEXT_OSLO_DOWNLOAD_URL = "https://live.euronext.com/pd_es/data/stocks/download?mics=XOSL%2CMERK%2CXOAS"
EURONEXT_OSLO_MAIN_MARKET_LABEL = "Oslo Børs"
OFFICIAL_FINAL_COLUMNS = ["security_id", "isin", "company_name", "exchange_group", "notes", "include"]
OFFICIAL_RECONCILIATION_COLUMNS = [
    "security_id",
    "exchange_group",
    "isin",
    "company_name",
    "official_symbol",
    "official_name",
    "official_segment",
    "official_market_label",
    "official_source",
    "match_reason",
    "include",
    "notes",
]
COMPANY_KEY_NOISE_TOKENS = {
    "a", "b", "c", "d", "series", "share", "shares", "sdb", "sek", "dkk",
    "ab", "abp", "asa", "oy", "oyj", "as", "aps", "plc", "ltd", "limited", "hf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or finalize the curated main-market allowlist used by Phase 1 downloads."
    )
    parser.add_argument("--mode", choices=("review", "finalize", "official"), required=True)
    parser.add_argument("--include-delisted", default=str(config.INCLUDE_DELISTED).lower())
    parser.add_argument("--existing-allowlist", default=config.MAIN_MARKET_ALLOWLIST_PATH)
    parser.add_argument("--review-input", default="data/main_market_allowlist_review.csv")
    parser.add_argument("--candidate-out", default="data/main_market_allowlist_candidates.csv")
    parser.add_argument("--reconciliation-out", default="data/main_market_allowlist_official_reconciliation.csv")
    parser.add_argument("--out")
    return parser.parse_args()


def atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_path.parent / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / output_path.name
    frame.to_csv(staging_path, index=False)
    shutil.move(str(staging_path), output_path)


def atomic_write_excel_friendly_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_path.parent / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / output_path.name
    frame.to_csv(staging_path, index=False, sep=";", encoding="utf-8-sig")
    shutil.move(str(staging_path), output_path)


def excel_friendly_path(output_path: Path) -> Path:
    if output_path.suffix.lower() == ".csv":
        return output_path.with_name(f"{output_path.stem}_excel.csv")
    return output_path.with_name(f"{output_path.name}_excel.csv")


def read_delimited_frame(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False, sep=None, engine="python")
    frame = frame.rename(columns=lambda value: str(value).replace("\ufeff", "").strip())
    return frame


def official_headers(*, referer: str) -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": referer,
    }


def normalize_symbol_key(value: Any) -> str:
    return "".join(character for character in normalize_text(value).upper() if character.isalnum())


def normalize_company_key(value: Any) -> str:
    ascii_text = unicodedata.normalize("NFKD", normalize_text(value)).encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    tokens = [token for token in ascii_text.split() if token and token not in COMPANY_KEY_NOISE_TOKENS]
    return " ".join(tokens)


def security_id_stem(security_id: str) -> str:
    return normalize_text(security_id).split(".", 1)[0]


def fetch_official_nasdaq_main_market(session: requests.Session) -> pd.DataFrame:
    response = session.get(
        NASDAQ_OFFICIAL_MAIN_MARKET_URL,
        params={"category": "MAIN_MARKET", "tableonly": "false"},
        timeout=60,
        headers=official_headers(referer=NASDAQ_OFFICIAL_REFERER),
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("instrumentListing", {}).get("rows", [])
    if not rows:
        raise Phase1Error("Nasdaq official main-market screener returned no rows.")

    records: list[dict[str, str]] = []
    exchange_by_currency = {
        "SEK": "Nasdaq Stockholm",
        "DKK": "Nasdaq Copenhagen",
    }
    for row in rows:
        orderbook_id = normalize_text(row.get("orderbookId"))
        if not orderbook_id:
            continue
        currency = normalize_text(row.get("currency"))
        exchange_group = exchange_by_currency.get(currency, "")
        segment = ""
        header: dict[str, Any] = {}
        if not exchange_group:
            detail_response = session.get(
                NASDAQ_OFFICIAL_INSTRUMENT_INFO_URL.format(orderbook_id=orderbook_id),
                params={"assetClass": "SHARES"},
                timeout=60,
                headers=official_headers(referer=NASDAQ_OFFICIAL_REFERER),
            )
            detail_response.raise_for_status()
            detail_payload = detail_response.json()
            header = detail_payload.get("data", {}).get("qdHeader", {})
            exchange_group = normalize_text(header.get("exchange"))
            segment = normalize_text(header.get("segment"))
        else:
            segment = "Main Market"
        if exchange_group not in NASDAQ_EXCHANGE_GROUPS:
            continue
        if segment and segment not in NASDAQ_MAIN_MARKET_SEGMENTS and segment != "Main Market":
            continue
        records.append(
            {
                "exchange_group": exchange_group,
                "isin": normalize_text(header.get("isin") or row.get("isin")).upper(),
                "official_symbol": normalize_text(header.get("symbol") or row.get("symbol")),
                "official_name": normalize_text(header.get("companyName") or row.get("fullName")),
                "currency": normalize_text(header.get("currency") or row.get("currency")),
                "official_segment": segment,
                "official_market_label": exchange_group,
                "official_source": "Nasdaq Nordic official main-market screener",
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise Phase1Error("Nasdaq official main-market reconciliation produced no in-scope Nordic rows.")
    return frame.drop_duplicates(subset=["exchange_group", "official_symbol", "isin"]).reset_index(drop=True)


def fetch_official_oslo_main_market(session: requests.Session) -> pd.DataFrame:
    response = session.get(
        EURONEXT_OSLO_DOWNLOAD_URL,
        timeout=60,
        headers=official_headers(referer=EURONEXT_OSLO_REFERER),
    )
    response.raise_for_status()
    raw_text = response.content.decode("utf-8-sig")
    download_frame = pd.read_csv(io.StringIO(raw_text), sep=";")
    download_frame = download_frame.loc[download_frame["Market"].eq(EURONEXT_OSLO_MAIN_MARKET_LABEL)].copy()
    download_frame["isin"] = download_frame["ISIN"].map(lambda value: normalize_text(value).upper())
    download_frame = download_frame.loc[download_frame["isin"] != ""].copy()
    if download_frame.empty:
        raise Phase1Error("Euronext official Oslo Bors download returned no Oslo Bors rows.")
    frame = pd.DataFrame(
        {
            "exchange_group": "Oslo Bors",
            "isin": download_frame["isin"],
            "official_symbol": download_frame["Symbol"].map(normalize_text),
            "official_name": download_frame["Name"].map(normalize_text),
            "currency": download_frame["Currency"].map(normalize_text),
            "official_segment": EURONEXT_OSLO_MAIN_MARKET_LABEL,
            "official_market_label": download_frame["Market"].map(normalize_text),
            "official_source": "Euronext Live Oslo equities download",
        }
    )
    return frame.drop_duplicates(subset=["exchange_group", "official_symbol", "isin"]).reset_index(drop=True)


def build_official_current_main_market_frame() -> pd.DataFrame:
    session = requests.Session()
    official = pd.concat(
        [
            fetch_official_nasdaq_main_market(session),
            fetch_official_oslo_main_market(session),
        ],
        ignore_index=True,
    )
    official["official_symbol_key"] = official["official_symbol"].map(normalize_symbol_key)
    official["official_name_key"] = official["official_name"].map(normalize_company_key)
    return official.sort_values(["exchange_group", "official_symbol"]).reset_index(drop=True)


def reconcile_official_allowlist(
    roster_frame: pd.DataFrame,
    official_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    roster = roster_frame.copy()
    roster["isin"] = roster["isin"].map(lambda value: normalize_text(value).upper())
    roster["symbol_key"] = roster["security_id"].map(security_id_stem).map(normalize_symbol_key)
    roster["company_key"] = roster["company_name"].map(normalize_company_key)

    matched_rows: list[dict[str, str]] = []
    reconciliation_rows: list[dict[str, str]] = []
    asof_date = pd.Timestamp.now(tz="Europe/Stockholm").date().isoformat()

    for official in official_frame.itertuples(index=False):
        exchange_subset = roster.loc[roster["exchange_group"] == official.exchange_group].copy()
        candidates = exchange_subset.loc[exchange_subset["isin"] == official.isin] if official.isin else exchange_subset.iloc[0:0]
        match_reason = "exchange_plus_isin"
        if len(candidates) != 1:
            candidates = exchange_subset.loc[exchange_subset["symbol_key"] == official.official_symbol_key]
            match_reason = "exchange_plus_symbol"
        if len(candidates) != 1:
            candidates = exchange_subset.loc[exchange_subset["company_key"] == official.official_name_key]
            match_reason = "exchange_plus_company"

        notes = (
            f"official_current_main_market_asof={asof_date}; "
            f"source={official.official_source}; "
            f"market={official.official_market_label}; "
            f"segment={official.official_segment}"
        )
        if len(candidates) == 1:
            matched = candidates.iloc[0]
            resolved_isin = normalize_text(matched.get("isin")).upper() or official.isin
            matched_rows.append(
                {
                    "security_id": matched["security_id"],
                    "isin": resolved_isin,
                    "company_name": matched["company_name"],
                    "exchange_group": matched["exchange_group"],
                    "notes": notes,
                    "include": "true",
                }
            )
            reconciliation_rows.append(
                {
                    "security_id": matched["security_id"],
                    "exchange_group": matched["exchange_group"],
                    "isin": resolved_isin,
                    "company_name": matched["company_name"],
                    "official_symbol": official.official_symbol,
                    "official_name": official.official_name,
                    "official_segment": official.official_segment,
                    "official_market_label": official.official_market_label,
                    "official_source": official.official_source,
                    "match_reason": match_reason,
                    "include": "true",
                    "notes": notes,
                }
            )
            continue

        reconciliation_rows.append(
            {
                "security_id": "",
                "exchange_group": official.exchange_group,
                "isin": official.isin,
                "company_name": "",
                "official_symbol": official.official_symbol,
                "official_name": official.official_name,
                "official_segment": official.official_segment,
                "official_market_label": official.official_market_label,
                "official_source": official.official_source,
                "match_reason": "unmatched",
                "include": "",
                "notes": notes,
            }
        )

    final_allowlist = (
        pd.DataFrame(matched_rows, columns=OFFICIAL_FINAL_COLUMNS)
        .drop_duplicates(subset=["security_id"])
        .sort_values(["exchange_group", "security_id", "isin"])
        .reset_index(drop=True)
    )
    reconciliation = (
        pd.DataFrame(reconciliation_rows, columns=OFFICIAL_RECONCILIATION_COLUMNS)
        .sort_values(["exchange_group", "official_symbol", "security_id"])
        .reset_index(drop=True)
    )
    if final_allowlist.empty:
        raise Phase1Error("Official current main-market reconciliation did not resolve any in-scope securities.")
    return final_allowlist, reconciliation


def broad_nordic_security_universe(*, include_delisted: bool) -> pd.DataFrame:
    api_key = load_eodhd_api_key()
    session = requests.Session()
    combined: dict[str, dict[str, Any]] = {}
    for exchange_code in active_nordic_exchanges():
        active_rows = load_exchange_roster(session, api_key, exchange_code, delisted=False)
        delisted_rows = load_exchange_roster(session, api_key, exchange_code, delisted=True) if include_delisted else []
        if not active_rows:
            raise Phase1Error(f"Exchange roster for {exchange_code} is empty.")

        for row in active_rows:
            normalized = normalize_security_row(row, exchange_code)
            if normalized is None:
                continue
            combined.setdefault(normalized["security_id"], normalized)["_active"] = True

        for row in delisted_rows:
            normalized = normalize_security_row(row, exchange_code)
            if normalized is None:
                continue
            combined.setdefault(normalized["security_id"], normalized)["_delisted"] = True

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
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise Phase1Error("No common-share Nordic securities were found in the broad EODHD rosters.")
    return frame.sort_values(["exchange_group", "ticker_local", "security_id"]).reset_index(drop=True)


def _notes_from_allowlist(allowlist: MainMarketAllowlist, matched_row_indices: set[int]) -> str:
    if not matched_row_indices or "notes" not in allowlist.entries.columns:
        return ""
    notes = []
    for value in allowlist.entries.loc[sorted(matched_row_indices), "notes"]:
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in notes:
            notes.append(text)
    return " | ".join(notes)


def recommend_review_decision(security: Any) -> tuple[str, str]:
    if security.vendor_roster_status != "active_only":
        return "exclude", "not_currently_active"

    haystack = " ".join(
        [
            str(getattr(security, "company_name", "") or ""),
            str(getattr(security, "ticker_local", "") or ""),
            str(getattr(security, "security_id", "") or ""),
        ]
    )
    for pattern, reason in LIKELY_EXCLUDE_PATTERNS:
        if pattern.search(haystack):
            return "exclude", reason
    return "candidate", "active_common_share_needs_segment_review"


def build_allowlist_review_frame(
    roster_frame: pd.DataFrame,
    *,
    existing_allowlist_path: Path | None,
) -> pd.DataFrame:
    allowlist = load_main_market_allowlist(existing_allowlist_path)
    matched_allowlist_rows: set[int] = set()
    rows: list[dict[str, str]] = []
    for security in roster_frame.itertuples(index=False):
        include = ""
        recommendation, recommendation_reason = recommend_review_decision(security)
        review_status = "review"
        seed_source = ""
        notes = ""
        if allowlist is not None:
            matched_rows = allowlist.match_row_indices(
                security_id=security.security_id,
                isin=security.isin,
                exchange_group=security.exchange_group,
            )
            if matched_rows:
                matched_allowlist_rows.update(matched_rows)
                include = "true"
                review_status = "seeded_include"
                seed_source = "existing_allowlist"
                notes = _notes_from_allowlist(allowlist, matched_rows)
        rows.append(
            {
                "include": include,
                "recommendation": recommendation,
                "recommendation_reason": recommendation_reason,
                "review_status": review_status,
                "seed_source": seed_source,
                "security_id": security.security_id,
                "isin": security.isin or "",
                "company_name": security.company_name,
                "exchange_group": security.exchange_group,
                "country_code": security.country_code,
                "currency": security.currency,
                "vendor_roster_status": security.vendor_roster_status,
                "ticker_local": security.ticker_local,
                "notes": notes,
            }
        )
    review = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    if allowlist is not None:
        unmatched = allowlist.unmatched_entries(matched_allowlist_rows)
        if not unmatched.empty:
            preview = []
            for row in unmatched.head(10).itertuples(index=False):
                preview.append(getattr(row, "security_id", "") or getattr(row, "isin", ""))
            suffix = " ..." if len(unmatched) > 10 else ""
            raise Phase1Error(
                "Existing allowlist contains rows that were not found in the Nordic EODHD rosters: "
                + ", ".join(preview)
                + suffix
            )
    return review


def shortlist_candidate_frame(review_frame: pd.DataFrame) -> pd.DataFrame:
    frame = review_frame.copy()
    if "recommendation" not in frame.columns:
        raise Phase1Error("Review frame is missing recommendation column.")
    candidate_mask = frame["recommendation"].map(lambda value: str(value).strip().lower()) == "candidate"
    candidates = frame.loc[candidate_mask].copy()
    if candidates.empty:
        raise Phase1Error("No candidate rows were identified in the review frame.")
    return candidates.reset_index(drop=True)


def finalize_allowlist_frame(review_frame: pd.DataFrame) -> pd.DataFrame:
    frame = review_frame.copy()
    renamed = {column: str(column).strip().lower() for column in frame.columns}
    frame = frame.rename(columns=renamed)
    for required in ("include", "security_id", "isin", "company_name", "exchange_group"):
        if required not in frame.columns:
            raise Phase1Error(f"Review file is missing required column '{required}'.")
    if "notes" not in frame.columns:
        frame["notes"] = ""

    include_mask = []
    for value in frame["include"]:
        normalized = str(value).strip()
        include_mask.append(parse_bool(normalized) if normalized else False)
    selected = frame.loc[include_mask, ["security_id", "isin", "company_name", "exchange_group", "notes"]].copy()
    if selected.empty:
        raise Phase1Error("Review file does not mark any rows with include=true.")

    for column in ("security_id", "isin", "company_name", "exchange_group", "notes"):
        selected[column] = selected[column].map(lambda value: str(value).strip())
    selected = selected.loc[(selected["security_id"] != "") | (selected["isin"] != "")]
    selected["include"] = "true"
    selected = (
        selected.drop_duplicates(subset=["security_id", "isin"])
        .sort_values(["exchange_group", "security_id", "isin"])
        .reset_index(drop=True)
    )
    return selected[FINAL_COLUMNS]


def review_output_path(args: argparse.Namespace) -> Path:
    return Path(args.out) if args.out else Path(args.review_input)


def finalize_output_path(args: argparse.Namespace) -> Path:
    return Path(args.out) if args.out else Path(config.MAIN_MARKET_ALLOWLIST_PATH)


def main() -> int:
    args = parse_args()
    if args.mode == "official":
        roster = broad_nordic_security_universe(include_delisted=False)
        official = build_official_current_main_market_frame()
        final_allowlist, reconciliation = reconcile_official_allowlist(roster, official)
        output_path = finalize_output_path(args)
        atomic_write_csv(final_allowlist, output_path)
        output_excel_path = excel_friendly_path(output_path)
        atomic_write_excel_friendly_csv(final_allowlist, output_excel_path)
        reconciliation_path = Path(args.reconciliation_out)
        atomic_write_csv(reconciliation, reconciliation_path)
        reconciliation_excel_path = excel_friendly_path(reconciliation_path)
        atomic_write_excel_friendly_csv(reconciliation, reconciliation_excel_path)
        matched_count = int(reconciliation["security_id"].ne("").sum())
        unmatched_count = int(reconciliation["security_id"].eq("").sum())
        print(
            f"Wrote {output_path} with {len(final_allowlist)} officially matched securities "
            f"({matched_count} matched, {unmatched_count} unmatched official rows)."
        )
        print(
            "Reconciliation reports saved to "
            f"{reconciliation_path} and {reconciliation_excel_path}."
        )
        return 0

    if args.mode == "review":
        roster = broad_nordic_security_universe(include_delisted=parse_bool(args.include_delisted))
        review = build_allowlist_review_frame(
            roster,
            existing_allowlist_path=Path(args.existing_allowlist) if args.existing_allowlist else None,
        )
        output_path = review_output_path(args)
        atomic_write_csv(review, output_path)
        review_excel_path = excel_friendly_path(output_path)
        atomic_write_excel_friendly_csv(review, review_excel_path)
        candidate_path = Path(args.candidate_out)
        candidates = shortlist_candidate_frame(review)
        atomic_write_csv(candidates, candidate_path)
        candidate_excel_path = excel_friendly_path(candidate_path)
        atomic_write_excel_friendly_csv(candidates, candidate_excel_path)
        include_count = int(review["include"].eq("true").sum())
        candidate_count = len(candidates)
        print(
            f"Wrote {output_path} with {len(review)} review rows, "
            f"{candidate_count} candidate rows, and {include_count} seeded includes."
        )
        print(f"Candidate shortlist saved to {candidate_path}.")
        print(f"Excel-friendly review files saved to {review_excel_path} and {candidate_excel_path}.")
        return 0

    review_path = Path(args.review_input)
    if not review_path.exists():
        raise Phase1Error(f"Review file does not exist: {review_path}")
    review_frame = read_delimited_frame(review_path)
    final_allowlist = finalize_allowlist_frame(review_frame)
    output_path = finalize_output_path(args)
    atomic_write_csv(final_allowlist, output_path)
    print(f"Wrote {output_path} with {len(final_allowlist)} included securities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
