"""Build a point-in-time CSI 300 universe and matching daily price panel.

The membership snapshots are taken on every fifth exchange trading day, which
is exactly the research rebalance grid.  A snapshot is never backfilled from
the latest constituent list.  Downloads are cached by date/code so an
interrupted Baostock request can be resumed without discarding completed data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_DIR = ROOT / "data" / "reference"
CACHE_DIR = ROOT / "data" / "cache" / "historical"
MEMBERSHIP_CACHE = CACHE_DIR / "membership"
PRICE_CACHE = CACHE_DIR / "prices"

PRICE_START = "2019-07-01"
DATA_START = "2020-01-02"
DATA_END = "2025-12-31"
REBALANCE_STEP = 5
PRICE_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
    "turn,tradestatus,pctChg,isST"
)
PRICE_COLUMNS = set(PRICE_FIELDS.split(","))


def _baostock():
    import baostock as bs

    return bs


def _login_checked():
    bs = _baostock()
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {result.error_msg}")
    return bs


def _reconnect(bs):
    try:
        bs.logout()
    except Exception:
        pass
    time.sleep(1)
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"Baostock reconnect failed: {result.error_msg}")
    return bs


def normalize_code(code: str) -> str:
    """Convert sh.600000 / sz.000001 to the six-digit project code."""
    return str(code).split(".")[-1].zfill(6)


def baostock_code(code: str) -> str:
    code = normalize_code(code)
    return ("sh." if code.startswith(("5", "6", "9")) else "sz.") + code


def fetch_trading_dates(start: str, end: str) -> list[str]:
    bs = _login_checked()
    try:
        result = bs.query_trade_dates(start_date=start, end_date=end)
        if result.error_code != "0":
            raise RuntimeError(result.error_msg)
        rows = []
        while result.next():
            row = dict(zip(result.fields, result.get_row_data()))
            if row["is_trading_day"] == "1":
                rows.append(row["calendar_date"])
        return rows
    finally:
        bs.logout()


def research_grid(trading_dates: list[str]) -> list[str]:
    """Use a single phase for every lookback and the time holdout."""
    eligible = [date for date in trading_dates if DATA_START <= date <= DATA_END]
    return eligible[::REBALANCE_STEP]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")
    os.replace(temp, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _membership_cache_valid(path: Path, date: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, dtype={"code": str})
        return (
            {"query_date", "index_effective_date", "code", "name"}.issubset(
                frame.columns
            )
            and len(frame) == 300
            and frame["code"].nunique() == 300
            and frame["query_date"].astype(str).eq(date).all()
        )
    except Exception:
        return False


def _price_cache_valid(path: Path, code: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
        required = PRICE_COLUMNS | {"date", "code"}
        if not required.issubset(frame.columns) or frame.empty:
            return False
        dates = pd.to_datetime(frame["date"], errors="coerce")
        return (
            frame["code"].astype(str).str.zfill(6).eq(code).all()
            and dates.notna().all()
            and dates.between(PRICE_START, DATA_END).all()
            and not frame.duplicated(["date", "code"]).any()
        )
    except Exception:
        return False


def _fetch_membership_chunk(dates: list[str]) -> dict:
    bs = _login_checked()
    completed = 0
    try:
        for date in dates:
            destination = MEMBERSHIP_CACHE / f"{date}.csv"
            if _membership_cache_valid(destination, date):
                completed += 1
                continue
            last_error = None
            for attempt in range(3):
                result = bs.query_hs300_stocks(date=date)
                if result.error_code == "0":
                    rows = []
                    while result.next():
                        row = dict(zip(result.fields, result.get_row_data()))
                        rows.append(
                            {
                                "query_date": date,
                                "index_effective_date": row["updateDate"],
                                "code": normalize_code(row["code"]),
                                "name": row["code_name"],
                            }
                        )
                    frame = pd.DataFrame(rows)
                    if len(frame) != 300 or frame["code"].nunique() != 300:
                        last_error = (
                            f"{date}: expected 300 unique constituents, got "
                            f"{len(frame)} rows / {frame['code'].nunique()} codes"
                        )
                    else:
                        _atomic_csv(frame, destination)
                        completed += 1
                        break
                else:
                    last_error = f"{date}: {result.error_msg}"
                    if "登录" in result.error_msg or result.error_code != "0":
                        bs = _reconnect(bs)
                time.sleep(1 + attempt)
            else:
                raise RuntimeError(f"Membership download failed: {last_error}")
    finally:
        bs.logout()
    return {"dates": len(dates), "completed": completed}


def download_memberships(dates: list[str], workers: int) -> pd.DataFrame:
    MEMBERSHIP_CACHE.mkdir(parents=True, exist_ok=True)
    missing = [
        date
        for date in dates
        if not _membership_cache_valid(
            MEMBERSHIP_CACHE / f"{date}.csv", date
        )
    ]
    if missing:
        chunks = [missing[index::workers] for index in range(workers)]
        chunks = [chunk for chunk in chunks if chunk]
        with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            futures = [pool.submit(_fetch_membership_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                print(
                    f"membership worker: {result['completed']}/"
                    f"{result['dates']} snapshots",
                    flush=True,
                )

    frames = []
    for date in dates:
        path = MEMBERSHIP_CACHE / f"{date}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing membership snapshot: {date}")
        frame = pd.read_csv(path, dtype={"code": str})
        frames.append(frame)
    membership = pd.concat(frames, ignore_index=True)
    membership["query_date"] = pd.to_datetime(membership["query_date"])
    membership["index_effective_date"] = pd.to_datetime(
        membership["index_effective_date"]
    )
    membership["code"] = membership["code"].str.zfill(6)
    duplicates = membership.duplicated(["query_date", "code"]).sum()
    if duplicates:
        raise ValueError(f"Duplicate membership rows: {duplicates}")
    counts = membership.groupby("query_date")["code"].nunique()
    if not (counts == 300).all():
        raise ValueError(
            f"Membership snapshots are not all 300 rows: {counts[counts != 300]}"
        )
    return membership.sort_values(["query_date", "code"]).reset_index(drop=True)


def _fetch_price_chunk(codes: list[str]) -> dict:
    bs = _login_checked()
    completed = 0
    rows = 0
    failures = []
    try:
        for code in codes:
            destination = PRICE_CACHE / f"{code}.parquet"
            if _price_cache_valid(destination, code):
                completed += 1
                continue
            bs_code = baostock_code(code)
            last_error = None
            succeeded = False
            for attempt in range(3):
                result = bs.query_history_k_data_plus(
                    bs_code,
                    PRICE_FIELDS,
                    start_date=PRICE_START,
                    end_date=DATA_END,
                    frequency="d",
                    adjustflag="1",
                )
                if result.error_code != "0":
                    last_error = result.error_msg
                    bs = _reconnect(bs)
                    time.sleep(1 + attempt)
                    continue
                records = []
                while result.next():
                    records.append(dict(zip(result.fields, result.get_row_data())))
                frame = pd.DataFrame(records)
                if frame.empty:
                    last_error = "empty price response"
                    time.sleep(1 + attempt)
                    continue
                numeric = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "preclose",
                    "volume",
                    "amount",
                    "turn",
                    "pctChg",
                    "tradestatus",
                    "isST",
                ]
                frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
                frame["date"] = pd.to_datetime(frame["date"])
                frame["code"] = normalize_code(code)
                _atomic_parquet(frame, destination)
                completed += 1
                rows += len(frame)
                succeeded = True
                break
            if not succeeded:
                failures.append(f"{code}: {last_error}")
    finally:
        bs.logout()
    return {
        "codes": len(codes),
        "completed": completed,
        "rows": rows,
        "failures": failures,
    }


def download_prices(codes: list[str], workers: int) -> pd.DataFrame:
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    missing = [
        code
        for code in codes
        if not _price_cache_valid(PRICE_CACHE / f"{code}.parquet", code)
    ]
    if missing:
        chunks = [missing[index::workers] for index in range(workers)]
        chunks = [chunk for chunk in chunks if chunk]
        with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            futures = {pool.submit(_fetch_price_chunk, chunk) for chunk in chunks}
            for future in as_completed(futures):
                result = future.result()
                print(
                    f"price worker: {result['completed']}/{result['codes']} "
                    f"new codes, {result['rows']:,} rows",
                    flush=True,
                )
                if result["failures"]:
                    raise RuntimeError("; ".join(result["failures"]))

    frames = []
    for code in codes:
        path = PRICE_CACHE / f"{code}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing price cache: {code}")
        frames.append(pd.read_parquet(path))
    panel = pd.concat(frames, ignore_index=True)
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.drop_duplicates(["date", "code"]).sort_values(
        ["date", "code"]
    ).reset_index(drop=True)
    return panel


def write_manifest(
    membership: pd.DataFrame,
    panel: pd.DataFrame,
    trading_dates: list[str],
    grid: list[str],
    workers: int,
) -> None:
    latest_effective = (
        membership.groupby("query_date")["index_effective_date"]
        .max()
        .reset_index()
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Baostock",
        "membership_endpoint": "query_hs300_stocks(date=signal_date)",
        "price_endpoint": (
            "query_history_k_data_plus(frequency='d', adjustflag='1')"
        ),
        "adjustment": "后复权",
        "requested_data_period": [PRICE_START, DATA_END],
        "research_start": DATA_START,
        "research_end": DATA_END,
        "rebalance_step_trading_days": REBALANCE_STEP,
        "workers": workers,
        "trading_dates": len(trading_dates),
        "membership_snapshots": int(membership["query_date"].nunique()),
        "membership_rows": int(len(membership)),
        "membership_union_codes": int(membership["code"].nunique()),
        "price_codes": int(panel["code"].nunique()),
        "price_rows": int(len(panel)),
        "price_start": str(panel["date"].min().date()),
        "price_end": str(panel["date"].max().date()),
        "latest_effective_date_by_snapshot": {
            str(row.query_date.date()): str(row.index_effective_date.date())
            for row in latest_effective.itertuples(index=False)
        },
        "research_grid": [grid[0], grid[-1], len(grid)],
        "sha256": {
            "membership_file": hashlib.sha256(
                (
                    REFERENCE_DIR / "csi300_historical_membership.csv"
                ).read_bytes()
            ).hexdigest(),
            "price_file": hashlib.sha256(
                (
                    RAW_DIR / "csi300_prices_historical.parquet"
                ).read_bytes()
            ).hexdigest(),
        },
    }
    destination = REFERENCE_DIR / "historical_dataset_manifest.json"
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent Baostock sessions; keep at 4 or below.",
    )
    args = parser.parse_args()
    workers = max(1, min(args.workers, 8))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching exchange calendar...", flush=True)
    trading_dates = fetch_trading_dates(DATA_START, DATA_END)
    grid = research_grid(trading_dates)
    print(
        f"Downloading {len(grid)} point-in-time constituent snapshots...",
        flush=True,
    )
    membership = download_memberships(grid, workers=workers)
    _atomic_csv(
        membership,
        REFERENCE_DIR / "csi300_historical_membership.csv",
    )

    codes = sorted(membership["code"].unique())
    print(f"Downloading prices for {len(codes)} historical constituents...", flush=True)
    panel = download_prices(codes, workers=workers)
    _atomic_parquet(panel, RAW_DIR / "csi300_prices_historical.parquet")
    write_manifest(membership, panel, trading_dates, grid, workers)

    print(
        f"PASS: {len(membership):,} membership rows, "
        f"{len(codes)} unique codes, {len(panel):,} price rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
