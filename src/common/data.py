"""Data loading and dense A-share market matrices.

The source files contain all 481 stocks that appear in the sampled historical
CSI 300 membership snapshots. Membership is still sampled on the five-day
research grid, not reconstructed on every calendar day. Results should be
described as a point-in-time membership-grid study rather than a daily
constituent or fully executable trading simulation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import project_path


REQUIRED_PRICE_COLUMNS = {
    "date",
    "code",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "preclose",
    "tradestatus",
    "isST",
}


@dataclass(frozen=True)
class PriceMatrices:
    calendar: pd.DatetimeIndex
    codes: pd.Index
    open: pd.DataFrame
    close: pd.DataFrame
    close_marked: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    preclose: pd.DataFrame
    volume: pd.DataFrame
    turnover: pd.DataFrame
    amount: pd.DataFrame
    trade_status: pd.DataFrame
    is_st: pd.DataFrame
    traded: pd.DataFrame
    valid_ohlc: pd.DataFrame


@dataclass(frozen=True)
class DataBundle:
    prices: pd.DataFrame
    membership: pd.DataFrame
    market: PriceMatrices
    membership_mask: pd.DataFrame
    price_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_price_panel(
    path: str | Path,
    expected_sha256: str | None = None,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, str]:
    price_path = project_path(path)
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")
    actual_hash = sha256_file(price_path)
    if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
        raise ValueError(
            "Price file hash mismatch: "
            f"expected {expected_sha256}, got {actual_hash}"
        )

    prices = pd.read_parquet(price_path)
    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise ValueError(f"Price file is missing columns: {sorted(missing)}")
    prices = prices.copy()
    if "turnover" not in prices.columns:
        if "turn" not in prices.columns:
            raise ValueError("Price file needs either 'turnover' or 'turn'")
        prices["turnover"] = prices["turn"]
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    prices = prices.sort_values(["date", "code"]).reset_index(drop=True)

    duplicates = prices.duplicated(["date", "code"])
    if duplicates.any():
        raise ValueError(f"Duplicate date-code rows: {int(duplicates.sum())}")
    if not prices["code"].str.fullmatch(r"\d{6}").all():
        raise ValueError("Stock codes must contain exactly six digits")

    finite_prices = prices[["open", "high", "low", "close"]].apply(
        np.isfinite
    ).all(axis=1)
    positive_prices = prices[["open", "high", "low", "close"]].gt(0).all(axis=1)
    legal_ranges = (
        prices["high"].ge(prices[["open", "close", "low"]].max(axis=1))
        & prices["low"].le(prices[["open", "close", "high"]].min(axis=1))
    )
    invalid = ~(finite_prices & positive_prices & legal_ranges)
    if invalid.any():
        message = f"Invalid OHLC rows: {int(invalid.sum())}"
        if strict:
            raise ValueError(message)
        prices = prices.loc[~invalid].copy()
    return prices, actual_hash


def load_membership(path: str | Path) -> pd.DataFrame:
    membership_path = project_path(path)
    if not membership_path.exists():
        raise FileNotFoundError(f"Membership file not found: {membership_path}")
    membership = pd.read_csv(
        membership_path,
        dtype={"code": str},
        parse_dates=["query_date", "index_effective_date"],
    )
    required = {"query_date", "index_effective_date", "code"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(
            f"Membership file is missing columns: {sorted(missing)}"
        )
    membership = membership.copy()
    membership["code"] = membership["code"].astype(str).str.zfill(6)
    if membership.duplicated(["query_date", "code"]).any():
        raise ValueError("Duplicate query_date-code membership rows")
    return membership.sort_values(["query_date", "code"]).reset_index(drop=True)


def _validate_matrices(field: pd.DataFrame, label: str) -> pd.DataFrame:
    if field.index.has_duplicates or not field.index.is_monotonic_increasing:
        raise ValueError(f"{label} matrix has an invalid date index")
    if field.columns.has_duplicates:
        raise ValueError(f"{label} matrix has duplicate stock columns")
    return field


def build_price_matrices(prices: pd.DataFrame) -> PriceMatrices:
    calendar = pd.DatetimeIndex(sorted(pd.unique(prices["date"])))
    codes = pd.Index(sorted(pd.unique(prices["code"])), name="code")
    fields: dict[str, pd.DataFrame] = {}
    for name in (
        "open",
        "close",
        "high",
        "low",
        "preclose",
        "volume",
        "turnover",
        "amount",
        "tradestatus",
        "isST",
    ):
        matrix = (
            prices.pivot(index="date", columns="code", values=name)
            .reindex(index=calendar, columns=codes)
            .astype(float)
        )
        fields[name] = _validate_matrices(matrix, name)

    valid_ohlc = pd.DataFrame(True, index=calendar, columns=codes)
    for name in ("open", "close", "high", "low", "preclose"):
        frame = fields[name]
        valid_ohlc &= frame.notna() & np.isfinite(frame) & frame.gt(0)
    high_legal = (
        fields["high"].ge(fields["open"])
        & fields["high"].ge(fields["close"])
        & fields["high"].ge(fields["low"])
    )
    low_legal = (
        fields["low"].le(fields["open"])
        & fields["low"].le(fields["close"])
        & fields["low"].le(fields["high"])
    )
    valid_ohlc &= high_legal & low_legal
    trade_status = fields.pop("tradestatus")
    is_st = fields.pop("isST")
    traded = (
        valid_ohlc
        & trade_status.eq(1)
        & fields["volume"].fillna(0).gt(0)
    )
    return PriceMatrices(
        calendar=calendar,
        codes=codes,
        open=fields["open"],
        close=fields["close"],
        close_marked=fields["close"].ffill(),
        high=fields["high"],
        low=fields["low"],
        preclose=fields["preclose"],
        volume=fields["volume"],
        turnover=fields["turnover"],
        amount=fields["amount"],
        trade_status=trade_status,
        is_st=is_st.eq(1),
        traded=traded,
        valid_ohlc=valid_ohlc,
    )


def build_membership_mask(
    market: PriceMatrices, membership: pd.DataFrame
) -> pd.DataFrame:
    snapshot = membership.assign(included=True).pivot_table(
        index="query_date",
        columns="code",
        values="included",
        aggfunc="any",
        fill_value=False,
    ).astype(bool)
    dense_dates = snapshot.index.union(market.calendar).sort_values()
    dense = (
        snapshot.reindex(index=dense_dates, columns=market.codes)
        .astype("boolean")
        .ffill()
        .reindex(index=market.calendar, columns=market.codes)
        .fillna(False)
        .astype(bool)
    )
    dense.index = market.calendar
    return dense


def load_bundle(config: dict[str, Any]) -> DataBundle:
    prices, price_hash = load_price_panel(
        config["paths"]["prices"],
        expected_sha256=config["data"].get("expected_price_sha256"),
    )
    membership = load_membership(config["paths"]["membership"])
    market = build_price_matrices(prices)
    mask = build_membership_mask(market, membership)
    overlap = len(set(membership["code"]).intersection(set(market.codes)))
    if overlap == 0:
        raise ValueError("Membership and price files have no stock-code overlap")
    return DataBundle(
        prices=prices,
        membership=membership,
        market=market,
        membership_mask=mask,
        price_sha256=price_hash,
    )
