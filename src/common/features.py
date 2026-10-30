"""Point-in-time feature construction for weekly A-share cross-sections."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataBundle, PriceMatrices


PRICE_FEATURES = (
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ret_120",
    "vol_5",
    "vol_20",
    "vol_60",
    "ma_dev_5",
    "ma_dev_20",
    "ma_dev_60",
    "range_pos_20",
    "range_pos_60",
    "amplitude",
)

LIQUIDITY_FEATURES = (
    "turnover_5",
    "turnover_20",
    "amount_ratio_5_60",
    "return_volume_corr_20",
)


@dataclass(frozen=True)
class ResearchPanel:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    rank_feature_columns: tuple[str, ...]
    signal_dates: pd.DatetimeIndex
    diagnostics: dict[str, float | int | str]


def _consecutive_valid(mask: pd.DataFrame, window: int) -> pd.DataFrame:
    return (
        mask.astype(float)
        .rolling(window, min_periods=window)
        .sum()
        .eq(window)
    )


def compute_price_features(
    market: PriceMatrices,
    *,
    include_liquidity: bool,
    min_history_days: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    close = market.close_marked
    high = market.high.ffill()
    low = market.low.ffill()
    ret_1 = close.pct_change(fill_method=None)
    features: dict[str, pd.DataFrame] = {}

    for window in (1, 5, 10, 20, 60, 120):
        features[f"ret_{window}"] = close.div(close.shift(window)).sub(1)
    for window in (5, 20, 60):
        features[f"vol_{window}"] = ret_1.rolling(
            window, min_periods=window
        ).std(ddof=1)
        rolling_mean = close.rolling(window, min_periods=window).mean()
        features[f"ma_dev_{window}"] = close.div(rolling_mean).sub(1)
    for window in (20, 60):
        rolling_low = low.rolling(window, min_periods=window).min()
        rolling_high = high.rolling(window, min_periods=window).max()
        width = rolling_high.sub(rolling_low)
        features[f"range_pos_{window}"] = close.sub(rolling_low).div(
            width.where(width > 0)
        )
    features["amplitude"] = market.high.sub(market.low).div(
        market.close.where(market.close > 0)
    )

    if include_liquidity:
        features["turnover_5"] = market.turnover.rolling(
            5, min_periods=5
        ).mean()
        features["turnover_20"] = market.turnover.rolling(
            20, min_periods=20
        ).mean()
        amount_5 = market.amount.rolling(5, min_periods=5).mean()
        amount_60 = market.amount.rolling(60, min_periods=60).mean()
        features["amount_ratio_5_60"] = amount_5.div(
            amount_60.where(amount_60 > 0)
        )
        volume_change = market.volume.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        features["return_volume_corr_20"] = ret_1.rolling(
            20, min_periods=20
        ).corr(volume_change)

    for name, frame in features.items():
        features[name] = frame.replace([np.inf, -np.inf], np.nan)

    active_history = (
        market.traded.astype(float)
        .rolling(min_history_days, min_periods=min_history_days)
        .sum()
        .shift(1)
    )
    history_valid = active_history.ge(min_history_days)
    adv20 = (
        market.amount.where(market.traded)
        .rolling(20, min_periods=15)
        .mean()
    )
    return features, history_valid, adv20


def compute_forward_labels(
    market: PriceMatrices,
    *,
    entry_delay: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entry_price = market.open.shift(-entry_delay)
    exit_price = market.open.shift(-(entry_delay + horizon))
    future_return = exit_price.div(entry_price.where(entry_price > 0)).sub(1)
    positions = np.arange(len(market.calendar))

    def shifted_dates(offset: int) -> pd.DataFrame:
        end = positions + offset
        values = np.full(
            len(positions),
            np.datetime64("NaT", "ns"),
            dtype="datetime64[ns]",
        )
        valid = end < len(market.calendar)
        values[valid] = market.calendar.to_numpy()[end[valid]]
        matrix = np.broadcast_to(values[:, None], future_return.shape).copy()
        return pd.DataFrame(
            matrix, index=market.calendar, columns=market.codes
        )

    entry_dates = shifted_dates(entry_delay)
    label_end_dates = shifted_dates(entry_delay + horizon)
    return future_return, entry_dates, label_end_dates


def add_cross_sectional_ranks(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    result = frame[
        [
            "signal_date",
            "code",
            "eligible",
            "future_return",
            "entry_date",
            "label_end_date",
        ]
    ].copy()
    eligible = result["eligible"].fillna(False).astype(bool)
    for column in feature_columns:
        ranked = (
            frame.loc[eligible]
            .groupby("signal_date", sort=False)[column]
            .rank(method="average", pct=True)
            .sub(0.5)
            .reindex(frame.index)
        )
        result[f"cs_{column}"] = ranked.where(eligible, np.nan)
    result["rank_label"] = (
        result.loc[eligible]
        .groupby("signal_date", sort=False)["future_return"]
        .rank(method="average", pct=True)
        .sub(0.5)
        .reindex(result.index)
    )
    for column in feature_columns:
        result[f"cs_{column}"] = result[f"cs_{column}"].fillna(0.0)
    return result


def build_research_panel(
    bundle: DataBundle,
    config: dict,
) -> ResearchPanel:
    data_config = config["data"]
    min_history = int(data_config["min_history_days"])
    include_liquidity = bool(data_config.get("include_liquidity_features", False))

    feature_matrices, history_valid, adv20 = compute_price_features(
        bundle.market,
        include_liquidity=include_liquidity,
        min_history_days=min_history,
    )
    future_return, entry_dates, label_end_dates = compute_forward_labels(
        bundle.market,
        entry_delay=int(data_config["entry_delay"]),
        horizon=int(data_config["label_horizon"]),
    )
    feature_columns = list(feature_matrices)
    research_start = pd.Timestamp(
        data_config.get("research_start", "2020-01-02")
    )
    query_dates = pd.DatetimeIndex(
        sorted(pd.unique(bundle.membership["query_date"]))
    )
    signal_dates = bundle.market.calendar[
        bundle.market.calendar.isin(query_dates)
        & (bundle.market.calendar >= research_start)
    ]
    if signal_dates.empty:
        raise ValueError("No signal dates remain after the warm-up window")

    base = pd.MultiIndex.from_product(
        [signal_dates, bundle.market.codes], names=["signal_date", "code"]
    ).to_frame(index=False)
    for column, matrix in feature_matrices.items():
        base[column] = matrix.loc[signal_dates].to_numpy().reshape(-1)
    base["future_return"] = (
        future_return.loc[signal_dates].to_numpy().reshape(-1)
    )
    base["entry_date"] = (
        entry_dates.loc[signal_dates].to_numpy().reshape(-1)
    )
    base["label_end_date"] = (
        label_end_dates.loc[signal_dates].to_numpy().reshape(-1)
    )
    base["member"] = (
        bundle.membership_mask.loc[signal_dates].to_numpy().reshape(-1)
    )
    base["history_valid"] = (
        history_valid.loc[signal_dates].fillna(False).to_numpy().reshape(-1)
    )
    base["adv20"] = adv20.loc[signal_dates].to_numpy().reshape(-1)
    signal_traded = bundle.market.traded.loc[signal_dates].to_numpy().reshape(-1)
    signal_not_st = (
        ~bundle.market.is_st.loc[signal_dates].to_numpy().reshape(-1)
    )
    base["eligible"] = (
        base["member"].astype(bool)
        & base["history_valid"].astype(bool)
        & signal_traded.astype(bool)
        & signal_not_st.astype(bool)
        & base["adv20"].ge(float(data_config["min_adv20"])).fillna(False)
    )
    base = base.loc[base["eligible"]].copy()
    if base.empty:
        raise ValueError("No eligible signal rows were constructed")

    base = add_cross_sectional_ranks(base, feature_columns)
    base["signal_year"] = base["signal_date"].dt.year
    base = base.sort_values(["signal_date", "code"]).reset_index(drop=True)
    rank_columns = tuple(f"cs_{column}" for column in feature_columns)
    diagnostics = {
        "price_rows": int(len(bundle.prices)),
        "stocks_in_prices": int(len(bundle.market.codes)),
        "stocks_in_membership": int(bundle.membership["code"].nunique()),
        "stocks_membership_missing_prices": int(
            len(set(bundle.membership["code"]) - set(bundle.market.codes))
        ),
        "calendar_days": int(len(bundle.market.calendar)),
        "signal_dates": int(base["signal_date"].nunique()),
        "eligible_rows": int(len(base)),
        "mean_stocks_per_signal": float(
            base.groupby("signal_date").size().mean()
        ),
        "label_coverage": float(base["future_return"].notna().mean()),
        "first_signal_date": str(base["signal_date"].min().date()),
        "last_signal_date": str(base["signal_date"].max().date()),
        "include_liquidity_features": include_liquidity,
        "universe_label": (
            "point_in_time_membership_grid_2019_2025_price_panel"
        ),
        "price_sha256": bundle.price_sha256,
    }
    return ResearchPanel(
        frame=base,
        feature_columns=tuple(feature_columns),
        rank_feature_columns=rank_columns,
        signal_dates=pd.DatetimeIndex(sorted(base["signal_date"].unique())),
        diagnostics=diagnostics,
    )


def split_masks(frame: pd.DataFrame, periods: dict) -> dict[str, pd.Series]:
    signal_date = pd.to_datetime(frame["signal_date"])
    label_end = pd.to_datetime(frame["label_end_date"])
    development_end = pd.Timestamp(periods["development_end"])
    validation_start = pd.Timestamp(periods["validation_start"])
    validation_end = pd.Timestamp(periods["validation_end"])
    test_start = pd.Timestamp(periods["test_start"])
    test_end = pd.Timestamp(periods["test_end"])
    observed = frame["future_return"].notna() & frame["rank_label"].notna()

    train = (
        signal_date.le(development_end)
        & label_end.lt(validation_start)
        & observed
    )
    validation = (
        signal_date.ge(validation_start)
        & signal_date.le(validation_end)
        & label_end.lt(test_start)
        & observed
    )
    test = (
        signal_date.ge(test_start)
        & signal_date.le(test_end)
    )
    return {
        "development": train,
        "validation": validation,
        "test": test,
        "train_plus_validation": train | validation,
    }
