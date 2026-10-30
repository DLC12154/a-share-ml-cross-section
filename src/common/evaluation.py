"""Cross-sectional signal evaluation and analytical long-only portfolios."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rank_ic_series(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    return_column: str = "future_return",
) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for signal_date, group in predictions.groupby("signal_date", sort=True):
        clean = group[[prediction_column, return_column]].dropna()
        if len(clean) < 3:
            continue
        if clean[prediction_column].nunique() < 2:
            continue
        if clean[return_column].nunique() < 2:
            continue
        values[pd.Timestamp(signal_date)] = float(
            clean[prediction_column].corr(
                clean[return_column], method="spearman"
            )
        )
    return pd.Series(values, dtype=float, name="rank_ic").sort_index()


def signal_summary(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
) -> dict[str, float | int]:
    ic = rank_ic_series(
        predictions, prediction_column=prediction_column
    ).dropna()
    standard_deviation = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    mean = float(ic.mean()) if not ic.empty else np.nan
    return {
        "ic_observations": int(len(ic)),
        "mean_rank_ic": mean,
        "rank_ic_std": standard_deviation,
        "icir_unannualized": (
            mean / standard_deviation
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else np.nan
        ),
        "rank_ic_positive_ratio": (
            float(ic.gt(0).mean()) if not ic.empty else np.nan
        ),
    }


def _target_weights(
    group: pd.DataFrame,
    top_fraction: float,
    *,
    benchmark: bool = False,
) -> pd.Series:
    eligible = group.loc[group["eligible"].fillna(False)].copy()
    if eligible.empty:
        return pd.Series(dtype=float, name="weight")
    if benchmark:
        selected = eligible
    else:
        n_select = max(1, math.ceil(len(eligible) * top_fraction))
        selected = eligible.sort_values(
            ["prediction", "code"], ascending=[False, True]
        ).head(n_select)
    count = len(selected)
    return pd.Series(1.0 / count, index=selected["code"], name="weight")


def _simulate_weight_series(
    predictions: pd.DataFrame,
    *,
    top_fraction: float,
    cost_bps: float,
    benchmark: bool,
) -> pd.DataFrame:
    previous_weights = pd.Series(dtype=float)
    previous_returns = pd.Series(dtype=float)
    rows: list[dict] = []
    for signal_date, group in predictions.groupby("signal_date", sort=True):
        target = _target_weights(
            group, top_fraction, benchmark=benchmark
        )
        if target.empty:
            continue
        returns = (
            group.drop_duplicates("code")
            .set_index("code")["future_return"]
            .reindex(target.index)
        )
        observed = returns.notna()
        gross_return = float(
            target.mul(returns.fillna(0.0)).sum()
        )
        if previous_weights.empty:
            pretrade_weights = previous_weights
        else:
            value_multipliers = 1.0 + previous_returns.reindex(
                previous_weights.index, fill_value=0.0
            )
            previous_values = previous_weights.mul(value_multipliers)
            total_value = float(previous_values.sum())
            pretrade_weights = (
                previous_values.div(total_value)
                if total_value > 0
                else previous_weights
            )
        all_codes = target.index.union(pretrade_weights.index)
        delta = target.reindex(all_codes, fill_value=0.0).sub(
            pretrade_weights.reindex(all_codes, fill_value=0.0)
        )
        traded_fraction = float(delta.abs().sum())
        one_way_turnover = 0.5 * traded_fraction
        cost = traded_fraction * float(cost_bps) / 10_000.0
        rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "gross_return": gross_return,
                "transaction_cost": cost,
                "net_return": gross_return - cost,
                "one_way_turnover": one_way_turnover,
                "traded_fraction": traded_fraction,
                "holdings": int(len(target)),
                "return_coverage": float(target[observed].sum()),
            }
        )
        previous_weights = target
        previous_returns = returns.fillna(0.0)
    if not rows:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "gross_return",
                "transaction_cost",
                "net_return",
                "one_way_turnover",
                "traded_fraction",
                "holdings",
                "return_coverage",
            ]
        )
    return pd.DataFrame(rows).set_index("signal_date").sort_index()


def performance_metrics(
    period_returns: pd.DataFrame,
    *,
    periods_per_year: float,
) -> dict[str, float | int]:
    if period_returns.empty:
        return {
            "periods": 0,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "mean_one_way_turnover": np.nan,
            "mean_return_coverage": np.nan,
        }
    returns = period_returns["net_return"].astype(float)
    nav = (1.0 + returns).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annual_return = (
        (1.0 + total_return) ** (periods_per_year / len(returns)) - 1.0
        if total_return > -1.0
        else -1.0
    )
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(periods_per_year))
        if len(returns) > 1
        else np.nan
    )
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    with_initial = pd.concat(
        [pd.Series([1.0], index=["initial"]), nav.reset_index(drop=True)]
    ).reset_index(drop=True)
    drawdown = with_initial.div(with_initial.cummax()).sub(1.0).min()
    return {
        "periods": int(len(returns)),
        "total_return": total_return,
        "annual_return": float(annual_return),
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown),
        "mean_one_way_turnover": float(
            period_returns["one_way_turnover"].mean()
        ),
        "mean_return_coverage": float(
            period_returns["return_coverage"].mean()
        ),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    split_name: str,
    top_fraction: float,
    cost_bps: float,
    signal_every: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    required = {
        "signal_date",
        "code",
        "eligible",
        "prediction",
        "future_return",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing: {sorted(missing)}")
    predictions = predictions.copy()
    predictions["signal_date"] = pd.to_datetime(predictions["signal_date"])
    predictions["code"] = predictions["code"].astype(str).str.zfill(6)
    outcome_dates = predictions.groupby("signal_date")["future_return"].apply(
        lambda values: values.notna().any()
    )
    complete_dates = outcome_dates.loc[outcome_dates].index
    predictions = predictions.loc[
        predictions["signal_date"].isin(complete_dates)
    ].copy()
    strategy = _simulate_weight_series(
        predictions,
        top_fraction=top_fraction,
        cost_bps=cost_bps,
        benchmark=False,
    )
    benchmark = _simulate_weight_series(
        predictions,
        top_fraction=top_fraction,
        cost_bps=cost_bps,
        benchmark=True,
    )
    periods_per_year = 252.0 / signal_every
    strategy_metrics = performance_metrics(
        strategy, periods_per_year=periods_per_year
    )
    benchmark_metrics = performance_metrics(
        benchmark, periods_per_year=periods_per_year
    )
    result = {
        "model": model_name,
        "split": split_name,
        "cost_bps_one_way": float(cost_bps),
        "top_fraction": float(top_fraction),
        "evaluated_dates": int(predictions["signal_date"].nunique()),
        **signal_summary(predictions),
        **strategy_metrics,
        "benchmark_annual_return": benchmark_metrics["annual_return"],
        "benchmark_annual_volatility": benchmark_metrics[
            "annual_volatility"
        ],
        "benchmark_sharpe": benchmark_metrics["sharpe"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "excess_annual_return": (
            strategy_metrics["annual_return"]
            - benchmark_metrics["annual_return"]
        ),
    }
    strategy = strategy.assign(
        model=model_name, split=split_name, series="strategy"
    ).reset_index()
    benchmark = benchmark.assign(
        model=model_name, split=split_name, series="benchmark"
    ).reset_index()
    return result, strategy, benchmark


def quantile_returns(
    predictions: pd.DataFrame,
    *,
    n_quantiles: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for signal_date, group in predictions.groupby("signal_date", sort=True):
        clean = group.dropna(subset=["prediction", "future_return"]).copy()
        if len(clean) < n_quantiles:
            continue
        ranked = clean["prediction"].rank(method="first")
        clean["quantile"] = pd.qcut(
            ranked, q=n_quantiles, labels=False
        ) + 1
        means = clean.groupby("quantile")["future_return"].mean()
        for quantile, value in means.items():
            rows.append(
                {
                    "signal_date": pd.Timestamp(signal_date),
                    "quantile": int(quantile),
                    "mean_future_return": float(value),
                }
            )
    return pd.DataFrame(rows)
