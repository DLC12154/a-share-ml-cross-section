"""Small, dependency-light reporting helpers for reproducible experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _display(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "NA"
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> str:
    selected = frame.loc[:, list(columns)] if columns else frame.copy()
    if selected.empty:
        return "_No observations._"
    header = "| " + " | ".join(str(column) for column in selected.columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| "
        + " | ".join(_display(value) for value in row)
        + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def save_rank_ic_plot(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    split: str = "test",
    cost_bps: float = 10.0,
) -> None:
    selected = metrics.loc[metrics["split"] == split].copy()
    if selected.empty:
        return
    selected = selected.loc[
        selected["cost_bps_one_way"].eq(cost_bps)
    ].sort_values("mean_rank_ic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis = selected.plot.barh(
        x="model",
        y="mean_rank_ic",
        legend=False,
        figsize=(8, 4.5),
        color="#2f6f9f",
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean RankIC")
    axis.set_ylabel("")
    split_label = (
        "2024-2025 temporal validation"
        if split in {"test", "temporal_validation"}
        else split.replace("_", " ")
    )
    title = f"Mean RankIC: {split_label}"
    benchmark_returns = selected["benchmark_annual_return"].dropna()
    if not benchmark_returns.empty:
        title += (
            "\nEqual-weight benchmark annual return: "
            f"{benchmark_returns.iloc[0]:.2%}"
        )
    axis.set_title(title)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)


def save_nav_plot(
    returns: pd.DataFrame,
    output_path: Path,
    *,
    split: str = "test",
    cost_bps: float = 10.0,
) -> None:
    strategy = returns.loc[
        (returns["split"] == split)
        & (returns["series"] == "strategy")
        & (returns["cost_bps_one_way"] == cost_bps)
    ].copy()
    if strategy.empty:
        return
    strategy = strategy.sort_values(["model", "signal_date"])
    axis = None
    for model, group in strategy.groupby("model", sort=True):
        nav = (1.0 + group.set_index("signal_date")["net_return"]).cumprod()
        axis = nav.plot(
            ax=axis,
            figsize=(10, 5),
            linewidth=1.6,
            label=model,
        )
    benchmark = returns.loc[
        (returns["split"] == split)
        & (returns["series"] == "benchmark")
        & (returns["cost_bps_one_way"] == cost_bps)
    ].copy()
    if not benchmark.empty:
        benchmark_model = sorted(benchmark["model"].unique())[0]
        benchmark = benchmark.loc[benchmark["model"].eq(benchmark_model)]
        benchmark_nav = (
            1.0
            + benchmark.sort_values("signal_date")
            .set_index("signal_date")["net_return"]
        ).cumprod()
        benchmark_nav.plot(
            ax=axis,
            color="black",
            linestyle="--",
            linewidth=2.2,
            label="equal-weight benchmark",
        )
    axis.set_title("Temporal validation weekly net asset value")
    axis.set_xlabel("")
    axis.set_ylabel("Growth of 1")
    axis.legend(loc="best")
    axis.figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)
