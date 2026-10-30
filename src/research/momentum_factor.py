"""Momentum factor study using the shared data and evaluation stack."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import ensure_output_dir, load_config, project_path
from src.common.data import load_bundle
from src.common.evaluation import (
    evaluate_predictions,
    quantile_returns,
    rank_ic_series,
)
from src.common.features import build_research_panel, split_masks
from src.common.reporting import markdown_table, save_nav_plot, write_json


def _prediction_slice(
    frame: pd.DataFrame,
    mask: pd.Series,
    factor_column: str,
    factor_name: str,
    split: str,
) -> pd.DataFrame:
    output = frame.loc[
        mask,
        [
            "signal_date",
            "code",
            "eligible",
            "future_return",
            "label_end_date",
        ],
    ].copy()
    output["prediction"] = frame.loc[mask, factor_column].to_numpy()
    output["model"] = factor_name
    output["split"] = split
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = ensure_output_dir(config["paths"]["momentum_output"])
    figure_dir = ensure_output_dir(config["paths"]["figures"])
    bundle = load_bundle(config)
    panel = build_research_panel(bundle, config)
    splits = split_masks(panel.frame, config["periods"])
    signal_every = int(config["data"]["signal_every"])
    cost = float(config["portfolio"]["primary_cost_bps"])
    top_fraction = float(config["portfolio"]["top_fraction"])
    lookbacks = [int(value) for value in config["models"]["momentum_lookbacks"]]

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    return_frames: list[pd.DataFrame] = []
    quantile_frames: list[pd.DataFrame] = []
    for lookback in lookbacks:
        factor_column = f"cs_ret_{lookback}"
        factor_name = f"momentum_{lookback}d"
        for split in ("validation", "test"):
            predictions = _prediction_slice(
                panel.frame,
                splits[split],
                factor_column,
                factor_name,
                split,
            )
            metric, strategy, benchmark = evaluate_predictions(
                predictions,
                model_name=factor_name,
                split_name=split,
                top_fraction=top_fraction,
                cost_bps=cost,
                signal_every=signal_every,
            )
            metric_rows.append(metric)
            prediction_frames.append(predictions)
            return_frames.extend(
                [
                    strategy.assign(cost_bps_one_way=cost),
                    benchmark.assign(cost_bps_one_way=cost),
                ]
            )
            if lookback == int(config["models"]["primary_momentum"]):
                groups = quantile_returns(predictions, n_quantiles=5)
                groups["model"] = factor_name
                groups["split"] = split
                quantile_frames.append(groups)

    primary_name = f"momentum_{int(config['models']['primary_momentum'])}d"
    primary_test = pd.concat(
        [
            frame
            for frame in prediction_frames
            if frame["split"].eq("test").all()
            and frame["model"].eq(primary_name).all()
        ],
        ignore_index=True,
    )
    cost_rows: list[dict] = []
    cost_returns: list[pd.DataFrame] = []
    for cost_bps in config["portfolio"]["cost_bps"]:
        metric, strategy, benchmark = evaluate_predictions(
            primary_test,
            model_name=primary_name,
            split_name="test",
            top_fraction=top_fraction,
            cost_bps=float(cost_bps),
            signal_every=signal_every,
        )
        cost_rows.append(metric)
        if float(cost_bps) != cost:
            cost_returns.extend(
                [
                    strategy.assign(cost_bps_one_way=float(cost_bps)),
                    benchmark.assign(cost_bps_one_way=float(cost_bps)),
                ]
            )

    metrics = pd.DataFrame(metric_rows)
    cost_metrics = pd.DataFrame(cost_rows)
    returns = pd.concat(return_frames + cost_returns, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    quantiles = (
        pd.concat(quantile_frames, ignore_index=True)
        if quantile_frames
        else pd.DataFrame()
    )
    metrics.to_csv(output_dir / "factor_metrics.csv", index=False)
    cost_metrics.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    returns.to_csv(output_dir / "period_returns.csv", index=False)
    predictions.to_csv(output_dir / "factor_predictions.csv", index=False)
    quantiles.to_csv(output_dir / "quantile_returns.csv", index=False)

    ic_rows: list[dict] = []
    for (model, split), group in predictions.groupby(["model", "split"]):
        ic = rank_ic_series(group)
        for date, value in ic.items():
            ic_rows.append(
                {
                    "model": model,
                    "split": split,
                    "signal_date": date,
                    "rank_ic": value,
                }
            )
    ic_frame = pd.DataFrame(ic_rows)
    ic_frame.to_csv(output_dir / "rank_ic_series.csv", index=False)

    test_metrics = metrics.loc[metrics["split"].eq("test")].sort_values(
        "mean_rank_ic", ascending=False
    )
    axis = test_metrics.plot.barh(
        x="model", y="mean_rank_ic", legend=False, figsize=(8, 4)
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean RankIC")
    axis.set_ylabel("")
    axis.set_title("Locked test period: momentum RankIC")
    axis.figure.tight_layout()
    axis.figure.savefig(
        figure_dir / "momentum_rank_ic.png", dpi=160
    )
    plt.close(axis.figure)
    save_nav_plot(
        returns.assign(
            split=returns["split"],
            series=returns["series"],
        ),
        figure_dir / "momentum_test_nav.png",
        split="test",
        cost_bps=cost,
    )

    diagnostics = {
        **panel.diagnostics,
        "train_rows": int(splits["development"].sum()),
        "validation_rows": int(splits["validation"].sum()),
        "test_rows": int(splits["test"].sum()),
        "label": "open[t+6]/open[t+1]-1",
        "signal": "close[t]",
        "universe_note": (
            "The price panel covers all 481 historical membership-union "
            "stocks. Membership is sampled on the five-day research grid "
            "rather than reconstructed for every calendar day."
        ),
    }
    write_json(output_dir / "run_diagnostics.json", diagnostics)

    report = f"""# 动量因子复现实验

## 结论

实验在历史名单查询日、481只历史成员并集和信号日可交易过滤上运行。名单为五日研究网格，不是逐日成分股。

### 时间外验证结果

{markdown_table(test_metrics, ["model", "mean_rank_ic", "icir_unannualized", "rank_ic_positive_ratio", "annual_return", "sharpe", "max_drawdown", "mean_one_way_turnover", "excess_annual_return"])}

### 成本敏感性

{markdown_table(cost_metrics, ["cost_bps_one_way", "annual_return", "sharpe", "max_drawdown", "mean_one_way_turnover", "benchmark_annual_return", "excess_annual_return"])}

## 时间定义

- t日收盘形成信号。
- t+1日开盘进入，t+6日开盘退出。
- 标签为 `open[t+6] / open[t+1] - 1`。
- 每5个市场交易日调仓，预测分数最高的20%股票等权。
- 成本按单边bps作用于买入和卖出成交额。

## 数据限制

- 行情为2019-07至2025-12，覆盖历史名单并集481只股票；研究截面从2020年开始。
- 历史名单按五日研究网格的查询日期做可得信息过滤，不是逐日成员快照。
- 信号日要求股票可交易、非ST、至少有120个可交易历史日，且20日平均成交额不低于3,000万元。
- 主实验默认不使用成交量类特征；这是共享信号评价，不是逐日现金账户仿真。
- 回测是周频组合收益研究，不是逐日现金账户或可执行交易仿真。

## 输出

- `factor_metrics.csv`：各窗口验证期和时间外验证期指标。
- `cost_sensitivity.csv`：主因子成本情景。
- `rank_ic_series.csv`：逐调仓日RankIC。
- `quantile_returns.csv`：主因子五分组收益。
- `period_returns.csv`：策略与同股票池等权基准。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(test_metrics.to_string(index=False))
    print(f"PASS: momentum outputs written to {output_dir}")


if __name__ == "__main__":
    main()
