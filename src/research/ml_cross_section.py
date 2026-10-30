"""Neural-network cross-sectional return forecasting experiment."""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import ensure_output_dir, load_config
from src.common.data import load_bundle
from src.common.evaluation import (
    evaluate_predictions,
    performance_metrics,
    rank_ic_series,
)
from src.common.features import build_research_panel, split_masks
from src.common.models import run_models
from src.common.reporting import (
    markdown_table,
    save_nav_plot,
    save_rank_ic_plot,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = ensure_output_dir(config["paths"]["ml_output"])
    figure_dir = ensure_output_dir(config["paths"]["figures"])

    bundle = load_bundle(config)
    panel = build_research_panel(bundle, config)
    splits = split_masks(panel.frame, config["periods"])
    model_run = run_models(panel, splits, config["models"])
    predictions = model_run.predictions.sort_values(
        ["split", "signal_date", "model", "code"]
    )

    metric_rows: list[dict] = []
    return_frames: list[pd.DataFrame] = []
    for (model, split), group in predictions.groupby(["model", "split"]):
        for cost_bps in config["portfolio"]["cost_bps"]:
            metric, strategy, benchmark = evaluate_predictions(
                group,
                model_name=model,
                split_name=split,
                top_fraction=float(config["portfolio"]["top_fraction"]),
                cost_bps=float(cost_bps),
                signal_every=int(config["data"]["signal_every"]),
            )
            metric_rows.append(metric)
            return_frames.extend(
                [
                    strategy.assign(cost_bps_one_way=float(cost_bps)),
                    benchmark.assign(cost_bps_one_way=float(cost_bps)),
                ]
            )

    metrics = pd.DataFrame(metric_rows)
    returns = pd.concat(return_frames, ignore_index=True)
    primary_cost = float(config["portfolio"]["primary_cost_bps"])
    primary_metrics = metrics.loc[
        metrics["cost_bps_one_way"].eq(primary_cost)
    ].copy()

    yearly_rows: list[dict] = []
    for model, test_group in predictions.loc[
        predictions["split"].eq("test")
    ].groupby("model"):
        test_group = test_group.copy()
        test_group["year"] = pd.to_datetime(
            test_group["signal_date"]
        ).dt.year
        for year, year_group in test_group.groupby("year"):
            metric, _, _ = evaluate_predictions(
                year_group,
                model_name=model,
                split_name=f"temporal_validation_{int(year)}",
                top_fraction=float(config["portfolio"]["top_fraction"]),
                cost_bps=primary_cost,
                signal_every=int(config["data"]["signal_every"]),
            )
            yearly_rows.append(metric)
    yearly = pd.DataFrame(yearly_rows)

    stress_dates = pd.to_datetime(
        config.get("diagnostics", {}).get("stress_exclusion_dates", [])
    )
    stress_rows: list[dict] = []
    primary_returns = returns.loc[
        (returns["split"] == "test")
        & (returns["cost_bps_one_way"] == primary_cost)
    ].copy()
    primary_returns["signal_date"] = pd.to_datetime(
        primary_returns["signal_date"]
    )
    for model, model_returns in primary_returns.groupby("model"):
        strategy = model_returns.loc[model_returns["series"] == "strategy"]
        benchmark = model_returns.loc[model_returns["series"] == "benchmark"]
        strategy_stress = strategy.loc[
            ~strategy["signal_date"].isin(stress_dates)
        ]
        benchmark_stress = benchmark.loc[
            ~benchmark["signal_date"].isin(stress_dates)
        ]
        periods_per_year = 252.0 / int(config["data"]["signal_every"])
        full_metrics = performance_metrics(
            strategy, periods_per_year=periods_per_year
        )
        stressed_metrics = performance_metrics(
            strategy_stress, periods_per_year=periods_per_year
        )
        benchmark_metrics = performance_metrics(
            benchmark_stress, periods_per_year=periods_per_year
        )
        stress_rows.append(
            {
                "model": model,
                "excluded_dates": ",".join(
                    date.strftime("%Y-%m-%d") for date in stress_dates
                ),
                "full_annual_return": full_metrics["annual_return"],
                "full_sharpe": full_metrics["sharpe"],
                "stressed_annual_return": stressed_metrics["annual_return"],
                "stressed_sharpe": stressed_metrics["sharpe"],
                "stressed_max_drawdown": stressed_metrics["max_drawdown"],
                "benchmark_stressed_annual_return": benchmark_metrics[
                    "annual_return"
                ],
                "benchmark_stressed_sharpe": benchmark_metrics["sharpe"],
            }
        )
    stress = pd.DataFrame(stress_rows)

    ic_rows: list[dict] = []
    for (model, split), group in predictions.groupby(["model", "split"]):
        for date, value in rank_ic_series(group).items():
            ic_rows.append(
                {
                    "model": model,
                    "split": split,
                    "signal_date": date,
                    "rank_ic": value,
                }
            )
    ic_series = pd.DataFrame(ic_rows)

    split_labels = {"test": "temporal_validation"}
    reported_predictions = predictions.assign(
        split=predictions["split"].replace(split_labels)
    )
    reported_metrics = metrics.assign(
        split=metrics["split"].replace(split_labels)
    )
    reported_primary_metrics = primary_metrics.assign(
        split=primary_metrics["split"].replace(split_labels)
    )
    reported_returns = returns.assign(
        split=returns["split"].replace(split_labels)
    )
    reported_ic = ic_series.assign(
        split=ic_series["split"].replace(split_labels)
    )
    reported_predictions.to_csv(output_dir / "predictions.csv", index=False)
    reported_metrics.to_csv(output_dir / "metrics_all_costs.csv", index=False)
    reported_primary_metrics.to_csv(
        output_dir / "metrics_primary_cost.csv", index=False
    )
    reported_returns.to_csv(output_dir / "period_returns.csv", index=False)
    yearly.to_csv(
        output_dir / "yearly_temporal_validation_metrics.csv", index=False
    )
    stress.to_csv(output_dir / "extreme_period_sensitivity.csv", index=False)
    reported_ic.to_csv(output_dir / "rank_ic_series.csv", index=False)
    write_json(output_dir / "run_diagnostics.json", {
        **panel.diagnostics,
        **model_run.diagnostics,
        "split_rows": {
            name: int(mask.sum()) for name, mask in splits.items()
        },
        "label": "open[t+6]/open[t+1]-1",
        "signal": "close[t]",
        "portfolio": {
            "long_only": True,
            "top_fraction": float(config["portfolio"]["top_fraction"]),
            "primary_cost_bps_one_way": primary_cost,
            "universe_benchmark": "equal_weight_eligible_stocks",
            "incomplete_holding_periods": "excluded from performance",
            "turnover": "computed from return-drifted pre-trade weights",
            "lightgbm_subsample_freq": int(
                config["models"]["lgbm_subsample_freq"]
            ),
        },
        "universe_note": (
            "The price panel covers all 481 historical membership-union "
            "stocks. Membership is sampled on the five-day research grid, "
            "not reconstructed for every calendar day. Portfolio results "
            "remain analytical and do not model blocked exits."
        ),
    })

    save_rank_ic_plot(
        reported_metrics,
        figure_dir / "ml_temporal_validation_rank_ic.png",
        split="temporal_validation",
        cost_bps=primary_cost,
    )
    save_nav_plot(
        reported_returns,
        figure_dir / "ml_temporal_validation_nav.png",
        split="temporal_validation",
        cost_bps=primary_cost,
    )

    test_table = primary_metrics.loc[
        primary_metrics["split"].eq("test")
    ].sort_values("mean_rank_ic", ascending=False)
    validation_table = primary_metrics.loc[
        primary_metrics["split"].eq("validation")
    ].sort_values("mean_rank_ic", ascending=False)
    mlp_test = test_table.loc[test_table["model"].eq("mlp")]
    mlp_row = mlp_test.iloc[0] if not mlp_test.empty else None
    if mlp_row is not None:
        mlp_sentence = (
            "MLP时间外验证期Mean RankIC为"
            f"{mlp_row['mean_rank_ic']:.4f}，ICIR为"
            f"{mlp_row['icir_unannualized']:.4f}；20%多头组合在"
            f"{primary_cost:g}bps单边成本下年化收益"
            f"{mlp_row['annual_return']:.2%}，Sharpe为"
            f"{mlp_row['sharpe']:.3f}，最大回撤为"
            f"{mlp_row['max_drawdown']:.2%}。"
        )
    else:
        mlp_sentence = "MLP未产生可用时间外验证预测。"

    mlp_stress = stress.loc[stress["model"].eq("mlp")]
    stress_sentence = ""
    if not mlp_stress.empty:
        row = mlp_stress.iloc[0]
        stress_sentence = (
            "剔除"
            f"{row['excluded_dates']}两个事后识别的高波动周期后，"
            f"MLP年化收益为{row['stressed_annual_return']:.2%}，"
            f"Sharpe为{row['stressed_sharpe']:.3f}；同期等权基准"
            f"年化收益为{row['benchmark_stressed_annual_return']:.2%}，"
            f"Sharpe为{row['benchmark_stressed_sharpe']:.3f}。"
            "这表明组合收益明显依赖少数极端周期，不能据此宣称稳定Alpha。"
        )

    report = f"""# A股横截面收益预测：MLP与线性基准的时间外比较

## 结论

该实验比较20日动量、等权因子复合、Ridge、MLP和LightGBM。所有模型使用相同的周频信号、未来5日开盘到开盘收益标签与多头组合规则。数据已同步至历史名单并集481只股票。

{mlp_sentence}

### 验证期

{markdown_table(validation_table, ["model", "mean_rank_ic", "icir_unannualized", "rank_ic_positive_ratio", "annual_return", "sharpe", "max_drawdown", "mean_one_way_turnover", "excess_annual_return"])}

### 时间外验证期

{markdown_table(test_table, ["model", "mean_rank_ic", "icir_unannualized", "rank_ic_positive_ratio", "annual_return", "sharpe", "max_drawdown", "mean_one_way_turnover", "excess_annual_return"])}

### 分年份时间外验证结果

{markdown_table(yearly, ["model", "split", "mean_rank_ic", "annual_return", "sharpe", "max_drawdown"])}

### 极端周期敏感性

{stress_sentence}

{markdown_table(stress, ["model", "full_annual_return", "full_sharpe", "stressed_annual_return", "stressed_sharpe", "benchmark_stressed_annual_return", "benchmark_stressed_sharpe"])}

## 研究设计

- 信号日t：收盘后形成横截面特征。
- 入场：t+1开盘；退出：t+6开盘；预测目标为 `open[t+6] / open[t+1] - 1`。
- 训练：2020年至2022年末；验证：2023年；时间外验证：2024-2025年。
- 训练样本的标签结束日必须早于验证期，避免跨边界标签泄漏。
- 特征在每个交易日的可选股票池内转成横截面百分位。
- MLP使用两层64/32隐藏单元、ReLU、Adam、L2正则和3个随机种子平均。
- Ridge的alpha仅在验证期选择；最终在训练加验证样本上重训，然后一次性预测时间外验证期。
- LightGBM的轮数仅由训练期与验证期决定，最终在训练加验证样本上按固定轮数重训。
- 组合每5个市场交易日调仓，选预测最高的20%股票等权持有，按收益漂移后的调仓前权重计算换手。
- 主成本为{primary_cost:g}bps单边成交额成本；同时报告5、10、20bps。
- 最后一个未来收益完全缺失的未完成持有期不进入组合绩效。
- `2024-09-23`和`2024-09-30`的敏感性分析属于事后诊断，不用于模型选择。

## 数据与限制

- 本地行情覆盖2019-07至2025-12，历史名单并集481只股票；研究截面从2020年开始。
- 历史名单按五日研究网格的查询日期取得，不是逐日成员快照。
- 信号日剔除停牌、ST、历史不足120个可交易日和20日成交额低于3,000万元的股票。
- 行情复权口径和量额单位未独立核验，因此主实验默认不使用成交量类特征。
- 无逐笔盘口、涨跌停阻塞后的持仓延期与容量限制，组合评价仍为简化研究回测。
- 期末标签缺失的股票不会因未来标签缺失从预测池删除；组合评价中缺失未来收益按0处理，并输出覆盖率。
- 结果用于面试项目和方法复现，不宣称实盘有效性。

## 主要输出

- `predictions.csv`：所有模型的验证与时间外验证预测。
- `metrics_all_costs.csv`：全部成本情景。
- `rank_ic_series.csv`：逐信号日RankIC。
- `period_returns.csv`：策略和等权基准的组合收益。
- `yearly_temporal_validation_metrics.csv`：分年份稳定性。
- `extreme_period_sensitivity.csv`：剔除指定极端周期后的描述性结果。

## 简历可用表述

> 基于多层感知机研究A股横截面收益预测，使用历史名单查询日、信号日可交易与流动性过滤、横截面特征转换和严格时间切分；与动量、Ridge及LightGBM在统一开盘到开盘5日标签和交易成本下比较，输出RankIC、ICIR、Top20%组合收益、换手率及回撤。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(test_table.to_string(index=False))
    print(f"PASS: ML outputs written to {output_dir}")


if __name__ == "__main__":
    main()
