from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd

from src.common.data import (
    DataBundle,
    build_membership_mask,
    build_price_matrices,
)
from src.common.evaluation import (
    evaluate_predictions,
    performance_metrics,
    rank_ic_series,
    signal_summary,
)
from src.common.features import (
    add_cross_sectional_ranks,
    build_research_panel,
    compute_forward_labels,
    compute_price_features,
    split_masks,
)
from src.common.models import _fit_lightgbm, _fit_mlp


def make_prices(
    *,
    codes: tuple[str, ...] = ("000001", "000002", "600000"),
    periods: int = 130,
) -> pd.DataFrame:
    calendar = pd.bdate_range("2022-01-03", periods=periods)
    rows: list[dict] = []
    for code_index, code in enumerate(codes):
        base = 10.0 + code_index
        for date_index, date in enumerate(calendar):
            close = base + date_index * 0.1 + code_index * 0.01
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": close - 0.02,
                    "close": close,
                    "high": close + 0.05,
                    "low": close - 0.05,
                    "preclose": close - 0.1,
                    "volume": 1_000_000.0 + date_index * 1_000.0,
                    "turnover": 0.01,
                    "amount": 100_000_000.0,
                    "tradestatus": 1,
                    "isST": 0,
                }
            )
    return pd.DataFrame(rows)


class DataAndFeatureTests(unittest.TestCase):
    def setUp(self):
        self.prices = make_prices()
        self.market = build_price_matrices(self.prices)

    def test_forward_label_uses_one_day_delay_and_five_day_holding(self):
        future, entry, end = compute_forward_labels(
            self.market, entry_delay=1, horizon=5
        )
        signal_position = 10
        open_price = self.market.open.iloc[:, 0]
        expected = open_price.iloc[16] / open_price.iloc[11] - 1
        self.assertAlmostEqual(
            future.iloc[signal_position, 0], expected, places=12
        )
        self.assertEqual(entry.iloc[signal_position, 0], self.market.calendar[11])
        self.assertEqual(end.iloc[signal_position, 0], self.market.calendar[16])

    def test_future_price_mutation_does_not_change_past_features(self):
        original, _, _ = compute_price_features(
            self.market,
            include_liquidity=False,
            min_history_days=120,
        )
        changed_prices = self.prices.copy()
        cutoff = pd.Timestamp(self.market.calendar[60])
        changed_prices.loc[changed_prices["date"] > cutoff, "close"] *= 7.0
        changed_prices.loc[changed_prices["date"] > cutoff, "high"] *= 7.0
        changed = build_price_matrices(changed_prices)
        mutated, _, _ = compute_price_features(
            changed,
            include_liquidity=False,
            min_history_days=120,
        )
        for name in original:
            pd.testing.assert_frame_equal(
                original[name].loc[:cutoff],
                mutated[name].loc[:cutoff],
                check_names=False,
            )

    def test_membership_uses_latest_available_snapshot(self):
        membership = pd.DataFrame(
            {
                "query_date": pd.to_datetime(
                    ["2022-01-03", "2022-01-05"]
                ),
                "index_effective_date": pd.to_datetime(
                    ["2022-01-01", "2022-01-05"]
                ),
                "code": ["000001", "000002"],
            }
        )
        mask = build_membership_mask(self.market, membership)
        self.assertTrue(mask.at[pd.Timestamp("2022-01-04"), "000001"])
        self.assertFalse(mask.at[pd.Timestamp("2022-01-04"), "000002"])
        self.assertTrue(mask.at[pd.Timestamp("2022-01-05"), "000002"])
        self.assertFalse(mask.at[pd.Timestamp("2022-01-05"), "000001"])

    def test_cross_sectional_ranks_are_date_local(self):
        frame = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    ["2023-01-06"] * 3 + ["2023-01-13"] * 2
                ),
                "code": ["000001", "000002", "600000", "000001", "000002"],
                "eligible": [True, True, True, True, True],
                "future_return": [0.01, 0.02, 0.03, 0.03, -0.01],
                "entry_date": pd.Timestamp("2023-01-09"),
                "label_end_date": pd.Timestamp("2023-01-16"),
                "factor": [1.0, 2.0, 3.0, 1.0, 0.0],
            }
        )
        ranked = add_cross_sectional_ranks(frame, ["factor"])
        first = ranked.loc[
            ranked["signal_date"].eq(pd.Timestamp("2023-01-06")),
            "cs_factor",
        ]
        second = ranked.loc[
            ranked["signal_date"].eq(pd.Timestamp("2023-01-13")),
            "cs_factor",
        ]
        np.testing.assert_allclose(
            first.to_numpy(), [-1 / 6, 1 / 6, 0.5]
        )
        np.testing.assert_allclose(second.to_numpy(), [0.5, 0.0])

    def test_test_mask_does_not_require_future_outcome(self):
        frame = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2025-12-31"]),
                "label_end_date": pd.to_datetime(["2026-01-09"]),
                "future_return": [np.nan],
                "rank_label": [np.nan],
            }
        )
        periods = {
            "development_end": "2022-12-31",
            "validation_start": "2023-01-01",
            "validation_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2025-12-31",
        }
        masks = split_masks(frame, periods)
        self.assertTrue(masks["test"].iloc[0])
        self.assertFalse(masks["validation"].iloc[0])

    def test_signal_filters_suspended_and_st_stocks(self):
        signal_date = pd.Timestamp(self.market.calendar[120])
        membership = pd.DataFrame(
            {
                "query_date": [signal_date] * 3,
                "index_effective_date": [signal_date] * 3,
                "code": ["000001", "000002", "600000"],
            }
        )
        prices = self.prices.copy()
        prices.loc[
            prices["date"].eq(signal_date) & prices["code"].eq("000002"),
            "tradestatus",
        ] = 0
        prices.loc[
            prices["date"].eq(signal_date) & prices["code"].eq("600000"),
            "isST",
        ] = 1
        market = build_price_matrices(prices)
        mask = build_membership_mask(market, membership)
        bundle = DataBundle(
            prices=prices,
            membership=membership,
            market=market,
            membership_mask=mask,
            price_sha256="synthetic",
        )
        config = {
            "data": {
                "min_history_days": 120,
                "signal_every": 5,
                "entry_delay": 1,
                "label_horizon": 5,
                "include_liquidity_features": False,
                "min_adv20": 0,
                "research_start": str(signal_date.date()),
            }
        }
        panel = build_research_panel(bundle, config)
        self.assertEqual(panel.frame["code"].tolist(), ["000001"])

    def test_train_and_validation_masks_purge_boundary_labels(self):
        frame = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    [
                        "2022-12-23",
                        "2022-12-30",
                        "2023-01-03",
                        "2023-12-29",
                        "2024-01-02",
                    ]
                ),
                "label_end_date": pd.to_datetime(
                    [
                        "2022-12-30",
                        "2023-01-06",
                        "2023-01-10",
                        "2024-01-05",
                        "2024-01-09",
                    ]
                ),
                "future_return": [0.01, 0.01, 0.01, 0.01, 0.01],
                "rank_label": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        periods = {
            "development_end": "2022-12-31",
            "validation_start": "2023-01-01",
            "validation_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2025-12-31",
        }
        masks = split_masks(frame, periods)
        self.assertEqual(
            masks["development"].tolist(),
            [True, False, False, False, False],
        )
        self.assertEqual(
            masks["validation"].tolist(),
            [False, False, True, False, False],
        )
        self.assertEqual(
            masks["test"].tolist(),
            [False, False, False, False, True],
        )


class EvaluationTests(unittest.TestCase):
    def test_rank_ic_direction(self):
        frame = pd.DataFrame(
            {
                "signal_date": pd.Timestamp("2024-01-05"),
                "code": ["000001", "000002", "600000"],
                "eligible": True,
                "prediction": [3.0, 2.0, 1.0],
                "future_return": [0.01, 0.02, 0.03],
            }
        )
        self.assertAlmostEqual(rank_ic_series(frame).iloc[0], -1.0)
        self.assertAlmostEqual(signal_summary(frame)["mean_rank_ic"], -1.0)

    def test_full_switch_pays_both_sides_of_turnover(self):
        rows = []
        for date, selected, other in (
            ("2024-01-05", "000001", "000002"),
            ("2024-01-12", "000002", "000001"),
        ):
            for code in ("000001", "000002"):
                rows.append(
                    {
                        "signal_date": pd.Timestamp(date),
                        "code": code,
                        "eligible": True,
                        "prediction": 2.0 if code == selected else 1.0,
                        "future_return": 0.0,
                    }
                )
        result, strategy, _ = evaluate_predictions(
            pd.DataFrame(rows),
            model_name="test",
            split_name="test",
            top_fraction=0.5,
            cost_bps=10,
            signal_every=5,
        )
        self.assertAlmostEqual(
            strategy["transaction_cost"].iloc[0], 0.001
        )
        self.assertAlmostEqual(
            strategy["transaction_cost"].iloc[1], 0.002
        )
        self.assertAlmostEqual(strategy["one_way_turnover"].iloc[1], 1.0)
        self.assertAlmostEqual(
            result["total_return"],
            (1.0 - 0.001) * (1.0 - 0.002) - 1.0,
            places=12,
        )

    def test_incomplete_holding_period_is_not_evaluated(self):
        rows = []
        for date in ("2024-01-05", "2024-01-12", "2024-01-19"):
            for code in ("000001", "000002"):
                rows.append(
                    {
                        "signal_date": pd.Timestamp(date),
                        "code": code,
                        "eligible": True,
                        "prediction": 2.0 if code == "000001" else 1.0,
                        "future_return": (
                            np.nan
                            if date == "2024-01-19"
                            else (0.01 if code == "000001" else 0.00)
                        ),
                    }
                )
        result, strategy, _ = evaluate_predictions(
            pd.DataFrame(rows),
            model_name="test",
            split_name="test",
            top_fraction=0.5,
            cost_bps=10,
            signal_every=5,
        )
        self.assertEqual(result["evaluated_dates"], 2)
        self.assertEqual(result["periods"], 2)
        self.assertNotIn(
            pd.Timestamp("2024-01-19"), strategy["signal_date"].tolist()
        )

    def test_turnover_uses_return_drifted_weights(self):
        rows = []
        for date, returns in (
            ("2024-01-05", {"000001": 0.20, "000002": 0.0}),
            ("2024-01-12", {"000001": 0.0, "000002": 0.0}),
        ):
            for code in ("000001", "000002"):
                rows.append(
                    {
                        "signal_date": pd.Timestamp(date),
                        "code": code,
                        "eligible": True,
                        "prediction": 2.0 if code == "000001" else 1.0,
                        "future_return": returns[code],
                    }
                )
        _, strategy, _ = evaluate_predictions(
            pd.DataFrame(rows),
            model_name="test",
            split_name="test",
            top_fraction=1.0,
            cost_bps=0.0,
            signal_every=5,
        )
        self.assertAlmostEqual(
            strategy["one_way_turnover"].iloc[1], 1.0 / 22.0
        )

    def test_max_drawdown_includes_initial_nav(self):
        returns = pd.DataFrame(
            {
                "net_return": [-0.10, 0.20],
                "one_way_turnover": [0.0, 0.0],
                "return_coverage": [1.0, 1.0],
            }
        )
        metrics = performance_metrics(
            returns, periods_per_year=50.4
        )
        self.assertAlmostEqual(metrics["max_drawdown"], -0.10)


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = pd.DataFrame(
            rng.normal(size=(120, 5)),
            columns=[f"x_{index}" for index in range(5)],
        )
        self.y = pd.Series(
            0.2 * self.x["x_0"] - 0.1 * self.x["x_1"]
            + rng.normal(scale=0.05, size=len(self.x))
        )
        self.model_config = {
            "mlp_hidden_layers": [8],
            "mlp_alpha": 0.0001,
            "mlp_batch_size": 32,
            "mlp_learning_rate": 0.001,
            "mlp_max_iter": 10,
            "lgbm_num_leaves": 7,
            "lgbm_learning_rate": 0.05,
            "lgbm_min_child_samples": 5,
            "lgbm_subsample": 0.8,
            "lgbm_subsample_freq": 1,
            "lgbm_colsample_bytree": 0.8,
            "lgbm_reg_alpha": 0.0,
            "lgbm_reg_lambda": 1.0,
            "lgbm_max_rounds": 20,
            "lgbm_early_stopping_rounds": 3,
        }

    def test_mlp_training_smoke(self):
        model = _fit_mlp(self.x, self.y, self.model_config, seed=7)
        self.assertEqual(model.predict(self.x).shape, (len(self.x),))

    @unittest.skipUnless(
        importlib.util.find_spec("lightgbm") is not None,
        "lightgbm is not installed",
    )
    def test_lightgbm_training_smoke_with_row_subsampling(self):
        booster, best_iteration = _fit_lightgbm(
            self.x.iloc[:90],
            self.y.iloc[:90],
            self.x.iloc[90:],
            self.y.iloc[90:],
            self.model_config,
        )
        self.assertGreater(best_iteration, 0)
        self.assertEqual(
            booster.predict(self.x.iloc[90:]).shape, (30,)
        )


if __name__ == "__main__":
    unittest.main()
