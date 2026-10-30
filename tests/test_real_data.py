from __future__ import annotations

import unittest

from src.common.config import PROJECT_ROOT, load_config
from src.common.data import load_bundle
from src.common.evaluation import rank_ic_series
from src.common.features import build_research_panel, split_masks


FROZEN_PRICE_PATH = PROJECT_ROOT / "data/raw/csi300_prices_historical.parquet"


@unittest.skipUnless(
    FROZEN_PRICE_PATH.exists(),
    "Frozen historical price file is not present; run scripts/build_historical_dataset.py",
)
class FrozenDataIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.bundle = load_bundle(cls.config)
        cls.panel = build_research_panel(cls.bundle, cls.config)
        cls.splits = split_masks(
            cls.panel.frame, cls.config["periods"]
        )

    def test_frozen_data_coverage(self):
        self.assertEqual(self.bundle.price_sha256.lower(), (
            "3f13a8eafb9138ff33a8c0807f143782"
            "f229c25f5fac8e3a7e4f7a74f4820a23"
        ))
        self.assertEqual(len(self.bundle.market.codes), 481)
        self.assertEqual(self.panel.diagnostics["signal_dates"], 291)
        self.assertEqual(
            self.panel.diagnostics["stocks_membership_missing_prices"], 0
        )

    def test_20d_oos_rank_ic_matches_source_project(self):
        test = self.panel.frame.loc[self.splits["test"]].copy()
        test["prediction"] = test["cs_ret_20"]
        rank_ic = rank_ic_series(test).dropna()
        self.assertEqual(len(rank_ic), 96)
        self.assertAlmostEqual(float(rank_ic.mean()), -0.014042, places=6)


if __name__ == "__main__":
    unittest.main()
