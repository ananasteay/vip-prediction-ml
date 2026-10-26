import unittest

import numpy as np
import pandas as pd

from src.features import build_features, fit_feature_params
from tests._fixtures import make_raw_features


class BuildFeaturesTests(unittest.TestCase):
    def test_basic_shape_and_uid_dropped(self):
        raw = make_raw_features(10)
        params = fit_feature_params(raw)
        X = build_features(raw, params)
        self.assertEqual(len(X), 10)
        self.assertNotIn("uid", X.columns)

    def test_different_row_counts(self):
        for n in [1, 3, 50]:
            raw = make_raw_features(n)
            params = fit_feature_params(raw)
            X = build_features(raw, params)
            self.assertEqual(len(X), n)

    def test_missing_bet_and_spins(self):
        raw = make_raw_features(5, avg_bet_size=[np.nan] * 5, spins_count=[np.nan] * 5)
        params = fit_feature_params(raw)
        X = build_features(raw, params)
        self.assertTrue((X["bet_spins_missing"] == 1).all())
        self.assertTrue(X["deposit_per_spin"].isna().all())

    def test_missing_install_date(self):
        raw = make_raw_features(5, app_installed_at=[pd.NaT] * 5)
        params = fit_feature_params(raw)
        X = build_features(raw, params)
        self.assertTrue((X["has_install_date"] == 0).all())
        self.assertTrue(X["install_to_ftd_days"].isna().all())

    def test_unknown_email_domain_consistent_encoding(self):
        train_raw = make_raw_features(20, email_domain=["gmail.com"] * 15 + ["wp.pl"] * 5)
        params = fit_feature_params(train_raw)

        seen_raw = make_raw_features(2, email_domain=["gmail.com", "wp.pl"])
        unseen_raw = make_raw_features(2, email_domain=["totally-new-domain.io", "another-one.com"])

        X_seen = build_features(seen_raw, params)
        X_unseen = build_features(unseen_raw, params)

        self.assertTrue((X_unseen["email_domain"] == "other").all())
        self.assertEqual(list(X_seen["email_domain"].cat.categories), list(X_unseen["email_domain"].cat.categories))

    def test_division_by_zero_is_safe(self):
        raw = make_raw_features(3, tx_count=[0, 0, 0], failed_deposits=[0, 0, 0], spins_count=[0, 0, 0])
        params = fit_feature_params(raw)
        X = build_features(raw, params)
        self.assertFalse(np.isinf(X["failed_deposit_ratio"]).any())
        self.assertFalse(np.isinf(X["spins_per_tx"]).any())
        self.assertFalse(np.isinf(X["deposit_per_spin"]).any())

    def test_registration_after_ftd_is_clipped(self):
        raw = make_raw_features(3)
        raw["registered_at"] = raw["ftd_at"] + pd.Timedelta(days=2)
        params = fit_feature_params(raw)
        X = build_features(raw, params)
        self.assertTrue((X["reg_to_ftd_days"] >= 0).all())

    def test_unparseable_dates_become_missing(self):
        raw = make_raw_features(3)
        raw["app_installed_at"] = ["not-a-date", "also-bad", None]
        params = fit_feature_params(raw)
        X = build_features(raw, params)
        self.assertTrue(X["install_to_ftd_days"].isna().all())

    def test_exclude_ftd_amount(self):
        raw = make_raw_features(5)
        params = fit_feature_params(raw)
        X = build_features(raw, params, include_ftd_amount=False)
        self.assertNotIn("ftd_amount", X.columns)
        self.assertNotIn("deposit_per_spin", X.columns)
        self.assertNotIn("bet_to_deposit_ratio", X.columns)


if __name__ == "__main__":
    unittest.main()
