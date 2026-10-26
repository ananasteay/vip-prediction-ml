import unittest

import numpy as np
import pandas as pd

from src.train import split_train_calib_holdout
from tests._fixtures import make_raw_features


class SplitTrainCalibHoldoutTests(unittest.TestCase):
    def test_temporal_split_rejects_missing_ftd_dates(self):
        raw = make_raw_features(10)
        raw.loc[3, "ftd_at"] = pd.NaT
        y = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "missing or unparseable FTD date"):
            split_train_calib_holdout(raw, y, split_mode="temporal")

    def test_temporal_split_accepts_valid_ftd_dates(self):
        raw = make_raw_features(10)
        y = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        X_train, X_calib, X_holdout, y_train, y_calib, y_holdout = split_train_calib_holdout(
            raw, y, split_mode="temporal"
        )
        self.assertEqual(len(X_train) + len(X_calib) + len(X_holdout), 10)

    def test_temporal_split_keeps_tied_ftd_timestamps_together(self):
        base = pd.Timestamp("2025-01-01")
        ftd_values = [base + pd.Timedelta(hours=i) for i in range(10)]
        ftd_values[6] = ftd_values[5]  # tie exactly at the naive 60% boundary (n_train=6)
        raw = make_raw_features(10, ftd_at=ftd_values)
        y = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0])

        X_train, X_calib, X_holdout, y_train, y_calib, y_holdout = split_train_calib_holdout(
            raw, y, split_mode="temporal"
        )
        # the boundary is pushed past the tied pair (rows 5 and 6) instead
        # of splitting them across train/calibration
        self.assertEqual(len(X_train), 7)
        self.assertEqual(len(X_calib), 1)
        self.assertEqual(len(X_holdout), 2)
        self.assertEqual(int(y_train.sum()), 1)
        self.assertEqual(int(y_calib.sum()), 1)

    def test_temporal_split_raises_if_train_has_no_positives(self):
        raw = make_raw_features(10)
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "training split has no positive"):
            split_train_calib_holdout(raw, y, split_mode="temporal")

    def test_temporal_split_raises_if_calib_has_no_positives(self):
        raw = make_raw_features(10)
        y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "calibration split has no positive"):
            split_train_calib_holdout(raw, y, split_mode="temporal")

    def test_invalid_split_mode_raises(self):
        raw = make_raw_features(10)
        y = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "split_mode must be"):
            split_train_calib_holdout(raw, y, split_mode="bogus")

    def test_empty_input_raises(self):
        raw = make_raw_features(0)
        y = np.array([], dtype=int)
        with self.assertRaisesRegex(ValueError, "no rows to split"):
            split_train_calib_holdout(raw, y, split_mode="temporal")

    def test_mismatched_length_raises(self):
        raw = make_raw_features(10)
        y = np.zeros(5, dtype=int)
        with self.assertRaisesRegex(ValueError, "must have the same length"):
            split_train_calib_holdout(raw, y, split_mode="temporal")


if __name__ == "__main__":
    unittest.main()
