import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.features import CATEGORICAL_FEATURES, build_features, fit_feature_params
from src.predict import load_artifact, main, predict, predict_vip_uids
from tests._fixtures import make_raw_features


def _toy_artifact(include_ftd_amount: bool = True) -> dict:
    n = 60
    raw = make_raw_features(n)
    raw["ftd_amount"] = np.linspace(5, 500, n)
    raw["spins_count"] = np.linspace(1, 2000, n)
    y = (raw["ftd_amount"] > 300).astype(int).to_numpy()

    params = fit_feature_params(raw)
    X = build_features(raw, params, include_ftd_amount)
    model = CatBoostClassifier(iterations=5, depth=3, cat_features=CATEGORICAL_FEATURES, verbose=False)
    model.fit(X, y)

    return {
        "model": model,
        "feature_params": params,
        "threshold": 0.5,
        "include_ftd_amount": include_ftd_amount,
        "feature_columns": list(X.columns),
    }


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.model_path = Path(self.tmpdir.name) / "model.pkl"
        with open(self.model_path, "wb") as f:
            pickle.dump(_toy_artifact(), f)

    def _write_excel(self, raw: pd.DataFrame) -> Path:
        path = Path(self.tmpdir.name) / f"input_{id(raw)}.xlsx"
        raw.to_excel(path, index=False, sheet_name="Sheet1")
        return path

    def test_predict_on_different_row_counts(self):
        for n in [1, 5, 40]:
            raw = make_raw_features(n)
            path = self._write_excel(raw)
            result = predict(path, self.model_path)
            self.assertEqual(len(result), n)
            self.assertTrue({"uid", "vip_probability", "predicted_vip"}.issubset(result.columns))
            self.assertTrue(result["vip_probability"].between(0, 1).all())

    def test_predict_with_unknown_domain_and_missing_values(self):
        raw = make_raw_features(
            6,
            email_domain=["gmail.com", "unknown-brand.io", "wp.pl", "yet-another.xyz", "gmail.com", "o2.pl"],
            avg_bet_size=[1.0, np.nan, 2.0, np.nan, 3.0, 4.0],
            spins_count=[10, np.nan, 20, np.nan, 30, 40],
            app_installed_at=[pd.NaT] * 6,
        )
        path = self._write_excel(raw)
        result = predict(path, self.model_path)
        self.assertEqual(len(result), 6)
        self.assertFalse(result["vip_probability"].isna().any())

    def test_csv_export_contains_only_predicted_vip(self):
        raw = make_raw_features(20)
        path = self._write_excel(raw)
        vip_uids = predict_vip_uids(path, self.model_path)
        full = predict(path, self.model_path)
        self.assertEqual(set(vip_uids), set(full.loc[full["predicted_vip"] == 1, "uid"]))

    def test_predict_with_non_english_column_a_header(self):
        raw = make_raw_features(10).rename(columns={"uid": "ідентифікатор_користувача"})
        path = self._write_excel(raw)
        result = predict(path, self.model_path)
        self.assertEqual(len(result), 10)
        self.assertEqual(list(result["uid"]), [f"user-{i}" for i in range(10)])

    def test_arbitrary_filename_and_headers_single_sheet(self):
        raw = make_raw_features(8).rename(
            columns={
                "uid": "Клієнт №",
                "ftd_at": "дата_першого_депозиту",
                "registered_at": "created",
                "email_domain": "mail_host",
            }
        )
        path = Path(self.tmpdir.name) / "довільна назва 2024 (1).xlsx"
        raw.to_excel(path, index=False, sheet_name="OnlySheet")
        result = predict(path, self.model_path)
        self.assertEqual(len(result), 8)
        self.assertEqual(list(result["uid"]), [f"user-{i}" for i in range(8)])

    def test_empty_file_produces_no_predictions(self):
        raw = make_raw_features(0)
        path = self._write_excel(raw)
        result = predict(path, self.model_path)
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), ["uid", "vip_probability", "predicted_vip"])

    def test_too_few_columns_raises_clear_error(self):
        path = Path(self.tmpdir.name) / "short.xlsx"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(path, index=False, sheet_name="Sheet1")
        with self.assertRaisesRegex(ValueError, "expected 11 columns"):
            predict(path, self.model_path)

    def test_missing_model_file_raises(self):
        raw = make_raw_features(3)
        path = self._write_excel(raw)
        missing_model = Path(self.tmpdir.name) / "does_not_exist.pkl"
        with self.assertRaises(FileNotFoundError):
            predict(path, missing_model)

    def test_invalid_artifact_missing_keys_raises(self):
        bad_model_path = Path(self.tmpdir.name) / "bad_model.pkl"
        with open(bad_model_path, "wb") as f:
            pickle.dump({"model": "not-a-model"}, f)
        with self.assertRaisesRegex(ValueError, "missing keys"):
            load_artifact(bad_model_path)

    def test_invalid_artifact_not_a_dict_raises(self):
        bad_model_path = Path(self.tmpdir.name) / "bad_model2.pkl"
        with open(bad_model_path, "wb") as f:
            pickle.dump([1, 2, 3], f)
        with self.assertRaisesRegex(ValueError, "missing keys"):
            load_artifact(bad_model_path)

    def test_empty_and_duplicate_uid_pass_through_unchanged(self):
        raw = make_raw_features(5)
        raw.loc[1, "uid"] = None
        raw.loc[3, "uid"] = raw.loc[0, "uid"]
        path = self._write_excel(raw)
        result = predict(path, self.model_path)
        self.assertEqual(len(result), 5)
        self.assertTrue(pd.isna(result.loc[1, "uid"]))
        self.assertEqual(result.loc[3, "uid"], result.loc[0, "uid"])

    def test_csv_output_has_single_uid_column_no_index_and_matching_row_count(self):
        raw = make_raw_features(20)
        path = self._write_excel(raw)
        output_path = Path(self.tmpdir.name) / "out.csv"
        argv = ["predict.py", "--input", str(path), "--output", str(output_path), "--model", str(self.model_path)]
        with patch.object(sys, "argv", argv):
            main()
        written = pd.read_csv(output_path)
        full = predict(path, self.model_path)
        expected_count = int((full["predicted_vip"] == 1).sum())
        self.assertEqual(list(written.columns), ["uid"])
        self.assertEqual(len(written), expected_count)

    def test_output_directory_created_if_missing(self):
        raw = make_raw_features(3)
        path = self._write_excel(raw)
        nested_output = Path(self.tmpdir.name) / "nested" / "dir" / "predictions.csv"
        argv = ["predict.py", "--input", str(path), "--output", str(nested_output), "--model", str(self.model_path)]
        with patch.object(sys, "argv", argv):
            main()
        self.assertTrue(nested_output.exists())


if __name__ == "__main__":
    unittest.main()
