import unittest

import numpy as np

from src.evaluate import (
    evaluate_predictions,
    min_count_for_recall,
    recall_at_ratio,
    select_threshold,
)


class RecallAtRatioTests(unittest.TestCase):
    def test_basic(self):
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.3, 0.1])
        recall, k = recall_at_ratio(y_true, y_proba, 1.0)
        self.assertEqual(k, 2)
        self.assertAlmostEqual(recall, 1.0)

    def test_zero_positives_returns_nan(self):
        y_true = np.array([0, 0, 0])
        y_proba = np.array([0.5, 0.3, 0.1])
        recall, k = recall_at_ratio(y_true, y_proba, 1.0)
        self.assertTrue(np.isnan(recall))
        self.assertEqual(k, 0)

    def test_negative_ratio_raises(self):
        with self.assertRaises(ValueError):
            recall_at_ratio(np.array([1, 0]), np.array([0.5, 0.4]), -1)

    def test_k_clamped_to_length(self):
        y_true = np.array([1, 0])
        y_proba = np.array([0.5, 0.4])
        recall, k = recall_at_ratio(y_true, y_proba, 10)
        self.assertEqual(k, 2)
        self.assertAlmostEqual(recall, 1.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            recall_at_ratio(np.array([]), np.array([]), 1.0)

    def test_mismatched_length_raises(self):
        with self.assertRaises(ValueError):
            recall_at_ratio(np.array([1, 0]), np.array([0.5]), 1.0)


class MinCountForRecallTests(unittest.TestCase):
    def test_basic(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertEqual(min_count_for_recall(y_true, y_proba, min_recall=0.8), 2)

    def test_zero_positives_raises(self):
        with self.assertRaises(ValueError):
            min_count_for_recall(np.array([0, 0, 0]), np.array([0.5, 0.3, 0.1]))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            min_count_for_recall(np.array([]), np.array([]))

    def test_invalid_min_recall_raises(self):
        y_true = np.array([1, 0])
        y_proba = np.array([0.5, 0.4])
        with self.assertRaises(ValueError):
            min_count_for_recall(y_true, y_proba, min_recall=1.5)
        with self.assertRaises(ValueError):
            min_count_for_recall(y_true, y_proba, min_recall=-0.1)

    def test_full_recall_requires_full_count(self):
        y_true = np.array([1, 0, 1])
        y_proba = np.array([0.9, 0.5, 0.1])
        # the second positive is ranked last -> need all 3 rows for recall 1.0
        self.assertEqual(min_count_for_recall(y_true, y_proba, min_recall=1.0), 3)


class SelectThresholdTests(unittest.TestCase):
    def test_basic_no_ties(self):
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.95, 0.9, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
        threshold = select_threshold(y_true, y_proba, min_recall=0.8, max_count_ratio_deviation=0.1)
        self.assertAlmostEqual(threshold, 0.9)
        predicted = (y_proba >= threshold).astype(int)
        self.assertEqual(predicted.sum(), 2)
        self.assertEqual((predicted & y_true).sum(), 2)

    def test_ties_at_cutoff_do_not_understate_predicted_count(self):
        # 4 rows tied at the top score, only 3 of them are true positives.
        # A naive rank-based search could believe count=3 is achievable;
        # `proba >= threshold` actually selects all 4 tied rows.
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.9, 0.9, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        threshold = select_threshold(y_true, y_proba, min_recall=0.8, max_count_ratio_deviation=0.1)
        self.assertAlmostEqual(threshold, 0.9)
        predicted = (y_proba >= threshold).astype(int)
        # realized count honestly reflects the tie - it is 4, not the 3 a
        # naive rank-based count would have assumed.
        self.assertEqual(predicted.sum(), 4)
        self.assertEqual((predicted & y_true).sum(), 3)

    def test_zero_positives_raises(self):
        with self.assertRaises(ValueError):
            select_threshold(np.array([0, 0, 0]), np.array([0.5, 0.3, 0.1]))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            select_threshold(np.array([]), np.array([]))

    def test_mismatched_length_raises(self):
        with self.assertRaises(ValueError):
            select_threshold(np.array([1, 0]), np.array([0.5]))

    def test_invalid_min_recall_raises(self):
        y_true = np.array([1, 0])
        y_proba = np.array([0.5, 0.4])
        with self.assertRaises(ValueError):
            select_threshold(y_true, y_proba, min_recall=1.5)

    def test_invalid_max_count_ratio_deviation_raises(self):
        y_true = np.array([1, 0])
        y_proba = np.array([0.5, 0.4])
        with self.assertRaises(ValueError):
            select_threshold(y_true, y_proba, max_count_ratio_deviation=-0.1)


class EvaluatePredictionsTests(unittest.TestCase):
    def test_basic(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["actual_vip_count"], 2)
        self.assertEqual(metrics["predicted_vip_count"], 2)
        self.assertFalse(metrics["meets_recall_requirement"])
        self.assertTrue(metrics["meets_count_requirement"])

    def test_zero_actual_vip(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 1, 0])
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertEqual(metrics["actual_vip_count"], 0)
        self.assertTrue(np.isnan(metrics["predicted_to_actual_ratio"]))
        self.assertFalse(metrics["meets_count_requirement"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(np.array([]), np.array([]))

    def test_mismatched_length_raises(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(np.array([1, 0]), np.array([1]))


if __name__ == "__main__":
    unittest.main()
