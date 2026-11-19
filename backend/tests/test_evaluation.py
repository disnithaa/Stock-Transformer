import unittest

import numpy as np

from app.evaluation import (
    backtest_long_flat,
    chronological_train_validation_indices,
    direction_metrics,
    evaluation_suite,
    fit_minmax_scaler,
    apply_minmax_scaler,
    regression_metrics,
    wilson_interval,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_reference_models_share_one_evaluation_index(self):
        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(60, 20, 9))
        test_x = rng.normal(size=(6, 20, 9))
        train_y = np.tile([-0.01, 0.008], 30)
        test_y = np.array([0.01, -0.006, 0.004, -0.008, 0.005, -0.003])
        selected = {
            "predictions": (test_y > 0).astype(int),
            "confidence": np.full(6, 0.6),
        }
        output = evaluation_suite(
            train_x, train_y, test_x, test_y,
            np.full(6, 100.0), np.full(6, 100.2), np.full(6, 100.1),
            np.zeros(6), selected, np.full(60, 0.02), np.full(6, 0.02),
        )
        self.assertEqual(output["selected_direction"]["n"], 6)
        self.assertTrue(any(row["model"] == "Persistence · previous close" for row in output["regression_models"]))
        self.assertTrue(any(row["model"] == "Expanding logistic · refit each session" for row in output["direction_models"]))
        self.assertEqual(output["three_class_experiment"]["n"], 6)
        self.assertEqual(output["trading_diagnostic"]["observations"], 6)

    def test_chronological_split_is_contiguous_and_disjoint(self):
        train, validation = chronological_train_validation_indices(10, 0.8)
        self.assertEqual(train.tolist(), list(range(8)))
        self.assertEqual(validation.tolist(), [8, 9])
        self.assertFalse(set(train) & set(validation))

    def test_scaler_bounds_are_fitted_only_on_training_sequences(self):
        train = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        x_min, x_max = fit_minmax_scaler(train)
        validation = np.array([[[1000.0, -1000.0]]])
        transformed = apply_minmax_scaler(validation, x_min, x_max)
        np.testing.assert_allclose(x_min, [1.0, 2.0])
        np.testing.assert_allclose(x_max, [3.0, 4.0])
        self.assertGreater(transformed[0, 0, 0], 1.0)

    def test_binary_metrics_use_explicit_down_up_order(self):
        report = direction_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(report["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(report["accuracy"], 75.0)
        self.assertEqual(report["balanced_accuracy"], 75.0)
        self.assertEqual(report["class_distribution"], {"down": 2, "up": 2})
        lo, hi = report["accuracy_ci95"]
        self.assertLess(lo, 75.0)
        self.assertGreater(hi, 75.0)

    def test_wilson_interval_handles_empty_sample(self):
        self.assertEqual(wilson_interval(0, 0), [None, None])

    def test_regression_metrics_against_known_values(self):
        result = regression_metrics([1, 3], [2, 2])
        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["rmse"], 1.0)
        self.assertEqual(result["r2"], 0.0)

    def test_backtest_delays_execution_one_session(self):
        result = backtest_long_flat([0.05, 0.05, 0.05], [1, 0, 1], cost_bps=0)
        # First signal cannot earn the first observed return; only prior signals act.
        expected = (1 + np.expm1(0.05))
        self.assertAlmostEqual(result["cumulative_return_pct"], (expected - 1) * 100, places=2)
        self.assertEqual(result["turnover_units"], 2.0)


if __name__ == "__main__":
    unittest.main()
