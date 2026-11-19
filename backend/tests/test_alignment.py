import unittest

import numpy as np
import pandas as pd

from app.features import FEATURES, compute_features, make_sequences


class TargetAlignmentTests(unittest.TestCase):
    def test_window_ends_at_known_close_and_target_is_next_close_return(self):
        dates = pd.bdate_range("2024-01-01", periods=120)
        close = 100 * np.exp(np.cumsum(np.linspace(-0.002, 0.003, len(dates))))
        raw = pd.DataFrame({"Date": dates, "Close": close})
        features = compute_features(raw)
        window = 10
        X, y, target_dates, anchor_close = make_sequences(features, window)

        first_origin_row = window - 1
        first_target_row = window
        self.assertEqual(pd.Timestamp(target_dates[0]), features.loc[first_target_row, "Date"])
        self.assertAlmostEqual(anchor_close[0], features.loc[first_origin_row, "Close"])
        self.assertAlmostEqual(
            y[0], np.log(features.loc[first_target_row, "Close"] / features.loc[first_origin_row, "Close"])
        )
        np.testing.assert_allclose(X[0], features.loc[: first_origin_row, FEATURES].to_numpy())

    def test_live_feature_window_uses_only_observed_rows(self):
        dates = pd.bdate_range("2024-01-01", periods=120)
        close = np.linspace(100, 120, len(dates))
        features = compute_features(pd.DataFrame({"Date": dates, "Close": close}), require_target=False)
        self.assertTrue(pd.isna(features.iloc[-1]["target_ret"]))
        self.assertTrue(np.isfinite(features.iloc[-1][FEATURES].to_numpy(dtype=float)).all())

    def test_missing_close_does_not_leave_nonfinite_training_features(self):
        dates = pd.bdate_range("2024-01-01", periods=120)
        close = np.linspace(100, 120, len(dates))
        close[60] = np.nan
        features = compute_features(pd.DataFrame({"Date": dates, "Close": close}))
        self.assertTrue(np.isfinite(features[FEATURES + ["target_ret"]].to_numpy(dtype=float)).all())

if __name__ == "__main__":
    unittest.main()
