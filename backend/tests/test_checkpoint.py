import unittest
import json
from pathlib import Path

import numpy as np

from app.data import REAL_DATA_SOURCE
from app.direction import classifier_input
from app.inference import _load_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_real_checkpoint_loads_transformer_and_direction_head(self):
        checkpoint = _load_checkpoint("INFY")
        self.assertEqual(checkpoint["data_source"], REAL_DATA_SOURCE)
        self.assertIsNotNone(checkpoint.get("model"))
        self.assertIsNotNone(checkpoint.get("direction_model"))

        sample = np.zeros((1, 20, checkpoint["n_features"]), dtype=float)
        first = checkpoint["direction_model"].predict(
            classifier_input(sample, checkpoint["direction_flatten"])
        )
        second = checkpoint["direction_model"].predict(
            classifier_input(sample, checkpoint["direction_flatten"])
        )
        np.testing.assert_array_equal(first, second)
        self.assertIn(int(first[0]), (0, 1))

    def test_dashboard_payload_metrics_match_selected_model_and_dates(self):
        base = Path(__file__).resolve().parents[1] / "precomputed"
        for ticker in ("INFY", "HUL"):
            with (base / f"{ticker}.json").open(encoding="utf-8") as stream:
                payload = json.load(stream)
            evaluation = payload["evaluation"]
            self.assertEqual(round(payload["metrics"]["DA"], 2), evaluation["selected_direction"]["accuracy"])
            self.assertEqual(payload["evaluation_metadata"]["observations"], len(payload["test_dates"]))
            self.assertEqual(payload["test_dates"][0], payload["evaluation_metadata"]["evaluation_start"])
            self.assertEqual(payload["test_dates"][-1], payload["evaluation_metadata"]["evaluation_end"])


if __name__ == "__main__":
    unittest.main()
