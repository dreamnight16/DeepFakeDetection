"""Server-only fixture checks and strict legacy result failures."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


class EvaluationContracts(unittest.TestCase):
    def records(self):
        return [
            {
                "dataset_id": "toy",
                "video_id": "real/v1",
                "frame_id": "0",
                "label": 0,
                "fake_probability": 0.1,
            },
            {
                "dataset_id": "toy",
                "video_id": "fake/v1",
                "frame_id": "0",
                "label": 1,
                "fake_probability": 0.9,
            },
        ]

    def test_fixture_auc_ties_and_full_ids(self):
        from g21.evaluation import aggregate_predictions, binary_auc

        rows = self.records()
        videos, metrics = aggregate_predictions(rows, rows)
        self.assertEqual(len(videos), 2)
        self.assertEqual(metrics["video_auc"], 1.0)
        self.assertEqual(binary_auc([0.5, 0.5], [0, 1]), 0.5)

    def test_duplicate_missing_and_label_conflict(self):
        from g21.evaluation import aggregate_predictions

        rows = self.records()
        for invalid in [rows[:1], rows + rows[:1]]:
            with self.assertRaises(ValueError):
                aggregate_predictions(invalid, rows)
        bad = copy.deepcopy(rows)
        bad[0]["label"] = 1
        with self.assertRaises(ValueError):
            aggregate_predictions(bad, rows)

    def test_missing_dataset_does_not_change_mean_definition(self):
        from g21.evaluation import summarize_datasets

        result = summarize_datasets({"DFDC": {"video_auc": 0.9}})
        self.assertIsNone(result["AUC_cross"])
        self.assertIsNone(result["mean7"])


if __name__ == "__main__":
    unittest.main()
