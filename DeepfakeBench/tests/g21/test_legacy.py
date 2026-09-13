"""Run on the server: compatibility parsing/config tests do not run a CLIP model."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


class LegacyCompatibility(unittest.TestCase):
    def log(self):
        from g21.config import TEST_DATASETS

        names = [*TEST_DATASETS, "average (over 8 datasets: historical)"]
        return "\n".join(
            f"dataset: {n}\nacc: 0.8\nauc: 0.9\nvideo_auc: 0.95" for n in names
        )

    def test_all_eight_and_average_are_required(self):
        from g21.legacy import parse_testall_log

        results = parse_testall_log(self.log())
        self.assertEqual(len(results), 9)
        self.assertEqual(results["DFDC"]["video_auc"], 0.95)
        with self.assertRaises(ValueError):
            parse_testall_log(
                self.log().replace(
                    "dataset: DFDC\nacc: 0.8\nauc: 0.9\nvideo_auc: 0.95", ""
                )
            )

    def test_child_failure_is_not_hidden_by_parent_exit_zero(self):
        from g21.legacy import parse_testall_log

        with self.assertRaises(ValueError):
            parse_testall_log(
                self.log() + "\n[WARNING] test.py exited with code 1 for DFDC"
            )

    def test_original_loader_configuration_is_retained(self):
        from g21.config import DEFAULT_CONFIG
        from g21.legacy import legacy_config

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["paths"].update(
            data_root="/example/data",
            dataset_json_folder="/example/index",
            clip_pretrained_path="/example/clip",
        )
        legacy = legacy_config(cfg)
        self.assertEqual(legacy["frame_num"]["test"], 8)
        self.assertFalse(legacy["multi_crop"])
        self.assertEqual(legacy["metric_scoring"], "auc")
        self.assertEqual(legacy["test_dataset"], ["Celeb-DF-v2"])
        self.assertEqual(legacy["dataset_json_folder"], "/example/index")


if __name__ == "__main__":
    unittest.main()
