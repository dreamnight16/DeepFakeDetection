import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
from test_contracts import pair


class DataContracts(unittest.TestCase):
    def test_uniform_frames_cover_timeline(self):
        from g21.data import uniform_indices
        self.assertEqual(uniform_indices(100, 8), [0, 14, 28, 42, 56, 70, 84, 99])
        self.assertEqual(uniform_indices(2, 8), [0, 1])

    def test_window_identical_across_arms(self):
        from g21.config import DEFAULT_CONFIG, resolve_arm
        from g21.data import make_window
        rows = [pair(f"r{i}", m) for m in DEFAULT_CONFIG["data"]["train_methods"] for i in range(6)]
        plans = [make_window(rows, resolve_arm(DEFAULT_CONFIG, arm, 42), 7) for arm in "ABCD"]
        self.assertTrue(all(p == plans[0] for p in plans))
        for p in plans[0]:
            self.assertTrue(all(i != j for i, j in enumerate(p["permutation"])))
            self.assertTrue(all(len(set(x)) == 2 for x in p["frame_indices"]))

    def test_data_changes_over_updates_but_repeats_on_resume(self):
        from g21.config import DEFAULT_CONFIG
        from g21.data import make_window
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        rows = [pair(f"r{i}", m) for m in cfg["data"]["train_methods"] for i in range(8)]
        self.assertEqual(make_window(rows, cfg, 9), make_window(rows, cfg, 9))
        self.assertNotEqual(make_window(rows, cfg, 8), make_window(rows, cfg, 9))


if __name__ == "__main__":
    unittest.main()
