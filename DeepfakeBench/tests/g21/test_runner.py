"""Server-side runner metadata tests; no subprocess training is launched."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def runner_module():
    path = Path(__file__).resolve().parents[2] / "experiments/run_g21.py"
    spec = importlib.util.spec_from_file_location("g21_runner_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunnerContracts(unittest.TestCase):
    def test_different_scientific_config_is_rejected(self):
        runner = runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "protocol": "g18_g19_legacy_testall_v1",
                "arms": ["A"],
                "seeds": [1],
                "input_digest": "data",
                "code_digest": "code",
                "arm_config_digests": {"1/A": "original_lr"},
            }
            (root / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            out = root / "seed_1/A"
            out.mkdir(parents=True)
            record = {
                "status": "OK",
                "exp_name": "G21/A",
                "seed": 1,
                "protocol": plan["protocol"],
                "input_digest": "data",
                "code_digest": "code",
                "config_digest": "changed_lr",
                "initial_state_digest": "initial",
                "completed_updates": 2,
                "counters": {"forward_images": 512, "backward_images": 256},
                "testall": {"DFDC": {"video_auc": 0.9}},
            }
            p = out / "result.json"
            p.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.aggregate(root)
            record["config_digest"] = "original_lr"
            p.write_text(json.dumps(record), encoding="utf-8")
            result = runner.aggregate(root)
            self.assertTrue(result["all_completed"])
            self.assertIsNone(result["seed_summary"]["A"]["DFDC"]["seed_std"])


if __name__ == "__main__":
    unittest.main()
