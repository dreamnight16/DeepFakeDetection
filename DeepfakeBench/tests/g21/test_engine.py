"""Server-only tiny CPU model integration; never downloads or loads real CLIP."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
from test_contracts import pair
from test_numerics import TinyModel


class EngineContracts(unittest.TestCase):
    def config(self, root):
        from g21.config import DEFAULT_CONFIG, resolve_arm

        cfg = resolve_arm(copy.deepcopy(DEFAULT_CONFIG), "D", 12)
        cfg["paths"] = {k: str(root / k) for k in cfg["paths"]}
        cfg["training"].update(
            device="cpu", max_updates=4, val_every_updates=1, save_last_every_updates=1
        )
        return cfg

    def inputs(self, cfg, **kwargs):
        pairs = [
            pair(f"r{i}", m) for m in cfg["data"]["train_methods"] for i in range(6)
        ]
        return (
            pairs,
            [{"fixture": True}],
            {},
            {"input_digest": "fixture", "manifest_hashes": {}},
        )

    def render(self, plan, root):
        generator = torch.Generator().manual_seed(int(plan["digest"][:8], 16))
        return torch.randn(4, 2, 2, 2, 3, 4, 4, generator=generator)

    def evaluate(self, model, *args, **kwargs):
        # Constant scores exercise the earlier-checkpoint tie rule without any real data.
        return {"auc": 0.8, "video_auc": 0.9}

    def test_exact_update_boundary_resume_matches_continuous(self):
        from g21.engine import train_arm

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root)
            with (
                patch("g21.engine.inspect_inputs", self.inputs),
                patch("g21.engine.render_plan", self.render),
                patch("g21.engine.evaluate_model", self.evaluate),
            ):
                train_arm(
                    cfg, root / "continuous", model_factory=lambda _: TinyModel(0.2)
                )
                train_arm(
                    cfg,
                    root / "resumed",
                    model_factory=lambda _: TinyModel(0.2),
                    stop_after=2,
                )
                train_arm(
                    cfg,
                    root / "resumed",
                    model_factory=lambda _: TinyModel(0.2),
                    resume=True,
                )
            a = torch.load(root / "continuous/checkpoints/last.pth", weights_only=True)
            b = torch.load(root / "resumed/checkpoints/last.pth", weights_only=True)
            self.assertEqual(a["best_update"], 1)
            self.assertEqual(a["completed_updates"], b["completed_updates"])
            for key in a["model_state_dict"]:
                torch.testing.assert_close(
                    a["model_state_dict"][key],
                    b["model_state_dict"][key],
                    atol=0,
                    rtol=0,
                )
            self.assertEqual(a["counters"], b["counters"])

    def test_truncated_tail_can_resume_but_middle_corruption_is_rejected(self):
        from g21.engine import _truncate_log

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "log.jsonl"
            p.write_text('{"update":1}\n{"update":2}\n{"upd', encoding="utf-8")
            _truncate_log(p, 1)
            self.assertEqual(json.loads(p.read_text()), {"update": 1})
            p.write_text('bad\n{"update":1}\n', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                _truncate_log(p, 1)

    def test_missing_resume_does_not_start_new_training(self):
        from g21.engine import train_arm

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp))
            with self.assertRaisesRegex(ValueError, "last.pth"):
                train_arm(cfg, Path(tmp) / "missing", resume=True)


if __name__ == "__main__":
    unittest.main()
