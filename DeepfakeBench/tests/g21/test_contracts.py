"""Public G21 contracts; no real model or training data is needed."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


def pair(source="r0", method="FF-DF"):
    return {
        "schema_version": 1, "pair_id": f"{method}/{source}", "split": "train",
        "method": method, "content_reference_id": source,
        "lineage_source_ids": [source], "real_video_id": f"real/{source}",
        "fake_video_id": f"{method}/{source}", "mapping_verified": True,
        "mapping_evidence": "test fixture explicit mapping",
        "frame_pairs": [{"sample_id": str(t), "real_path": f"r/{source}/{t}.png",
                         "fake_path": f"f/{method}/{source}/{t}.png"} for t in range(3)],
    }


class ManifestContracts(unittest.TestCase):
    def test_unverified_mapping_is_rejected(self):
        from g21.manifest import validate_pair
        item = pair()
        item["mapping_verified"] = False
        with self.assertRaisesRegex(ValueError, "mapping"):
            validate_pair(item, min_frames=2)

    def test_path_and_identity_contracts(self):
        from g21.manifest import validate_pair
        validate_pair(pair(), min_frames=2)
        item = pair()
        item["frame_pairs"][0]["real_path"] = "../outside.png"
        with self.assertRaises(ValueError):
            validate_pair(item, min_frames=2)

    def test_split_lineage_is_checked(self):
        from g21.manifest import validate_split_isolation
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_isolation([pair()], [{"video_id": "val/v", "label": 0,
                "lineage_source_ids": ["r0"]}])

    def test_jsonl_duplicates_are_not_silent(self):
        from g21.manifest import load_pairs
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pairs.jsonl"
            p.write_text("\n".join([json.dumps(pair())] * 2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_pairs(p, min_frames=2)


class ConfigurationContracts(unittest.TestCase):
    def test_all_arms_share_window_and_data(self):
        from g21.config import DEFAULT_CONFIG, resolve_arm
        arms = [resolve_arm(copy.deepcopy(DEFAULT_CONFIG), arm, 1024) for arm in "ABCD"]
        for cfg in arms:
            self.assertEqual(cfg["training"]["gradient_mode"], "two_pass_replay")
            self.assertEqual(cfg["data"], arms[0]["data"])
        self.assertEqual(arms[0]["loss"]["pair_lambda"], 0)
        self.assertEqual(arms[2]["loss"]["pairing"], "shuffled")
        self.assertEqual(arms[3]["loss"]["pair_mode"], "group_softmax")

    def test_unknown_arm_is_not_ignored(self):
        from g21.config import DEFAULT_CONFIG, resolve_arm
        with self.assertRaises(ValueError):
            resolve_arm(DEFAULT_CONFIG, "E", 1024)


if __name__ == "__main__":
    unittest.main()
