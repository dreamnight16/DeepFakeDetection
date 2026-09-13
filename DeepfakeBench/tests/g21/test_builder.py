import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


def fixture():
    data = {label: {s: {"c23": {}} for s in ("train", "val", "test")}
            for label in ("FF-real", "FF-DF", "FF-F2F", "FF-FS", "FF-NT")}
    for split, ids in [("train", ["001", "002"]), ("val", ["010", "011"])]:
        for sid in ids:
            data["FF-real"][split]["c23"][sid] = {"label": "FF-real", "frames": [f"real/{sid}/{t}.png" for t in (0, 10, 20)]}
        for m in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT"):
            key = "_".join(ids)
            data[m][split]["c23"][key] = {"label": m, "frames": [f"{m}/{key}/{t}.png" for t in (0, 20, 30)]}
    return {"FaceForensics++": data}


class BuilderContracts(unittest.TestCase):
    def test_mapping_template_never_asserts_verification(self):
        from g21.builder import mapping_template
        self.assertTrue(all(not r["verified"] for r in mapping_template()["rules"].values()))

    def test_builder_uses_frame_intersection_and_val_split(self):
        from g21.builder import build_ffpp, mapping_template
        mapping = mapping_template()
        for rule in mapping["rules"].values():
            rule.update(verified=True, reference_component=0, same_original_index_verified=True, evidence="synthetic test metadata")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "data.json").write_text(json.dumps(fixture()), encoding="utf-8")
            (p / "map.json").write_text(json.dumps(mapping), encoding="utf-8")
            report = build_ffpp(p / "data.json", p / "map.json", p / "out")
            pairs = [json.loads(x) for x in (p / "out/pairs_train.jsonl").read_text().splitlines()]
            self.assertEqual(report["pair_count"], 3)
            self.assertEqual([f["sample_id"] for f in pairs[0]["frame_pairs"]], ["0", "20"])
            self.assertEqual({x["method"] for x in pairs}, {"FF-DF", "FF-F2F", "FF-FS"})


if __name__ == "__main__":
    unittest.main()
