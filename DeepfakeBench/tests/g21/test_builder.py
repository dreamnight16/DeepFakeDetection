import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


def fixture():
    data = {
        label: {s: {"c23": {}} for s in ("train", "val", "test")}
        for label in ("FF-real", "FF-DF", "FF-F2F", "FF-FS", "FF-NT")
    }
    for split, ids in [("train", ["001", "002"]), ("val", ["010", "011"])]:
        for sid in ids:
            data["FF-real"][split]["c23"][sid] = {
                "label": "FF-real",
                "frames": [f"real/{sid}/{t}.png" for t in (0, 10, 20)],
            }
        for m in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT"):
            key = "_".join(ids)
            data[m][split]["c23"][key] = {
                "label": m,
                "frames": [f"{m}/{key}/{t}.png" for t in (0, 20, 30)],
            }
    return {"FaceForensics++": data}


class BuilderContracts(unittest.TestCase):
    def test_mapping_template_never_asserts_verification(self):
        from g21.builder import mapping_template

        self.assertTrue(
            all(not r["verified"] for r in mapping_template()["rules"].values())
        )

    def test_builder_uses_frame_intersection_and_all_four_methods(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = mapping_template()
        for rule in mapping["rules"].values():
            rule.update(
                verified=True,
                reference_component=0,
                same_original_index_verified=True,
                evidence="synthetic test metadata",
            )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "data.json").write_text(json.dumps(fixture()), encoding="utf-8")
            (p / "map.json").write_text(json.dumps(mapping), encoding="utf-8")
            report = build_ffpp(p / "data.json", p / "map.json", p / "out")
            pairs = [
                json.loads(x)
                for x in (p / "out/pairs_train.jsonl").read_text().splitlines()
            ]
            self.assertEqual(report["pair_count"], 4)
            self.assertEqual(
                [f["sample_id"] for f in pairs[0]["frame_pairs"]], ["0", "20"]
            )
            self.assertEqual(
                {x["method"] for x in pairs}, {"FF-DF", "FF-F2F", "FF-FS", "FF-NT"}
            )


def ffpp_fixture(real_train, fake_train):
    """Minimal FF++ JSON with caller-chosen frame indices for every split."""
    data = {
        label: {s: {"c23": {}} for s in ("train", "val", "test")}
        for label in ("FF-real", "FF-DF", "FF-F2F", "FF-FS", "FF-NT")
    }
    for sid, frames in real_train.items():
        data["FF-real"]["train"]["c23"][sid] = {
            "label": "FF-real",
            "frames": [f"real/{sid}/{t}.png" for t in frames],
        }
    for method, entries in fake_train.items():
        for key, frames in entries.items():
            data[method]["train"]["c23"][key] = {
                "label": method,
                "frames": [f"{method}/{key}/{t}.png" for t in frames],
            }
    return {"FaceForensics++": data}


def two_pair_fixture():
    """Four real ids and two fake keys per method so one key can be excluded."""
    real = {sid: (0, 10, 20) for sid in ("001", "002", "003", "004")}
    fake = {
        method: {"001_002": (0, 20, 30), "003_004": (0, 20, 30)}
        for method in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")
    }
    return ffpp_fixture(real, fake)


def verify_all(mapping):
    """Mark every method-level rule verified, the way a human reviewer would."""
    for rule in mapping["rules"].values():
        rule.update(
            verified=True,
            reference_component=0,
            same_original_index_verified=True,
            evidence="synthetic test metadata",
        )
    return mapping


def write_inputs(tmp, data, mapping):
    data_path = tmp / "data.json"
    map_path = tmp / "map.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    map_path.write_text(json.dumps(mapping), encoding="utf-8")
    return data_path, map_path


class BuilderExclusionContracts(unittest.TestCase):
    def test_template_carries_no_opt_out_marker(self):
        from g21.builder import EXCLUDED_KEY, mapping_template

        for rule in mapping_template()["rules"].values():
            self.assertNotIn(EXCLUDED_KEY, rule)
            self.assertFalse(rule["verified"])

    def test_exclusion_marker_on_override_excludes_one_video_and_continues(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = verify_all(mapping_template())
        mapping["overrides"]["FF-DF/001_002"] = {
            "excluded": True,
            "exclude_reason": "manual review: source scene does not match",
            "exclude_detail": {"reviewer": "a"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data_path, map_path = write_inputs(p, two_pair_fixture(), mapping)
            report = build_ffpp(data_path, map_path, p / "out")
            pairs = [
                json.loads(x)
                for x in (p / "out/pairs_train.jsonl").read_text().splitlines()
            ]
        self.assertEqual(report["pair_count"], 7)
        self.assertEqual(report["pairs_by_method"]["FF-DF"], 1)
        self.assertEqual({x["method"] for x in pairs}, {"FF-DF", "FF-F2F", "FF-FS", "FF-NT"})
        self.assertEqual(len(report["exclusions"]), 1)
        record = report["exclusions"][0]
        self.assertEqual(record["method"], "FF-DF")
        self.assertEqual(record["fake_key"], "001_002")
        self.assertEqual(record["reason_code"], "explicit_exclusion")
        self.assertEqual(record["reason"], "manual review: source scene does not match")
        self.assertEqual(record["detail"], {"reviewer": "a"})

    def test_unverified_rule_without_marker_still_raises(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = verify_all(mapping_template())
        mapping["rules"]["FF-F2F"]["verified"] = False
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data_path, map_path = write_inputs(p, two_pair_fixture(), mapping)
            with self.assertRaisesRegex(
                ValueError, "unverified mapping rule: FF-F2F/001_002"
            ):
                build_ffpp(data_path, map_path, p / "out")

    def test_marker_wins_over_unverified_per_video_override(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = verify_all(mapping_template())
        mapping["overrides"]["FF-F2F/001_002"] = {
            "verified": False,
            "evidence": "",
            "excluded": True,
            "exclude_reason": "opted out after review",
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data_path, map_path = write_inputs(p, two_pair_fixture(), mapping)
            report = build_ffpp(data_path, map_path, p / "out")
        self.assertEqual(report["pair_count"], 7)
        self.assertEqual(report["pairs_by_method"]["FF-F2F"], 1)
        self.assertEqual(report["exclusions"][0]["reason_code"], "explicit_exclusion")

    def test_malformed_exclusion_marker_is_rejected(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = verify_all(mapping_template())
        mapping["rules"]["FF-DF"] = {"excluded": True, "verified": False}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data_path, map_path = write_inputs(p, two_pair_fixture(), mapping)
            with self.assertRaisesRegex(ValueError, "exclude_reason"):
                build_ffpp(data_path, map_path, p / "out")


class BuilderFunnelContracts(unittest.TestCase):
    def test_funnel_reports_seen_paired_and_excluded_by_reason(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = verify_all(mapping_template())
        mapping["overrides"]["FF-DF/001_002"] = {
            "excluded": True,
            "exclude_reason": "manual review",
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data_path, map_path = write_inputs(p, two_pair_fixture(), mapping)
            report = build_ffpp(data_path, map_path, p / "out")
        funnel = report["funnel"]
        self.assertEqual(
            funnel["overall"],
            {
                "fake_videos_seen": 8,
                "paired": 7,
                "excluded": 1,
                "excluded_by_reason": {
                    "explicit_exclusion": 1,
                    "insufficient_frame_intersection": 0,
                },
            },
        )
        self.assertEqual(
            funnel["by_method"]["FF-DF"],
            {
                "fake_videos_seen": 2,
                "paired": 1,
                "excluded": 1,
                "excluded_by_reason": {
                    "explicit_exclusion": 1,
                    "insufficient_frame_intersection": 0,
                },
            },
        )
        self.assertEqual(funnel["by_method"]["FF-F2F"]["paired"], 2)
        self.assertEqual(funnel["by_method"]["FF-F2F"]["excluded"], 0)
        self.assertEqual(
            sum(v["fake_videos_seen"] for v in funnel["by_method"].values()),
            funnel["overall"]["fake_videos_seen"],
        )
        self.assertEqual(
            sum(v["paired"] for v in funnel["by_method"].values()),
            funnel["overall"]["paired"],
        )

    def test_insufficient_frame_intersection_is_structured(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = verify_all(mapping_template())
        data = ffpp_fixture(
            {sid: (0, 10, 20) for sid in ("001", "002", "003", "004")},
            {
                method: {"001_002": (50, 60, 70), "003_004": (0, 20, 30)}
                for method in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data_path, map_path = write_inputs(p, data, mapping)
            report = build_ffpp(data_path, map_path, p / "out")
        self.assertEqual(report["pair_count"], 4)
        self.assertEqual(len(report["exclusions"]), 4)
        record = report["exclusions"][0]
        self.assertEqual(record["reason_code"], "insufficient_frame_intersection")
        self.assertEqual(record["reason"], "fewer than two matched frames")
        self.assertEqual(record["real_available"], 3)
        self.assertEqual(record["fake_available"], 3)
        self.assertEqual(record["intersection"], 0)
        self.assertNotIn("matched_moments", record)
        self.assertEqual(
            report["funnel"]["overall"]["excluded_by_reason"][
                "insufficient_frame_intersection"
            ],
            4,
        )

    def test_explicit_frame_matches_records_matched_moments(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = verify_all(mapping_template())
        mapping["overrides"]["FF-NT/001_002"] = {"frame_matches": [[0, 0]]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data_path, map_path = write_inputs(p, two_pair_fixture(), mapping)
            report = build_ffpp(data_path, map_path, p / "out")
        self.assertEqual(report["pair_count"], 7)
        record = report["exclusions"][0]
        self.assertEqual(record["fake_key"], "001_002")
        self.assertEqual(record["matched_moments"], 1)
        self.assertEqual(record["intersection"], 2)

    def test_report_keeps_existing_keys_and_is_byte_deterministic(self):
        from g21.builder import build_ffpp, mapping_template

        mapping = verify_all(mapping_template())
        mapping["overrides"]["FF-FS/003_004"] = {
            "excluded": True,
            "exclude_reason": "manual review",
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data_path, map_path = write_inputs(p, two_pair_fixture(), mapping)
            first = build_ffpp(data_path, map_path, p / "out1")
            second = build_ffpp(data_path, map_path, p / "out2")
            for name in ("pairs_train.jsonl", "manifest_report.json"):
                left = (p / "out1" / name).read_bytes()
                right = (p / "out2" / name).read_bytes()
                self.assertEqual(left, right)
                self.assertEqual(first, second)
        for key in (
            "schema_version",
            "protocol",
            "pair_count",
            "validation_dataset",
            "exclusions",
            "pairs_by_method",
            "source_sha256",
            "funnel",
        ):
            self.assertIn(key, first)

    def test_builder_module_stays_free_of_torch(self):
        import g21.builder as builder_module

        lines = Path(builder_module.__file__).read_text(encoding="utf-8").splitlines()
        imports = [
            line.strip()
            for line in lines
            if line.strip().startswith(("import ", "from "))
        ]
        forbidden = [
            line
            for line in imports
            if any(name in line for name in ("torch", "numpy", "cv2"))
        ]
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
