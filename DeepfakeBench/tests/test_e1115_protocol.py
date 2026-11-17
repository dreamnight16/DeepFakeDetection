import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "experiments"))

from e1115.statistics import build_moments
from e1115.evaluation import aggregate_video, calibrate, fusion_score, evaluate_scores


def test_temporal_comparisons_share_matched_sources():
    audit = {"complete_sources": ["r"], "real_frames": {"fit": {"r": []}}, "fit_videos": []}
    lookup = {}
    for i in range(4):
        p = f"r/{i}.png"
        audit["real_frames"]["fit"]["r"].append({"frame": i, "path": p})
        lookup[p] = np.array([0., float(i)])
    for m in ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]:
        frames = []
        for i in range(4):
            p = f"{m}/{i}.png"
            frames.append({"frame": i, "fake_path": p})
            lookup[p] = lookup[f"r/{i}.png"] + np.array([(-1.) ** i, 0.])
        audit["fit_videos"].append({"source": "r", "method": m, "frames": frames})
    moments, info = build_moments(audit, lookup)
    np.testing.assert_allclose(moments["matched_frame"], moments["matched_mean"] + moments["matched_variation"])
    assert moments["mean"][0, 0] == 0 and moments["block4"][0, 0] == 1
    assert info["matched_sources"] == ["r"]


def test_baseline_zero_fusion_is_exact_and_threshold_uses_real_videos():
    p = np.array([.2, .4, .7, .9])
    np.testing.assert_array_equal(fusion_score(p, np.arange(4), np.arange(10), 0), p)
    rows = {"labels": np.array([0, 0, 1, 1]), "video_id": np.array(["r", "r", "f", "f"]),
            "cls_prob": p, "path": np.array(["r/0", "r/1", "f/0", "f/1"])}
    ids, labels, scores = aggregate_video(rows, p)
    assert dict(zip(ids, scores))["r"] == pytest.approx(.3)
    out = evaluate_scores(rows, p, {"0.1": .5})
    assert out["video_auc_fullpath"] == 1 and out["operating_points"]["0.1"]["FPR"] == 0
    assert calibrate(np.arange(30), .1) == 27
    assert calibrate(np.arange(4), .05) is None


def test_full_cli_is_dependency_light_and_all_is_default():
    from run_e1115 import build_parser, experiment_plan
    args = build_parser().parse_args(["--dry_run"])
    design = experiment_plan(args)
    assert design["groups"] == ["sampling", "geometry", "adaptation", "baseline"]
    assert "FULL_REAL70_COVERAGE" in design["train_arms"]
    assert design["development_gate"] is False


def test_float32_exports_keep_temporal_identity():
    rng = np.random.default_rng(19)
    audit = {"complete_sources": ["a", "b"], "real_frames": {"fit": {}}, "fit_videos": []}
    lookup = {}
    for source in audit["complete_sources"]:
        audit["real_frames"]["fit"][source] = []
        for i in range(31):
            path = f"{source}/{i}"
            lookup[path] = rng.normal(size=16).astype(np.float32)
            audit["real_frames"]["fit"][source].append({"frame": i, "path": path})
        for method in ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]:
            rows = []
            for i in range(31):
                path = f"{source}/{method}/{i}"
                lookup[path] = rng.normal(size=16).astype(np.float32)
                rows.append({"frame": i, "fake_path": path})
            audit["fit_videos"].append({"source": source, "method": method, "frames": rows})
    moments, _ = build_moments(audit, lookup)
    np.testing.assert_allclose(moments["block4"], moments["block_mean4"] + moments["variation4"], rtol=1e-10, atol=1e-10)
