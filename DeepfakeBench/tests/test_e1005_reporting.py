"""Evidence gates and identity-aware same-seed comparison for E1005."""

import copy
import csv
import importlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
DOMAINS = ("WDF", "FFIW", "DeepFakeDetection", "DFDC", "DFDCP", "DeeperForensics-1.0")


def reporting():
    return importlib.import_module("e1005_reporting")


def reports(deltas=None, fpr_deltas=None):
    deltas = [.01, .009, .008, .006, 0., -.003] if deltas is None else deltas
    fpr_deltas = [.01] * 6 if fpr_deltas is None else fpr_deltas
    return {name: {"video_auc": .8 + delta, "video_fpr_at_val_threshold": .05 + fpr_delta,
                   "baseline": {"video_auc": .8, "video_fpr_at_val_threshold": .05},
                   "ranking_changes": {"repair": .02, "damage": .02 - delta, "delta_auc": delta}}
            for name, delta, fpr_delta in zip(DOMAINS, deltas, fpr_deltas)}


def assess(data=None, **kwargs):
    module = reporting()
    assert hasattr(module, "assess_breakthrough"), "Breakthrough assessment implementation is missing"
    arguments = {"bootstrap": {"ci95": [.001, .02]}, "reproduction": {"passed": True},
                 "ordinary_comparison": {"ordinary_baseline_id": "G33_VIDEO_B0", "mean_delta": .002,
                                         "ci95": [.0001, .004]}}
    arguments.update(kwargs)
    return module.assess_breakthrough(reports() if data is None else data, **arguments)


def export(scores=(.1, .4, .3, .8)):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.array([0, 0, 1, 1]) if len(scores) == 4 else np.r_[np.zeros(len(scores) - 1, dtype=int), 1]
    video_id = np.array([f"video/{i}" for i in range(len(scores))])
    return {"labels": labels, "path": np.array([f"{video}/0.png" for video in video_id]),
            "video_id": video_id, "cls_prob": np.full(len(scores), .5),
            "global_log_odds": np.zeros(len(scores)), "score": scores}


def compare(before, after, first_step=5750, repeat_step=5750):
    module = reporting()
    assert hasattr(module, "compare_reproduction"), "Reproduction comparison implementation is missing"
    return module.compare_reproduction(before, after, first_step, repeat_step)


def test_reused_higher_development_score_can_win_champion_selection():
    rows = {"G32_J_M": {"status": "OK", "primary_auc": .9, "selected_step": 5750,
                         "trainable_parameters": 2048},
            "G35_MID_LOCAL_SWIGLU": {"status": "REUSED", "primary_auc": .95, "selected_step": 5750,
                                      "trainable_parameters": 3072},
            "G33_VIDEO_B0": {"status": "FAILED", "primary_auc": .99, "selected_step": 100,
                              "trainable_parameters": 1024}}
    assert reporting().choose_champion(rows) == "G35_MID_LOCAL_SWIGLU"
    assert reporting().choose_champion(rows, allowed_families={"G32"}) == "G32_J_M"
    rows["G32_J_M"]["primary_auc"] = .95
    rows["G32_J_M"]["selected_step"] = 5175
    assert reporting().choose_champion(rows) == "G32_J_M"
    rows["G35_MID_LOCAL_SWIGLU"]["selected_step"] = 5175
    assert reporting().choose_champion(rows) == "G32_J_M"


def test_all_auc_gates_and_reproduction_allow_claim_at_exact_mean_threshold():
    result = assess()
    assert result["auc_breakthrough"]
    assert result["cross_domain_fpr_controlled"]
    assert result["can_claim_breakthrough"]
    assert result["method_increment_supported"] and result["can_claim_method_increment"]
    assert result["can_claim_base_improvement"]
    assert result["mean6_delta_auc"] == pytest.approx(.005)
    assert result["nondecreasing_domains"] == 5


def test_missing_ordinary_comparison_preserves_b0_evidence_without_full_claim():
    result = assess(ordinary_comparison=None)
    assert result["auc_breakthrough"] and result["can_claim_base_improvement"]
    assert not result["can_claim_breakthrough"]
    assert not result["method_increment_supported"]
    assert any("ordinary" in limitation.lower() for limitation in result["limitations"])


@pytest.mark.parametrize("mean_delta,ci95,reference_recipe,full_claim,increment", [
    (-.001, [-.002, 0.], False, False, False),
    (0., [-.001, .001], False, True, False),
    (.002, [0., .004], False, True, False),
    (.002, [.0001, .004], False, True, True),
    (.002, [.0001, .004], True, True, False),
])
def test_ordinary_baseline_gate_distinguishes_breakthrough_from_new_method_increment(
        mean_delta, ci95, reference_recipe, full_claim, increment):
    comparison = {"ordinary_baseline_id": "G33_VIDEO_B0", "mean_delta": mean_delta,
                  "ci95": ci95, "champion_is_reference_recipe": reference_recipe}
    result = assess(ordinary_comparison=comparison)
    assert result["auc_breakthrough"] and result["can_claim_base_improvement"]
    assert result["can_claim_breakthrough"] is full_claim
    assert result["method_increment_supported"] is increment
    assert result["can_claim_method_increment"] is increment
    assert result["ordinary_comparison"] == comparison


def test_statistical_method_increment_still_needs_formal_repeat_for_a_claim():
    result = assess(formal_budget=False, reproduction={"passed": False})
    assert result["method_increment_supported"]
    assert not result["can_claim_method_increment"]
    assert not result["can_claim_base_improvement"]


@pytest.mark.parametrize("comparison", [
    "unverified", {}, {"ordinary_baseline_id": "G33_VIDEO_B0", "mean_delta": math.nan, "ci95": [0., .001]},
    {"ordinary_baseline_id": "G33_VIDEO_B0", "mean_delta": 0., "ci95": [.001]},
    {"ordinary_baseline_id": "G33_VIDEO_B0", "mean_delta": 0., "ci95": [.002, .001]},
    {"ordinary_baseline_id": "G33_VIDEO_B0", "mean_delta": 0., "ci95": [0., .001],
     "champion_is_reference_recipe": "True"},
])
def test_invalid_ordinary_comparison_cannot_support_a_claim(comparison):
    with pytest.raises(ValueError):
        assess(ordinary_comparison=comparison)


def test_inclusive_auc_drop_and_fpr_boundaries_survive_float_subtraction_roundoff():
    result = assess(reports([-.01, .015, .015, .01, 0., 0.], [.02] * 6))
    assert result["auc_breakthrough"]
    assert result["cross_domain_fpr_controlled"]


@pytest.mark.parametrize("deltas,gate", [([.004] * 6, "mean"),
                                        ([-.005] * 3 + [.035] * 3, "four"),
                                        ([-.02] + [.02] * 5, "domain")])
def test_each_auc_gate_independently_blocks_a_breakthrough(deltas, gate):
    result = assess(reports(deltas))
    assert not result["auc_breakthrough"]
    assert not result["can_claim_breakthrough"]
    assert any(gate in limitation.lower() for limitation in result["limitations"])


@pytest.mark.parametrize("lower", [0., -.001])
def test_bootstrap_lower_bound_must_be_strictly_positive(lower):
    result = assess(bootstrap={"ci95": [lower, .02]})
    assert not result["auc_breakthrough"]
    assert not result["can_claim_breakthrough"]


def test_auc_claim_and_fpr_control_are_separate_decisions():
    result = assess(reports(fpr_deltas=[.021] + [.0] * 5))
    assert result["auc_breakthrough"] and result["can_claim_breakthrough"]
    assert not result["cross_domain_fpr_controlled"]


@pytest.mark.parametrize("kwargs,reason", [({"formal_budget": False}, "budget"),
                                          ({"reproduction": {"passed": False}}, "reproduction"),
                                          ({"reproduction": {"passed": 1}}, "reproduction"),
                                          ({"reproduction": None}, "reproduction")])
def test_smoke_budget_or_unverified_reproduction_prevents_claim(kwargs, reason):
    result = assess(**kwargs)
    assert result["auc_breakthrough"]
    assert not result["can_claim_breakthrough"]
    assert any(reason in limitation.lower() for limitation in result["limitations"])


@pytest.mark.parametrize("change", ["missing", "extra", "alias"])
def test_assessment_rejects_an_incomplete_or_renamed_six_domain_panel(change):
    data = reports()
    if change == "missing":
        del data["WDF"]
    elif change == "extra":
        data["CDF"] = data["WDF"]
    else:
        data["DFD"] = data.pop("DeepFakeDetection")
    with pytest.raises(ValueError, match="six|domain|dataset"):
        assess(data)


@pytest.mark.parametrize("change", ["auc_nan", "fpr_inf", "auc_range", "missing_baseline", "invalid_ci"])
def test_assessment_rejects_invalid_numeric_evidence(change):
    data = reports()
    kwargs = {}
    if change == "auc_nan":
        data["WDF"]["video_auc"] = math.nan
    elif change == "fpr_inf":
        data["WDF"]["baseline"]["video_fpr_at_val_threshold"] = math.inf
    elif change == "auc_range":
        data["WDF"]["video_auc"] = 1.1
    elif change == "missing_baseline":
        del data["WDF"]["baseline"]
    else:
        kwargs["bootstrap"] = {"ci95": [.03, .01]}
    with pytest.raises(ValueError):
        assess(data, **kwargs)


def test_identical_repeat_records_bitwise_predictions_auc_and_selection_step():
    before = {domain: export() for domain in DOMAINS}
    result = compare(before, copy.deepcopy(before))
    assert result["passed"] and result["bitwise_equal"]
    assert result["max_abs_difference"] == 0.
    assert result["macro_auc_difference"] == 0.
    assert result["selected_step_consistent"]
    assert result["datasets"]["WDF"]["reference_video_auc"] == .75
    assert result["individual_auc_differences"] == {domain: 0. for domain in DOMAINS}


def test_numerically_reproducible_repeat_can_differ_bitwise_and_in_selected_step():
    before = {"WDF": export()}
    after = copy.deepcopy(before)
    after["WDF"]["score"][0] += 1e-9
    result = compare(before, after, repeat_step=5175)
    assert result["passed"]
    assert not result["bitwise_equal"]
    assert not result["selected_step_consistent"]
    assert result["reference_selected_step"] == 5750 and result["repeated_selected_step"] == 5175
    assert result["max_abs_difference"] == pytest.approx(1e-9)


def test_signed_zero_score_difference_is_equal_numerically_but_not_bitwise():
    before = {"WDF": export([0., .4, .3, .8])}
    after = copy.deepcopy(before)
    after["WDF"]["score"][0] = -0.
    result = compare(before, after)
    assert result["passed"] and not result["bitwise_equal"]
    assert result["max_abs_difference"] == 0.


def test_repeat_auc_thresholds_use_macro_and_per_domain_differences():
    # One fake against 1000 distinct reals: moving .5000 to .5010 repairs one
    # of 1000 comparisons, so domain AUC increases by exactly .001.
    before = {domain: export(np.r_[np.arange(1000) / 1000., .5]) for domain in DOMAINS}
    after = copy.deepcopy(before)
    for domain in DOMAINS[:3]:
        after[domain]["score"][-1] = .501
    boundary = compare(before, after)
    assert boundary["passed"]
    assert boundary["macro_auc_difference"] == pytest.approx(.0005)
    for domain in DOMAINS[3:]:
        after[domain]["score"][-1] = .501
    over_macro = compare(before, after)
    assert not over_macro["passed"]
    assert over_macro["macro_auc_difference"] == pytest.approx(.001)
    after = copy.deepcopy(before)
    after["WDF"]["score"][-1] = .502
    over_domain = compare(before, after)
    assert not over_domain["passed"]
    assert abs(over_domain["macro_auc_difference"]) < .0005
    assert over_domain["individual_auc_differences"]["WDF"] == pytest.approx(.002)


@pytest.mark.parametrize("key", ["labels", "path", "video_id", "cls_prob", "global_log_odds"])
def test_repeat_rejects_changed_frame_identity_or_original_b0_arrays(key):
    before = {"WDF": export()}
    after = copy.deepcopy(before)
    if key in {"path", "video_id"}:
        after["WDF"][key][0] = "other"
    elif key == "labels":
        after["WDF"][key][0] = 1
    else:
        after["WDF"][key][0] += .01
    with pytest.raises(ValueError, match="identity|B0|paired"):
        compare(before, after)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1.1])
def test_repeat_rejects_invalid_score_arrays_even_when_identity_matches(value):
    before = {"WDF": export()}
    after = copy.deepcopy(before)
    after["WDF"]["score"][0] = value
    with pytest.raises(ValueError, match="finite|probabilit"):
        compare(before, after)


def test_repeat_rejects_missing_domains_and_malformed_base_arrays():
    with pytest.raises(ValueError, match="dataset|domain"):
        compare({"WDF": export()}, {"FFIW": export()})
    before = {"WDF": export()}
    del before["WDF"]["global_log_odds"]
    with pytest.raises(ValueError, match="global_log_odds|B0"):
        compare(before, copy.deepcopy(before))
    before = {"WDF": export()}
    before["WDF"]["cls_prob"][0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        compare(before, copy.deepcopy(before))


def test_analysis_csv_preserves_g_ids_aliases_and_json_metric_values_atomically(tmp_path):
    module = reporting()
    assert hasattr(module, "write_analysis_csv"), "Analysis CSV implementation is missing"
    results = {"G32_J_M": {"status": "OK", "design_id": "B_J_M", "selected_step": 5750,
                             "trainable_parameters": 2048, "primary_auc": .9},
               "G36_NATIVE_448": {"status": "NOT_ELIGIBLE_ASSET", "design_id": "E6_NATIVE_448"}}
    evaluation = {"reports": {"G32_J_M": {"WDF": reports([.01] * 6)["WDF"]}},
                  "champion": "G32_J_M", "assessment": {"auc_breakthrough": True,
                  "cross_domain_fpr_controlled": True, "can_claim_breakthrough": False}}
    original = copy.deepcopy((results, evaluation))
    path = tmp_path / "nested" / "analysis.csv"
    module.write_analysis_csv(path, results, evaluation)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    candidate = next(row for row in rows if row["g_id"] == "G32_J_M")
    assert candidate["family"] == "G32" and candidate["design_id"] == "B_J_M"
    assert candidate["dataset"] == "WDF"
    assert float(candidate["delta_video_auc"]) == pytest.approx(.01)
    assert float(candidate["baseline_video_auc"]) == .8
    assert float(candidate["ranking_repair"]) == .02
    blocked = next(row for row in rows if row["g_id"] == "G36_NATIVE_448")
    assert blocked["dataset"] == "" and blocked["status"] == "NOT_ELIGIBLE_ASSET"
    assert (results, evaluation) == original
    assert list(path.parent.iterdir()) == [path]
