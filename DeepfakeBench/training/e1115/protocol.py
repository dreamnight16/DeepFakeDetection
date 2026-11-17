"""Predeclared E1115 experiment matrix; no development pass/fail gate."""

import hashlib
import numpy as np

TRAIN_ARMS = {
    "B0_NATIVE": dict(ratio=.3, loss="batch", frames="legacy", fixed_budget=False),
    "B0_CONTROL": dict(ratio=.3, loss="batch", frames="legacy", fixed_budget=True),
    "CLASS_BALANCED": dict(ratio=.3, loss="class_mean", frames="legacy", fixed_budget=True),
    "REAL50": dict(ratio=.5, loss="batch", frames="legacy", fixed_budget=True),
    "REAL70": dict(ratio=.7, loss="batch", frames="legacy", fixed_budget=True),
    "FRAME_COVERAGE": dict(ratio=.3, loss="batch", frames="epoch_stratified", fixed_budget=True),
    "FULL_REAL70_COVERAGE": dict(ratio=.7, loss="batch", frames="epoch_stratified", fixed_budget=True),
}


def metric_specs():
    full = dict(statistic="block4", rank=2, eta=1., weighting="soft", compression=True,
                project_normalize=True, real_weight=0., direction_weight=1.)
    arms = {"FULL": full, "VM_LEGACY": {**full, "statistic": "mean", "legacy": True,
                                       "eta": 0., "weighting": "legacy_cap99",
                                       "project_normalize": False, "real_weight": 1., "direction_weight": 2.}}
    interventions = {
        "NO_REAL_CONSTRAINT": dict(eta=0.), "NO_COMPRESSION": dict(compression=False),
        "NO_PROJECT_NORMALIZATION": dict(project_normalize=False),
        "EQUAL_SPECTRUM": dict(weighting="equal"), "CAPPED_SPECTRUM": dict(weighting="cap99"),
        "REAL_ONLY": dict(real_weight=1., direction_weight=0.),
        "JOINT": dict(real_weight=1., direction_weight=2.),
        "MEAN_ONLY": dict(statistic="block_mean4"), "VARIATION_ONLY": dict(statistic="variation4"),
        "VIDEO_MEAN": dict(statistic="mean"),
        "BLOCK2": dict(statistic="block2"), "BLOCK8": dict(statistic="block8"),
        "RANK4": dict(rank=4), "RANK8": dict(rank=8),
        "ETA0.1": dict(eta=.1), "ETA10": dict(eta=10.),
        "MATCHED_MEAN": dict(statistic="matched_mean"),
        "MATCHED_BLOCK": dict(statistic="matched_block"),
        "MATCHED_FRAME": dict(statistic="matched_frame"),
        "MATCHED_VARIATION": dict(statistic="matched_variation"),
    }
    arms.update({name: {**full, **change} for name, change in interventions.items()})
    return arms


def plan():
    return {"experiment": "E1115", "version": "E1115-v1", "train_arms": TRAIN_ARMS,
            "metric_arms": metric_specs(), "primary_training_arm": "FULL_REAL70_COVERAGE",
            "primary_metric": "FULL", "tau": .1, "fit_frames": 32, "test_frames": 8,
            "adaptation_modes": ["threshold", "mean", "cov", "mean_cov"],
            "adaptation_fit_budgets": [10, 30, 100], "calibration_real_videos": 30,
            "adaptation_repeats": 3, "covariance_mix": .5,
            "fusion_weights": [0., .05, .1, .25], "alpha": [.1, .05],
            "selection": "fixed matrix; no development gate, test-based selection, or promoted winner",
            "target_protocol": "trusted real fit/calibration videos removed from held-out evaluation; matched B0 controls",
            "source_split_seed": 1024, "baseline_preservation": "unchanged checkpoint and separate raw predictions"}


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


def split_target_real(video_ids, labels, budget, calibration_count, seed, reserve_budget=None):
    """Nested fits, fixed calibration, no selected video in evaluation."""
    video_ids, labels = np.asarray(video_ids), np.asarray(labels)
    if labels.shape != video_ids.shape or budget < 1 or calibration_count < 1:
        raise ValueError("Invalid target split inputs")
    unique_ids, first, inverse = np.unique(video_ids, return_index=True, return_inverse=True)
    video_labels = labels[first]
    if not np.isin(labels, [0, 1]).all() or not np.array_equal(labels, video_labels[inverse]):
        raise ValueError("Video has conflicting or nonbinary labels")
    unique = unique_ids.tolist()
    real = unique_ids[video_labels == 0].tolist()
    reserve_budget = budget if reserve_budget is None else reserve_budget
    if reserve_budget < budget:
        raise ValueError("Reserved adaptation pool is smaller than fit budget")
    if len(real) < reserve_budget + calibration_count + 20:
        raise ValueError("Insufficient target real videos: require budget + calibration + 20 held-out")
    order = np.random.default_rng(seed).permutation(real).tolist()
    calibration = order[:calibration_count]
    fit = order[calibration_count:calibration_count + budget]
    reserved = order[calibration_count:calibration_count + reserve_budget]
    excluded = set(reserved + calibration)
    return {"fit": fit, "calibration": calibration, "test": [v for v in unique if v not in excluded],
            "reserved_fit_pool": reserved, "total_trusted_real_budget": budget + calibration_count,
            "scope": "video-disjoint, not an identity-disjoint claim"}
