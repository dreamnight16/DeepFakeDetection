"""Dependency-free E1005 G31--G36 registry and reproducible arm identities."""

from copy import deepcopy
import hashlib
import json

FAMILIES = {
    "G31": "更新位置与初始化对照", "G32": "参数更新范围与视频监督矩阵",
    "G33": "同协议强基线", "G34": "监督机制消融", "G35": "内部层与预训练表示",
    "G36": "空间取证与原生分辨率",
}
REGRESSION_DATASETS = ["WDF", "FFIW", "DeepFakeDetection", "DFDC", "DFDCP", "DeeperForensics-1.0"]
OBJECTIVES = ("V", "S", "M", "MK", "MG", "MN", "MKGN")


def objective(code="V", classification="video"):
    if code not in OBJECTIVES:
        raise ValueError(f"Unknown E1005 objective: {code}")
    return {"classification": classification, "rank": "shuffled" if code == "S" else
            ("matched" if code != "V" else "none"), "keep": "K" in code,
            "method_risk": "G" in code, "nuisance_risk": "N" in code}


def _arm(gid, design_id, kind="warm", scope="J", **extras):
    return {"id": gid, "family": gid.split("_", 1)[0], "design_id": design_id,
            "kind": kind, "scope": scope, "objective": objective(), "requires_pairs": False,
            "requires_pristine": False, "requires_mask": False, "requires_native": False,
            "depends_on_leader": False, "metric": False, **extras}


def _catalog():
    rows = []
    for name, alias, scope in (("HEAD", "A1_HEAD", "H"), ("LORA", "A2_LORA", "L"),
            ("JOINT", "A3_JOINT", "J"), ("RESET_HEAD", "A4_RESET_HEAD", "reset_head"),
            ("NEW_RESIDUAL", "A5_NEW_RESIDUAL", "new_residual"),
            ("RESIDUAL_HEAD", "A6_RESIDUAL_HEAD", "residual_head")):
        rows.append(_arm(f"G31_{name}", alias, scope=scope, objective=objective(classification="frame")))
    for scope in ("H", "L", "J"):
        for code in OBJECTIVES:
            rows.append(_arm(f"G32_{scope}_{code}", f"B_{scope}_{code}", scope=scope,
                             objective=objective(code), requires_pairs=True))
    rows.extend([
        _arm("G33_MATCHED_B0", "R1_MATCHED_B0", "cold_b0", requires_pristine=True,
             objective=objective(classification="frame")),
        _arm("G33_VIDEO_B0", "R2_VIDEO_B0", "cold_video", requires_pristine=True, requires_pairs=True),
        _arm("G33_GEND", "R3_GEND_RECIPE", "gend", requires_pristine=True, metric=True,
             objective=objective(classification="frame")),
    ])
    for name, alias, factor in (
            ("SHUFFLE", "C1_SHUFFLE", "shuffle"), ("INDEPENDENT_NUISANCE", "C2_INDEPENDENT_NUISANCE", "nuisance"),
            ("NO_KEEP", "C3_NO_KEEP", "keep"), ("NO_METHOD_RISK", "C4_NO_METHOD_RISK", "method"),
            ("NO_NUISANCE_RISK", "C5_NO_NUISANCE_RISK", "condition"),
            ("FRAME_OBJECTIVE", "C6_FRAME_OBJECTIVE", "frame")):
        rows.append(_arm(f"G34_{name}", alias, requires_pairs=True, depends_on_leader=True, factor=factor))
    for name, alias, kind, pristine in (
            ("MID_LOCAL_SWIGLU", "D1_MID_LOCAL_SWIGLU", "local_swiglu", False),
            ("SHUFFLED_NEIGHBORS", "D2_SHUFFLED_NEIGHBORS", "shuffle_local", False),
            ("LATE_LOCAL", "D3_LATE_LOCAL", "late_local", False),
            ("MID_GEGLU", "D4_MID_GEGLU", "local_geglu", False),
            ("PRETRAINED_LINEAR_DELTA", "D5_PRETRAINED_LINEAR_DELTA", "pretrained_delta", True),
            ("B0_LINEAR_DELTA", "D6_B0_LINEAR_DELTA", "b0_delta", False),
            ("PRETRAINED_LN_DELTA", "D7_PRETRAINED_LN_DELTA", "ln_delta", True),
            ("LN_METRIC_DELTA", "D8_LN_METRIC_DELTA", "ln_metric", True)):
        rows.append(_arm(f"G35_{name}", alias, kind, depends_on_leader=True, requires_pairs=True,
                         requires_pristine=pristine, metric=kind == "ln_metric"))
    for name, alias, kind, mask, native in (
            ("PIXEL_224", "E1_PIXEL_224", "pixel224", False, False),
            ("TAMPER_MASK", "E2_TAMPER_MASK", "tamper", True, False),
            ("BOUNDARY", "E3_BOUNDARY", "boundary", True, False),
            ("SHUFFLED_MASK", "E4_SHUFFLED_MASK", "shuffled_mask", True, False),
            ("INTERPOLATED_448", "E5_INTERPOLATED_448", "interpolated448", False, True),
            ("NATIVE_448", "E6_NATIVE_448", "native448", False, True)):
        rows.append(_arm(f"G36_{name}", alias, kind, depends_on_leader=True, requires_pairs=True,
                         requires_mask=mask, requires_native=native))
    return {row["id"]: row for row in rows}


ARMS = _catalog()
ALIASES = {row["design_id"]: gid for gid, row in ARMS.items()}


def resolve_arms(values=None):
    values = list(ARMS) if values is None else values
    expanded = []
    for value in values:
        if value in FAMILIES:
            expanded.extend(gid for gid, row in ARMS.items() if row["family"] == value)
        else:
            gid = ALIASES.get(value, value)
            if gid not in ARMS:
                raise ValueError(f"Unknown G ID/design alias: {value}")
            expanded.append(gid)
    if len(set(expanded)) != len(expanded):
        raise ValueError("Duplicate experiment identities after alias expansion")
    if not expanded:
        raise ValueError("At least one experiment must be requested")
    return [deepcopy(ARMS[gid]) for gid in expanded]


def materialize_arm(template, leader):
    result = deepcopy(template)
    if not result["depends_on_leader"]:
        return result
    if leader is None:
        raise ValueError("This arm needs a completed G32 development leader")
    result["objective"] = deepcopy(leader["objective"])
    result["leader_id"] = leader["id"]
    if result["family"] == "G34":
        result["kind"], result["scope"] = leader["kind"], leader["scope"]
        factor = result["factor"]
        required = {"shuffle": result["objective"]["rank"] == "matched",
                    "keep": result["objective"]["keep"],
                    "method": result["objective"]["method_risk"],
                    "condition": result["objective"]["nuisance_risk"],
                    "frame": result["objective"]["classification"] == "video", "nuisance": True}
        if not required[factor]:
            raise ValueError(f"NOT_APPLICABLE: leader does not contain {factor}")
        if factor == "shuffle":
            result["objective"]["rank"] = "shuffled"
        elif factor == "keep":
            result["objective"]["keep"] = False
        elif factor == "method":
            result["objective"]["method_risk"] = False
        elif factor == "condition":
            result["objective"]["nuisance_risk"] = False
        elif factor == "frame":
            result["objective"]["classification"] = "frame"
        elif factor == "nuisance":
            result["independent_nuisance"] = True
    return result


def training_signature(spec, settings, source_identities):
    # Labels such as G34_NO_KEEP and an equivalent G32 arm may share training,
    # but must retain separate reporting identities; selectors are not training.
    excluded = {"id", "family", "design_id", "factor", "depends_on_leader", "leader_id"}
    value = {"training": {key: value for key, value in spec.items() if key not in excluded},
             "settings": settings, "sources": source_identities}
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def plan(arms=None, seed=1024, steps=5750):
    resolved = resolve_arms(arms)
    return {"experiment": "E1005", "seed": seed, "families": FAMILIES,
            "arms": {row["id"]: row for row in resolved}, "additional_optimizer_steps": steps,
            "reference": "immutable original B0 re-scored on the same full-path frame manifests",
            "selection": "S3=Celeb-DF-v2 development video_auc; no historical-six-domain selection",
            "selectors": ["S1_source_frame", "S2_source_video", "S3_development_video", "S4_final_step"],
            "primary_sampling": "numeric sorted uniform_metadata8", "historical_sampling": "legacy_prefix8",
            "pair_gate": "explicit time/target-face audit receipt; candidates are not verified pairs",
            "regression_datasets": REGRESSION_DATASETS,
            "reproduction": "same seed/config/manifests; no multi-seed scan",
            "stages": ["audit", "screen", "ablate", "represent", "conditional", "evaluate", "reproduce"]}
