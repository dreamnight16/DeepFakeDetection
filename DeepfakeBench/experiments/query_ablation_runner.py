"""Shared one-arm runner for the isolated G22/G23 experiments."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEEPFAKE = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _DEEPFAKE)

from experiment_utils import build_config, evaluate_model, train_model


TRAIN_DS = "FaceForensics++"
VAL_DS = "Celeb-DF-v2"
CROSS_DS = [
    "WDF",
    "FFIW",
    "Celeb-DF-v2",
    "DeepFakeDetection",
    "DFDC",
    "DFDCP",
    "DeeperForensics-1.0",
]
TEST_DS = CROSS_DS + [TRAIN_DS]
CROSS_METRIC_DS = ["Celeb-DF-v2", "DFDC"]

EXPERIMENTS = {
    "G22": {
        "model_name": "effort_g22_last_block_input",
        "default_output": "./experiment_results/g22_last_block_input",
        "description": "full query blocks over hidden_states[-2]",
    },
    "G23": {
        "model_name": "effort_g23_cross_only",
        "default_output": "./experiment_results/g23_cross_only",
        "description": "cross-attention-only query blocks over final output",
    },
}


def build_parser(experiment: str) -> argparse.ArgumentParser:
    spec = EXPERIMENTS[experiment]
    parser = argparse.ArgumentParser(description=f"{experiment}: {spec['description']}")
    parser.add_argument("--output_dir", default=spec["default_output"])
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--sampler_real_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=1024)
    return parser


def _build_kwargs(args, model_name, *, for_training, test_dataset):
    return dict(
        use_mixup=False,
        mixup_loss_strip=False,
        sampler_real_ratio=args.sampler_real_ratio,
        model_name=model_name,
        log_dir=os.path.join(args.run_dir, "logs"),
        train_dataset=TRAIN_DS,
        test_dataset=test_dataset,
        n_epochs=args.n_epochs if for_training else 0,
        for_training=for_training,
        lfeq_hidden_dim=256,
        lfeq_num_evidence_tokens=8,
        lfeq_depth=2,
        lfeq_num_heads=8,
        lfeq_dropout=0.1,
    )


def run(experiment: str, argv=None):
    spec = EXPERIMENTS[experiment]
    args = build_parser(experiment).parse_args(argv)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.run_dir = os.path.join(args.output_dir, f"seed{args.seed}_{run_id}")
    os.makedirs(args.run_dir, exist_ok=False)
    print(
        f"{'=' * 72}\n"
        f"  {experiment}: {spec['description']}\n"
        f"  K=8 hidden=256 depth=2 heads=8 mean-readout\n"
        f"  sampler_real_ratio={args.sampler_real_ratio} seed={args.seed}\n"
        f"  output={args.run_dir}\n"
        f"{'=' * 72}"
    )

    train_config = build_config(
        **_build_kwargs(args, spec["model_name"], for_training=True, test_dataset=VAL_DS)
    )
    train_config["manualSeed"] = args.seed
    checkpoint = train_model(train_config, TRAIN_DS, VAL_DS)
    if checkpoint is None:
        result = {
            "exp_name": experiment,
            "model_name": spec["model_name"],
            "status": "TRAIN_FAILED",
        }
    else:
        eval_config = build_config(
            **_build_kwargs(
                args, spec["model_name"], for_training=False, test_dataset=TEST_DS
            )
        )
        eval_config["manualSeed"] = args.seed
        try:
            result = evaluate_model(
                eval_config, checkpoint, TEST_DS, TRAIN_DS, args.run_dir, experiment
            )
        except Exception as exc:  # preserve an auditable failed-run artifact
            print(f"  [eval] FAILED: {exc}")
            result = {
                "exp_name": experiment,
                "model_name": spec["model_name"],
                "status": "EVAL_FAILED",
                "ckpt": checkpoint,
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            metrics = result.get("testall", {})
            missing = []
            for dataset in TEST_DS:
                value = metrics.get(dataset, {}).get("video_auc")
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    missing.append(dataset)
            result.update(
                exp_name=experiment, model_name=spec["model_name"], ckpt=checkpoint
            )
            if missing:
                result["status"] = "EVAL_FAILED"
                result["missing_video_auc"] = missing
            else:
                result["status"] = "OK"

    results_path = os.path.join(args.run_dir, "all_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump([result], handle, indent=2, default=str)

    if result["status"] == "OK":
        metrics = result.get("testall", {})
        cross_values = [
            metrics[name]["video_auc"]
            for name in CROSS_METRIC_DS
            if name in metrics and "video_auc" in metrics[name]
        ]
        cross = float(np.mean(cross_values)) if len(cross_values) == 2 else None
        print(f"  AUC_cross={cross:.6f}" if cross is not None else "  AUC_cross=N/A")
    print(f"  results={results_path}")
    return result
