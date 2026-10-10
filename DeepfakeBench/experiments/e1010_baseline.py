"""Train E1010's fresh B0 with the original E0924/G26_B0 protocol."""

from pathlib import Path
from types import SimpleNamespace

import run_g26
from run_e0924 import write_json


INDEPENDENT_DS = [dataset for dataset in run_g26.baseline.CROSS_DS
                  if dataset != run_g26.baseline.VAL_DS]


def baseline_args(args):
    """Keep B0's budget independent from the later auxiliary training budget."""
    cli = ["--arms", "B0", "--seed", str(args.seed),
           "--n_epochs", str(args.base_n_epochs),
           "--sampler_real_ratio", str(args.base_sampler_real_ratio)]
    if args.clip_pretrained_path:
        cli += ["--clip_pretrained_path", str(args.clip_pretrained_path)]
    return run_g26.build_parser().parse_args(cli)


def _apply_overrides(args, config):
    config["e0924_protocol"] = True
    if args.dataset_json_folder is not None:
        config["dataset_json_folder"] = str(Path(args.dataset_json_folder).resolve())
    if args.rgb_root:
        config["rgb_root_override"] = str(args.rgb_root)
    if args.base_batch_size is not None:
        config["train_batchSize"] = args.base_batch_size
    return config


def build_baseline_config(args, folder, build_config, for_training=True):
    """Reuse the historical builder without altering augmentation or losses."""
    def historical_config(**kwargs):
        return _apply_overrides(args, build_config(**kwargs))

    return run_g26.build_arm_config(baseline_args(args), "B0", Path(folder),
                                   historical_config, for_training)


def _seed_evaluation(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_baseline(args, folder, utilities):
    """Save a fresh B0 and its legacy metrics before auxiliary-only evaluation."""
    folder = Path(folder)

    def build_config(**kwargs):
        return _apply_overrides(args, utilities.build_config(**kwargs))

    def train_model(*values):
        checkpoint = utilities.train_model(*values)
        if checkpoint is not None and not Path(checkpoint).is_file():
            raise FileNotFoundError(f"B0 training returned a missing checkpoint: {checkpoint}")
        return str(Path(checkpoint).resolve()) if checkpoint is not None else None

    adapter = SimpleNamespace(build_config=build_config, train_model=train_model,
                              evaluate_model=utilities.evaluate_model,
                              seed_evaluation=getattr(utilities, "seed_evaluation", _seed_evaluation))
    result = run_g26.run_one(baseline_args(args), "B0", folder, adapter)
    result.update(config_path=str((folder / "train_config.json").resolve()),
                  eval_config_path=str((folder / "eval_config.json").resolve()),
                  reproduction_path=str((folder / "reproduction.json").resolve()),
                  initialization="fresh CLIP pretrained; no detector warm start")
    report = {"status": result["status"], "protocol": "E0924/G26_B0",
              "checkpoint": result.get("ckpt"), "config_path": result["config_path"],
              "initialization": result["initialization"],
              "nEpochs": args.base_n_epochs,
              "epoch_loop": {"start": 0, "end_inclusive": args.base_n_epochs,
                             "training_passes": args.base_n_epochs + 1},
              "checkpoint_selection": {"dataset": run_g26.baseline.VAL_DS, "metric": "frame_auc"},
              "metric_protocol": "project legacy basename video grouping",
              "seven_mean_datasets": list(run_g26.baseline.CROSS_DS),
              "independent_datasets": INDEPENDENT_DS,
              "excluded_from_independent_mean": [run_g26.baseline.VAL_DS, run_g26.baseline.TRAIN_DS],
              "reproduction_status": "NOT_EVALUATED"}
    if result["status"] == "OK":
        report.update(datasets={dataset: result["testall"][dataset]
                                for dataset in run_g26.baseline.TEST_DS},
                      video_auc_seven_mean=result["video_auc_seven_mean"],
                      AUC_cross=result["AUC_cross"], G=result["G"],
                      independent_video_auc_mean=sum(result["testall"][dataset]["video_auc"]
                                                     for dataset in INDEPENDENT_DS) / len(INDEPENDENT_DS),
                      reproduction_status="REFERENCE_NOT_PROVIDED")
    if result.get("error"):
        report["error"] = result["error"]
    write_json(folder / "reproduction.json", report)
    write_json(folder / "result.json", result)
    print(f"E1010 B0 original-protocol evaluation: {result['status']}", flush=True)
    if result["status"] == "OK":
        for dataset, metrics in report["datasets"].items():
            print(f"  {dataset}: legacy_video_auc={metrics['video_auc']:.6f}", flush=True)
        print(f"  seven_mean={report['video_auc_seven_mean']:.6f} "
              f"independent_six_mean={report['independent_video_auc_mean']:.6f} "
              f"AUC_cross={report['AUC_cross']:.6f} G={report['G']:.6f}\n"
              "  REFERENCE_NOT_PROVIDED: historical reference unavailable; reproduction not yet established.\n"
              f"  Report: {result['reproduction_path']}", flush=True)
    return result
