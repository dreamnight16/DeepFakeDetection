"""Check E1021's CUDA training memory with a few synthetic batches."""

import gc
import json
import sys
from types import SimpleNamespace

import run_e1021 as experiment


def check_arm(args, arm, utilities):
    import torch
    from metrics.base_metrics_class import Recorder
    from trainer.trainer_v2 import Trainer

    config = experiment.build_arm_config(args, arm, args.output_dir / arm, utilities.build_config, True)
    experiment.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = config["cudnn"]
    model = utilities.DETECTOR[config["model_name"]](config).cuda().train()
    settings = config["optimizer"]["adam"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"],
                                 betas=(settings["beta1"], settings["beta2"]), eps=settings["eps"],
                                 amsgrad=settings["amsgrad"])
    trainer = SimpleNamespace(config=config, model=model, optimizer=optimizer)
    batch_size = 2 * max(1, config["train_batchSize"] // 2)
    num_real = max(1, round(batch_size * config["sampler_real_ratio"]))
    batch = {"image": torch.randn(batch_size, 3, config["resolution"], config["resolution"], device="cuda"),
             "label": (torch.arange(batch_size, device="cuda") >= num_real).long()}
    recorders = {key: Recorder() for key in ("overall", "real_loss", "fake_loss")}
    allocated = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        losses, predictions = Trainer.train_step(trainer, batch)
        if not torch.isfinite(losses["overall"]).item():
            raise RuntimeError(f"{arm}: nonfinite loss at batch {step}")
        for key, value in losses.items():
            recorders[key].update(value)
        if step % 300 == 0:
            for recorder in recorders.values():
                recorder.clear()
        del losses, predictions, value
        torch.cuda.synchronize()
        allocated.append(torch.cuda.memory_allocated())
        print(f"{arm} batch {step + 1}/{args.steps}: allocated={allocated[-1] / 2**20:.1f} MiB", flush=True)
    # Ignore allocator/optimizer warm-up; a retained encoder graph grows by
    # much more than the few scalar tensors held by the loss recorders.
    growth = max(allocated[2:]) - min(allocated[2:])
    if growth > 16 * 2**20:
        raise RuntimeError(f"{arm}: live CUDA allocations grew by {growth / 2**20:.1f} MiB after warm-up")
    free, total = torch.cuda.mem_get_info()
    return {"arm": arm, "status": "OK", "input": "synthetic", "steps": args.steps,
            "batch_size": batch_size, "real_samples": num_real, "fake_samples": batch_size - num_real,
            "allocated_mib": [value / 2**20 for value in allocated],
            "steady_growth_mib": growth / 2**20, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "device_free_mib": free / 2**20, "device_total_mib": total / 2**20}


def main(argv=None):
    parser = experiment.build_parser()
    parser.description = __doc__
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)
    experiment.validate_arguments(parser, args)
    if args.steps < 4:
        parser.error("The memory check requires at least 4 steps")
    if args.dry_run:
        print(json.dumps({"mode": "synthetic CUDA memory check; no experiment results or checkpoint",
                          "steps": args.steps, **experiment.plan(args)}, indent=2))
        return 0
    import torch

    if not torch.cuda.is_available():
        print("CUDA is required; GPU memory remains unverified. No training started.", file=sys.stderr)
        return 2
    import experiment_utils as utilities

    for arm in args.arms:
        try:
            report = check_arm(args, arm, utilities)
            print(json.dumps(report), flush=True)
        except Exception as exc:
            print(json.dumps({"arm": arm, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}),
                  file=sys.stderr, flush=True)
            return 1
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
