"""E1111A complete frozen-B0 experiment, with optional individual stage commands."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from e1111.protocol import protocol


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    predictor_candidates = [ROOT / "preprocessing/shape_predictor_81_face_landmarks.dat",
                            ROOT / "preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"]
    default_predictor = next((p for p in predictor_candidates if p.is_file()), predictor_candidates[0])
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--base_run", type=Path)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--base_config", type=Path)
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root", type=Path)
    parser.add_argument("--clip_pretrained_path", type=Path)
    parser.add_argument("--pair_map", type=Path)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1111/E1111A")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_d8", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    stages = parser.add_subparsers(dest="stage")
    for name in ("inventory", "audit"):
        child = stages.add_parser(name)
        child.add_argument("--dataset-json", type=Path, default=ROOT / "preprocessing/dataset_json/FaceForensics++.json")
        child.add_argument("--output-dir", type=Path, required=True)
        if name == "inventory":
            child.add_argument("--data-root", type=Path, help="Use the same crop files via a normalized index view")
        if name == "audit":
            child.add_argument("--data-root", type=Path, required=True)
            child.add_argument("--pair-map", type=Path, required=True)
    for name in ("extract", "s0", "calibrate"):
        child = stages.add_parser(name)
        child.add_argument("--audit", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if name in ("s0", "calibrate"):
            child.add_argument("--cache", type=Path, required=True)
        if name in ("extract", "calibrate"):
            child.add_argument("--base-config", type=Path, required=True)
            child.add_argument("--checkpoint", type=Path, required=True)
            child.add_argument("--clip-pretrained-path", type=Path, required=True)
            child.add_argument("--device", default="cuda:0")
            child.add_argument("--batch-size", type=int, default=8)
        if name == "extract":
            child.add_argument("--d8", action="store_true")
        if name == "calibrate":
            child.add_argument("--metric", type=Path, required=True)
    child = stages.add_parser("score")
    child.add_argument("--features", type=Path, required=True)
    child.add_argument("--metric", type=Path, required=True)
    child.add_argument("--output-dir", type=Path, required=True)
    child = stages.add_parser("export")
    child.add_argument("--dataset-json", type=Path, required=True)
    child.add_argument("--dataset-id", required=True)
    child.add_argument("--split", choices=("train", "val", "test"), required=True)
    child.add_argument("--data-root", type=Path, required=True)
    child.add_argument("--base-config", type=Path, required=True)
    child.add_argument("--checkpoint", type=Path, required=True)
    child.add_argument("--clip-pretrained-path", type=Path, required=True)
    child.add_argument("--device", default="cuda:0")
    child.add_argument("--batch-size", type=int, default=8)
    child.add_argument("--output-dir", type=Path, required=True)
    child = stages.add_parser("attribution")
    child.add_argument("--evaluations", nargs="+", type=Path, required=True)
    child.add_argument("--output-dir", type=Path, required=True)
    child = stages.add_parser("prepare", help="Legacy shared-crop experiment, not part of the normal A run")
    child.add_argument("--dataset-json", type=Path, default=ROOT / "preprocessing/dataset_json/FaceForensics++.json")
    child.add_argument("--pair-map", type=Path, required=True)
    child.add_argument("--raw-root", type=Path, required=True)
    child.add_argument("--predictor-path", type=Path, default=default_predictor)
    child.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(protocol(), ensure_ascii=False, indent=2))
        return 0
    try:
        if args.stage is None:
            from e1111_runner import run
            result = run(args)
        elif args.stage == "inventory":
            from e1111.audit import inventory
            result = inventory(args.dataset_json, args.output_dir, data_root=args.data_root)
        elif args.stage == "audit":
            from e1111.audit import audit
            result = audit(args.dataset_json, args.data_root, args.pair_map, args.output_dir)
        elif args.stage == "extract":
            from e1111.features import extract
            if args.output_dir.exists():
                raise FileExistsError(args.output_dir)
            result = extract(args.audit, args.base_config, args.checkpoint, args.clip_pretrained_path,
                             args.output_dir / "features.npz", device=args.device,
                             batch_size=args.batch_size, d8=args.d8)
        elif args.stage == "s0":
            from e1111.pipeline import run_s0
            result = run_s0(args.audit, args.cache, args.output_dir)
        elif args.stage == "calibrate":
            from e1111.pipeline import calibrate
            result = calibrate(args.audit, args.cache, args.metric, args.base_config, args.checkpoint,
                               args.clip_pretrained_path, args.output_dir,
                               device=args.device, batch_size=args.batch_size)
        elif args.stage == "score":
            from e1111.pipeline import score_export
            result = score_export(args.features, args.metric, args.output_dir)
        elif args.stage == "export":
            from e1111.features import export_dataset
            if args.output_dir.exists():
                raise FileExistsError(args.output_dir)
            result = export_dataset(args.dataset_json, args.dataset_id, args.split, args.data_root,
                                    args.base_config, args.checkpoint, args.clip_pretrained_path,
                                    args.output_dir / "features.npz",
                                    device=args.device, batch_size=args.batch_size)
        elif args.stage == "attribution":
            from e1111.pipeline import attribution
            result = attribution(args.evaluations, args.output_dir)
        else:
            from e1111.shared_crop import prepare
            result = prepare(args.dataset_json, args.pair_map, args.raw_root, args.predictor_path, args.output_dir)
        summary = {key: result[key] for key in ("status", "N4", "Ncell", "K_pair", "gamma") if key in result}
        print(json.dumps({"output": result.get("results_dir", str(args.output_dir)), **summary}, ensure_ascii=False))
        return 0 if result.get("passed", True) and result.get("status") not in (
            "FAILED", "S1_FAILED", "S0_STOPPED", "S0_PAUSED_P_D7",
            "S0_PAUSED_UNREADABLE_REAL", "D1_FAILED_REGULARIZATION_ONLY") else 2
    except (ValueError, OSError, KeyError) as exc:
        print(f"E1111A stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
