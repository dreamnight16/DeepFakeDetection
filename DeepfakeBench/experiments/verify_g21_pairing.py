"""Content-first FF++ lineage verification: turn a dataset JSON into a pair map.

Runs the pairing in ``training/g21/pairing.py``: every train fake key is
compared against both named candidate originals using only the eligible
background region, the extracted-frame index set is kept as a candidate hint
and a timeline-compatibility check, and any content/timeline disagreement is
reported instead of hidden.  Only a decisive, close content win with a
compatible timeline becomes a verified rule; everything else is an explicit
exclusion, so nothing is guessed and no method is silently dropped.

The resulting ``pair_map.json`` is accepted by ``build_g21_manifest.py build``.
Never imports torch and never loads a detection model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from g21.pairing import PairingConfig, run_pairing

DEFAULT_FACE_BOX = (0.25, 0.75, 0.15, 0.85)


def build_config(args: argparse.Namespace) -> PairingConfig:
    """Translate CLI options into the validated pairing configuration."""
    return PairingConfig(
        sample_frames=args.sample_frames,
        min_verification_frames=args.min_verification_frames,
        comparison_size=args.comparison_size,
        blur_sigma=args.blur_sigma,
        mask_dilation=args.mask_dilation,
        eligible_floor=args.eligible_floor,
        face_box=tuple(args.face_box),
        win_rate_min=args.win_rate_min,
        margin_min=args.margin_min,
        max_winner_distance=args.max_winner_distance,
        max_single_side_distance=args.max_single_side_distance,
        timeline_jaccard_min=args.timeline_jaccard_min,
    )


def _summary(args: argparse.Namespace, report: dict) -> dict:
    by_method = report["funnel"]["by_method"]
    references = report["references_by_method"]
    # Preflight demands pairs_per_microbatch distinct content sources per method,
    # so a pair map that cannot satisfy it must not report success here.
    missing = sorted(m for m in by_method if by_method[m]["usable_pairs"] == 0)
    thin = sorted(m for m in by_method if references[m] < args.min_sources)
    return {
        "status": "PAIRING_OK" if not missing and not thin else "PAIRING_INCOMPLETE",
        "output": str(Path(args.output)),
        "usable_pairs": report["usable_pairs"],
        "decision_counts": report["decision_counts"],
        "methods_without_usable_pairs": missing,
        "references_by_method": references,
        "methods_below_min_sources": thin,
        "min_sources": args.min_sources,
        "funnel": report["funnel"],
        "warnings": report["warnings"],
        "distributions": report["distributions"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json", required=True, help="original FF++ JSON with all four methods"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="root that frame and mask relative paths resolve against",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new directory for pair_map.json and pairing_audit.json",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=12,
        help="comparison frames per fake video, clamped to 8..16",
    )
    parser.add_argument(
        "--min-verification-frames",
        type=int,
        default=8,
        help="fewest same-index frames a decision may rest on, clamped to 2..sample-frames",
    )
    parser.add_argument("--comparison-size", type=int, default=224)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--mask-dilation", type=int, default=8)
    parser.add_argument("--eligible-floor", type=float, default=0.10)
    parser.add_argument(
        "--face-box",
        type=float,
        nargs=4,
        default=list(DEFAULT_FACE_BOX),
        metavar=("X0", "X1", "Y0", "Y1"),
        help="fallback face box as fractions of the frame",
    )
    parser.add_argument("--win-rate-min", type=float, default=0.9)
    parser.add_argument("--margin-min", type=float, default=0.5)
    parser.add_argument(
        "--max-winner-distance",
        type=float,
        default=12.0,
        help="absolute background-distance ceiling for the winner, 0..255 scale",
    )
    parser.add_argument(
        "--max-single-side-distance",
        type=float,
        default=4.0,
        help=(
            "tighter ceiling used when fewer than min_verification_frames frames "
            "are shared by both candidates; a side confirmed here must also not "
            "lose to the other on any frame they do share, so this bound only "
            "carries a decision alone when the two share no frame at all"
        ),
    )
    parser.add_argument("--timeline-jaccard-min", type=float, default=0.9)
    parser.add_argument(
        "--min-sources",
        type=int,
        default=4,
        help="distinct content sources each method must keep for the microbatch",
    )
    parser.add_argument(
        "--require-usable",
        action="store_true",
        help="exit nonzero unless every method produced at least one verified pair",
    )
    args = parser.parse_args(argv)
    try:
        report = run_pairing(
            args.dataset_json, args.data_root, args.output, config=build_config(args)
        )
    except (ValueError, OSError, KeyError) as exc:
        print(
            json.dumps(
                {"status": "PAIRING_FAILED", "error": str(exc)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 1
    summary = _summary(args, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_usable and summary["status"] != "PAIRING_OK":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
