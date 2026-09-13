"""G21 metadata tools. Never imports torch or loads a detection model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from g21.builder import build_ffpp, mapping_template
from g21.io import atomic_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser(
        "template", help="write unverified mapping rules for review"
    )
    template.add_argument("--output", required=True)
    build = commands.add_parser(
        "build", help="build FF++ four-method training pairs from verified rules"
    )
    build.add_argument("--dataset-json", required=True)
    build.add_argument("--pair-map", required=True)
    build.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            atomic_json(args.output, mapping_template())
            report = {"status": "TEMPLATE_ONLY", "mapping_verified": False}
        else:
            report = build_ffpp(args.dataset_json, args.pair_map, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(
            json.dumps(
                {"status": "MANIFEST_FAILED", "error": str(exc)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
