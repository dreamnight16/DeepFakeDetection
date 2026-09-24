"""Create a checksummed source overlay for an existing Effort repository."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from e0924_protocol import plan
from run_e0924 import ROOT, source_hashes


def bundle(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    files = set(source_hashes())
    # Runtime files are a source overlay, not a self-contained environment.
    for name in ("G26_README.md", "G27_README.md", "E0924_README.md", "make_e0924_bundle.py"):
        files.add(f"experiments/{name}")
    for pattern in ("test_e0924.py", "test_g25*.py", "test_g26*.py", "test_g27*.py", "test_g22_g23.py"):
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob(pattern))
    # Old-runner tests import these; ship the exact local source with the tests.
    files.update({"experiments/run_g25v2.py", "experiments/run_g22.py", "experiments/run_g23.py"})
    revision = subprocess.run(["git", "-c", f"safe.directory={ROOT.parent.as_posix()}", "rev-parse", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True, check=False)
    manifest = {"name": "E0924", "format": "source overlay for existing Effort-AIGI-Detection-main",
                "base_commit": revision.stdout.strip() if revision.returncode == 0 else None,
                "plan": plan(), "sha256": {}}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(files):
            data = (ROOT / relative).read_bytes()
            name = f"E0924/DeepfakeBench/{relative}"
            manifest["sha256"][name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
        archive.writestr("E0924/manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        archive.writestr("E0924/README.md", (ROOT / "experiments/E0924_README.md").read_bytes())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity check failed")
        for name, checksum in manifest["sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError(f"Archive checksum mismatch: {name}")
    return {"archive": str(destination), "files": len(files), "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "base_commit": manifest["base_commit"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.output), indent=2))
