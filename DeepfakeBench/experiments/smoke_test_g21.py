"""Server-side G21 verification: synthetic CPU tensors only, no CLIP download."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "training"))
    tests = unittest.defaultTestLoader.discover(
        str(root / "tests/g21"), pattern="test_*.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful() and result.testsRun > 0:
        print(
            f"ALL PASS: {result.testsRun} G21 server checks; no real model was loaded."
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
