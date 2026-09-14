"""Run G23: replace full query blocks with cross-attention-only blocks."""

import sys

from query_ablation_runner import run


if __name__ == "__main__":
    sys.exit(0 if run("G23")["status"] == "OK" else 1)
