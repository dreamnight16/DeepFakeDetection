"""Run G22: query the representation entering CLIP ViT's final block."""

import sys

from query_ablation_runner import run


if __name__ == "__main__":
    sys.exit(0 if run("G22")["status"] == "OK" else 1)
