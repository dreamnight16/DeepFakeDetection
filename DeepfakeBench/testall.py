"""
Usage:
    python3 testall.py \
        --detector_path ./training/config/detector/effort.yaml \
        --weights_path  /path/to/ckpt_best.pth \
        --test_datasets Celeb-DF-v1 Celeb-DF-v2 DFDC DFDCP FaceForensics++ UADFV
"""

import subprocess
import sys
import re
import argparse
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

TEST_SCRIPT = "training/test.py"

parser = argparse.ArgumentParser()
parser.add_argument("--detector_path", required=True)
parser.add_argument("--weights_path",  required=True)
parser.add_argument("--test_datasets", nargs="+", required=True)
parser.add_argument("--artifact_dir", default=None, help="Optional isolated output directory; metric computation is unchanged")
parser.add_argument("--dataset_json_folder", default=None)
parser.add_argument("--data_root", default=None)
args = parser.parse_args()

METRIC_RE = re.compile(r"^([a-zA-Z_]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")

def parse_metrics(lines):
    result = {}
    for line in lines:
        m = METRIC_RE.match(line.strip())
        if m:
            result[m.group(1)] = float(m.group(2))
    return result

def run_dataset(dataset):
    cmd = [
        sys.executable, TEST_SCRIPT,
        "--detector_path", args.detector_path,
        "--test_dataset",  dataset,
        "--weights_path",  args.weights_path,
    ]
    for option in ("artifact_dir", "dataset_json_folder", "data_root"):
        value = getattr(args, option)
        if value is not None:
            cmd.extend(["--" + option, value])

    print(f"\n{'='*60}")
    print(f"dataset: {dataset}")
    print(f"{'='*60}")
    sys.stdout.flush()

    import os
    # stderr goes straight to terminal → tqdm sees a real TTY → progress bar renders normally
    # stdout is captured for metric parsing
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,   # inherit parent's stderr = real terminal
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    collected = []
    for line in proc.stdout:
        line = line.decode("utf-8", errors="replace")
        sys.stdout.write(line)
        sys.stdout.flush()
        collected.append(line.rstrip("\n"))
    proc.wait()

    if proc.returncode != 0:
        print(f"[WARNING] test.py exited with code {proc.returncode} for {dataset}")

    return parse_metrics(collected)


# ── Run all datasets ──────────────────────────────────────────────────────────
all_metrics = {}
all_probs = {}   # 收集各数据集概率，用于最后画图

for ds in args.test_datasets:
    metrics = run_dataset(ds)
    if metrics:
        all_metrics[ds] = metrics
    else:
        print(f"[WARNING] No metrics parsed for {ds} — skipping from average.")
    # 读取 test.py 保存的临时概率文件（col0=prob, col1=label）
    npy_path = os.path.join(args.artifact_dir or "/tmp", f"effort_probs_{ds}.npy")
    if os.path.exists(npy_path):
        data = np.load(npy_path)   # shape [N, 2]
        all_probs[ds] = data  # shape [N, 2]：col0=prob, col1=label

# ── Print averages ────────────────────────────────────────────────────────────
if all_metrics:
    metric_names = list(next(iter(all_metrics.values())).keys())
    avg = {}
    for m in metric_names:
        vals = [all_metrics[ds][m] for ds in all_metrics if m in all_metrics[ds]]
        if vals:
            avg[m] = float(np.mean(vals))

    print(f"\n{'='*60}")
    print(f"dataset: average (over {len(all_metrics)} datasets: {', '.join(all_metrics.keys())})")
    for k, v in avg.items():
        print(f"{k}: {v}")

print("\n===> All Done!")

# ── 画各数据集 Probability Density 分布图（按 Real / Fake 分标签）────────────
if all_probs:
    n_ds  = len(all_probs)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=False)
    if n_ds == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    xs = np.linspace(0, 1, 500)

    for idx, (ds, data) in enumerate(all_probs.items()):
        ax    = axes[idx]
        color = colors[idx % len(colors)]
        probs  = data[:, 0]
        labels = data[:, 1].astype(int)

        for lbl, lname, ls in [(0, "Real", "--"), (1, "Fake", "-")]:
            subset = probs[labels == lbl]
            if len(subset) >= 2:
                kde = gaussian_kde(subset, bw_method=0.08)
                ys  = kde(xs)
                ax.plot(xs, ys, color=color, linewidth=2,
                        linestyle=ls, label=lname)
                ax.fill_between(xs, ys, alpha=0.12, color=color)

        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("Fake Probability", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Predicted Probability Density — Real vs Fake per Dataset",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(args.artifact_dir, "prob_density.png") if args.artifact_dir else "prob_density.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n概率密度分布图已保存至 {out_path}")
