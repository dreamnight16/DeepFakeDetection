"""The CUDA check is separate from long training and never creates a run."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/check_e1021.py"


def test_memory_check_dry_run_is_dependency_free_and_does_not_create_output(tmp_path):
    output = tmp_path / "unused"
    process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--dry_run", "--output_dir", str(output)],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    assert plan["mode"] == "synthetic CUDA memory check; no experiment results or checkpoint"
    assert plan["steps"] == 10
    assert list(plan["arms"]) == ["B0", "R2F2", "R4F4", "R2F6", "R6F2"]
    assert not output.exists()


def test_memory_check_requires_enough_steps_to_detect_cross_batch_growth():
    process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--steps", "3"],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 2
    assert "at least 4" in process.stderr


@pytest.mark.skipif(torch.cuda.is_available(), reason="This tests the CPU-only refusal path")
def test_no_cuda_is_reported_as_unverified_without_starting_training(tmp_path):
    output = tmp_path / "unused"
    process = subprocess.run([sys.executable, str(SCRIPT), "--arms", "R2F2", "--output_dir", str(output)],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 2
    assert "CUDA is required" in process.stderr
    assert not output.exists()
