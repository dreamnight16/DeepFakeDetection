"""Binary mask encoding cannot silently attenuate spatial supervision."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from e1005_data import make_mask_reader


def test_binary_zero_one_and_zero_255_masks_have_identical_targets(tmp_path):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    Image.fromarray(mask).save(tmp_path / "binary01.png")
    Image.fromarray(mask * 255).save(tmp_path / "binary255.png")
    reader = make_mask_reader(tmp_path)
    first, second = reader("binary01.png"), reader("binary255.png")
    assert torch.equal(first, second)
    assert first.max() == 1 and first.sum() == 16


def test_resize_cannot_change_the_declared_grayscale_mask_scale(tmp_path):
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[0, 0] = 255  # This pixel is omitted by nearest 4->2 sampling.
    Image.fromarray(mask).save(tmp_path / "soft.png")
    target = make_mask_reader(tmp_path, resolution=2)("soft.png")
    assert torch.all(target == 1 / 255)
