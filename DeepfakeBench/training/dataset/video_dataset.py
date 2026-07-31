"""Thin wrapper: fixes video_mode texture_score_tensors for collate_fn."""
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class VideoDeepfakeDataset(DeepfakeAbstractBaseDataset):
    """Same as DeepfakeAbstractBaseDataset, but forces texture_scores=None
    in video_level mode so the static collate_fn handles it correctly."""

    def __getitem__(self, index, no_norm=False):
        img, lbl, lmk, msk, ts = super().__getitem__(index, no_norm)
        if self.video_level:
            ts = None
        return img, lbl, lmk, msk, ts
