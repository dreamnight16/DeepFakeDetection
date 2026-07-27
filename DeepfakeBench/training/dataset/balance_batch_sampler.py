"""
BalanceBatchSampler: ensures every batch contains exactly `batch_size_per_class`
samples from each class, producing strictly class-balanced mini-batches.

Usage (DataLoader):
    sampler = BalanceBatchSampler(dataset.label_list, batch_size_per_class=2)
    loader = DataLoader(dataset, batch_sampler=sampler, ...)
    # effective batch_size = num_classes * batch_size_per_class

When combined with DDP, wrap this sampler in a DistributedBatchSampler or
partition indices manually per rank — BalanceBatchSampler itself is rank-agnostic.
"""

import numpy as np
from torch.utils.data import Sampler


class BalanceBatchSampler(Sampler):
    """Sampler that yields batches with a controlled real:fake ratio per batch.

    Each batch contains ``n_real`` real and ``n_fake`` fake samples, where
    ``n_real / (n_real + n_fake) ≈ real_ratio`` and the total is kept at
    ``2 * batch_size_per_class``.  The number of full batches is limited by
    the minority-class count: ``num_batches = min(N_real // n_real, N_fake // n_fake)``.

    Args:
        labels: list or np.ndarray of integer class labels (0=real, 1=fake).
        batch_size_per_class: control for total batch size; N = 2 × this value.
        shuffle: if True, shuffle intra-class indices each epoch and shuffle
                 batch order.
        real_ratio: proportion of real samples per batch in (0,1).
                    Default 0.5 → equal real/fake (backward compatible).
    """

    def __init__(self, labels, batch_size_per_class: int, shuffle: bool = True,
                 real_ratio: float = 0.5):
        if batch_size_per_class < 1:
            raise ValueError(f"batch_size_per_class must be >= 1, got {batch_size_per_class}")
        if not (0.0 < real_ratio < 1.0):
            raise ValueError(f"real_ratio must be in (0, 1), got {real_ratio}")

        self.labels = np.asarray(labels)
        self.batch_size_per_class = batch_size_per_class
        self.shuffle = shuffle
        self.real_ratio = real_ratio

        # Group indices by class label
        unique_classes = np.unique(self.labels)
        self.class_indices: dict = {}
        for cls in unique_classes:
            self.class_indices[cls] = np.where(self.labels == cls)[0].tolist()

        # Compute per-class counts: keep total ≈ 2 * batch_size_per_class
        total = 2 * batch_size_per_class
        self.n_real = max(1, int(round(total * real_ratio)))
        self.n_fake = total - self.n_real

        sorted_cls = sorted(self.class_indices.keys())  # [0, 1]
        self.n_per_class = {
            sorted_cls[0]: self.n_real,   # class 0 = real
            sorted_cls[1]: self.n_fake,   # class 1 = fake
        }

        self.num_batches = min(
            len(self.class_indices[sorted_cls[0]]) // self.n_real,
            len(self.class_indices[sorted_cls[1]]) // self.n_fake,
        )

        if self.num_batches == 0:
            min_len = min(len(self.class_indices[c]) for c in sorted_cls)
            raise ValueError(
                f"Smallest class has only {min_len} samples, "
                f"but batch needs n_real={self.n_real} + n_fake={self.n_fake}. "
                f"Reduce batch_size_per_class or add more data."
            )

    def __iter__(self):
        # 1. Shuffle within each class (independently)
        indices_per_class = {}
        for cls, idx_list in self.class_indices.items():
            perm = idx_list.copy()
            if self.shuffle:
                np.random.shuffle(perm)
            indices_per_class[cls] = perm

        # 2. Build batches: take n_per_class[cls] from each class per step
        batches = []
        for step in range(self.num_batches):
            batch = []
            for cls in sorted(indices_per_class.keys()):
                n = self.n_per_class[cls]
                start = step * n
                end = start + n
                batch.extend(indices_per_class[cls][start:end])
            batches.append(batch)

        # 3. Shuffle batch order
        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches


class BalancePairSampler(Sampler):
    """v2: Each batch consists of N explicit (real, fake) pairs, interleaved.

    Yields index lists: [r1, f1, r2, f2, ..., rN, fN] (2N total per batch).
    When used with mixup, real[i] pairs with fake[i] directly — no randperm,
    so only RF pairs are produced (no RR, FF, or FR waste).

    Args:
        labels: list or np.ndarray of integer class labels (0=real, 1=fake).
        batch_size_per_class: number of pairs per batch (N).
        shuffle: if True, shuffle intra-class indices each epoch and shuffle
                 batch order.
    """

    def __init__(self, labels, batch_size_per_class: int, shuffle: bool = True):
        if batch_size_per_class < 1:
            raise ValueError(f"batch_size_per_class must be >= 1, got {batch_size_per_class}")

        self.labels = np.asarray(labels)
        self.batch_size_per_class = batch_size_per_class
        self.shuffle = shuffle

        self.real_indices = np.where(self.labels == 0)[0].tolist()
        self.fake_indices = np.where(self.labels == 1)[0].tolist()

        min_len = min(len(self.real_indices), len(self.fake_indices))
        self.num_batches = min_len // self.batch_size_per_class

        if self.num_batches == 0:
            raise ValueError(
                f"Smallest class has only {min_len} samples, "
                f"but batch_size_per_class={batch_size_per_class}. "
                f"Reduce batch_size_per_class or add more data."
            )

    def __iter__(self):
        r_idx = self.real_indices.copy()
        f_idx = self.fake_indices.copy()
        if self.shuffle:
            np.random.shuffle(r_idx)
            np.random.shuffle(f_idx)

        batches = []
        for step in range(self.num_batches):
            start = step * self.batch_size_per_class
            end = start + self.batch_size_per_class
            batch = []
            for r, f in zip(r_idx[start:end], f_idx[start:end]):
                batch.append(r)
                batch.append(f)
            batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches
