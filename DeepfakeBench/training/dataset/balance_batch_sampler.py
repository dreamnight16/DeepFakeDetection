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
    """Sampler that yields batches with exactly N samples per class.

    Each batch contains `batch_size_per_class` samples from every class present
    in the dataset.  The total number of batches is determined by the class with
    the fewest samples:  num_batches = floor(min_class_count / batch_size_per_class).
    Samples beyond the last full batch are dropped (one epoch covers all full batches).

    Args:
        labels: list or np.ndarray of integer class labels for the entire dataset.
        batch_size_per_class: number of samples drawn from each class per batch.
        shuffle: if True, shuffle intra-class indices each epoch and shuffle
                 batch order.
    """

    def __init__(self, labels, batch_size_per_class: int, shuffle: bool = True):
        if batch_size_per_class < 1:
            raise ValueError(f"batch_size_per_class must be >= 1, got {batch_size_per_class}")

        self.labels = np.asarray(labels)
        self.batch_size_per_class = batch_size_per_class
        self.shuffle = shuffle

        # Group indices by class label
        unique_classes = np.unique(self.labels)
        self.class_indices: dict = {}
        for cls in unique_classes:
            self.class_indices[cls] = np.where(self.labels == cls)[0].tolist()

        # Number of full balanced batches = floor(min_class_count / batch_size_per_class)
        min_class_len = min(len(v) for v in self.class_indices.values())
        self.num_batches = min_class_len // self.batch_size_per_class

        if self.num_batches == 0:
            raise ValueError(
                f"Smallest class has only {min_class_len} samples, "
                f"but batch_size_per_class={batch_size_per_class}. "
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

        # 2. Build batches: take batch_size_per_class from each class per step
        batches = []
        for step in range(self.num_batches):
            batch = []
            for cls in sorted(indices_per_class.keys()):
                start = step * self.batch_size_per_class
                end = start + self.batch_size_per_class
                batch.extend(indices_per_class[cls][start:end])
            batches.append(batch)

        # 3. Shuffle batch order
        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches
