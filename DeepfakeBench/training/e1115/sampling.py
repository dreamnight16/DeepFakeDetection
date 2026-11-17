"""Fixed-step class sampling and deterministic epoch-wise frame coverage."""

from pathlib import PurePosixPath
import numpy as np

from .protocol import stable_seed


def select_frames(paths, count, policy, seed, epoch, video):
    if count < 1 or not paths:
        raise ValueError("Need nonempty frame candidates and a positive count")
    if policy == "legacy":
        return list(paths[:count])
    if policy != "epoch_stratified":
        raise ValueError(f"Unknown frame policy: {policy}")
    ordered = sorted(paths, key=lambda p: int(PurePosixPath(p.replace("\\", "/")).stem))
    if len(ordered) <= count:
        return ordered
    rng = np.random.default_rng(stable_seed(seed, f"{epoch}:{video}"))
    bins = np.array_split(np.arange(len(ordered)), count)
    return [ordered[int(rng.choice(group))] for group in bins]


class FixedStepBatchSampler:
    """Shuffle each class without replacement; reshuffle on exhaustion, keep steps fixed."""

    def __init__(self, labels, batch_size, real_ratio, steps, seed=1024):
        self.labels = np.asarray(labels)
        if batch_size < 2 or steps < 2 or not 0 < real_ratio < 1:
            raise ValueError("Invalid fixed-budget sampler parameters")
        self.n_real = round(batch_size * real_ratio)
        self.n_fake = batch_size - self.n_real
        if min(self.n_real, self.n_fake) < 1 or set(np.unique(self.labels)) != {0, 1}:
            raise ValueError("Both binary classes are required in every batch")
        self.steps, self.seed, self.epoch = int(steps), seed, 0

    def set_epoch(self, epoch, labels=None):
        self.epoch = int(epoch)
        if labels is not None:
            labels = np.asarray(labels)
            if sorted(labels.tolist()) != sorted(self.labels.tolist()):
                raise ValueError("Epoch refresh changed class counts")
            self.labels = labels

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(stable_seed(self.seed, f"sampler:{self.epoch}"))
        def stream(label):
            indices = np.flatnonzero(self.labels == label)
            while True:
                yield from rng.permutation(indices).tolist()
        real, fake = stream(0), stream(1)
        for _ in range(self.steps):
            batch = [next(real) for _ in range(self.n_real)] + [next(fake) for _ in range(self.n_fake)]
            yield batch


def refresh_epoch(dataset, sampler, epoch):
    """Workers are recreated by the native loader; refresh before constructing its iterator."""
    if dataset.config.get("e1115_frame_policy") == "epoch_stratified":
        import random
        state = random.getstate()
        dataset.config["e1115_epoch"] = epoch
        try:
            random.seed(stable_seed(dataset.config["manualSeed"], f"dataset:{epoch}"))
            images, labels = [], []
            for name in dataset.config["train_dataset"]:
                paths, truth, _ = dataset.collect_img_and_label_for_one_dataset(name)
                images.extend(paths)
                labels.extend(truth)
        finally:
            random.setstate(state)
        dataset.image_list, dataset.label_list = images, labels
        dataset.data_dict = {"image": images, "label": labels}
    sampler.set_epoch(epoch, dataset.label_list)
    if dataset.config.get("log_dir"):
        import hashlib
        import json
        from pathlib import Path
        path = Path(dataset.config["log_dir"]) / "e1115_sampling.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows_hash = hashlib.sha256(json.dumps(dataset.image_list, ensure_ascii=False).encode()).hexdigest()
        record = {"epoch": epoch, "steps": len(sampler), "real_per_batch": sampler.n_real,
                  "fake_per_batch": sampler.n_fake, "real_draws": len(sampler) * sampler.n_real,
                  "fake_draws": len(sampler) * sampler.n_fake, "frame_list_sha256": rows_hash,
                  "frame_policy": dataset.config.get("e1115_frame_policy", "legacy")}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
