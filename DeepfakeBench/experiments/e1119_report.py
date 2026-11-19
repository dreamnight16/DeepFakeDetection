"""E1119 main, replication, and paired-bootstrap tables."""

import csv
import io
import json

import numpy as np

from g21.io import atomic_json, atomic_text
from run_g25 import CROSS_DS, TEST_DS, VAL_DS


PRIMARY = "independent_video_auc_mean_fullpath"
COMPARISONS = (
    ("B02", "B00"), ("B03", "B00"), ("B04", "B00"), ("B02", "B03"),
    ("B04", "B02"), ("B04", "B03"), ("M01", "B00"), ("M02", "M01"),
    ("M04", "M05"), ("M04", "M06"),
    *((arm, comparator) for arm in ("M03", "M04", "M05", "M06")
      for comparator in ("B00", "M03") if arm != comparator),
    *(("P01", arm) for arm in ("B00", "P02", "P03", "P04")),
    ("T01", "B00"), ("T01", "T02"), ("T01", "T03"),
    ("I01", "B00"), ("I02", "I01"), ("I03", "B00"), ("I01", "I03"),
    ("I04", "I01"), ("I04", "I02"), ("I04", "B00"),
    ("I05", "I03"), ("I05", "B00"), ("I04", "I05"),
)
BOOTSTRAPS = 2000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path, rows):
    if rows:
        stream = io.StringIO()
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(path, stream.getvalue())


def difference(value, reference):
    return None if value is None or reference is None else 100 * (value - reference)


def report(folder, arms, seeds, jobs, history=None):
    historical = read(history) if history else None
    if historical and (historical.get("seed") != 1024 or not historical.get("source")):
        raise ValueError("The historical record needs seed=1024 and a source")
    material = read(folder / "inputs.json")
    verified = historical and historical.get("input_identity") == material["input_identity"]
    if historical:
        atomic_json(folder / "historical_b0.json", {**historical, "input_identity_matched": bool(verified)})
    all_rows, long_form, protocol_checks = [], [], []
    baselines = {}
    per_run = {}
    for seed in seeds:
        for arm in arms:
            train = jobs[f"train:{arm}:{seed}"]
            evaluation = jobs[f"eval:{arm}:{seed}"]
            train_folder, eval_folder = train.get("folder"), evaluation.get("folder")
            training = read(train_folder / "training.json") if train_folder and (train_folder / "training.json").is_file() else {}
            protocol_path = train_folder / "training_protocol.json" if train_folder else None
            protocol = read(protocol_path) if protocol_path and protocol_path.is_file() else {}
            result = read(eval_folder / "evaluation.json") if eval_folder and (eval_folder / "evaluation.json").is_file() else {}
            per_run[(arm, seed)] = (protocol, result)
            if arm == "B00":
                baselines[seed] = result.get(PRIMARY)
            row = {"arm": arm, "seed": seed, "status": result.get("status", evaluation["status"]),
                   "train_status": train["status"], "evaluation_status": evaluation["status"],
                   "training_dir": str(train_folder or ""), "evaluation_dir": str(eval_folder or ""),
                   PRIMARY: result.get(PRIMARY), "same_seed_B00": baselines[seed],
                   "delta_same_seed_B00_pp": difference(result.get(PRIMARY), baselines[seed]),
                   "video_auc_seven_mean_fullpath": result.get("video_auc_seven_mean_fullpath"),
                   "missing_datasets": ",".join(result.get("missing_datasets", [])),
                   "trainable": protocol.get("trainable_count"), **training}
            if seed == 1024:
                row["historical_B0_source"] = historical["source"] if historical else "unavailable"
                row["historical_B0"] = historical.get(PRIMARY) if historical else None
                row["delta_historical_B0_pp"] = difference(result.get(PRIMARY), row["historical_B0"]) if verified else None
            all_rows.append(row)
            for dataset in TEST_DS:
                m = result.get("datasets", {}).get(dataset, {})
                long_form.append({"arm": arm, "seed": seed, "dataset": dataset, **m})
    for (arm, seed), (protocol, _) in per_run.items():
        baseline = per_run[("B00", seed)][0]
        if protocol and baseline:
            keys = ("head_initial_sha256", "loader_rng_sha256", "train_list_order_sha256", "steps_per_epoch",
                    "total_steps", "real_per_batch", "fake_per_batch")
            protocol_checks.append({"arm": arm, "seed": seed,
                                    **{key: protocol[key] == baseline[key] for key in keys}})
    table(folder / "runs.csv", all_rows)
    table(folder / "main_seed1024.csv", [row for row in all_rows if row["seed"] == 1024])
    table(folder / "datasets.csv", long_form)
    table(folder / "protocol_alignment.csv", protocol_checks)
    replica = []
    for arm in arms:
        rows = [row for row in all_rows if row["arm"] == arm]
        aggregate = {"arm": arm}
        for key in (PRIMARY, "delta_same_seed_B00_pp"):
            values = [row.get(key) for row in rows]
            aggregate.update({f"{key}_{seed}": v for seed, v in zip(seeds, values)})
            aggregate[f"{key}_mean"] = float(np.mean(values)) if None not in values else None
        aggregate.update({f"status_{seed}": row["status"] for seed, row in zip(seeds, rows)})
        replica.append(aggregate)
    table(folder / "replication.csv", replica)
    if bootstrap_rows := intervals(jobs, seeds):
        table(folder / "paired_bootstrap.csv", bootstrap_rows)

    # D01 is always reported against the frozen backbone at the same layer.
    reference = jobs["D01:CLIP"].get("folder")
    reference = read(reference / "analysis.json") if reference and (reference / "analysis.json").is_file() else {}
    analyses, drift_rows = [], []
    for key, job in jobs.items():
        if key.split(":")[0] not in ("D01", "D02", "D03", "mechanism"):
            continue
        path = job.get("folder")
        row = {"task": key, "status": job["status"], "directory": str(path or "")}
        analyses.append(row)
        if key.startswith("D01:") and path and (path / "analysis.json").is_file():
            rows = []
            for dataset, item in read(path / "analysis.json")["datasets"].items():
                reference_layers = reference.get("datasets", {}).get(dataset, {}).get("layers", {})
                for depth, values in item.get("layers", {}).items():
                    auc = values["video_auc_fullpath"]
                    ref = reference_layers.get(depth, {}).get("video_auc_fullpath")
                    rows.append({"dataset": dataset, "block": depth, "video_auc_fullpath": auc,
                                 "frozen_clip_video_auc": ref, "delta_frozen_clip_pp": difference(auc, ref)})
            table(path / "layers.csv", rows)
        if key.startswith("D03:") and path and (path / "analysis.json").is_file():
            _, arm, seed = key.split(":")
            result = per_run[(arm, int(seed))][1]
            for dataset, item in read(path / "analysis.json")["datasets"].items():
                auc = result.get("datasets", {}).get(dataset, {}).get("video_auc_fullpath")
                for depth, values in item.get("layers", {}).items():
                    drift_rows.append({"arm": arm, "seed": seed, "dataset": dataset, "block": depth,
                                       "video_auc_fullpath": auc, "real_drift": values["real"]["mean"],
                                       "fake_drift": values["fake"]["mean"],
                                       "real_videos": values["real"]["videos"], "fake_videos": values["fake"]["videos"]})
    table(folder / "analyses.csv", analyses)
    table(folder / "d03_drift.csv", drift_rows)


def auc_resamples(scores, labels, counts):
    order = np.argsort(scores, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    counts = counts[:, order]
    tp = np.add.reduceat(counts * labels[order], boundaries, axis=1)
    fp = np.add.reduceat(counts * (1 - labels[order]), boundaries, axis=1)
    return (tp * (fp.cumsum(axis=1) - .5 * fp)).sum(axis=1) / (tp.sum(axis=1) * fp.sum(axis=1))


def paired_samples(first, second, state):
    if not np.array_equal(first["video_id"], second["video_id"]) or not np.array_equal(first["label"], second["label"]):
        raise ValueError("Paired intervals require exactly aligned full-path video rows")
    y = first["label"]
    real, fake = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(real) or not len(fake):
        raise ValueError("Paired intervals require both classes")
    values = []
    for begin in range(0, BOOTSTRAPS, 64):
        n = min(64, BOOTSTRAPS - begin)
        weights = np.zeros((n, len(y)), dtype=np.int32)
        for group in (real, fake):
            weights[:, group] = state.multinomial(len(group), np.full(len(group), 1 / len(group)), size=n)
        values.extend(auc_resamples(first["prob_fake"], y, weights) -
                      auc_resamples(second["prob_fake"], y, weights))
    return np.asarray(values)


def intervals(jobs, seeds):
    named = [name for name in CROSS_DS if name != VAL_DS]
    datasets = {}

    def video_data(arm, seed, dataset):
        key = (arm, seed, dataset)
        if key not in datasets:
            job = jobs[f"eval:{arm}:{seed}"]
            folder = job.get("folder")
            if job["status"] in ("OK", "PARTIAL") and folder and (folder / dataset / "videos.npz").is_file():
                with np.load(folder / dataset / "videos.npz", allow_pickle=False) as handle:
                    datasets[key] = {name: handle[name] for name in ("video_id", "label", "prob_fake")}
            else:
                datasets[key] = None
        return datasets[key]

    rows = []
    for seed in seeds:
        for first, second in COMPARISONS:
            entry = {"seed": seed, "arm": first, "comparison": second, "resamples": BOOTSTRAPS}
            try:
                samples = []
                for i, dataset in enumerate(named):
                    a, b = video_data(first, seed, dataset), video_data(second, seed, dataset)
                    if a is None or b is None:
                        raise ValueError(f"Independent predictions unavailable for {dataset}")
                    samples.append(paired_samples(a, b, np.random.default_rng(1119 + i)))
                lower, upper = np.percentile(np.mean(samples, axis=0) * 100, [2.5, 97.5])
                entry.update(status="OK", low_pp=float(lower), high_pp=float(upper))
            except ValueError as exc:
                entry.update(status="UNAVAILABLE", error=str(exc))
            rows.append(entry)
    return rows
