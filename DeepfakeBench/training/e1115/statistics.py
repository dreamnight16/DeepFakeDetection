"""Video/segment/exact-index paired moments with explicit shared cohorts."""

import numpy as np

METHODS = ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")


def build_moments(audit, lookup):
    bags = {(b["source"], b["method"]): b for b in audit["fit_videos"]}
    sources = audit["complete_sources"]
    if not sources:
        raise ValueError("No complete source-method cells")
    d = len(next(iter(lookup.values())))
    names = ["mean", "block2", "block4", "block8", "variation4", "block_mean4",
             "matched_mean", "matched_block", "matched_frame", "matched_variation"]
    sums = {name: np.zeros((d, d)) for name in names}
    counts = dict.fromkeys(names, 0)
    matched_sources, common_counts, block_counts = [], {}, {}
    def add(name, delta):
        sums[name] += delta.T @ delta / len(delta)
        counts[name] += 1
    for source in sources:
        real_rows = sorted(audit["real_frames"]["fit"][source], key=lambda x: x["frame"])
        real_map = {x["frame"]: np.asarray(lookup[x["path"]], dtype=np.float64) for x in real_rows}
        real = np.stack(list(real_map.values()))
        fake_maps = {}
        for method in METHODS:
            rows = sorted(bags[source, method]["frames"], key=lambda x: x["frame"])
            fake_map = {x["frame"]: np.asarray(lookup[x["fake_path"]], dtype=np.float64) for x in rows}
            fake_maps[method] = fake_map
            fake = np.stack(list(fake_map.values()))
            mean = fake.mean(0) - real.mean(0)
            add("mean", mean[None])
            for blocks in (2, 4, 8):
                k = min(blocks, len(real), len(fake))
                # Equal-count ordered bins are weak temporal matching, not equal timestamps.
                delta = np.stack([f.mean(0) - r.mean(0)
                                  for f, r in zip(np.array_split(fake, k), np.array_split(real, k))])
                add(f"block{blocks}", delta)
                if blocks == 4:
                    add("block_mean4", delta.mean(0)[None])
                    add("variation4", delta - delta.mean(0))
                block_counts.setdefault(str(blocks), []).append(k)
        if "e1115_matched" in audit:
            aligned = audit["e1115_matched"].get(source)
            if aligned is None:
                continue
            real_map = {x["frame"]: np.asarray(lookup[x["path"]], dtype=np.float64) for x in aligned["real"]}
            fake_maps = {m: {x["frame"]: np.asarray(lookup[x["fake_path"]], dtype=np.float64) for x in rows}
                         for m, rows in aligned["fake"].items()}
        common = sorted(set(real_map).intersection(*(set(x) for x in fake_maps.values())))
        if len(common) < 2:
            continue
        matched_sources.append(source)
        common_counts[source] = len(common)
        for method in METHODS:
            delta = np.stack([fake_maps[method][t] - real_map[t] for t in common])
            mean = delta.mean(0)
            add("matched_mean", mean[None])
            add("matched_frame", delta)
            add("matched_variation", delta - mean)
            add("matched_block", np.stack([x.mean(0) for x in np.array_split(delta, min(4, len(delta)))]))
    moments = {key: sums[key] / count for key, count in counts.items() if count}
    np.testing.assert_allclose(moments["block4"], moments["block_mean4"] + moments["variation4"],
                               rtol=1e-9, atol=1e-10)
    if "matched_frame" in moments:
        np.testing.assert_allclose(moments["matched_frame"], moments["matched_mean"] + moments["matched_variation"],
                                   rtol=1e-9, atol=1e-10)
    return moments, {"full_sources": list(sources), "matched_sources": matched_sources,
                     "matched_common_frames": common_counts, "cell_counts": counts,
                     "effective_block_counts": {k: {"min": min(v), "max": max(v)} for k, v in block_counts.items()},
                     "paired_scope": "same saved frame index and official lineage; existing independent crops, not shared recropping",
                     "block_scope": "independent ordered equal-count bins; no framewise correspondence claim"}
