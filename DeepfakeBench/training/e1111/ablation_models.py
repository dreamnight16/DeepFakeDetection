"""Source-only ablation fitting and batched sufficient-direction scoring."""

import numpy as np

from .metric import ProtocolError, fit_space, inverse_sqrt, real_statistics


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ProtocolError("Spherical prototypes require finite nonzero features")
    return values / norms


def spherical_kmeans(values, k, weights=None, seed=1024, iterations=100):
    x = unit(values)
    if not 1 <= k <= len(x):
        raise ProtocolError("Prototype count exceeds available real samples")
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(x),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ProtocolError("Prototype weights must be finite and positive")
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.choice(len(x), p=weights))]]
    while len(centers) < k:
        distance = np.maximum(1 - (x @ np.stack(centers).T).max(axis=1), 0)
        mass = weights * distance
        if mass.sum() <= 1e-14:
            raise ProtocolError("Prototype K exceeds distinct spherical support")
        centers.append(x[int(rng.choice(len(x), p=mass / mass.sum()))])
    centers = np.stack(centers)
    for _ in range(iterations):
        similarities = x @ centers.T
        labels = similarities.argmax(axis=1)
        updated, empty = [], []
        for j in range(k):
            keep = labels == j
            if not keep.any():
                empty.append(j)
                updated.append(None)
            else:
                updated.append(unit(np.average(x[keep], axis=0, weights=weights[keep])))
        active = [value for value in updated if value is not None]
        for j in empty:
            distance = np.maximum(1 - (x @ np.stack(active).T).max(axis=1), 0)
            if distance.max() <= 1e-14:
                raise ProtocolError("Prototype K exceeds distinct spherical support")
            value = x[int(np.argmax(weights * distance))]
            updated[j] = value
            active.append(value)
        updated = unit(updated)
        if np.max(np.abs(updated - centers)) <= 1e-8:
            centers = updated
            break
        centers = updated
    dispersion = float(np.sum(weights * np.maximum(1 - (x @ centers.T).max(axis=1), 0)))
    return centers, dispersion


def gap_diagnostic(rad_groups, seed=1024, repeats=20):
    x = unit(np.stack([np.mean(group, axis=0) for group in rad_groups]))
    mean = x.mean(axis=0)
    _, _, axes = np.linalg.svd(x - mean, full_matrices=False)
    projected = (x - mean) @ axes.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    rng = np.random.default_rng(seed)
    references = [unit(mean + rng.uniform(low, high, size=projected.shape) @ axes) for _ in range(repeats)]
    rows = []
    for k in range(1, min(8, len(x)) + 1):
        _, observed = spherical_kmeans(x, k, seed=seed)
        reference = [spherical_kmeans(sample, k, seed=seed + i)[1] for i, sample in enumerate(references)]
        logs = np.log(np.maximum(reference, 1e-15))
        rows.append({"K": k, "dispersion": observed,
                     "gap": float(logs.mean() - np.log(max(observed, 1e-15))),
                     "se": float(logs.std(ddof=1) * np.sqrt(1 + 1 / repeats))})
    selected = rows[-1]["K"]
    for left, right in zip(rows, rows[1:]):
        if left["gap"] >= right["gap"] - right["se"]:
            selected = left["K"]
            break
    chosen = rows[selected - 1]
    if selected > 1 and chosen["gap"] - rows[0]["gap"] <= chosen["se"]:
        selected = 1
    return {"status": "COMPUTED", "selected_K": selected, "table": rows,
            "fit_split": "D_rad real video means only", "seed": seed, "reference_repeats": repeats,
            "reference": "uniform PCA bounding box, reconstructed and normalized to the sphere",
            "selection": "smallest K satisfying gap one-SE rule; K>1 also needs gain over K=1 > its SE"}


def compact(model, gamma, covariance_key="main", vectors=None, weights=None):
    vectors = model["vectors"] if vectors is None else vectors
    weights = model["weights"] if weights is None else weights
    return {"kind": "quadratic", "mu": model["mu"], "white": model["white"],
            "directions": model["transform"].T @ vectors, "weights": weights,
            "gamma": gamma, "covariance_key": covariance_key}


def from_space(space, covariance, mu, rank, gamma, key, white=None):
    if rank > space.diagnostics["positive_rank"]:
        raise ProtocolError(f"Requested rank {rank} exceeds positive rank {space.diagnostics['positive_rank']}")
    return {"kind": "quadratic", "mu": mu, "white": inverse_sqrt(covariance) if white is None else white,
            "directions": space.transform.T @ space.vectors[:, :rank], "weights": space.weights[:rank],
            "gamma": gamma, "covariance_key": key}


def fit_variants(main, identity, s0, settings, delta, fit_real, rad_real, dc):
    tau, lam, rank, gamma = s0["tau_R"], identity["lambda_m"], identity["rank"], identity["gamma"]
    models = {"D_J": compact(main, gamma), "D_R": compact(main, 0.), "D_F": compact(main, "inf"),
              "D_P": compact(main, "inf", weights=np.ones_like(main["weights"]))}
    for name in ("C_random", "C_shuffle", "C_real"):
        if name + "_V" in main:
            models[name] = compact(main, "inf", vectors=main[name + "_V"], weights=main[name + "_w"])
    models["M_style"] = {"kind": "gaussian_nll", "mu": main["mu"], "white": main["white"],
                         "covariance_key": "main", "nll_constant": float(
                             -np.linalg.slogdet(main["white"])[1] + len(main["mu"]) / 2 * np.log(2 * np.pi))}
    specs = {name: {"status": "FITTED", "scope": "source_only"} for name in models}
    for name in ("C_random", "C_shuffle", "C_real"):
        if name not in models:
            specs[name] = {"status": "UNDEFINED", "scope": "source_only",
                           "reason": identity.get("omitted_controls", {}).get(name, "Control absent from frozen model")}
    specs["M_style"].update(note="single-Gaussian NLL M-style reimplementation; affine in D_R, not a separate flow")
    spaces, covariances, whites = {}, {}, {}
    def variant(name, r=rank, l=lam, t=tau, g=gamma):
        try:
            if t not in covariances:
                covariances[t] = real_statistics(fit_real, t)
                whites[t] = inverse_sqrt(covariances[t][1])
            mu, covariance, _ = covariances[t]
            if (t, l) not in spaces:
                print(f"E1111A/ablations/fit: tau={t}, lambda_m={l}", flush=True)
                spaces[(t, l)] = fit_space(delta, covariance, dc, l, identity["family"])
            space = spaces[(t, l)]
            key = "main" if t == tau else f"tau_{t}"
            models[name] = from_space(space, covariance, mu, r, g, key, white=whites[t])
            specs[name] = {"status": "FITTED", "scope": "source_only", "rank": r,
                           "lambda_m": l, "lambda_t": 0., "tau_R": t, "gamma": g,
                           "numeric_diagnostics": space.diagnostics}
        except (ProtocolError, np.linalg.LinAlgError) as exc:
            specs[name] = {"status": "UNDEFINED", "reason": str(exc), "scope": "source_only"}
    variant("C4_no_method_reg", l=0.)
    specs["C4_no_method_reg"]["definition"] = "lambda_m=lambda_t=0; D_c retained; rank/tau/gamma fixed at main"
    for r in settings["ranks"] if identity["family"] == "B" else [1]:
        variant(f"rank_{r}", r=r)
    for l in settings["lambda_m"]:
        variant(f"lambda_{l:g}", l=l)
    for t in settings["tau_R"]:
        variant(f"tau_{t:g}", t=t)
    for g in settings["gamma"]:
        name = f"gamma_{g}"
        models[name] = compact(main, g)
        specs[name] = {"status": "FITTED", "scope": "source_only", "gamma": g,
                       "fixed": "main rank/lambda/tau and learned directions"}
    try:
        print("E1111A/ablations/D4_gap: RUNNING", flush=True)
        d4 = gap_diagnostic(rad_real, settings["seed"])
    except (ProtocolError, np.linalg.LinAlgError) as exc:
        d4 = {"status": "UNDEFINED", "reason": str(exc), "selected_K": 1}
    points = np.concatenate(fit_real)
    weights = np.concatenate([np.full(len(group), 1 / len(group)) for group in fit_real])
    for k in range(1, 9):
        name = f"R_style_K{k}"
        try:
            print(f"E1111A/ablations/{name}: FITTING", flush=True)
            centers, dispersion = spherical_kmeans(points, k, weights, settings["seed"])
            models[name] = {"kind": "prototypes", "centers": centers}
            specs[name] = {"status": "FITTED", "scope": "source_only", "K": k,
                           "dispersion": dispersion, "definition": "spherical prototypes; nearest cosine distance; no covariance; equal real-source weight"}
        except ProtocolError as exc:
            specs[name] = {"status": "UNDEFINED", "reason": str(exc), "scope": "source_only"}
    chosen = f"R_style_K{d4['selected_K']}"
    if chosen in models:
        models["R_style"] = models[chosen]
        specs["R_style"] = {**specs[chosen], "selected_by": "D_rad gap; never target labels or test AUC"}
    else:
        specs["R_style"] = {"status": "UNDEFINED", "scope": "source_only", "reason": "D_rad-selected prototype fit unavailable"}
    return models, specs, d4


def score_models(features, models, batch_size=512):
    result = {name: np.empty(len(features), dtype=np.float64) for name in models}
    for start in range(0, len(features), batch_size):
        z = np.asarray(features[start:start + batch_size], dtype=np.float64)
        real_cache = {}
        for name, model in models.items():
            if model["kind"] == "prototypes":
                values = np.maximum(1 - (unit(z) @ model["centers"].T).max(axis=1), 0)
            else:
                centered = z - model["mu"]
                d = z.shape[1]
                needs_real = model["kind"] == "gaussian_nll" or model["gamma"] != "inf"
                if needs_real and model["covariance_key"] not in real_cache:
                    real_cache[model["covariance_key"]] = np.square(centered @ model["white"].T).sum(axis=1) / d
                if model["kind"] == "gaussian_nll":
                    values = .5 * d * real_cache[model["covariance_key"]] + model["nll_constant"]
                else:
                    energy = np.square(centered @ model["directions"]) @ model["weights"] / d
                    values = energy if model["gamma"] == "inf" else real_cache[model["covariance_key"]] + float(model["gamma"]) * energy
            if not np.isfinite(values).all():
                raise ProtocolError(f"Nonfinite ablation score: {name}")
            result[name][start:start + len(z)] = values
    return result
