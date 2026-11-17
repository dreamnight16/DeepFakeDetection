"""Float64 closed-form E1115 statistics; fit inputs never include evaluation labels."""

import copy
import numpy as np
from scipy.linalg import eigh


def moment_components(delta):
    """Equal-cell E[dd^T] = E[mean(d)mean(d)^T] + E[within-cell covariance]."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 3 or min(delta.shape) < 1 or not np.isfinite(delta).all():
        raise ValueError("Expected cells x equal-time-bins x features")
    means = delta.mean(axis=1)
    centered = delta - means[:, None]
    total = np.einsum("cti,ctj->ij", delta, delta) / np.prod(delta.shape[:2])
    mean = means.T @ means / len(means)
    variation = np.einsum("cti,ctj->ij", centered, centered) / np.prod(delta.shape[:2])
    return {"total": total, "mean": mean, "variation": variation}


def real_statistics(groups, tau=.1):
    groups = [np.asarray(x, dtype=np.float64) for x in groups]
    if not groups or not 0 < tau <= 1 or any(x.ndim != 2 or not len(x) or not np.isfinite(x).all() for x in groups):
        raise ValueError("Finite, nonempty real video groups and positive shrinkage required")
    mu = np.mean([x.mean(0) for x in groups], 0)
    covariance = sum((x - mu).T @ (x - mu) / len(x) for x in groups) / len(groups)
    scale = np.trace(covariance) / len(mu)
    if scale <= 0:
        raise ValueError("Degenerate real covariance")
    return mu, (1 - tau) * covariance + tau * scale * np.eye(len(mu))


def inverse_sqrt(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    floor = max(np.trace(matrix) / len(matrix) * 1e-8, np.finfo(float).tiny)
    return (vectors / np.sqrt(np.maximum(values, floor))) @ vectors.T


def spectral_weights(values, rank, mode):
    positive = np.asarray(values)[np.asarray(values) > 0]
    if rank > len(positive) or rank < 1:
        raise ValueError("Requested rank exceeds positive spectrum")
    selected = positive[:rank]
    if mode == "equal":
        weights = np.ones(rank)
    elif mode == "soft":
        weights = np.sqrt(selected)
    elif mode == "cap99":
        weights = np.minimum(selected, np.quantile(positive, .99))
    else:
        raise ValueError(f"Unknown spectrum mode: {mode}")
    return weights / weights.mean()


def fit_direction(moment, compression, covariance, rank, eta=1., weighting="soft"):
    moment, compression, covariance = [np.asarray(x, dtype=np.float64) for x in (moment, compression, covariance)]
    d = len(covariance)
    if (moment.shape != (d, d) or compression.shape != (d, d) or covariance.shape != (d, d)
            or not np.isfinite([moment, compression, covariance]).all() or not np.isfinite(eta) or eta < 0):
        raise ValueError("Invalid direction matrices or real-variance weight")
    def normalize(x):
        trace = np.trace(x)
        return x * d / trace if trace > 0 else np.zeros_like(x)
    denominator = normalize(compression) + eta * normalize(covariance) + 1e-6 * np.eye(d)
    values, vectors = eigh((moment + moment.T) / 2, (denominator + denominator.T) / 2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    tolerance = np.finfo(float).eps * d * max(np.abs(values).max(), 1e-300)
    positive = values[values > tolerance]
    weights = spectral_weights(positive, rank, weighting)
    return {"directions": vectors[:, :rank], "weights": weights,
            "raw_eigenvalues": values[:rank], "eta": float(eta), "weighting": weighting,
            "positive_rank": len(positive)}


def prepare_model(model):
    model = dict(model)
    model["white"] = (inverse_sqrt(model["covariance"]) if model.get("real_weight", 0.)
                      else np.empty((0, len(model["mu"]))))
    U = model["directions"]
    model["project_white"] = (inverse_sqrt(U.T @ model["covariance"] @ U)
                              if model.get("project_normalize", True) else np.eye(U.shape[1]))
    return model


def score_model(features, model, batch_size=1024):
    model = model if "project_white" in model else prepare_model(model)
    output = np.empty(len(features), dtype=np.float64)
    for begin in range(0, len(features), batch_size):
        centered = np.asarray(features[begin:begin + batch_size], dtype=np.float64) - model["mu"]
        y = centered @ model["directions"] @ model["project_white"].T
        normalizer = len(model["mu"]) if model.get("legacy", False) else len(model["weights"])
        direction = np.square(y) @ model["weights"] / normalizer
        real = (np.square(centered @ model["white"].T).sum(1) / len(model["mu"])
                if model.get("real_weight", 0.) else 0.)
        output[begin:begin + len(centered)] = model.get("real_weight", 0.) * real + model.get("direction_weight", 1.) * direction
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite metric scores")
    return output


def adapt_model(source, real_groups, mode, mix=.5, tau=.1):
    if mode not in ("threshold", "mean", "cov", "mean_cov") or not 0 <= mix <= 1:
        raise ValueError("Invalid target adaptation mode or covariance mix")
    model = copy.deepcopy(source)
    if mode == "threshold":
        return prepare_model(model)
    mu, covariance = real_statistics(real_groups, tau)
    if mode in ("mean", "mean_cov"):
        model["mu"] = mu
    if mode in ("cov", "mean_cov"):
        model["covariance"] = (1 - mix) * source["covariance"] + mix * covariance
    return prepare_model(model)
