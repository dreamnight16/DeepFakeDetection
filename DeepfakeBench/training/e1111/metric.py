"""Float64 E1111A empirical decomposition, nested selection and frozen scoring."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_triangular

from .protocol import protocol


class ProtocolError(ValueError):
    """An undefined statistic or unmet design condition; never a numeric zero."""


class FisherRequired(ProtocolError):
    """Registered Cholesky recovery exhausted; restart the entire Fisher D1."""


def second_moment(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or not np.isfinite(values).all() or values.size == 0:
        raise ProtocolError("Expected finite, nonempty feature vectors")
    values = values.reshape(-1, values.shape[-1])
    return values.T @ values / len(values)


def decompose(delta):
    delta = np.asarray(delta, dtype=np.float64)
    if (delta.ndim not in (3, 4) or min(delta.shape[:2]) < 2
            or (delta.ndim == 4 and delta.shape[2] < 2)):
        raise ProtocolError("Decomposition requires video x method x feature, or legacy equal-frame differences")
    B = second_moment(delta)
    cell = delta.mean(axis=2) if delta.ndim == 4 else delta
    mu = cell.mean(axis=(0, 1))
    method = cell.mean(axis=0) - mu
    video = cell.mean(axis=1) - mu
    interaction = cell - mu - method[None] - video[:, None]
    parts = {"mean": np.outer(mu, mu), "method": second_moment(method),
             "video": second_moment(video), "inter": second_moment(interaction)}
    if delta.ndim == 4:
        parts["resid"] = second_moment(delta - cell[:, :, None])
    trace = float(np.trace(B))
    if trace <= 0:
        raise ProtocolError("tr(B)=0: paired differences have no energy; stop S0")
    error = float(np.linalg.norm(B - sum(parts.values())) / np.linalg.norm(B))
    ratios = {key: float(np.trace(value) / trace) for key, value in parts.items()}
    if error > 1e-10 or abs(sum(ratios.values()) - 1) > 1e-10:
        raise ProtocolError("Empirical decomposition identity failed; check sample weights")
    return B, parts, {"rho": ratios, "relative_reconstruction_error": error,
                      "rho_sum": sum(ratios.values())}


def real_statistics(groups, tau):
    groups = [np.asarray(x, dtype=np.float64) for x in groups]
    if not groups or any(x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all() for x in groups):
        raise ProtocolError("Every real source must have finite features")
    mu = np.mean([x.mean(axis=0) for x in groups], axis=0)
    d = len(mu)
    # Accumulate dxd matrices instead of materializing 395 copies per statistic.
    total, within = np.zeros((d, d)), np.zeros((d, d))
    for x in groups:
        total += second_moment(x - mu)
        within += second_moment(x - x.mean(axis=0))
    total /= len(groups)
    within /= len(groups)
    if np.trace(total) <= 0:
        raise ProtocolError("Real features are degenerate; stop S0")
    covariance = (1 - tau) * total + tau * np.trace(total) / d * np.eye(d)
    return mu, covariance, within


def inverse_sqrt(covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = 1e-4 * np.trace(covariance) / len(covariance)
    if floor <= 0 or not np.isfinite(eigenvalues).all():
        raise ProtocolError("Real covariance cannot be whitened")
    return (eigenvectors / np.sqrt(np.maximum(eigenvalues, floor))) @ eigenvectors.T


def choose_tau(fit_real, sel_ffpp_real, candidates=None):
    candidates = protocol()["tau_R"] if candidates is None else candidates
    rows = []
    for tau in candidates:
        mu, covariance, _ = real_statistics(fit_real, tau)
        # Same eigenvalue floor used in all later white-coordinate operations.
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-4 * np.trace(covariance) / len(mu))
        white = inverse_sqrt(covariance)
        nll = np.mean([.5 * (np.sum(((x - mu) @ white.T) ** 2, axis=1).mean()
                                  + np.log(eigenvalues).sum()) for x in sel_ffpp_real])
        if not math.isfinite(nll):
            raise ProtocolError("Undefined real validation NLL")
        rows.append({"tau": float(tau), "real_video_mean_nll": float(nll)})
    minimum = min(x["real_video_mean_nll"] for x in rows)
    winner = min(x["tau"] for x in rows if x["real_video_mean_nll"] <= minimum + 1e-12)
    return winner, rows


@dataclass
class Space:
    transform: np.ndarray
    vectors: np.ndarray
    weights: np.ndarray
    diagnostics: dict


def fit_space(delta, covariance, dc, lam, family="B"):
    if np.ndim(delta) == 3 and family == "Fisher":
        raise ProtocolError("Video-mean mode has no frame-residual Fisher denominator")
    B, parts, _ = decompose(delta)
    d = len(covariance)
    dc = np.asarray(dc, dtype=np.float64)
    if dc.shape != (d, d) or not np.isfinite(dc).all():
        raise ProtocolError("D_c has wrong shape or nonfinite entries")
    if np.linalg.eigvalsh((dc + dc.T) / 2).min() < -1e-10 * max(1., np.linalg.norm(dc)):
        raise ProtocolError("D_c is not positive semidefinite")
    dm_trace = float(np.trace(parts["method"]))
    dc_trace = float(np.trace(dc))
    dm = parts["method"] * dc_trace / dm_trace if dm_trace > 0 else np.zeros_like(dc)
    white = inverse_sqrt(covariance)
    numerator = B if family == "B" else parts["method"]
    if family not in ("B", "Fisher") or np.trace(numerator) <= 0:
        raise ProtocolError("No numerator energy for requested model family")
    attempts = []
    # These are precisely the registered numeric recovery steps, each recorded.
    for epsilon_scale, effective_lam in ((1e-6, lam), (1e-5, lam), (1e-5, 0.)):
        epsilon = max(1e-6 * dc_trace / d, epsilon_scale * np.trace(covariance) / d)
        nuisance = dc + effective_lam * dm + epsilon * np.eye(d)
        denominator = nuisance if family == "B" else nuisance + parts["resid"]
        du = white @ denominator @ white.T
        du = (du + du.T) / 2
        values = np.linalg.eigvalsh(du)
        record = {"epsilon_scale": epsilon_scale, "requested_lambda_m": float(lam),
                  "effective_lambda_m": float(effective_lam), "tr_Dc": dc_trace,
                  "tr_Sigma_R": float(np.trace(covariance)),
                  "lambda_min_Dnuis": float(np.linalg.eigvalsh(nuisance).min()),
                  "lambda_min_Du": float(values.min()),
                  "condition_Du": float(values.max() / values.min()) if values.min() > 0 else None}
        attempts.append(record)
        try:
            L = np.linalg.cholesky(du)
            break
        except np.linalg.LinAlgError:
            continue
    else:
        raise FisherRequired(f"Cholesky failed after registered steps; rerun nested Fisher, not arbitrary jitter: {attempts}")
    transform = solve_triangular(L, white, lower=True)
    transformed = transform @ numerator @ transform.T
    eigenvalues, vectors = np.linalg.eigh((transformed + transformed.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    tolerance = np.finfo(np.float64).eps * d * max(float(np.abs(eigenvalues).max()), 1e-300)
    positive = eigenvalues[eigenvalues > tolerance]
    if positive.size == 0:
        raise ProtocolError("Generalized numerator has no positive eigenvalue")
    weights = np.minimum(np.where(eigenvalues > tolerance, eigenvalues, 0), np.quantile(positive, .99))
    return Space(transform, vectors, weights, {"family": family, "numeric_attempts": attempts,
                                              "positive_rank": int(len(positive)),
                                              "Dc_floor_dominated": dc_trace / np.trace(covariance) < 1e-5})


def capture(space, source_deltas, rank):
    positive_rank = space.diagnostics.get("positive_rank", space.vectors.shape[1])
    if not 0 < rank <= positive_rank or not source_deltas:
        raise ProtocolError("Invalid rank or empty validation method")
    contributions = {}
    for source, delta in sorted(source_deltas.items()):
        coords = np.asarray(delta, dtype=np.float64) @ space.transform.T
        numerator = float(np.square(coords @ space.vectors[:, :rank]).sum(axis=1).mean())
        denominator = float(np.square(coords).sum(axis=1).mean())
        if not math.isfinite(numerator + denominator):
            raise ProtocolError("Nonfinite capture energy")
        contributions[source] = (numerator, denominator)
    numerator, denominator = np.mean(list(contributions.values()), axis=0)
    if denominator <= 0:
        raise ProtocolError("rho_cap denominator has zero trace; stop S0")
    return float(numerator / denominator), contributions


def _best(rows):
    maximum = max(x["J"] for x in rows)
    tied = [x for x in rows if x["J"] >= maximum - 1e-12]
    return min(tied, key=lambda x: (x["rank"], x["lambda_m"]))


def _bootstrap_lower(outer, repeats, seed):
    # Draw one source-ID sample shared by all method folds, preserving clusters.
    sources = sorted({s for fold in outer for s in fold["contributions"]})
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        draw = rng.choice(sources, len(sources), replace=True)
        ratios = []
        for fold in outer:
            terms = [fold["contributions"][s] for s in draw if s in fold["contributions"]]
            if not terms or sum(x[1] for x in terms) <= 0:
                return None
            ratios.append(sum(x[0] for x in terms) / sum(x[1] for x in terms))
        means.append(np.mean(ratios))
    return float(np.quantile(means, .025))


def nested_d1(delta, covariance, dc, sel_deltas, family="B", settings=None):
    settings = protocol() if settings is None else settings
    methods, d = settings["methods"], delta.shape[-1]
    ranks = settings["fisher_ranks"] if family == "Fisher" else settings["ranks"]
    if delta.shape[1] != len(methods) or set(sel_deltas) != set(methods):
        raise ProtocolError("D1 requires the four registered methods")
    spaces = {}

    def refit(indices, lam):
        key = (tuple(indices), lam)
        if key not in spaces:
            spaces[key] = fit_space(delta[:, indices], covariance, dc, lam, family)
        return spaces[key]

    outer = []
    selection_tables = {}
    for m, method in enumerate(methods):
        remaining = [i for i in range(len(methods)) if i != m]
        candidates = []
        for lam in settings["lambda_m"]:
            for rank in ranks:
                inner = []
                for j in remaining:
                    training = [i for i in remaining if i != j]
                    space = refit(training, lam)
                    if rank > space.diagnostics["positive_rank"]:
                        break
                    rho, _ = capture(space, sel_deltas[methods[j]], rank)
                    inner.append({"held_method": methods[j], "fit_methods": [methods[i] for i in training],
                                  "rho": rho})
                if len(inner) != len(remaining):
                    continue
                candidates.append({"rank": rank, "lambda_m": lam,
                                   "J": float(np.mean([x["rho"] for x in inner]) / (rank / d)),
                                   "inner": inner})
        if not candidates:
            raise ProtocolError(f"{family}/{method}: no rank candidate fits every inner positive spectrum")
        winner = _best(candidates)
        space = refit(remaining, winner["lambda_m"])
        rho, terms = capture(space, sel_deltas[method], winner["rank"])
        outer.append({"method": method, "rank": winner["rank"], "lambda_m": winner["lambda_m"],
                      "rho": rho, "above_1_5_baseline": rho > 1.5 * winner["rank"] / d,
                      "contributions": terms})
        selection_tables[method] = candidates
    lower = _bootstrap_lower(outer, settings["bootstrap_repeats"], settings["seed"])
    baseline = float(np.mean([x["rank"] for x in outer]) / d)
    passed = sum(x["above_1_5_baseline"] for x in outer) >= 3 and lower is not None and lower > baseline
    ranks = sorted(x["rank"] for x in outer)
    rank = ranks[(len(ranks) - 1) // 2]
    # New four-fold fits at the fixed final rank; no outer metric is overwritten.
    final_candidates = []
    for lam in settings["lambda_m"]:
        ratios = []
        for m, method in enumerate(methods):
            training = [i for i in range(len(methods)) if i != m]
            space = refit(training, lam)
            if rank > space.diagnostics["positive_rank"]:
                break
            rho, _ = capture(space, sel_deltas[method], rank)
            ratios.append(rho)
        if len(ratios) != len(methods):
            continue
        final_candidates.append({"rank": rank, "lambda_m": lam,
                                 "J": float(np.mean(ratios) / (rank / d))})
    if not final_candidates:
        raise ProtocolError("Final rank is unsupported by every four-fold lambda candidate")
    final = _best(final_candidates)
    model = fit_space(delta, covariance, dc, final["lambda_m"], family)
    if rank > model.diagnostics["positive_rank"]:
        raise ProtocolError("Final full-method positive spectrum cannot support the selected rank")
    return model, {
        "family": family, "passed": bool(passed), "outer": outer,
        "inner_selection": selection_tables, "mean_bootstrap_lower_95": lower,
        "mean_rank_random_baseline": baseline, "final_rank": rank,
        "final_lambda_m": final["lambda_m"], "final_selection": final_candidates,
        "fit_diagnostics": {str(key): value.diagnostics for key, value in spaces.items()},
        "scope": "video-mean method transfer; D_sel_FF++ sources shared across methods" if np.ndim(delta) == 3
                 else "method transfer; D_sel_FF++ sources shared across methods",
    }


def overlap(space, moment, rank, name):
    transformed = space.transform @ moment @ space.transform.T
    denominator = float(np.trace(transformed))
    if denominator <= 0:
        return {"name": name, "value": None, "passed": False, "status": "UNDEFINED_ZERO_TRACE"}
    ratio = float(np.trace(space.vectors[:, :rank].T @ transformed @ space.vectors[:, :rank]) / denominator)
    if not math.isfinite(ratio):
        raise ProtocolError(f"Nonfinite {name} ratio")
    return {"name": name, "value": ratio, "passed": ratio < .3, "status": "DEFINED"}


def score(features, model, gamma):
    features = np.asarray(features, dtype=np.float64)
    d = len(model["mu"])
    if features.ndim == 1:
        features = features[None, :]
    if features.ndim == 3:
        if features.shape[1:] != (256, d):
            raise ProtocolError("Patch input must have shape (N,256,d), excluding CLS")
        features = features.mean(axis=1)
    if features.ndim != 2 or features.shape[1] != d or not np.isfinite(features).all():
        raise ProtocolError("Expected finite global feature matrix (N,d)")
    centered = features - model["mu"]
    u = centered @ model["white"].T
    v = centered @ model["transform"].T
    projection = np.square(v @ model["vectors"])
    real = np.square(u).sum(axis=1) / d
    weighted = projection @ model["weights"] / d
    unweighted = projection.sum(axis=1) / d
    joint = weighted if gamma == "inf" else real + float(gamma) * weighted
    result = {"D_R": real, "D_F": weighted, "D_J": joint, "D_P": unweighted}
    for control in ("C_random", "C_shuffle", "C_real"):
        if f"{control}_V" in model:
            result[control] = np.square(v @ model[f"{control}_V"]) @ model[f"{control}_w"] / d
    return result


def control_spectrum(space, numerator, rank):
    transformed = space.transform @ numerator @ space.transform.T
    values, vectors = np.linalg.eigh((transformed + transformed.T) / 2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    tolerance = np.finfo(np.float64).eps * len(values) * max(float(np.abs(values).max()), 1e-300)
    positive = values[values > tolerance]
    if len(positive) < rank:
        raise ProtocolError("Control numerator has insufficient positive rank")
    return vectors[:, :rank], np.minimum(np.maximum(values[:rank], 0), np.quantile(positive, .99))


def conformal_threshold(video_scores, alpha):
    values = np.asarray(video_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ProtocolError("Calibration requires nonempty finite real video scores")
    index = math.ceil((len(values) + 1) * (1 - alpha))
    if not 0 < alpha < 1 or index > len(values):
        raise ProtocolError("Insufficient real videos for requested conformal level")
    return float(np.sort(values)[index - 1])
