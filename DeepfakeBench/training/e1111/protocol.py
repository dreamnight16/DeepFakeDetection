"""Frozen E1111A choices, approved before any experiment results."""

import hashlib
import json
from pathlib import Path


def protocol():
    return {
        "experiment": "E1111", "branch": "E1111A", "version": "v4.4.3",
        "seed": 1024, "feature_dim": 1024, "candidate_frames": 32,
        "split_counts": {"fit": 395, "rad": 72, "sel": 58, "beta": 57, "theta": 137},
        "methods": ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"],
        "ranks": [2, 4, 8], "fisher_ranks": [1], "lambda_m": [0., .1, .3, 1.], "lambda_t": 0.,
        "tau_R": [.1, .3, .5], "gamma": [0., .25, .5, 1., 2., 4., 8., "inf"],
        "alpha": [.1, .05], "jpeg_quality": [15, 30, 40, 50],
        "bootstrap_repeats": 1000,
        "inner_objective": "mean_j(rho_cap(j)/(r/d)); maximize",
        "ties": "scores within 1e-12: smaller rank, then smaller lambda_m",
        "tau_objective": "source-equal held-out FF++ real Gaussian NLL; ties: smaller tau",
        "final_rank": "lower median of four outer inner-selected ranks",
        "final_lambda": "fixed final rank; four method-LOO refits; same normalized J",
        "gamma_selection": "maximize D_sel_FF++ video TPR at provisional alpha=.10; ties smaller gamma; freeze before D_theta",
        "fisher": "dedicated rank 1; rerun every nested D1 fold using method scatter / (residual + D_nuis)",
        "spectral_rank": "reject rank > positive eigenvalue count; tolerance = eps64*d*max(abs(eigenvalues))",
        "controls": "fixed Sigma_R/D_nuis/r; random basis+spectrum shuffle, shuffled real/fake pairs, real temporal differences",
        "cdf_real": "diagnostic only; excluded from tau, D1 and final lambda selection",
        "D3": "descriptive only; R3 disabled; no pass/fail",
        "zero_trace": {"D1": "stop S0", "D2": "undefined; no D2 pass or full mechanism",
                       "D8": "undefined; no R4 continuation"},
        "P_D7": {"min_complete_sources": 30, "min_complete_fraction": .5,
                 "min_cell_coverage": .9, "unverified": "stop before S0/S1"},
        "K_pair": "minimum common valid frame count over all complete sources; never tuned",
        "shared_crop": "raw c23 frames; one real-reference affine applied unchanged to real and fake; 256px ArcFace template scale 1.3",
        "execution": "local source/static checks only; all data/model execution on server",
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def implementation_digest(stage):
    root = Path(__file__).resolve().parents[2]
    groups = {
        "audit": ["training/e1111/audit.py", "training/e1111/protocol.py",
                  "training/e1111/shared_crop.py",
                  "training/g21/data.py", "training/g21/manifest.py", "training/g21/io.py"],
        "feature": ["training/e1111/features.py", "training/detectors/effort_detector.py",
                    "training/g21/data.py", "training/g21/manifest.py"],
        "metric": ["training/e1111/metric.py", "training/e1111/pipeline.py",
                   "training/e1111/protocol.py"],
        "crop": ["training/e1111/shared_crop.py", "training/e1111/protocol.py"],
    }
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in groups[stage]})
