"""
G17 sequence — combined runner for G17-1 and G17-2.

    G17-1 (model-side)  dual-line cross-complementary gated fusion
    G17-2 (data-side)   real-noise (high-pass residual) evidence isolation

Both share the same training protocol (frozen CLIP ViT-L/14 + LoRA, FF++ train,
Celeb-DF-v2 for best-ckpt selection) and the SAME 7-dataset cross-domain test
set, with FaceForensics++ reported as a SEPARATE in-domain column so the
generalization gap  G = AUC_FFpp - AUC_cross  can be computed.

──── ISOLATION ──────────────────────────────────────────────────────────────
Each arm is run fully in isolation so no two experiments can interfere:
  * per-arm  output dir / log dir / ckpt / results dir  (no shared namespace);
  * per-arm  config  (model_name + comp_* for G17-1, residual_* for G17-2 —
    the two parameter families are never mixed on one arm;
  * no cross-arm checkpoint reuse  (every arm trains its own ckpt, no
    init_ckpt / warm-start);
  * train_model() and evaluate_model() each submit a FRESH Python subprocess
    (train.py / testall.py), so every arm gets its own CUDA context that is
    released on process exit — the strongest available isolation between arms;
  * torch.cuda.empty_cache() is forced after each arm's in-process eval, and
    model tensors are explicitly freed (evaluate_model already does
    `del model; torch.cuda.empty_cache()` at the end of every call).

The single exception is the RGB anchor: G17-1 A0 and G17-2 01 are byte-identical
configs (effort, RGB, no band, no residual), so it is trained once and the same
summary is referenced by both sequences. That is de-duplication of one identical
run, not cross-arm contamination.

Usage:
    python3 experiments/run_g17_freq_sequence.py                 # all arms
    python3 experiments/run_g17_freq_sequence.py --arms A1 01 03 # subset
    python3 experiments/run_g17_freq_sequence.py --n_epochs 10 --g17-1 --g17-2
"""
import os
import sys
import argparse
import json

import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_deepfake_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _current_dir)
sys.path.insert(0, _deepfake_dir)

from experiment_utils import build_config, train_model, evaluate_model, compute_comp_band_sigma

TRAIN_DS = 'FaceForensics++'
VAL_DS = 'Celeb-DF-v2'

# The 7 cross-domain test sets (G15-consistent).  FF++ is NOT in this list: it is
# the in-domain training set, reported as a SEPARATE column (see TEST_DS below).
CROSS_DS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
            'DFDC', 'DFDCP', 'DeeperForensics-1.0']
IN_DOMAIN_DS = 'FaceForensics++'

# Full evaluation list = cross-domain 7 + in-domain FF++ column.
TEST_DS = CROSS_DS + [IN_DOMAIN_DS]

# Cross-domain aggregate = mean(Celeb-DF-v2, DFDC) — G16 discipline.  NOT the
# 7-set average (which would dilute through 5 other domains) and NEVER includes
# the in-domain FF++ column.
CROSS_METRIC_DS = ['Celeb-DF-v2', 'DFDC']


# ── G17-1: model-side dual-line gated fusion ────────────────────────────────
G17_1_ARMS = [
    # A0 — RGB-only anchor, effort.  Reproduces the effort RGB cross ≈ 0.928.
    {'name': 'A0', 'model_name': 'effort',
     'freq_ablation': None, 'residual_ablation': None},
    # A3 — effort frequency-only (data-side low band).  Standalone F-line ceiling
    # for the gate, expect ≈ G13 Low ≈ 0.796.
    {'name': 'A3', 'model_name': 'effort',
     'freq_ablation': 'low', 'residual_ablation': None},
    # A1 — PRIMARY: dualcomp gate, F=low band, pooler CLS feature.
    {'name': 'A1', 'model_name': 'effort_dualcomp',
     'freq_ablation': None, 'residual_ablation': None,
     'comp_freq_bands': ['low'], 'comp_fuse': 'gate'},
    # A2 — control: equal fusion w=0.5 (same dualcomp model, gate frozen).
    {'name': 'A2', 'model_name': 'effort_dualcomp',
     'freq_ablation': None, 'residual_ablation': None,
     'comp_freq_bands': ['low'], 'comp_fuse': 'equal'},
    # A1b — dualcomp gate fused over low+high bands (multi-band evidence).
    {'name': 'A1b', 'model_name': 'effort_dualcomp',
     'freq_ablation': None, 'residual_ablation': None,
     'comp_freq_bands': ['low', 'high'], 'comp_fuse': 'gate'},
]

# ── G17-2: data-side real-noise evidence isolation (effort UNCHANGED) ───────
G17_2_ARMS = [
    {'name': '01', 'model_name': 'effort', 'residual_ablation': None},   # RGB
    {'name': '02', 'model_name': 'effort', 'residual_ablation': 'gauss', 'residual_sigma': 1.0},
    {'name': '03', 'model_name': 'effort', 'residual_ablation': 'gauss', 'residual_sigma': 2.0},
    {'name': '04', 'model_name': 'effort', 'residual_ablation': 'gauss', 'residual_sigma': 4.0},
    {'name': '05', 'model_name': 'effort', 'residual_ablation': 'gauss', 'residual_sigma': 8.0},
    {'name': '06', 'model_name': 'effort', 'residual_ablation': 'fft_high', 'residual_fft_r0': 0.65},
    {'name': '07', 'model_name': 'effort', 'residual_ablation': 'gauss',
     'residual_sigma': 4.0, 'residual_shuffle': True},                     # spatial-structure control (shuffle on the σ=4 residual, G17-2 §3 row 07)
]


def _prefix(seq, exp):
    """Full arm id, e.g. G17-1/A1 — unique across both sequences."""
    return f"{seq}/{exp['name']}"


def run_one(seq, exp, args):
    """Train + eval a single arm in its own isolated dir. Returns summary dict."""
    exp_id = _prefix(seq, exp)
    output_dir = os.path.join(args.output_dir, seq, exp['name'])
    os.makedirs(output_dir, exist_ok=True)
    # Per-arm isolated log dir — ckpts and run logs never collide across arms.
    log_dir = os.path.join(args.output_dir, 'logs', seq, exp['name'])

    # Build config with ONLY this arm's params.  The seq's parameter family is
    # kept disjoint: G17-1 arms set comp_* (and never residual_*); G17-2 arms set
    # residual_* (and freq_ablation stays None).  No key leaks between arms.
    kwargs = dict(
        use_mixup=False, mixup_loss_strip=False,
        sampler_real_ratio=args.sampler_real_ratio,
        log_dir=log_dir, train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
    )
    if seq == 'G17-1':
        kwargs['model_name'] = exp['model_name']
        kwargs['freq_ablation'] = exp['freq_ablation']
        if exp['model_name'] == 'effort_dualcomp':
            kwargs.update({
                'comp_freq_bands': exp['comp_freq_bands'],
                'comp_fuse': exp['comp_fuse'],
                'comp_freq_norm': 'train_rms',
                'comp_cls_feature': 'pooler_output',
                'comp_lambda_freq': args.lambda_freq,
                'comp_lambda_max': args.lambda_max,
            })
    else:  # 'G17-2'
        kwargs['model_name'] = 'effort'
        kwargs['freq_ablation'] = None
        kwargs['residual_ablation'] = exp['residual_ablation']
        if 'residual_sigma' in exp:
            kwargs['residual_sigma'] = exp['residual_sigma']
        if 'residual_fft_r0' in exp:
            kwargs['residual_fft_r0'] = exp['residual_fft_r0']
        if 'residual_shuffle' in exp:
            kwargs['residual_shuffle'] = exp['residual_shuffle']

    config = build_config(**kwargs)

    # G17-1 model-side F line uses train-set-stats per-band RMS (G17-1 §2 — the
    # better normalisation: a FIXED scalar preserves inter-image amplitude, unlike
    # per-image re-scaling).  Calibrate the scalar over the training set once,
    # then rebuild the config so the SAME scalar threads into the training and
    # eval configs (and, via arch_keys, testall) — no per-image fallback, no
    # silent divergence between train / val / eval.
    if exp['model_name'] == 'effort_dualcomp':
        sigma = compute_comp_band_sigma(config, exp['comp_freq_bands'])
        kwargs['comp_band_sigma'] = sigma
        config = build_config(**kwargs)

    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_id}] TRAIN FAILED")
        return {'exp_name': exp_id, 'status': 'TRAIN_FAILED'}

    # Eval config mirrors train config exactly, so test.py rebuilds the model and
    # dataset identically (arch_keys propagate comp_*/residual_* to testall).
    config_eval = build_config(**{**kwargs, 'n_epochs': 0,
                                  'test_dataset': TEST_DS, 'for_training': False})
    summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS,
                             output_dir, exp_id)
    summary['ckpt'] = ckpt
    summary['status'] = 'OK'
    summary['seq'] = seq
    return summary


def _cross_and_gap(summary):
    """Return (cross_auc, in_auc, gap) from a summary, or (None, None, None)."""
    ta = summary.get('testall', {})
    cross_aucs = [ta[d]['video_auc'] for d in CROSS_METRIC_DS
                  if d in ta and 'video_auc' in ta[d]]
    cross = float(np.mean(cross_aucs)) if cross_aucs else None
    in_auc = ta.get(IN_DOMAIN_DS, {}).get('video_auc') \
        if IN_DOMAIN_DS in ta else None
    gap = (in_auc - cross) if (in_auc is not None and cross is not None) else None
    return cross, in_auc, gap


def _print_table(title, results):
    cols = TEST_DS if TEST_DS else []
    hdr = (f"\n  {'Arm':<10s} | {'Model':<15s} |"
           + " | ".join(f"{d[:11]:>11s}" for d in cols)
           + f" | {'AUC_cross':>10s} | {'In(+FF++)':>10s} | {'G':>7s}")
    sep = ("  " + "-"*10 + " | " + "-"*15 + " |"
           + " | ".join("-"*11 for _ in cols)
           + " | " + "-"*10 + " | " + "-"*10 + " | " + "-"*7)
    print(hdr)
    print(sep)
    for r in results:
        exp_id = r['exp_name']
        ta = r.get('testall', {})
        if r.get('status') != 'OK':
            print(f"  {str(exp_id):<10s} | {r.get('status','?'):>9s} | "
                  + " | ".join('N/A' for _ in cols)
                  + " |   N/A    |    N/A    |  N/A")
            continue
        model = r.get('seq', '').replace('G17-', 'G17_')
        cells = []
        for d in cols:
            v = ta.get(d, {}).get('video_auc') if d in ta else None
            cells.append(f"  {v:.4f}  " if v is not None else "     N/A    ")
        cross, in_auc, gap = _cross_and_gap(r)
        cross_s = f"  {cross:.4f}  " if cross is not None else "    N/A    "
        in_s = f"  {in_auc:.4f}  " if in_auc is not None else "    N/A    "
        gap_s = f"  {gap:.4f}  " if gap is not None else "   N/A  "
        print(f"  {str(exp_id):<10s} | {model:<15s} | "
              + " | ".join(cells) + f" | {cross_s:>10s} | {in_s:>10s} | {gap_s:>7s}")
    print(f"\n  AUC_cross = mean({', '.join(CROSS_METRIC_DS)}) — G16 discipline. "
          f"In(+FF++) = the in-domain FaceForensics++ column; G = In − AUC_cross "
          f"(the generalization gap).  Celeb-DF-v2 is partly selection-circular "
          f"(it is the best-ckpt val set); DFDC is the cleanest cross-domain read.")


def main():
    ap = argparse.ArgumentParser(description='G17 sequence combined runner')
    ap.add_argument('--arms', nargs='+', default=None,
                    help='Subset of arm ids (e.g. A1 01 03). Default: all.')
    ap.add_argument('--output_dir', type=str,
                    default='./experiment_results/g17_sequence',
                    help='Root output dir (per-arm subdirs created under it)')
    ap.add_argument('--n_epochs', type=int, default=10)
    ap.add_argument('--sampler_real_ratio', type=float, default=0.30)
    ap.add_argument('--lambda_freq', type=float, default=1.0,
                    help='G17-1 weight on per-line CLS losses')
    ap.add_argument('--lambda_max', type=float, default=1.0,
                    help='G17-1 weight on max-evidence patch loss')
    ap.add_argument('--g17-1', dest='g17_1', action='store_true',
                    help='Run G17-1 arms only')
    ap.add_argument('--g17-2', dest='g17_2', action='store_true',
                    help='Run G17-2 arms only')
    args = ap.parse_args()

    seqs = []
    if args.g17_1 or not (args.g17_1 or args.g17_2):
        seqs.append(('G17-1', G17_1_ARMS))
    if args.g17_2 or not (args.g17_1 or args.g17_2):
        seqs.append(('G17-2', G17_2_ARMS))

    # Build the full (seq, arm) order; filter by --arms if given.
    all_arms = []
    for seq, arms in seqs:
        for e in arms:
            all_arms.append((seq, e))
    if args.arms:
        wanted = set(args.arms)
        all_arms = [(seq, e) for (seq, e) in all_arms if e['name'] in wanted]

    total = len(all_arms)
    print(f"{'='*78}\n  G17 sequence — {total} arms  ({', '.join(f'{s}({len(a)})' for s, a in seqs)})\n"
          f"  test={len(TEST_DS)} sets ({len(CROSS_DS)} cross-domain + {IN_DOMAIN_DS} in-domain)\n"
          f"  cross=mean({', '.join(CROSS_METRIC_DS)})\n"
          f"  output={args.output_dir}\n{'='*78}")

    results = []
    for i, (seq, exp) in enumerate(all_arms):
        exp_id = _prefix(seq, exp)
        # RGB anchor de-duplication: G17-1 A0 and G17-2 01 are identical configs.
        if seq == 'G17-2' and exp['name'] == '01':
            anchor = next((r for r in results if r['exp_name'] == 'G17-1/A0'), None)
            if anchor is not None:
                print(f"\n[{exp_id}] identical to G17-1/A0 (RGB anchor) — reusing "
                      f"isolated run (not a cross-arm checkpoint share).")
                copied = dict(anchor)
                copied['exp_name'] = exp_id
                copied['seq'] = 'G17-2'
                results.append(copied)
                continue
        print(f"\n[{'='*62}]")
        print(f"  [{i+1}/{total}] {exp_id}  model={exp['model_name']}")
        print(f"[{'='*62}]")
        r = run_one(seq, exp, args)
        results.append(r)
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Print per-sequence tables (the anchor copy carried seq='G17-2').
    for seq, arms in seqs:
        seq_results = [r for r in results if r.get('seq') == seq]
        _print_table(f"G17-1 (model-side)" if seq == 'G17-1' else "G17-2 (data-side)",
                     seq_results)

    print(f"\n  Full per-arm per-dataset metrics: {args.output_dir}/all_results.json")
    print(f"  Note: each arm's frame-level confusion matrix + KDE score plots "
          f"live under {args.output_dir}/<seq>/<arm>/.  Between-arm CUDA cache "
          f"is cleared and each train/test runs in a fresh subprocess, so arms "
          f"cannot interfere.")


if __name__ == '__main__':
    main()
