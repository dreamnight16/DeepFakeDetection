"""
Unified experiment runner — 16-config pyramid + trajectory mixup matrix.

G1–G5 use the pyramid mixup family; G6 is the Diffusion Trajectory Mixup
(DTP-Mixup) 2×2 ablation (trajectory × pyramid); G7 is ordinary pixel mixup
with RF label forced 0; G8 is RF-only pyramid mixup (no RR/FF mixing);
G9 is base hard-CE plus an ordinal forensic ranking loss on generated RF
pixel-mixes; G10 is a HiMix replica (clean base + appended RF pixel mixes
with λ~Beta(0.1,0.1), hard-labeled fake); G12 is the Asymmetric Evidential
Patch Aggregation (AEPA) detector (see AEPA_method.md) — a patch-level
readout that replaces the CLS classification with a shared softmax head plus
a shared scalar evidence head, trained with an asymmetric
universal/existential likelihood.  All mixup experiments share:
BalanceBatchSampler v1 (real_ratio=0.30), mixup_alpha=5.0,
mixup_gamma=1.0, lap_num_levels=3, n_epochs=10.

Usage:
    python3 experiments/run_experiments.py                          # all 16
    python3 experiments/run_experiments.py --groups G1 G3           # specific
    python3 experiments/run_experiments.py --groups G6              # trajectory ablation
"""
import os
import sys
import argparse
import json

_current_dir = os.path.dirname(os.path.abspath(__file__))
_deepfake_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _current_dir)
sys.path.insert(0, _deepfake_dir)

from experiment_utils import build_config, train_model, evaluate_model

TRAIN_DS = 'FaceForensics++'
VAL_DS = 'Celeb-DF-v2'
TEST_DS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
           'DFDC', 'DFDCP', 'DeeperForensics-1.0']

# ═══════════════════════════════════════════════════════════════════════════
# Experiment definitions — 12 configs across 5 groups
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # ── G1: 2x2 label(0/1) x scope(top/bottom) ────────────────────────────
    {'group': 'G1', 'name': 'label0_top',    'mode': 'lap_pyramid_label0_top',    'strip': False},
    {'group': 'G1', 'name': 'label0_bottom', 'mode': 'lap_pyramid_label0_bottom', 'strip': False},
    {'group': 'G1', 'name': 'label1_top',    'mode': 'lap_pyramid_label1_top',    'strip': False},
    {'group': 'G1', 'name': 'label1_bottom', 'mode': 'lap_pyramid_label1_bottom', 'strip': False},

    # ── G2: Beta(2,5) on label1_top ───────────────────────────────────────
    {'group': 'G2', 'name': 'beta25',       'mode': 'lap_pyramid_label1_top', 'strip': False,
     'alpha': 2, 'beta_b': 5, 'beta_flip': False},
    {'group': 'G2', 'name': 'beta25_flip',  'mode': 'lap_pyramid_label1_top', 'strip': False,
     'alpha': 2, 'beta_b': 5, 'beta_flip': True},

    # ── G3: Full scope label 0/1 ───────────────────────────────────────────
    {'group': 'G3', 'name': 'label0_full',  'mode': 'lap_pyramid_label0_full', 'strip': False},
    {'group': 'G3', 'name': 'label1_full',  'mode': 'lap_pyramid_label1_full', 'strip': False},

    # ── G4: Pyramid loss ablation ──────────────────────────────────────────
    {'group': 'G4', 'name': 'exp1_soft_ce',  'mode': 'lap_pyramid', 'strip': False},
    {'group': 'G4', 'name': 'exp2_strip_rf', 'mode': 'lap_pyramid', 'strip': True},

    # ── G5: RR+FF pyramid (on strip-RF basis) ──────────────────────────────
    {'group': 'G5', 'name': 'g1_rf_stripped',      'mode': 'lap_pyramid_all',  'strip': True},
    {'group': 'G5', 'name': 'g2_rf_not_generated', 'mode': 'lap_pyramid_rrff', 'strip': False},

    # ── G6: Diffusion Trajectory Mixup (2×2 trajectory × pyramid) ──────────
    # real_ratio=0.30 (same as G1–G5) → pyramid_only is the direct对照 anchor
    # to G4_exp1_soft_ce (identical mode/hyperparams), so trajectory_pyramid
    # is directly comparable to the whole G1–G5 matrix.
    # Re-run under a fresh namespace — do NOT mix with the old buggy sweep:
    #   python3 experiments/run_experiments.py --groups G6 \
    #       --output_dir ./experiment_results/trajectory_mixup_sweep_v2_correct
    {'group': 'G6', 'name': 'baseline',           'mode': 'original',           'use_mixup': False, 'strip': False},
    {'group': 'G6', 'name': 'pyramid_only',       'mode': 'lap_pyramid',        'use_mixup': True,  'strip': False},
    {'group': 'G6', 'name': 'trajectory_only',    'mode': 'trajectory',         'use_mixup': True,  'strip': False,
     'traj_t_min': 50, 'traj_t_max': 700, 'traj_T': 1000},
    {'group': 'G6', 'name': 'trajectory_pyramid', 'mode': 'trajectory_pyramid', 'use_mixup': True,  'strip': False,
     'traj_t_min': 50, 'traj_t_max': 700, 'traj_T': 1000},

    # ── G7: ordinary pixel mixup for ALL pairs, RF label forced 0 ────────────
    # Isolates "label cross as real" from the pyramid: G7 (pixel + label0) vs
    # G3_label0_full (pyramid + label0) vs G6_baseline (no mixup).
    {'group': 'G7', 'name': 'pixel_label0', 'mode': 'pixel_label0', 'use_mixup': True, 'strip': False},

    # ── G8: pyramid mixup for RF pairs only (G4 exp1 basis) ────────────────
    # RR/FF pairs pass through unmixed (original image + hard label); only
    # real+fake pairs get the Laplacian-pyramid treatment. Isolates the RF
    # branch's contribution vs G4_exp1_soft_ce (RR+FF+RF).
    {'group': 'G8', 'name': 'rf_only', 'mode': 'lap_pyramid_rf_only', 'use_mixup': True, 'strip': False},

    # ── G9: base hard-CE on the original batch + ordinal ranking loss ──────
    # Base samples keep their normal loss (no mixup on the main batch); an
    # auxiliary ranking loss is added over 2·n_real generated RF pixel-mixes
    # (λ_a < λ_b per real anchor, fake partners sampled from the batch).
    # See ordinal_forensic_ranking_loss.md
    {'group': 'G9', 'name': 'base_plus_rank', 'mode': 'ordinal_rank', 'use_mixup': True, 'strip': False,
     'rank_margin': 1.0, 'rank_loss_weight': 1.0, 'rank_softplus': False,
     'rank_alpha': 1.0},

    # ── G10: HiMix replica — clean base + appended RF pixel mixes ──────────
    # Base batch untouched (normal hard CE); n_real extra RF pixel-mixes with
    # per-pair λ~Beta(0.1,0.1) (bimodal) are appended and hard-labeled fake.
    # α=0.1 is the key ingredient (HiMix's optimal regime); rr/ff untouched.
    {'group': 'G10', 'name': 'himix_replica', 'mode': 'pixel_rf_hardfake', 'use_mixup': True, 'strip': False,
     'alpha': 0.1},

    # ── G10-pyramid: HiMix MDA + Laplacian pyramid mixing ─────────────────
    # Same recipe as himix_replica (base batch untouched, n_real appended RF
    # mixes, hard fake label, per-pair α=0.1 bimodal), but the mixing uses the
    # project's Laplacian pyramid (real coarse structure preserved, residual
    # bands blended) instead of pixel interpolation.
    {'group': 'G10', 'name': 'pyramid_rf_hardfake', 'mode': 'pyramid_rf_hardfake', 'use_mixup': True, 'strip': False,
     'alpha': 0.1},

    # ── G11: RF data-augmentation variants on G9's base-CE structure ──────
    # Base batch untouched (hard CE, like G9); the appended RF mixes are
    # handled differently instead of G9's ordinal ranking loss. FR ("fake as
    # base") dropped — the real-anchor prior keeps real as the base.
    #   g0_baseline: no mixes at all — pure-LoRA control (reproduces G6_baseline)
    #   g1_label0:   pixel mix, hard-labeled real (0), α ∈ {0.1, 5.0}
    #   g3_pyr_soft: pyramid mix, soft energy-grounded label, α ∈ {0.1, 5.0}
    #   g4_pyr_label0: pyramid mix, hard-labeled real (0), α ∈ {0.1, 5.0}
    {'group': 'G11', 'name': 'g0_baseline', 'mode': 'original', 'use_mixup': False, 'strip': False},
    {'group': 'G11', 'name': 'g1_label0_a01', 'mode': 'pixel_rf_label0', 'use_mixup': True, 'strip': False, 'alpha': 0.1},
    {'group': 'G11', 'name': 'g1_label0_a5', 'mode': 'pixel_rf_label0', 'use_mixup': True, 'strip': False, 'alpha': 5.0},
    {'group': 'G11', 'name': 'g3_pyr_soft_a01', 'mode': 'pyramid_rf_soft', 'use_mixup': True, 'strip': False, 'alpha': 0.1},
    {'group': 'G11', 'name': 'g3_pyr_soft_a5', 'mode': 'pyramid_rf_soft', 'use_mixup': True, 'strip': False, 'alpha': 5.0},
    {'group': 'G11', 'name': 'g4_pyr_label0_a01', 'mode': 'pyramid_rf_label0', 'use_mixup': True, 'strip': False, 'alpha': 0.1},
    {'group': 'G11', 'name': 'g4_pyr_label0_a5', 'mode': 'pyramid_rf_label0', 'use_mixup': True, 'strip': False, 'alpha': 5.0},

    # ── G12: Asymmetric Evidential Patch Aggregation (AEPA) ──────────────
    # Full ablation B0→B3 (AEPA_method.md Section 8), no mixup.  B0 is the
    # CLS baseline; B1/B2/B3 seed their shared patch head from B0's trained
    # CLS head (Section 3: W_p ← W_cls), so each step changes one component.
    #   b0_cls:        CLS classification (B0)
    #   b1_patch_pool: patch softmax + symmetric mean pooling (B1)
    #   b2_asym_mil:   patch softmax + universal-existential MIL, no evidence (B2)
    #   b3_aepa:       full AEPA — shared scalar evidence head (B3)
    {'group': 'G12', 'name': 'b0_cls', 'model_name': 'effort',
     'mode': 'none', 'use_mixup': False, 'strip': False},
    {'group': 'G12', 'name': 'b1_patch_pool', 'model_name': 'effort_aepa',
     'aepa_mode': 'b1_pool', 'mode': 'none', 'use_mixup': False, 'strip': False,
     'init_from': 'G12_b0_cls'},
    {'group': 'G12', 'name': 'b2_asym_mil', 'model_name': 'effort_aepa',
     'aepa_mode': 'b2_mil', 'mode': 'none', 'use_mixup': False, 'strip': False,
     'init_from': 'G12_b0_cls'},
    {'group': 'G12', 'name': 'b3_aepa', 'model_name': 'effort_aepa',
     'aepa_mode': 'b3_evidence', 'mode': 'none', 'use_mixup': False, 'strip': False,
     'init_from': 'G12_b0_cls'},

    # ── G15: Maximum Fake-Evidence Selection Loss (最大伪造证据选择损失.md) ──
    # The method *keeps* the CLS classification head (L_cls) and adds a patch-
    # level auxiliary head supervised only on the most-suspect patch (L_max):
    #     L = CE(cls_logits, y) + lambda_max * CE(a_{i*}, y),
    #     i* = argmax_i q_{i,1}     q_i = softmax(W_e h_i + b_e)
    # The patch evidence head is applied directly to raw patch tokens — NO
    # LayerNorm ("ln 直接去掉").  Two variants differ ONLY in patch-head init:
    #     no_init: patch head random (default nn.Linear init)
    #     cls_init: patch head seeded from the trained CLS classifier (G15 b0),
    #               W_e <- W_cls, b_e <- b_cls  (AEPA-style Section-3 warm-start)
    # b0_cls_baseline is the pure CLS baseline: the anchor AND the init source.
    #
    # PRIMARY = strict per the method doc (§3/§4.1): the CLS head consumes the
    # *raw* CLS token (max_evidence_cls_feature='raw_token', the default).  So the
    # b0<->maxev CLS branch is NOT byte-identical (b0's head was trained on the
    # pooled CLS feature) — this is the accepted cost of doc fidelity.
    {'group': 'G15', 'name': 'b0_cls_baseline', 'model_name': 'effort',
     'mode': 'none', 'use_mixup': False, 'strip': False},
    {'group': 'G15', 'name': 'no_init', 'model_name': 'effort_maxev',
     'mode': 'none', 'use_mixup': False, 'strip': False},
    {'group': 'G15', 'name': 'cls_init', 'model_name': 'effort_maxev',
     'mode': 'none', 'use_mixup': False, 'strip': False,
     'init_from': 'G15_b0_cls_baseline'},
    # ── SUPPLEMENTARY (anchor-aligned CLS branch) ──
    # Same as cls_init but the CLS head consumes pooler_output (= LayerNorm(CLS)),
    # the exact feature b0's head was trained on, so the b0<->maxev CLS branch is
    # byte-identical.  Answers "does the doc-strict raw-token CLS branch, vs a
    # baseline-aligned one, change the result?"  Run after the PRIMARY trio.
    {'group': 'G15', 'name': 'cls_init_pooler', 'model_name': 'effort_maxev',
     'mode': 'none', 'use_mixup': False, 'strip': False,
     'init_from': 'G15_b0_cls_baseline', 'max_evidence_cls_feature': 'pooler_output'},
]


def run_one(exp, args, init_ckpt=None):
    """Train + eval a single experiment config. Returns summary dict.

    init_ckpt: optional path to a B0 (CLS baseline) checkpoint used to
    warm-start the AEPA patch head (Section 3: W_p ← W_cls).
    """
    exp_name = f"{exp['group']}_{exp['name']}"
    output_dir = os.path.join(args.output_dir, exp['group'], exp['name'])
    os.makedirs(output_dir, exist_ok=True)

    exp_alpha = exp.get('alpha', args.alpha)
    exp_use_mixup = exp.get('use_mixup', True)
    exp_model_name = exp.get('model_name', 'effort')
    traj_kwargs = {k: exp[k] for k in ('traj_t_min', 'traj_t_max', 'traj_T') if k in exp}
    rank_kwargs = {k: exp[k] for k in ('rank_margin', 'rank_loss_weight',
                                       'rank_softplus', 'rank_alpha') if k in exp}
    aepa_kwargs = {k: exp[k] for k in ('aepa_mode', 'aepa_lambda_f', 'aepa_eps') if k in exp}
    if init_ckpt is not None:
        aepa_kwargs['aepa_init_ckpt'] = init_ckpt
    maxev_kwargs = {k: exp[k] for k in ('max_evidence_lambda', 'max_evidence_eps',
                                        'max_evidence_cls_feature') if k in exp}
    if init_ckpt is not None:
        maxev_kwargs['max_evidence_init_ckpt'] = init_ckpt

    log_dir = os.path.join(args.output_dir, 'logs', exp['group'], exp['name'])
    config = build_config(
        pyramid_mode=exp['mode'],
        use_mixup=exp_use_mixup,
        mixup_loss_strip=exp['strip'],
        mixup_alpha=exp_alpha, mixup_gamma=args.gamma,
        mixup_beta_b=exp.get('beta_b'), mixup_beta_flip=exp.get('beta_flip', False),
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        model_name=exp_model_name, **aepa_kwargs, **maxev_kwargs,
        **traj_kwargs, **rank_kwargs,
        log_dir=log_dir,
        train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
    )

    # Train
    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_name}] TRAIN FAILED")
        return {'exp_name': exp_name, 'status': 'TRAIN_FAILED'}

    # Eval
    config_eval = build_config(
        pyramid_mode=exp['mode'],
        use_mixup=exp_use_mixup,
        mixup_loss_strip=exp['strip'],
        mixup_alpha=exp_alpha,
        mixup_beta_b=exp.get('beta_b'), mixup_beta_flip=exp.get('beta_flip', False),
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        model_name=exp_model_name, **aepa_kwargs, **maxev_kwargs,
        **traj_kwargs,
        n_epochs=0, train_dataset=TRAIN_DS, test_dataset=TEST_DS,
        for_training=False,
    )
    summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS, output_dir, exp_name)
    summary['ckpt'] = ckpt
    summary['status'] = 'OK'
    return summary


def main():
    parser = argparse.ArgumentParser(description='Unified pyramid mixup experiment runner')
    parser.add_argument('--groups', nargs='+', default=None,
                        help='Which groups to run (default: all)')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Which exact experiment names to run, e.g. G10_pyramid_rf_hardfake (default: all)')
    parser.add_argument('--output_dir', type=str,
                        default='./experiment_results/master_sweep',
                        help='Output directory for all results')
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--sampler_real_ratio', type=float, default=0.30)
    parser.add_argument('--n_epochs', type=int, default=10)
    args = parser.parse_args()

    exp_list = EXPERIMENTS
    if args.groups:
        exp_list = [e for e in EXPERIMENTS if e['group'] in args.groups]
    if args.names:
        exp_list = [e for e in exp_list if f"{e['group']}_{e['name']}" in args.names]

    total = len(exp_list)
    print(f"{'='*70}\n  Unified Experiment Runner — {total} configs\n{'='*70}")

    results = []
    results_by_name = {}
    for i, exp in enumerate(exp_list):
        exp_name = f"{exp['group']}_{exp['name']}"

        # Resolve dependency: experiments with `init_from` warm-start their
        # patch head from that experiment's checkpoint (e.g. B1/B2/B3 ← B0).
        init_ckpt = None
        init_from = exp.get('init_from')
        if init_from:
            init_ckpt = results_by_name.get(init_from)
            if init_ckpt is None:
                print(f"\n[{exp_name}] SKIP: dependency '{init_from}' not available "
                      f"(run it first / it failed)")
                results.append({'exp_name': exp_name, 'status': 'SKIP_NO_INIT'})
                continue

        print(f"\n[{'='*60}]")
        print(f"  [{i+1}/{total}] {exp_name}")
        print(f"  mode={exp['mode']}  strip={exp['strip']}"
              + (f"  beta=({args.alpha},{exp['beta_b']}) flip={exp['beta_flip']}"
                 if 'beta_b' in exp else "")
              + (f"  init_from={init_from}" if init_from else ""))
        print(f"[{'='*60}]")
        r = run_one(exp, args, init_ckpt=init_ckpt)
        results.append(r)
        if r.get('status') == 'OK':
            results_by_name[exp_name] = r.get('ckpt')
        # Save incremental results
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}\n  SUMMARY TABLE\n{'='*70}")
    hdr = f"  {'Exp':<28s} | {'Status':<12s} | {'testall_vAUC':>12s} | {'testall_AUC':>12s} | {'testall_ACC':>12s} | {'frame_ACC':>10s}"
    sep = f"  {'-'*28} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10}"
    print(hdr)
    print(sep)
    for r in results:
        status = r.get('status', '?')
        ta = r.get('testall', {}).get('average', {})
        print(f"  {r['exp_name']:<28s} | {status:<12s} | "
              f"{str(ta.get('video_auc','N/A')):>12s} | {str(ta.get('auc','N/A')):>12s} | "
              f"{str(ta.get('acc','N/A')):>12s} | {str(r.get('test_acc_avg','N/A')):>10}")

    print(f"\nResults saved to: {args.output_dir}/all_results.json")


if __name__ == '__main__':
    main()
