"""Runtime smoke test for EffortDetectorAEPA (G12).

Instantiates the real AEPA model, runs one forward/backward on a small real+fake
batch, and prints:

  1. Forward sanity  (patch/embed shapes, no NaN)
  2. Evidential identity   (r_i + f_i + u_i == 1) and E / f mass stats
  3. Score separation      (mean s_AEPA on real vs fake, rank-based AUC)
  4. Per-parameter gradient norms of the FULL loss  (is the patch head alive?)
  5. Which fake-loss variant gives a *live* fake branch gradient:
        -log(P_F)      (paper §6, saturating)   vs
        -log(max_i f_i)(the applied fix)        vs
        -log(mean_i f_i)  (the dense alternative)
     Gradient is measured on a FAKE-ONLY batch, so a near-zero norm on the patch /
     evidence head means that branch is dead for that loss form.
  6. Cross-dataset patch-level asymmetry diagnostic (real = uniform low /
     fake = spiky): per-image max/mean sparsity + top-5 mass concentration of
     f_i, compared real-vs-fake across all 7 datasets (or a `--dataset` list),
     to test the asymmetric-patch hypothesis behind AEPA (§1).

Run on the server:

    cd .../Effort-AIGI-Detection-main/DeepfakeBench
    /home/user1/miniconda3/envs/effort/bin/python3 experiments/smoke_test_aepa.py \
        --dataset WDF --mode b3_evidence --batch 4
    # or across all 7 datasets (for the [6] hypothesis test), on a trained ckpt:
    /home/user1/miniconda3/envs/effort/bin/python3 experiments/smoke_test_aepa.py \
        --dataset all --mode b3_evidence --batch 16 --load-ckpt <path/to/ckpt_best.pth>

Note on GPU memory: a ViT-L/14 graph at N=256 patches is huge; keep --batch small
(4-8 per class).  If you still hit CUDA OOM, set
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
and/or lower --batch.
"""
import os
import sys
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

# ── path setup (mirrors experiment_utils.py) ─────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_script_dir) == 'experiments':
    _deepfake_dir = os.path.dirname(_script_dir)
else:
    _deepfake_dir = _script_dir
_training_dir = os.path.join(_deepfake_dir, 'training')
for p in (_training_dir, _deepfake_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

DETECTOR_YAML = os.path.join(_training_dir, 'config', 'detector', 'effort.yaml')
TEST_YAML = os.path.join(_training_dir, 'config', 'test_config.yaml')

ALL_DATASETS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
                'DFDC', 'DFDCP', 'DeeperForensics-1.0']


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', nargs='+', default=['WDF'],
                    help="dataset(s) to test, or 'all' for the 7-dataset suite "
                         "(WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP "
                         "DeeperForensics-1.0)")
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--load-ckpt', default=None,
                    help='full trained AEPA ckpt (ckpt_best.pth) loaded with '
                         'strict=True — use for [6] on a TRAINED model (unlike '
                         '--ckpt, which only warm-starts the patch head).')
    ap.add_argument('--mode', default='b3_evidence',
                    choices=['b3_evidence', 'b2_mil', 'b1_pool'])
    ap.add_argument('--lambda-f', type=float, default=1.0)
    ap.add_argument('--eps', type=float, default=1e-8)
    ap.add_argument('--batch', type=int, default=4,
                    help='samples per class to pull (keep small for GPU memory)')
    ap.add_argument('--fallback-random', action='store_true',
                    help='skip real-data load, use random tensors')
    return ap


def load_config(args):
    import yaml
    with open(DETECTOR_YAML, 'r') as f:
        config = yaml.safe_load(f)
    with open(TEST_YAML, 'r') as f:
        config.update(yaml.safe_load(f))
    json_folder = os.path.join(_deepfake_dir, 'preprocessing', 'dataset_json')
    if os.path.isdir(json_folder):
        config['dataset_json_folder'] = json_folder
    config['model_name'] = 'effort_aepa'
    config['aepa_mode'] = args.mode
    config['aepa_lambda_f'] = args.lambda_f
    config['aepa_eps'] = args.eps
    config['use_mixup'] = False
    config['mixup_mode'] = 'none'
    if args.ckpt:
        config['aepa_init_ckpt'] = args.ckpt
    return config


def load_small_batch(config, args, dataset):
    """Pull ~2*batch real+fake samples for `dataset`, binarize labels."""
    from torch.utils.data import DataLoader
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    cfg = config.copy()
    cfg['test_dataset'] = dataset
    ds = DeepfakeAbstractBaseDataset(config=cfg, mode='test')
    # The dataset collate_fn handles None-valued keys that default_collate rejects.
    collate_fn = getattr(ds, 'collate_fn', None)
    loader = DataLoader(ds, batch_size=args.batch * 2, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    data_dict = next(iter(loader))
    label = torch.where(data_dict['label'] != 0, 1, 0).view(-1)
    images = data_dict['image'].float()
    if images.dim() == 5:          # [B, n_crops, C, H, W] -> keep first crop only
        images = images[:, 0].contiguous()
        print(f"[data] note: 5D multi-crop input, using first crop -> 4D")
    real_idx = (label == 0).nonzero(as_tuple=True)[0]
    fake_idx = (label == 1).nonzero(as_tuple=True)[0]
    print(f"[data] {dataset}: pulled {len(label)} samples "
          f"({real_idx.numel()} real / {fake_idx.numel()} fake)")
    n = args.batch
    r = real_idx[:n]
    f = fake_idx[:n]
    img = torch.cat([images[r], images[f]])
    lab = torch.cat([torch.zeros(r.numel()), torch.ones(f.numel())]).long()
    return img, lab


def make_random_batch(args):
    n = args.batch
    img = torch.randn(2 * n, 3, 224, 224) * 0.5
    lab = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return img, lab


def load_batch(config, args, dataset, device):
    """Pull one real+fake batch for `dataset` and move it to `device`.

    Returns (images, labels, using_random).
    """
    using_random = args.fallback_random
    if args.fallback_random:
        images, labels = make_random_batch(args)
        print("[data] USING RANDOM TENSORS (score separation / asymmetry "
              "diagnostic meaningless, gradient flow test still valid)")
    else:
        try:
            images, labels = load_small_batch(config, args, dataset)
        except Exception as e:
            print(f"[data] WARNING: {dataset} load failed ({e}); "
                  f"reverting to random tensors.", file=sys.stderr)
            images, labels = make_random_batch(args)
            using_random = True
    images = images.to(device)
    labels = labels.to(device)
    return images, labels, using_random


def rank_auc(scores, labels):
    """P(fake > real) via Mann-Whitney (labels: 1=fake, 0=real)."""
    fake = scores[labels == 1].detach().cpu().numpy()
    real = scores[labels == 0].detach().cpu().numpy()
    if fake.size == 0 or real.size == 0:
        return float('nan')
    return float((fake[:, None] > real[None, :]).mean())


def patch_asymmetry_stats(f, labels, eps=1e-8):
    """Per-image patch-level fake-mass statistics, split by class.

    Tests the asymmetric-patch hypothesis behind AEPA (AEPA_method.md §1):
        real  -> z_i = R for ALL patches   (f_i uniformly low, small spread)
        fake  -> z_i = F for SOME patch    (f_i spiky: a few patches dominate)

    Args:
        f: [B, N] patch fake-evidence mass (b3) or p_iF (b1/b2).
        labels: [B] 0=real, 1=fake.
    Returns:
        {'real': {...}, 'fake': {...}}; each value is None if no samples.
    """
    f = f.detach().float()
    n_patch = f.size(1)
    mean_f = f.mean(dim=1)
    max_f = f.max(dim=1).values
    std_f = f.std(dim=1)
    total = f.sum(dim=1) + eps
    top5 = f.topk(min(5, n_patch), dim=1).values.sum(dim=1)
    sparsity = max_f / (mean_f + eps)        # >>1 => a few patches dominate
    top5_conc = top5 / total                 # high => fake mass localized

    def agg(mask):
        if mask.sum() == 0:
            return None
        return {
            'n': int(mask.sum().item()),
            'mean_f': float(mean_f[mask].mean().item()),
            'max_f': float(max_f[mask].mean().item()),
            'sparsity': float(sparsity[mask].mean().item()),
            'top5_conc': float(top5_conc[mask].mean().item()),
            'std_f': float(std_f[mask].mean().item()),
        }

    return {'real': agg(labels == 0), 'fake': agg(labels == 1)}


def load_full_ckpt(model, path):
    """Load a full trained detector checkpoint (mirrors experiment_utils.load_model).

    The trained ``ckpt_best.pth`` may be a raw state_dict or ``{'state_dict': ...}``
    and may carry a DataParallel ``module.`` prefix.  Loads with ``strict=True`` so
    any key mismatch surfaces immediately.
    """
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    new_weights = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(new_weights, strict=True)
    return len(new_weights)


def grad_norms(model):
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        g = p.grad.detach().norm().item()
        if math.isnan(g) or math.isinf(g):
            g = float('nan')
        head = n.split('.')[0]
        out.setdefault(head, []).append(g)
    return {k: float(np.sum(v)) for k, v in out.items()}


def run_variant_loss(model, images, labels, variant, eps):
    """Fresh forward + fake-only loss for a given variant; return (loss, grad)."""
    model.zero_grad()
    dd = {'image': images, 'label': labels}
    pred = model.forward(dd, inference=False)
    f = pred['f']                      # [B, N]
    P_F = pred['P_F']                  # [B]
    mask = labels == 1                 # fake only
    if variant == 'paper_logPF':
        loss = (-torch.log(P_F[mask] + eps)).mean()
    elif variant == 'max_f':
        fmax = f[mask].max(dim=1).values
        loss = (-torch.log(fmax + eps)).mean()
    elif variant == 'mean_f':
        loss = (-torch.log(f[mask].mean(dim=1) + eps)).mean()
    else:
        raise ValueError(variant)
    loss.backward()
    return loss.item(), grad_norms(model)


def main():
    args = build_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[env] torch {torch.__version__} device={device} "
          f"cuda={'yes' if torch.cuda.is_available() else 'no'}")

    config = load_config(args)
    from detectors import DETECTOR
    model = DETECTOR['effort_aepa'](config)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] name=effort_aepa mode={args.mode} lambda_f={args.lambda_f} "
          f"eps={args.eps} trainable_params={n_train} "
          f"use_loralib={config.get('use_loralib')}")
    model.to(device)
    if args.load_ckpt:
        n_loaded = load_full_ckpt(model, args.load_ckpt)
        print(f"[model] loaded FULL trained ckpt ({n_loaded} tensors): "
              f"{args.load_ckpt}")
    model.train()

    # ── Data ─────────────────────────────────────────────────────────────
    datasets = ALL_DATASETS if 'all' in args.dataset else args.dataset
    print(f"[run] datasets ({len(datasets)}): {', '.join(datasets)}")

    images, labels, using_random = load_batch(config, args, datasets[0], device)
    print(f"[data] batch {list(images.shape)} real={int((labels==0).sum())} "
          f"fake={int((labels==1).sum())}")

    # ── 1. Forward sanity (backbone runs ONCE here, inside forward) ─────
    print("\n[1] FORWARD SANITY")
    dd = {'image': images, 'label': labels}
    pred = model.forward(dd, inference=False)
    print(f"    keys: {sorted(pred.keys())}")
    print(f"    patches {tuple(pred['p_iF'].shape)} (N={pred['p_iF'].shape[1]}) "
          f"embed {pred['feat'].shape[-1]}  "
          f"s_aepa mean {float(pred['s_aepa'].mean()):.4f}")
    assert torch.isfinite(pred['prob']).all(), "prob has NaN/Inf"
    del dd
    torch.cuda.empty_cache()

    # ── 2. Evidential identity r+f+u==1 ──────────────────────────────────
    print("\n[2] EVIDENTIAL IDENTITY")
    if args.mode == 'b3_evidence':
        E = pred['E']; r = pred['r']; f = pred['f']
        u = 2.0 / (2.0 + E)
        tot = (r + f + u)
        print(f"    E: min {float(E.min()):.4f} max {float(E.max()):.4f} "
              f"mean {float(E.mean()):.4f}")
        print(f"    r+f+u: max |err| {float((tot - 1).abs().max()):.2e} (should ~0)")
        print(f"    f (fake-evidence mass): mean {float(f.mean()):.4f} "
              f"max {float(f.max()):.4f}")
    else:
        print(f"    r=pR, f=pF (no evidence head); f mean "
              f"{float(pred['f'].mean()):.4f} max {float(pred['f'].max()):.4f}")

    # ── 3. Score separation ──────────────────────────────────────────────
    print("\n[3] SCORE SEPARATION  (s_AEPA)")
    s = pred['s_aepa']
    rs, fs = s[labels == 0], s[labels == 1]
    print(f"    real: n={rs.numel()} mean {float(rs.mean()):.5f} "
          f"(min {float(rs.min()):.5f})")
    print(f"    fake: n={fs.numel()} mean {float(fs.mean()):.5f} "
          f"(min {float(fs.min()):.5f})")
    print(f"    rank-AUC (P(fake>real)) = {rank_auc(s, labels):.4f} "
          f"(1.0 perfect / 0.5 random)")
    del pred
    torch.cuda.empty_cache()

    # ── 4. Full-loss backward (is the patch head alive?) ────────────────
    print("\n[4] FULL LOSS BACKWARD (per-head gradient norms)")
    dd = {'image': images, 'label': labels}
    pred = model.forward(dd, inference=False)
    losses = model.get_losses(dd, pred)
    loss = losses['overall']
    print(f"    overall={float(loss):.4f} real={float(losses['real_loss']):.4f} "
          f"fake={float(losses['fake_loss']):.4f}")
    model.zero_grad()
    loss.backward()
    for head, norm in grad_norms(model).items():
        print(f"    grad|{head}| = {norm:.6g}")
    assert torch.isfinite(loss), "Loss is NaN!"
    model.zero_grad()
    del dd, pred
    torch.cuda.empty_cache()

    # ── 5. Fake-branch gradient per variant (fake-only batch) ───────────
    print("\n[5] FAKE-BRANCH GRADIENT PER VARIANT  (fake-only batch)")
    print("    large norm => that loss form supervises the fake branch;")
    print("    ~1e-3 or below on patch_head/evidence_head => that branch is DEAD")
    for variant in ('paper_logPF', 'max_f', 'mean_f'):
        g = run_variant_loss(model, images, labels, variant, args.eps)
        norm = g[1]
        ph = norm.get('patch_head', float('nan'))
        eh = norm.get('evidence_head', float('nan'))
        print(f"    {variant:<15} loss={g[0]:.4f}  "
              f"grad|patch_head|={ph:.6g}  grad|evidence_head|={eh:.6g}")
        model.zero_grad()
        torch.cuda.empty_cache()

    # ── 6. Cross-dataset patch-level asymmetry (hypothesis test) ─────────
    print("\n[6] CROSS-DATASET PATCH-LEVEL ASYMMETRY DIAGNOSTIC")
    print("    Hypothesis (AEPA §1): real = uniform low f_i; fake = spiky f_i "
          "(local leak)")
    print(f"    {'dataset':<20} {'n_real':>6} {'n_fake':>6} {'rankAUC':>8} "
          f"{'real_spar':>10} {'fake_spar':>10} {'real_t5':>8} {'fake_t5':>8}  verdict")
    tally = {'SUPPORTS': 0, 'INVERTED': 0, 'TIE': 0, 'NO_DATA': 0}
    for ds in datasets:
        imgs, labs, is_rand = load_batch(config, args, ds, device)
        with torch.no_grad():
            pred_ds = model.forward({'image': imgs, 'label': labs},
                                    inference=False)
            stats = patch_asymmetry_stats(pred_ds['f'], labs, args.eps)
            rk = rank_auc(pred_ds['s_aepa'], labs)
        if is_rand or stats['real'] is None or stats['fake'] is None:
            verdict = 'NO_DATA'
        else:
            spr, spf = stats['real']['sparsity'], stats['fake']['sparsity']
            if spf > spr * 1.05:
                verdict = 'SUPPORTS'
            elif spr > spf * 1.05:
                verdict = 'INVERTED'
            else:
                verdict = 'TIE'
        tally[verdict] += 1
        r, f = stats['real'], stats['fake']
        print(f"    {ds:<20} "
              f"{r['n'] if r else 0:>6} "
              f"{f['n'] if f else 0:>6} "
              f"{rk:>8.3f} "
              f"{r['sparsity'] if r else float('nan'):>10.2f} "
              f"{f['sparsity'] if f else float('nan'):>10.2f} "
              f"{r['top5_conc'] if r else float('nan'):>8.3f} "
              f"{f['top5_conc'] if f else float('nan'):>8.3f}  {verdict}")
        del pred_ds
        torch.cuda.empty_cache()
    print(f"    tally -> SUPPORTS {tally['SUPPORTS']}, "
          f"INVERTED {tally['INVERTED']}, TIE {tally['TIE']}, "
          f"NO_DATA {tally['NO_DATA']}")

    print("\n[done] If grad|patch_head|/grad|evidence_head| is ~1e-3 or below for "
          "paper_logPF but clearly larger for max_f/mean_f, that confirms the "
          "paper's §6 fake loss starves the fake branch.")

    n_bad = 0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
            n_bad += 1
    print(f"    non-finite grads among trainable params: {n_bad}")


if __name__ == '__main__':
    main()
