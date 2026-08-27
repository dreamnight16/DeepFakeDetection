"""Runtime smoke test for EffortDetectorMaxEvidence (G15).

Instantiates the real max-fake-evidence model, runs one forward/backward on a
small real+fake batch, and prints:

  1. Forward sanity    (CLS/patch/selection shapes, no NaN; score separation)
  2. Architecture      (the patch evidence head has NO LayerNorm — "ln 直接去掉")
  3. Full-loss backward (both the CLS head and the patch head are alive)
  4. Selection         (max_index picks the largest fake probability; non-selected
                        patches get ~0 gradient through the patch head)
  5. Init (opt)        (when --ckpt points at a trained CLS baseline, verify the
                        patch head was warm-started from it: W_e <- W_cls)

Run on the server:

    cd .../Effort-AIGI-Detection-main/DeepfakeBench
    python3 experiments/smoke_test_maxev.py --fallback-random          # no data
    python3 experiments/smoke_test_maxev.py --dataset WDF --batch 4   # real data
    # warm-start ("with CLS init") check vs a trained CLS baseline ckpt:
    python3 experiments/smoke_test_maxev.py --fallback-random \
        --ckpt <path/to/G15_b0_cls_baseline/test/avg/ckpt_best.pth>

Note on GPU memory: a ViT-L/14 graph at N=256 patches is huge; keep --batch
small (4-8 per class).  If you still hit CUDA OOM, set
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
                    help="dataset(s) to test, or 'all' for the 7-dataset suite")
    ap.add_argument('--ckpt', default=None,
                    help='trained CLS-baseline ckpt used to warm-start the patch '
                         'head (W_e <- W_cls); only affects init, used for [5]')
    ap.add_argument('--load-ckpt', default=None,
                    help='full trained maxev ckpt loaded with strict=True')
    ap.add_argument('--lambda-max', type=float, default=1.0)
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
    config['model_name'] = 'effort_maxev'
    config['max_evidence_lambda'] = args.lambda_max
    config['max_evidence_eps'] = args.eps
    config['use_mixup'] = False
    config['mixup_mode'] = 'none'
    if args.ckpt:
        config['max_evidence_init_ckpt'] = args.ckpt
    return config


def load_small_batch(config, args, dataset):
    """Pull ~2*batch real+fake samples for `dataset`, binarize labels."""
    from torch.utils.data import DataLoader
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    cfg = config.copy()
    cfg['test_dataset'] = dataset
    ds = DeepfakeAbstractBaseDataset(config=cfg, mode='test')
    collate_fn = getattr(ds, 'collate_fn', None)
    loader = DataLoader(ds, batch_size=args.batch * 2, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    data_dict = next(iter(loader))
    label = torch.where(data_dict['label'] != 0, 1, 0).view(-1)
    images = data_dict['image'].float()
    if images.dim() == 5:          # multi-crop -> keep first crop only
        images = images[:, 0].contiguous()
    real_idx = (label == 0).nonzero(as_tuple=True)[0]
    fake_idx = (label == 1).nonzero(as_tuple=True)[0]
    n = args.batch
    r, f = real_idx[:n], fake_idx[:n]
    img = torch.cat([images[r], images[f]])
    lab = torch.cat([torch.zeros(r.numel()), torch.ones(f.numel())]).long()
    return img, lab


def make_random_batch(args):
    n = args.batch
    img = torch.randn(2 * n, 3, 224, 224) * 0.5
    lab = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return img, lab


def load_batch(config, args, dataset, device):
    if args.fallback_random:
        img, lab = make_random_batch(args)
        print("[data] USING RANDOM TENSORS (score separation meaningless, "
              "gradient flow test still valid)")
        return img.to(device), lab.to(device), True
    try:
        img, lab = load_small_batch(config, args, dataset)
        print(f"[data] {dataset}: fetched {lab.numel()} samples")
        return img.to(device), lab.to(device), False
    except Exception as e:
        print(f"[data] WARNING: {dataset} load failed ({e}); "
              f"reverting to random tensors.", file=sys.stderr)
        img, lab = make_random_batch(args)
        return img.to(device), lab.to(device), True


def rank_auc(scores, labels):
    fake = scores[labels == 1].detach().cpu().numpy()
    real = scores[labels == 0].detach().cpu().numpy()
    if fake.size == 0 or real.size == 0:
        return float('nan')
    return float((fake[:, None] > real[None, :]).mean())


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


def load_full_ckpt(model, path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    new_weights = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(new_weights, strict=True)
    return len(new_weights)


def main():
    args = build_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[env] torch {torch.__version__} device={device} "
          f"cuda={'yes' if torch.cuda.is_available() else 'no'}")

    config = load_config(args)
    from detectors import DETECTOR
    model = DETECTOR['effort_maxev'](config)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] name=effort_maxev lambda_max={args.lambda_max} "
          f"eps={args.eps} trainable_params={n_train} "
          f"patch_head=({model.patch_head.weight.shape[0]}x"
          f"{model.patch_head.weight.shape[1]})")
    model.to(device)
    if args.load_ckpt:
        n_loaded = load_full_ckpt(model, args.load_ckpt)
        print(f"[model] loaded FULL trained ckpt ({n_loaded} tensors)")
    model.train()

    images, labels, using_random = load_batch(config, args, args.dataset[0], device)
    print(f"[data] batch {list(images.shape)} real={int((labels == 0).sum())} "
          f"fake={int((labels == 1).sum())}")

    # ── 1. Forward sanity ────────────────────────────────────────────────
    print("\n[1] FORWARD SANITY")
    dd = {'image': images, 'label': labels}
    pred = model.forward(dd, inference=False)
    print(f"    keys: {sorted(pred.keys())}")
    print(f"    cls {tuple(pred['cls'].shape)}  prob {tuple(pred['prob'].shape)}")
    print(f"    patch_logits {tuple(pred['patch_logits'].shape)} "
          f"fake_prob_map {tuple(pred['fake_prob_map'].shape)} "
          f"max_index {tuple(pred['max_index'].shape)}")
    assert torch.isfinite(pred['prob']).all(), "prob has NaN/Inf"
    rs, fs = pred['prob'][labels == 0], pred['prob'][labels == 1]
    if rs.numel() and fs.numel():
        print(f"    CLS fake-prob: real mean {float(rs.mean()):.4f}, "
              f"fake mean {float(fs.mean()):.4f}")
        print(f"    rank-AUC (P(fake>real)) = {rank_auc(pred['prob'], labels):.4f}")
    else:
        print("    (single-class batch; score separation skipped)")
    del dd, pred
    torch.cuda.empty_cache()

    # ── 2. Architecture: no LayerNorm on the patch branch ───────────────
    print("\n[2] ARCHITECTURE (no LayerNorm on patch head)")
    has_ln = hasattr(model, 'patch_ln')
    print(f"    model has 'patch_ln': {has_ln}  -> "
          f"{'FAIL (LN present)' if has_ln else 'PASS (ln removed)'}")
    assert not has_ln, "patch_ln present — LN was supposed to be removed"

    # ── 3. Full-loss backward ────────────────────────────────────────────
    print("\n[3] FULL LOSS BACKWARD (per-head gradient norms)")
    dd = {'image': images, 'label': labels}
    pred = model.forward(dd, inference=False)
    losses = model.get_losses(dd, pred)
    print(f"    overall={float(losses['overall']):.4f} "
          f"loss_cls={float(losses['loss_cls']):.4f} "
          f"loss_max={float(losses['loss_max']):.4f}")
    model.zero_grad()
    losses['overall'].backward()
    for head, norm in grad_norms(model).items():
        print(f"    grad|{head}| = {norm:.6g}")
    assert torch.isfinite(losses['overall']), "Loss is NaN!"
    ok_head = grad_norms(model).get('head', 0.0) > 0
    ok_patch = grad_norms(model).get('patch_head', 0.0) > 0
    print(f"    CLS head alive: {ok_head}; patch head alive: {ok_patch}")
    assert ok_head and ok_patch, "one of head/patch_head gradient is dead"
    model.zero_grad()
    del dd, pred
    torch.cuda.empty_cache()

    # ── 4. Selection & patch-head gradient sparsity ──────────────────────
    print("\n[4] SELECTION & GRADIENT SPARSITY")
    dd = {'image': images, 'label': labels}
    pred = model.forward(dd, inference=False)
    fake_prob = pred['fake_prob_map']
    max_index = pred['max_index']
    B = fake_prob.size(0)
    gathered = fake_prob[torch.arange(B), max_index]
    row_max = fake_prob.max(dim=1)[0]
    print(f"    max_index == argmax fake-prob (err="
          f"{float((gathered - row_max).abs().max()):.2e})")
    assert torch.allclose(gathered, row_max, atol=1e-5), \
        "selection does not pick the max fake probability"
    # backward again to read patch_head gradient support
    losses = model.get_losses(dd, pred)
    model.zero_grad()
    losses['overall'].backward()
    g = model.patch_head.weight.grad
    print(f"    patch_head.weight grad: nonzero entries = "
          f"{int((g.abs() > 0).sum())}/{g.numel()}")
    del dd, pred
    torch.cuda.empty_cache()

    # ── 5. Init (with CLS init): W_e <- W_cls ──────────────────────────
    print("\n[5] CLS-INIT CHECK (W_e <- W_cls)")
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        new = {k.replace('module.', ''): v for k, v in ckpt.items()}
        w_cls = new.get('head.weight')
        w_patch = model.patch_head.weight.detach()
        if w_cls is None:
            print("    SKIP: ckpt has no 'head.weight' (not a CLS baseline?)")
        else:
            diff = float((w_cls - w_patch).abs().max())
            print(f"    patch_head.weight == ckpt head.weight (max |Δ| = {diff:.2e}) "
                  f"-> {'PASS (warm-started)' if diff < 1e-5 else 'FAIL'}")
    else:
        print("    (pass --ckpt <CLS-baseline.pth> to check warm-start)")

    n_bad = 0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
            n_bad += 1
    print(f"    non-finite grads among trainable params: {n_bad}")
    print("\n[done] If head & patch_head both have healthy gradient norms, the "
          "CLS branch and the max-fake-evidence branch are both trainable.")


if __name__ == '__main__':
    main()
