"""
Re-run testall for G15 cls_init_pooler (④) ONLY — no retraining.

Why: the original testall for ④ was scored on *raw_token*.  Before the
arch_keys fix, the eval path dropped ``max_evidence_cls_feature``, so test.py
fell back to the default 'raw_token' while ④'s CLS head was trained on
'pooler_output' — a train/eval input mismatch.  That makes ④'s cross-domain
video_auc/AUC/ACC in all_resultsG15.json invalid (only its frame-level test_acc
was correct, since that path carries the key).

This script re-evaluates the already-trained ④ checkpoint through
experiment_utils.run_testall with the fixed arch_keys, so the merged detector
YAML carries ``max_evidence_cls_feature: pooler_output`` and test.py rebuilds
EffortDetectorMaxEvidence on its pooler_output CLS branch — exactly how it was
trained.  No retraining.

Usage (from the DeepfakeBench root, on the server):
    nohup python3 experiments/rerun_g15_cls_init_pooler.py > rerun_04.log 2>&1 &
    # optionally pass the checkpoint explicitly:
    #   python3 experiments/rerun_g15_cls_init_pooler.py /abs/path/to/ckpt_best.pth

Prereq: the updated experiments/experiment_utils.py (with max_evidence_cls_feature
in arch_keys) must be uploaded first.
"""
import glob
import os
import sys

_cur = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _cur)
sys.path.insert(0, os.path.dirname(_cur))   # DeepfakeBench root

from run_experiments import EXPERIMENTS, TEST_DS, TRAIN_DS        # noqa: E402
from experiment_utils import build_config, run_testall            # noqa: E402

# ④ — exactly the G15 cls_init_pooler block in run_experiments (no config drift)
exp = next(e for e in EXPERIMENTS
           if e.get('group') == 'G15' and e.get('name') == 'cls_init_pooler')

# ── locate the trained ④ checkpoint ─────────────────────────────────────────
if len(sys.argv) > 1:
    ckpt = sys.argv[1]
else:
    cands = glob.glob(os.path.join('experiment_results', 'master_sweep', 'G15',
                                   'cls_init_pooler', '**', 'ckpt_best.pth'),
                      recursive=True)
    if not cands:
        cands = glob.glob(os.path.join('experiment_results', '**',
                                       'cls_init_pooler', '**', 'ckpt_best.pth'),
                          recursive=True)
    if not cands:
        sys.exit(f"[rerun] No ckpt_best.pth for cls_init_pooler under "
                 f"experiment_results. Pass it explicitly: "
                 f"python3 {os.path.basename(__file__)} /path/to/ckpt_best.pth")
    ckpt = cands[0]
print(f"[rerun] ④ checkpoint: {ckpt}", flush=True)

# ── build the eval config mirroring run_one (for_training=False) ────────────
maxev_kwargs = {k: exp[k] for k in ('max_evidence_lambda', 'max_evidence_eps',
                                    'max_evidence_cls_feature') if k in exp}
config_eval = build_config(
    pyramid_mode=exp['mode'], use_mixup=False, mixup_loss_strip=exp['strip'],
    mixup_alpha=5.0, mixup_gamma=1.0,
    mixup_beta_b=exp.get('beta_b'), mixup_beta_flip=exp.get('beta_flip', False),
    lap_num_levels=3, sampler_real_ratio=0.30,
    model_name=exp.get('model_name', 'effort'), **maxev_kwargs,
    n_epochs=0, train_dataset=TRAIN_DS, test_dataset=TEST_DS, for_training=False,
)

# Exactly the arch_keys subset experiment_utils.evaluate_model now propagates.
arch_keys = ('use_freq_split', 'freq_split_pool', 'model_name',
             'aepa_mode', 'aepa_lambda_f', 'aepa_eps',
             'max_evidence_cls_feature', 'max_evidence_lambda', 'max_evidence_eps',
             'freq_ablation', 'freq_norm', 'freq_energy_match', 'freq_after_aug')
extra_config = {k: config_eval[k] for k in arch_keys if k in config_eval}
print(f"[rerun] extra_config: {extra_config}", flush=True)

log_path = os.path.join(os.path.dirname(ckpt), 'testall_v2.log')
metrics = run_testall(ckpt, TEST_DS, log_path, extra_config=extra_config)
print(f"[rerun] testall_v2 log: {log_path}", flush=True)
for ds, m in metrics.items():
    print(f"  {ds}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
