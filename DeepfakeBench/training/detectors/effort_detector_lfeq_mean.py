"""G19-A: mean-pooled-token linear read-out detector.

Registered as ``effort_lfeq_mean``.  Keeps the frozen CLIP ViT-L/14 + LoRA
backbone and the LFEQ query-transformer body (decision token + K evidence token,
self/cross-attention over the RAW patch tokens, parameter-identical to G18 L1),
but replaces the LFEQ read-out with a single ``TokenReadout(mode='mean')``:

    pooled  = mean over ALL query tokens (decision + evidence)   [B, hidden]
    logits  = Linear(hidden, 2)(pooled)                          [B, 2]

No hard-argmax, no fusion weighting, no evidence/diversity loss.  All shared
machinery (forward, 5D TAA, config keys, frozen baseline head) lives in
``effort_detector_lfeq_readout_base.py``; this file only pins the arm's mode.
"""
from detectors import DETECTOR
from .effort_detector_lfeq_readout_base import EffortDetectorLFEQReadoutBase


@DETECTOR.register_module(module_name='effort_lfeq_mean')
class EffortDetectorLFEQMean(EffortDetectorLFEQReadoutBase):
    """G19-A: mean over ALL query tokens -> single Linear(hidden,2)."""
    readout_mode = 'mean'
