"""G19-C: concatenated-token linear read-out detector.

Registered as ``effort_lfeq_concat``.  Same frozen CLIP + LoRA backbone and
LFEQ query-transformer body as G19-A, but the read-out is
``TokenReadout(mode='concat')``:

    pooled  = flatten( ALL K+1 query tokens )                  [B, hidden*(K+1)]
    logits  = Linear(hidden*(K+1), 2)(pooled)                  [B, 2]

i.e. no pooling at all — the full per-token dimension is preserved and fed to a
wide linear head.  This is the highest-capacity read-out of the three and is the
"no information discarded" pole (A/B both collapse the K+1 tokens to one vector
first; C carries every token dimension into the head).

No hard-argmax, no fusion weighting, no evidence/diversity loss.  All shared
machinery lives in ``effort_detector_lfeq_readout_base.py``; this file only pins
the arm's mode.  Carries ~(K+1)*hidden*2 head params (~9x A), noted as a
capacity mis-match that G19 isolates on purpose.
"""
from detectors import DETECTOR
from .effort_detector_lfeq_readout_base import EffortDetectorLFEQReadoutBase


@DETECTOR.register_module(module_name='effort_lfeq_concat')
class EffortDetectorLFEQConcat(EffortDetectorLFEQReadoutBase):
    """G19-C: flatten ALL query tokens -> wide Linear(hidden*(K+1),2)."""
    readout_mode = 'concat'
