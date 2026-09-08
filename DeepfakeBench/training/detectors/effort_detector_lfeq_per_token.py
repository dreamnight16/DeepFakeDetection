"""G19-B: per-token independent-head read-out detector.

Registered as ``effort_lfeq_per_token``.  Same frozen CLIP + LoRA backbone and
LFEQ query-transformer body as G19-A, but the read-out is
``TokenReadout(mode='per_token')``:

    logits  = mean_i( Linear_i(queries_i) )                    [B, 2]

i.e. one DISTINCT Linear(hidden,2) per query token (K+1 heads), then a soft
average of the per-token logits.  A *shared* per-token linear would collapse to
the 'mean' arm (linear commutes with mean), so these heads are intentionally
independent — B is a genuinely different training-time operation from A.

No hard-argmax, no fusion weighting, no evidence/diversity loss.  All shared
machinery lives in ``effort_detector_lfeq_readout_base.py``; this file only pins
the arm's mode.  Carries ~(K+1)*hidden*2 head params (~9x A), noted as a
capacity mis-match that G19 isolates on purpose.
"""
from detectors import DETECTOR
from .effort_detector_lfeq_readout_base import EffortDetectorLFEQReadoutBase


@DETECTOR.register_module(module_name='effort_lfeq_per_token')
class EffortDetectorLFEQPerToken(EffortDetectorLFEQReadoutBase):
    """G19-B: per-token independent Linear heads, soft-averaged logits."""
    readout_mode = 'per_token'
