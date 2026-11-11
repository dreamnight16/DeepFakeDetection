"""E1025's single-LoRA model with CE-compatible PU regularization."""

from detectors import DETECTOR

from .e1111_losses import nnPUCEWithReg
from .effort_detector_e1025_pu import EffortDetectorE1025PU


@DETECTOR.register_module(module_name="effort_e1111b")
class EffortDetectorE1111B(EffortDetectorE1025PU):
    def __init__(self, config):
        objective = nnPUCEWithReg(config.get("e1025_prior", .3), config.get("e1111_lam", .05))
        super().__init__(config)
        # Keep E1025's initialization, frozen scope and CE validation logging.
        self.pu_loss = objective
