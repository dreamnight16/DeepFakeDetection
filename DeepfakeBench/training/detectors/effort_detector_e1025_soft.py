"""E1021 dual LoRA with weighted cross-class parameter gradients."""

from detectors import DETECTOR

from .e1021_lora import class_conditional_backward, lora_parameter_groups
from .effort_detector_e1021 import EffortDetectorE1021


@DETECTOR.register_module(module_name="effort_e1025_soft")
class EffortDetectorE1025Soft(EffortDetectorE1021):
    def __init__(self, config):
        beta = float(config.get("e1025_beta", .25))
        if not 0.0 <= beta <= 1.0:
            raise ValueError("E1025 beta must be finite and between 0 and 1")
        super().__init__(config)
        self.beta = beta

    def backward(self, losses):
        real, fake = lora_parameter_groups(self.backbone)
        class_conditional_backward(losses, real, fake, self.head.parameters(), beta=self.beta)
