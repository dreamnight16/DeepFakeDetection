"""Original single-LoRA B0 with nnPU replacing its training CE objective."""

from detectors import DETECTOR

from .effort_detector import EffortDetector
from .e1025_losses import NNPULoss


@DETECTOR.register_module(module_name="effort_e1025_pu")
class EffortDetectorE1025PU(EffortDetector):
    def __init__(self, config):
        if (config.get("ddp", False) or config.get("use_mixup", False)
                or config.get("optimizer_wrapper") is not None
                or config.get("optimizer", {}).get("type", "adam") != "adam"
                or not config.get("full_train_head", False)
                or config.get("use_freq_split", False)
                or config.get("margin_loss_mode", "off") != "off"
                or config.get("rank_loss_weight", 0) != 0):
            raise ValueError("E1025 nnPU requires B0's plain Adam, full head, hard labels, and single GPU")
        pu_loss = NNPULoss(config.get("e1025_prior", .3))
        super().__init__(config)
        self.pu_loss = pu_loss

    def get_losses(self, data_dict, pred_dict):
        losses = super().get_losses(data_dict, pred_dict)
        # Validation can have a single class; its logged loss remains ordinary
        # CE. Checkpoint selection still uses the unchanged validation AUC.
        if self.training:
            losses["overall"] = self.pu_loss(pred_dict["cls"], data_dict["label"])
        return losses
