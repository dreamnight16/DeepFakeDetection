"""B0 with real/fake additive LoRA and class-conditioned backward."""

from detectors import DETECTOR

from .effort_detector import EffortDetector
from .e1021_lora import class_conditional_backward, lora_parameter_groups, replace_lora


@DETECTOR.register_module(module_name="effort_e1021")
class EffortDetectorE1021(EffortDetector):
    def __init__(self, config):
        if (config.get("ddp", False) or config.get("use_mixup", False)
                or config.get("optimizer_wrapper") is not None
                or config.get("optimizer", {}).get("type", "adam") != "adam"
                or not config.get("full_train_head", False)
                or config.get("use_freq_split", False)
                or config.get("margin_loss_mode", "off") != "off"
                or config.get("rank_loss_weight", 0) != 0):
            raise ValueError("E1021 requires B0's plain Adam, full head, hard labels, and single GPU")
        super().__init__(config)
        replace_lora(self.backbone, config.get("e1021_real_rank", 4),
                     config.get("e1021_fake_rank", 4))

    def get_losses(self, data_dict, pred_dict):
        losses = super().get_losses(data_dict, pred_dict)
        # Match the requested separate class means, including the shared head.
        losses["overall"] = losses["real_loss"] + losses["fake_loss"]
        return losses

    def backward(self, losses):
        real, fake = lora_parameter_groups(self.backbone)
        class_conditional_backward(losses, real, fake, self.head.parameters())
