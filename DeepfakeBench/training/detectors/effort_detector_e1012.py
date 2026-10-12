"""Trainer registration for the jointly trained E1012 experiment."""

import torch
from torch import nn

from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train
from .effort_detector import EffortDetector
from .e1012_tokens import JointForgeryModel


@DETECTOR.register_module(module_name="effort_e1012")
class EffortDetectorE1012(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = dict(config or {})
        if config.get("use_mixup", False) or config.get("use_freq_split", False):
            raise ValueError("E1012 requires RGB inputs and hard labels without mixup")
        config.update(full_train_head=True, margin_loss_mode="off")
        self.config = config
        self.core = JointForgeryModel(EffortDetector(config), **config["e1012_settings"])

    def forward(self, data_dict, inference=False):
        images = data_dict["image"]
        if not (inference and images.ndim == 5):
            return self.core(data_dict, inference=inference)
        batch, crops, channels, height, width = images.shape
        result = self.core({"image": images.reshape(-1, channels, height, width)}, inference=True)
        probabilities = result["prob"].reshape(batch, crops)
        selected = (probabilities - .5).abs().argmax(1)
        rows = torch.arange(batch, device=images.device)
        return {"cls": result["cls"].reshape(batch, crops, 2)[rows, selected],
                "prob": probabilities[rows, selected],
                "feat": result["feat"].reshape(batch, crops, -1)[rows, selected],
                "e1012": {name: value.reshape(batch, crops, *value.shape[1:])[rows, selected]
                          if isinstance(value, torch.Tensor) else value
                          for name, value in result["e1012"].items()}}

    def get_losses(self, data_dict, pred_dict):
        if "label_soft" in data_dict:
            raise ValueError("E1012 requires hard binary labels")
        return self.core.losses(pred_dict, data_dict["label"])

    def get_train_metrics(self, data_dict, pred_dict):
        labels = data_dict["label"]
        score = pred_dict["prob"]
        accuracy = float(((score > .5).long() == labels).float().mean().detach())
        auc, eer, _, ap = calculate_metrics_for_train(labels.detach(), pred_dict["cls"].detach())
        return dict(acc=accuracy, auc=auc, eer=eer, ap=ap)
