import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

import loralib as lora_lib
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0, merge_weights=False, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
            
        self.lora_A = nn.Parameter(torch.Tensor(in_features, r))
        self.lora_B = nn.Parameter(torch.Tensor(r, out_features))
        self.scaling = lora_alpha / r
        
        self.reset_parameters()
        
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
            
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        if self.r > 0:
            nn.init.normal_(self.lora_A, mean=0, std=0.02)
            nn.init.zeros_(self.lora_B)    
    
    def forward(self, x):
        original = F.linear(x, self.weight, self.bias)
        
        if self.r > 0 and not self.merge_weights:
            lora_x = self.lora_dropout(x)
            lora_output = (lora_x @ self.lora_A @ self.lora_B) * self.scaling
            return original + lora_output
        
        return original
    
    def train(self, mode=True):
        return super(Linear, self).train(mode) 
            
class LoRAModule:
    Linear = Linear
    
lora = LoRAModule()
        
        
@DETECTOR.register_module(module_name='effort')
class EffortDetector(nn.Module):
    def __init__(self, config=None):
        super(EffortDetector, self).__init__()
        self.config = config
        self.use_loralib = config.get('use_loralib', False) if config else False
        
        self.backbone = self.build_backbone(config)
        
        # 根据配置选择使用 loralib 还是自编 LoRA
        LinearClass = lora_lib.Linear if self.use_loralib else lora.Linear
        
        self.head = LinearClass(
            in_features=1024,
            out_features=2,
            r=2,
            lora_alpha=8,
            lora_dropout=0,
            merge_weights=False,
            bias=True
        )
        self.loss_func = nn.CrossEntropyLoss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        # --- 自适应阈值队列 ---
        self.prediction_queue = []

        # --- Asymmetric Center Loss (margin loss) ---
        # margin_loss_mode: 'off' | 'add' | 'replace'
        self.margin_loss_mode = config.get('margin_loss_mode', 'off') if config else 'off'
        self.margin_m = float(config.get('margin_m', 0.5)) if config else 0.5
        self.margin_weight = float(config.get('margin_weight', 1.0)) if config else 1.0
        if self.margin_loss_mode != 'off':
            # learnable center c_R, normalized in loss; randn init puts it in feature space
            self.center = nn.Parameter(torch.randn(1024))

    def build_backbone(self, config):
        # ⚠⚠⚠ Download CLIP model using the below link
        # https://drive.google.com/drive/folders/1fm3Jd8lFMiSP1qgdmsxfqlJZGpr_bXsx?usp=drive_link 
        
        # mean: [0.48145466, 0.4578275, 0.40821073]
        # std: [0.26862954, 0.26130258, 0.27577711]
        
        # ViT-L/14 224*224
        clip_model = CLIPModel.from_pretrained("/home/user1/effort/effort_main/Effort-AIGI-Detection-main/DeepfakeBench/training/models--openai--clip-vit-large-patch14")

        for param in clip_model.vision_model.parameters():
            param.requires_grad = False

        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        
        # 根据配置选择使用 loralib 还是自编 LoRA
        LinearClass = lora_lib.Linear if self.use_loralib else lora.Linear
        
        for name, module in clip_model.vision_model.named_modules():
            if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = clip_model.vision_model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
            
                lora_layer = LinearClass(
                    module.in_features, 
                    module.out_features, 
                    r=4,
                    lora_alpha=16,
                    lora_dropout=0,
                    merge_weights=False
                )
      
                lora_layer.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    lora_layer.bias.data.copy_(module.bias.data)
                
                setattr(parent, child_name, lora_layer)

        for name, param in clip_model.vision_model.named_parameters():
            if 'lora_' not in name: 
                param.requires_grad = False
        
        return clip_model.vision_model

    # --- OWTTT 动态阈值计算 (ICCV23, Yushu-Li/OWTTT) ---
    def compute_adaptive_threshold(self, gap_weight=0.01, max_len=512):
        if len(self.prediction_queue) < 32:
            return 0.5

        os = np.array(self.prediction_queue[-max_len:], dtype=float)
        threshold_range = np.arange(0, 1, 0.01)
        best_th = 0.5
        min_crit = float('inf')

        for th in threshold_range:
            mask = os >= th
            nb = os.size
            nb1 = np.count_nonzero(mask)
            w1 = nb1 / nb
            w0 = 1 - w1
            if w1 == 0 or w0 == 0:
                continue

            v0 = np.var(os[~mask])
            v1 = np.var(os[mask])
            min_gap = np.min(np.abs(os - th))

            crit = w0 * v0 + w1 * v1 - gap_weight * min_gap
            if crit < min_crit:
                min_crit = crit
                best_th = th
        return best_th

    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)

    def asymmetric_center_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        L = (1/N) Σ [ y_i·‖f̂(x_i)-ĉ_R‖² + (1-y_i)·max(0, m-‖f̂(x_i)-ĉ_R‖)² ]
        Features and center are L2-normalized before distance computation so that
        dist ∈ [0, 2], making margin_m meaningful (recommended: 0.3~0.8).
        Convention: label=0 → Real (y_i=1), label=1 → Fake (y_i=0)
        """
        f_norm = F.normalize(features, dim=1)                              # [B, 1024]
        c_norm = F.normalize(self.center.unsqueeze(0), dim=1)             # [1, 1024]
        dist = torch.norm(f_norm - c_norm, dim=1)                         # [B], ∈ [0,2]
        real = (labels == 0).float()                                       # y_i=1
        fake = (labels == 1).float()                                       # y_i=0
        loss = (real * dist ** 2 +
                fake * torch.clamp(self.margin_m - dist, min=0) ** 2).mean()
        return loss

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']

        # ── Soft-label cross-entropy when Asymmetric Mixup is active ──────────
        if 'label_soft' in data_dict:
            log_probs = F.log_softmax(pred, dim=1)          # [B, 2]
            y_soft = data_dict['label_soft']                # [B], values in [0,1]
            per_sample = -(y_soft * log_probs[:, 1] +
                           (1.0 - y_soft) * log_probs[:, 0])  # [B]
            if data_dict.get('mixup_selection') == 'mean':
                K = data_dict['mixup_k']
                n_rf = data_dict.get('n_rf', 0)
                mask_real = label == 0
                mask_fake = label == 1
                real_all = per_sample[mask_real]
                # rr (real+real) first, then K * n_rf rf candidates
                n_rr = len(real_all) - K * n_rf
                rr_losses = real_all[:n_rr]
                rf_losses = real_all[n_rr:].view(K, n_rf).mean(dim=0)  # [n_rf]
                fake_losses = per_sample[mask_fake]
                loss = torch.cat([rr_losses, rf_losses, fake_losses]).mean()
            else:
                loss = per_sample.mean()
        else:
            loss = self.loss_func(pred, label)
        # ─────────────────────────────────────────────────────────────────────

        mask_real = label == 0
        mask_fake = label == 1

        if mask_real.sum() > 0:
            pred_real = pred[mask_real]
            label_real = label[mask_real]
            loss_real = self.loss_func(pred_real, label_real)
        else:
            loss_real = torch.tensor(0.0, device=pred.device)

        if mask_fake.sum() > 0:
            pred_fake = pred[mask_fake]
            label_fake = label[mask_fake]
            loss_fake = self.loss_func(pred_fake, label_fake)
        else:
            loss_fake = torch.tensor(0.0, device=pred.device)
        
        loss_dict = {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
        }

        if self.margin_loss_mode != 'off':
            margin_loss = self.asymmetric_center_loss(pred_dict['feat'], label)
            loss_dict['margin_loss'] = margin_loss
            if self.margin_loss_mode == 'replace':
                loss_dict['overall'] = margin_loss
            else:  # 'add'
                loss_dict['overall'] = loss + self.margin_weight * margin_loss

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        prob = pred_dict['prob']

        if 'label_soft' in data_dict:
            label_for_acc = (data_dict['label_soft'] >= 0.5).long()
        else:
            label_for_acc = data_dict['label']
        pred_label = (prob > 0.5).long()
        correct = (pred_label == label_for_acc).sum().item()
        acc = correct / len(label_for_acc)

        auc, eer, _, ap = calculate_metrics_for_train(label_for_acc.detach(), pred_dict['cls'].detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    # def forward(self, data_dict: dict, inference=False) -> dict:
    #     features = self.features(data_dict)
    #     pred = self.classifier(features)
    #     prob = torch.softmax(pred, dim=1)[:, 1]
    #     pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
    #     return pred_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        images = data_dict['image']
        
        # 兼容性处理：如果是测试模式且输入是 5 维 (Batch, Crops, C, H, W)
        if inference and len(images.shape) == 5:
            b, n, c, h, w = images.shape
            # 将 Batch 和 Crops 维度合并进行特征提取
            flat_images = images.view(-1, c, h, w)
            feats = self.backbone(flat_images)['pooler_output']  # [B*N, 1024]
            preds = self.classifier(feats)                        # [B*N, 2]
            probs = torch.softmax(preds, dim=1)[:, 1]            # [B*N]
            probs = probs.view(b, n)                              # [B, N]
            feats = feats.view(b, n, -1)                          # [B, N, 1024]
            preds = preds.view(b, n, -1)                          # [B, N, 2]

            texture_scores = data_dict.get('texture_scores', None)  # [B, N] or None

            if texture_scores is not None:
                # ── Texture-aware weighted ensemble (formula step 3 & 4) ──
                # Index 0: full image → s_full = f(I)
                # Index 1..N-1: selected patches P* → s_j = f(resize(P_j))
                s_full    = probs[:, 0]                                    # [B]
                s_patches = probs[:, 1:]                                   # [B, N-1]
                # t_j: patch texture scores (skip sentinel 0 at index 0)
                t_j = texture_scores[:, 1:].to(images.device).float()     # [B, N-1]

                gamma = self.config.get('texture_gamma', 1.5) if self.config else 1.5
                beta  = self.config.get('texture_beta',  0.5) if self.config else 0.5

                # w_j = t_j^γ / Σ_k t_k^γ
                t_gamma = t_j ** gamma                                      # [B, N-1]
                w = t_gamma / (t_gamma.sum(dim=1, keepdim=True) + 1e-8)   # [B, N-1]

                # S(I) = β·s_full + (1-β)·Σ_j w_j·s_j
                final_prob = beta * s_full + (1 - beta) * (w * s_patches).sum(dim=1)  # [B]
                final_feat = feats[:, 0, :]   # use full-image feat as representative
                final_pred = preds[:, 0, :]
            else:
                # ── Original TAA logic (unchanged for non-texture mode) ──
                conf = torch.abs(probs - 0.5)
                max_idx = torch.argmax(conf, dim=1)
                final_prob = probs[torch.arange(b), max_idx]
                final_feat = feats[torch.arange(b), max_idx, :]
                final_pred = preds[torch.arange(b), max_idx, :]

            return {'cls': final_pred, 'prob': final_prob, 'feat': final_feat}
    
        # 原有的正常训练/推理流程
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': features}