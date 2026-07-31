"""
Video-Level Information Graph Detectors
========================================

Three detectors for video deepfake detection, registered via DETECTOR.
Each follows the exact same interface as EffortDetector:
  - build_backbone(config)  → CLIP ViT
  - features(data_dict)     → temporal features
  - classifier(features)    → logits
  - forward(data_dict)      → {'cls', 'prob', 'feat'}
  - get_losses(...)         → {'overall', ...}
  - get_train_metrics(...)  → {'acc', 'auc', 'eer', 'ap'}

Config keys (in YAML):
  mi_temporal:   true    # MI-D: enable temporal MI
  mi_spatial:    false   # MI-D: enable spatial patch MI
  mi_frequency:  false   # MI-D: enable temporal-frequency MI
  gt_temporal:   true    # GT-D: enable temporal graph
  gt_spatial:    false   # GT-D: enable spatial patch graph
  gt_full:       false   # GT-D: enable spatio-temporal graph
  gnn_variant:   gcn     # GNN-D: mlp | gcn | gat | stgcn

Author: personal experiment
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel
import loralib as lora_lib

from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train

# ═══════════════════════════════════════════════════════════════════════════════
# LoRA helpers (copied from effort_detector.py)
# ═══════════════════════════════════════════════════════════════════════════════

class Linear(nn.Module):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1,
                 lora_dropout=0, merge_weights=False, bias=True):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Shared backbone builder (identical to EffortDetector.build_backbone)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_clip_backbone(config):
    """Build frozen CLIP ViT backbone with LoRA adapters."""
    use_loralib = config.get('use_loralib', False) if config else False
    LinearClass = lora_lib.Linear if use_loralib else lora.Linear

    clip_model = CLIPModel.from_pretrained(
        config['clip_model_path']
        if config and 'clip_model_path' in config
        else "/home/user1/effort/effort_main/Effort-AIGI-Detection-main/DeepfakeBench/training/models--openai--clip-vit-large-patch14"
    )

    for param in clip_model.vision_model.parameters():
        param.requires_grad = False

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    for name, module in clip_model.vision_model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = clip_model.vision_model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            lora_layer = LinearClass(
                module.in_features, module.out_features,
                r=4, lora_alpha=16, lora_dropout=0, merge_weights=False,
            )
            lora_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                lora_layer.bias.data.copy_(module.bias.data)
            setattr(parent, child_name, lora_layer)

    for name, param in clip_model.vision_model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    return clip_model.vision_model


# ═══════════════════════════════════════════════════════════════════════════════
# MI computation utilities (theory → features)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_temporal_mi(cls_seq, max_lag=4):
    """MI_k = -0.5 log(1 - corr(z_t, z_{t+k})²). [B,T,D] → [B, max_lag*2]"""
    B, T, D = cls_seq.shape
    eps = 1e-8
    z_norm = (cls_seq - cls_seq.mean(dim=2, keepdim=True)) / \
             (cls_seq.std(dim=2, keepdim=True, unbiased=True) + eps)
    corr = torch.bmm(z_norm, z_norm.transpose(1, 2)) / (D - 1)
    feats = []
    for lag in range(1, min(max_lag + 1, T)):
        lag_corr = torch.diagonal(corr, offset=lag, dim1=1, dim2=2)
        mi_lag = -0.5 * torch.log(1.0 - lag_corr.clamp(-1 + eps, 1 - eps) ** 2 + eps)
        feats.append(mi_lag.mean(dim=1))
        feats.append(mi_lag.std(dim=1))
    while len(feats) < max_lag * 2:
        feats.append(torch.zeros(B, device=cls_seq.device))
        feats.append(torch.zeros(B, device=cls_seq.device))
    return torch.stack(feats, dim=1)


def compute_spatial_mi(patch_seq):
    """Frame-averaged patch MI statistics. [B,T,N,D] → [B,5]"""
    B, T, N, D = patch_seq.shape
    eps = 1e-8
    all_stats = []
    for t in range(T):
        z = patch_seq[:, t]
        z_c = z - z.mean(dim=2, keepdim=True)
        z_n = z_c / (z_c.std(dim=2, keepdim=True, unbiased=True) + eps)
        rho = torch.bmm(z_n, z_n.transpose(1, 2)) / (D - 1)
        rho = torch.clamp(rho, -1 + eps, 1 - eps)
        M = -0.5 * torch.log(1.0 - rho ** 2 + eps)
        M[:, range(N), range(N)] = 0.0
        off_mask = ~torch.eye(N, dtype=torch.bool, device=M.device)
        off = M[:, off_mask].reshape(B, N * (N - 1))
        mu = off.mean(dim=1)
        sg = off.std(dim=1, unbiased=True).clamp(min=eps)
        fr = torch.norm(M.reshape(B, -1), dim=1)
        sk = ((off - mu.unsqueeze(1)) ** 3).mean(dim=1) / (sg ** 3 + eps)
        ku = ((off - mu.unsqueeze(1)) ** 4).mean(dim=1) / (sg ** 4 + eps) - 3
        all_stats.append(torch.stack([mu, sg, fr, sk, ku], dim=1))
    return torch.stack(all_stats, dim=1).mean(dim=1)


def compute_frequency_mi(cls_seq):
    """Temporal FFT energy bands. [B,T,D] → [B,6]"""
    B, T, D = cls_seq.shape
    z_fft = torch.fft.rfft(cls_seq.float(), dim=1)
    mag = torch.abs(z_fft)
    n_freq = mag.shape[1]
    low_e = max(1, n_freq // 4)
    mid_e = max(2, n_freq // 2)
    E_low  = mag[:, :low_e].mean(dim=(1, 2))
    E_mid  = mag[:, low_e:mid_e].mean(dim=(1, 2))
    E_high = mag[:, mid_e:].mean(dim=(1, 2))
    total = E_low + E_mid + E_high + 1e-8
    mag_norm = mag / (mag.sum(dim=1, keepdim=True) + 1e-8)
    spec_ent = -(mag_norm * torch.log(mag_norm + 1e-8)).sum(dim=(1, 2))
    return torch.stack([E_low, E_mid, E_high,
                        E_low / total, E_high / total, spec_ent], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph topology utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _frame_adjacency(cls_seq):
    """A_{ij} = exp(-||z_i-z_j||²). [B,T,D] → [B,T,T]"""
    dist_sq = torch.cdist(cls_seq, cls_seq) ** 2
    A = torch.exp(-dist_sq)
    B, T = A.shape[:2]
    A[:, range(T), range(T)] = 0.0
    return A


def _patch_adjacency(patches):
    """Per-frame cosine adjacency averaged over T. [B,T,N,D] → [B,N,N]"""
    B, T, N, D = patches.shape
    flat = patches.reshape(B * T, N, D)
    flat_n = F.normalize(flat, dim=2)
    A = torch.bmm(flat_n, flat_n.transpose(1, 2))
    A = F.relu(A)
    A[:, range(N), range(N)] = 0.0
    return A.reshape(B, T, N, N).mean(dim=1)


def _graph_features(A, z, eps=1e-8):
    """Laplacian spectrum features. [B,N,N],[B,N,D] → [B,11]"""
    B, N, _ = A.shape
    D = A.sum(dim=2) + eps
    D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(D))
    I = torch.eye(N, device=A.device).unsqueeze(0)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt

    eigvals = torch.linalg.eigvalsh(L)
    f_low  = eigvals[:, :4]
    f_high = eigvals[:, max(0, N - 4):]
    gap    = eigvals[:, 1] - eigvals[:, 0]
    dist_sq = torch.cdist(z, z) ** 2
    E_G = (A * dist_sq).sum(dim=(1, 2)) / max(N * (N - 1), 1)

    trace_L = torch.diagonal(L, dim1=1, dim2=2).sum(dim=1, keepdim=True)
    rho = L / (trace_L.unsqueeze(2) + eps)
    rho_eig = torch.linalg.eigvalsh(rho).clamp(min=eps)
    H_G = -(rho_eig * torch.log(rho_eig)).sum(dim=1)

    # Pad to fixed size
    if f_low.shape[1] < 4:
        f_low = F.pad(f_low, (0, 4 - f_low.shape[1]))
    if f_high.shape[1] < 4:
        f_high = F.pad(f_high, (0, 4 - f_high.shape[1]))

    return torch.cat([E_G.unsqueeze(1), f_low, f_high,
                      gap.unsqueeze(1), H_G.unsqueeze(1)], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# GNN layers
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, A):
        D = A.sum(dim=2, keepdim=True) + 1e-8
        return torch.bmm(A / D, x @ self.W)


class TemporalGAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, A_mask):
        B, T, _ = x.shape
        Wh = x @ self.W
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, T, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, T, -1, -1)
        e = F.leaky_relu(torch.cat([Wh_i, Wh_j], dim=-1) @ self.a, 0.2).squeeze(-1)
        e = e.masked_fill(A_mask <= 1e-8, -1e9)
        alpha = F.softmax(e, dim=-1)
        return torch.bmm(alpha, Wh)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MI-D: Mutual Information Detector
# ═══════════════════════════════════════════════════════════════════════════════

@DETECTOR.register_module(module_name='mi_video')
class MIDetector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.backbone = _build_clip_backbone(config)

        self.use_temporal  = self.config.get('mi_temporal', True)
        self.use_spatial   = self.config.get('mi_spatial', False)
        self.use_frequency = self.config.get('mi_frequency', False)

        in_dim = 0
        if self.use_temporal:  in_dim += 8   # 4 lags × (mean+std)
        if self.use_spatial:   in_dim += 5   # mean, std, frob, skew, kurt
        if self.use_frequency: in_dim += 6   # 3 bands + 2 ratios + entropy
        assert in_dim > 0, "At least one MI component required"

        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.loss_func = nn.CrossEntropyLoss()

    def build_backbone(self, config):
        return _build_clip_backbone(config)

    def features(self, data_dict):
        images = data_dict['image']  # [B, T, C, H, W]
        if len(images.shape) == 5:
            B, T, C, H, W = images.shape
            flat = images.reshape(B * T, C, H, W)
        else:
            B, C, H, W = images.shape
            T = 1
            flat = images
        feats = self.backbone(flat)['pooler_output']  # [B*T, D]
        D = feats.shape[1]
        return feats.reshape(B, T, D)  # [B, T, D]

    def classifier(self, features):
        return self.head(features)

    def forward(self, data_dict, inference=False):
        cls_seq = self.features(data_dict)  # [B, T, D]
        feats = []
        if self.use_temporal:
            feats.append(compute_temporal_mi(cls_seq))
        if self.use_spatial:
            images = data_dict['image']
            if len(images.shape) == 5:
                B, T, C, H, W = images.shape
                flat = images.reshape(B * T, C, H, W)
            else:
                B, C, H, W = images.shape
                T = 1
                flat = images
            # Get patch tokens via hidden states
            out = self.backbone(flat, output_hidden_states=True)
            if hasattr(out, 'hidden_states'):
                h = out.hidden_states[-1][:, 1:, :].reshape(B, T, -1, h.shape[-1])
            else:
                h = out[2][-1][:, 1:, :].reshape(B, T, -1, out[2][-1].shape[-1])
            feats.append(compute_spatial_mi(h))
        if self.use_frequency:
            feats.append(compute_frequency_mi(cls_seq))

        x = torch.cat(feats, dim=1)
        pred = self.classifier(x)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': x}

    def get_losses(self, data_dict, pred_dict):
        loss = self.loss_func(pred_dict['cls'], data_dict['label'])
        return {'overall': loss}

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        prob = pred_dict['prob']
        pred_label = (prob > 0.5).long()
        correct = (pred_label == label).sum().item()
        acc = correct / len(label)
        auc, eer, _, ap = calculate_metrics_for_train(label.detach(),
                                                       pred_dict['cls'].detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GT-D: Graph Topology Detector
# ═══════════════════════════════════════════════════════════════════════════════

@DETECTOR.register_module(module_name='gt_video')
class GTDetector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.backbone = _build_clip_backbone(config)

        self.use_temporal = self.config.get('gt_temporal', True)
        self.use_spatial  = self.config.get('gt_spatial', False)
        self.use_full     = self.config.get('gt_full', False)

        in_dim = 0
        n_per = 11  # E_G(1) + f_low(4) + f_high(4) + gap(1) + entropy(1)
        if self.use_temporal: in_dim += n_per
        if self.use_spatial:  in_dim += n_per
        if self.use_full:     in_dim += n_per
        assert in_dim > 0, "At least one graph variant required"

        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.loss_func = nn.CrossEntropyLoss()

    def build_backbone(self, config):
        return _build_clip_backbone(config)

    def features(self, data_dict):
        images = data_dict['image']
        if len(images.shape) == 5:
            B, T, C, H, W = images.shape
            flat = images.reshape(B * T, C, H, W)
        else:
            B, T = images.shape[0], 1
            flat = images
        feats = self.backbone(flat)['pooler_output']
        return feats.reshape(B, T, -1)

    def classifier(self, features):
        return self.head(features)

    def _get_patch_seq(self, data_dict):
        images = data_dict['image']
        if len(images.shape) == 5:
            B, T, C, H, W = images.shape
            flat = images.reshape(B * T, C, H, W)
        else:
            B, T = images.shape[0], 1
            flat = images
        out = self.backbone(flat, output_hidden_states=True)
        if hasattr(out, 'hidden_states'):
            h = out.hidden_states[-1]
        else:
            h = out[2][-1]
        Np = h.shape[1] - 1  # exclude CLS
        return h[:, 1:, :].reshape(B, T, Np, -1)

    def forward(self, data_dict, inference=False):
        cls_seq = self.features(data_dict)
        B, T, D = cls_seq.shape
        feats = []

        if self.use_temporal:
            A_t = _frame_adjacency(cls_seq)
            feats.append(_graph_features(A_t, cls_seq))

        if self.use_spatial:
            patch_seq = self._get_patch_seq(data_dict)
            B2, T2, N, D2 = patch_seq.shape
            A_s = torch.zeros(B2, N, N, device=cls_seq.device)
            all_gf = []
            for t in range(T2):
                pts = patch_seq[:, t]
                p_norm = F.normalize(pts, dim=2)
                A = torch.bmm(p_norm, p_norm.transpose(1, 2))
                A = F.relu(A)
                A[:, range(N), range(N)] = 0.0
                all_gf.append(_graph_features(A, pts))
            feats.append(torch.stack(all_gf, dim=1).mean(dim=1))

        if self.use_full:
            # Spatio-temporal: temporal graph features + spatial avg eigenvalues
            # Use temporal graph on cls_seq already computed; add patch spectrum stats
            patch_seq = self._get_patch_seq(data_dict)
            B2, T2, N, D2 = patch_seq.shape
            all_eig = []
            for t in range(T2):
                pts = patch_seq[:, t]
                p_norm = F.normalize(pts, dim=2)
                A = torch.bmm(p_norm, p_norm.transpose(1, 2))
                A = F.relu(A)
                A[:, range(N), range(N)] = 0.0
                D_deg = A.sum(dim=2) + 1e-8
                D_inv = torch.diag_embed(1.0 / torch.sqrt(D_deg))
                I = torch.eye(N, device=A.device).unsqueeze(0)
                L = I - D_inv @ A @ D_inv
                all_eig.append(torch.linalg.eigvalsh(L))
            eig_stack = torch.stack(all_eig, dim=2).mean(dim=2)  # [B, N]
            f_low  = eig_stack[:, :4]
            f_high = eig_stack[:, max(0, N-4):]
            gap    = eig_stack[:, 1] - eig_stack[:, 0]
            if f_low.shape[1] < 4:
                f_low = F.pad(f_low, (0, 4 - f_low.shape[1]))
            if f_high.shape[1] < 4:
                f_high = F.pad(f_high, (0, 4 - f_high.shape[1]))
            feats.append(torch.cat([f_low, f_high, gap.unsqueeze(1)], dim=1))

        x = torch.cat(feats, dim=1)
        pred = self.classifier(x)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': x}

    def get_losses(self, data_dict, pred_dict):
        loss = self.loss_func(pred_dict['cls'], data_dict['label'])
        return {'overall': loss}

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        prob = pred_dict['prob']
        pred_label = (prob > 0.5).long()
        correct = (pred_label == label).sum().item()
        acc = correct / len(label)
        auc, eer, _, ap = calculate_metrics_for_train(label.detach(),
                                                       pred_dict['cls'].detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GNN-D: Graph Neural Network Detector
# ═══════════════════════════════════════════════════════════════════════════════

@DETECTOR.register_module(module_name='gnn_video')
class GNNDetector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.backbone = _build_clip_backbone(config)
        self.variant = self.config.get('gnn_variant', 'gcn')

        D = 1024  # ViT-L/14 CLS dim (auto-detect would be better but matches effort)
        h_dim = 256
        out_dim = 128
        dropout = 0.3

        if self.variant == 'mlp':
            self.conv1 = None
            self.conv2 = None
            self.head = nn.Sequential(
                nn.Linear(D, h_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(h_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(out_dim, 2),
            )
        elif self.variant == 'gcn':
            self.conv1 = TemporalGCN(D, h_dim)
            self.conv2 = TemporalGCN(h_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 2),
            )
        elif self.variant == 'gat':
            self.conv1 = TemporalGAT(D, h_dim)
            self.conv2 = TemporalGAT(h_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 2),
            )
        else:
            raise ValueError(f"Unknown gnn_variant: {self.variant}")

        self.loss_func = nn.CrossEntropyLoss()

    def build_backbone(self, config):
        return _build_clip_backbone(config)

    def features(self, data_dict):
        images = data_dict['image']
        if len(images.shape) == 5:
            B, T, C, H, W = images.shape
            flat = images.reshape(B * T, C, H, W)
        else:
            B, T = images.shape[0], 1
            flat = images
        feats = self.backbone(flat)['pooler_output']
        return feats.reshape(B, T, -1)

    def classifier(self, features):
        return self.head(features)

    def forward(self, data_dict, inference=False):
        cls_seq = self.features(data_dict)  # [B, T, D]
        B, T, D = cls_seq.shape

        if self.variant == 'mlp':
            z = cls_seq.mean(dim=1)
            pred = self.head(z)
            prob = torch.softmax(pred, dim=1)[:, 1]
            return {'cls': pred, 'prob': prob, 'feat': z}

        # Build temporal adjacency
        dist_sq = torch.cdist(cls_seq, cls_seq) ** 2
        A = torch.exp(-dist_sq)
        A[:, range(T), range(T)] = 0.0

        h = F.relu(self.conv1(cls_seq, A))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, A))
        h = self.dropout(h)
        z = h.mean(dim=1)

        pred = self.head(z)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': z}

    def get_losses(self, data_dict, pred_dict):
        loss = self.loss_func(pred_dict['cls'], data_dict['label'])
        return {'overall': loss}

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        prob = pred_dict['prob']
        pred_label = (prob > 0.5).long()
        correct = (pred_label == label).sum().item()
        acc = correct / len(label)
        auc, eer, _, ap = calculate_metrics_for_train(label.detach(),
                                                       pred_dict['cls'].detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
