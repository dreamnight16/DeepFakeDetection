"""
Video-Level Information Graph Detectors for Deepfake Detection
===============================================================

Three independent detectors + fusion, each with ablation variants:

  1. MIDetector       — Temporal/Spatial/Frequency Mutual Information
  2. GTDetector       — Temporal/Spatial/Full Graph Topology
  3. GNNDetector      — MLP / GCN / GAT / ST-GCN
  4. FusionDetector   — MI+GT, GT+GNN, MI+GT+GNN

All operate on frozen CLIP ViT features + small trainable classifier.
Designed for per-module ablation to measure independent contribution.

Author: personal experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def frame_graph_adjacency(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Build frame-level graph: A_{ij} = exp(-||z_i - z_j||² / σ²).

    Args:
        z:     [B, T, D] frame CLS tokens
        sigma: scalar parameter (learnable)

    Returns:
        A: [B, T, T] adjacency (no self-loops)
    """
    B, T, _ = z.shape
    dist_sq = torch.cdist(z, z) ** 2
    A = torch.exp(-dist_sq / (sigma.view(-1, 1, 1) ** 2 + 1e-8))
    A[:, range(T), range(T)] = 0.0
    return A


def patch_cosine_adjacency(patches: torch.Tensor) -> torch.Tensor:
    """Build patch-level adjacency within each frame (cosine similarity).

    Args:
        patches: [B, N, D] patch tokens (single frame or time-averaged)

    Returns:
        A: [B, N, N] adjacency (ReLU thresholded, no self-loops)
    """
    B, N, D = patches.shape
    p_norm = F.normalize(patches, dim=2)
    A = torch.bmm(p_norm, p_norm.transpose(1, 2))
    A = F.relu(A)
    A[:, range(N), range(N)] = 0.0
    return A


def normalized_laplacian(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2}.

    Args:
        A: [B, N, N] adjacency

    Returns:
        L_norm: [B, N, N]
    """
    B, N, _ = A.shape
    D = A.sum(dim=2) + eps
    D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(D))
    I = torch.eye(N, device=A.device).unsqueeze(0)
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def laplacian_spectrum_features(L: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Extract spectral features from normalized Laplacian.

    Returns:
        eigvals: [B, N] sorted eigenvalues
        f_low:   [B, 4] first 4 eigenvalues (global structure)
        f_high:  [B, 4] last 4 eigenvalues (local anomalies)
        gap:     [B]    spectral gap λ₂−λ₁
        entropy: [B]    von Neumann entropy
    """
    B, N, _ = L.shape
    eps = 1e-8

    eigvals = torch.linalg.eigvalsh(L)  # [B, N]

    f_low  = eigvals[:, :4]
    f_high = eigvals[:, N - 4:] if N >= 4 else eigvals
    gap    = eigvals[:, 1] - eigvals[:, 0]

    trace_L = torch.diagonal(L, dim1=1, dim2=2).sum(dim=1, keepdim=True)
    rho = L / (trace_L.unsqueeze(2) + eps)
    rho_eig = torch.linalg.eigvalsh(rho).clamp(min=eps)
    entropy = -(rho_eig * torch.log(rho_eig)).sum(dim=1)

    # Pad low/high to constant size if N varies
    if f_low.shape[1] < 4:
        f_low = F.pad(f_low, (0, 4 - f_low.shape[1]))
    if f_high.shape[1] < 4:
        f_high = F.pad(f_high, (0, 4 - f_high.shape[1]))

    return {
        'eigvals': eigvals, 'f_low': f_low, 'f_high': f_high,
        'gap': gap, 'entropy': entropy,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MI-D: Mutual Information Deepfake Detector
# ═══════════════════════════════════════════════════════════════════════════════

class MIDetector(nn.Module):
    """Video deepfake detector based on mutual information.

    Ablation variants (controlled by flags):
      MI-T : temporal MI between frame CLS tokens at lags k=1..K
      MI-S : spatial patch MI within each frame (averaged over T)
      MI-F : temporal-frequency MI (FFT along time per feature dim)

    Hypothesis
    ----------
      I_real(z_t; z_{t+k}) > I_fake(z_t; z_{t+k})     [MI-T]
      M_real(p_i, p_j)    ≠ M_fake(p_i, p_j)          [MI-S]
      FFT(z_t) pattern differs real vs fake            [MI-F]
    """

    def __init__(self, feature_dim: int = 1024, max_lag: int = 4,
                 hidden_dim: int = 128,
                 use_temporal: bool = True,
                 use_spatial: bool = False,
                 use_frequency: bool = False):
        super().__init__()
        self.max_lag = max_lag
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency

        in_dim = 0
        if use_temporal:
            in_dim += max_lag * 2       # mean+std per lag
        if use_spatial:
            in_dim += 5                 # mean, std, frob, skew, kurt
        if use_frequency:
            in_dim += 6                 # low/mid/high band energies + ratios
        assert in_dim > 0, "At least one MI component required"

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    # ── MI-T: temporal mutual information ──────────────────────────────────

    def compute_temporal_mi(self, cls_seq: torch.Tensor) -> torch.Tensor:
        """Compute lag-k MI between frame CLS tokens.

        MI_k = -½ log(1 - ρ_k²) where ρ_k = corr(z_t, z_{t+k}) over t.

        Args:
            cls_seq: [B, T, D]

        Returns:
            feats: [B, max_lag*2]  [mean_1, std_1, ..., mean_K, std_K]
        """
        B, T, D = cls_seq.shape
        eps = 1e-8

        z_norm = (cls_seq - cls_seq.mean(dim=2, keepdim=True)) / \
                 (cls_seq.std(dim=2, keepdim=True, unbiased=True) + eps)
        corr = torch.bmm(z_norm, z_norm.transpose(1, 2)) / (D - 1)

        feats = []
        for lag in range(1, min(self.max_lag + 1, T)):
            lag_corr = torch.diagonal(corr, offset=lag, dim1=1, dim2=2)
            mi_lag = -0.5 * torch.log(1.0 - lag_corr.clamp(-1 + eps, 1 - eps) ** 2 + eps)
            feats.append(mi_lag.mean(dim=1))
            feats.append(mi_lag.std(dim=1))

        # Pad with zeros if T is small
        while len(feats) < self.max_lag * 2:
            feats.append(torch.zeros(B, device=cls_seq.device))
            feats.append(torch.zeros(B, device=cls_seq.device))

        return torch.stack(feats, dim=1)

    # ── MI-S: spatial patch mutual information ─────────────────────────────

    def compute_spatial_mi(self, patch_seq: torch.Tensor) -> torch.Tensor:
        """Compute frame-averaged patch MI statistics.

        For each frame, build pairwise MI matrix from patch tokens,
        extract statistics, average across T frames.

        Uses Gaussian MI approximation: M_{ij} = -½ log(1 - ρ_{ij}²)
        where ρ_{ij} is Pearson correlation between patch i and j features.

        Args:
            patch_seq: [B, T, N, D]

        Returns:
            feats: [B, 5]  [mean_M, std_M, ||M||_F, skew, kurt]
        """
        B, T, N, D = patch_seq.shape
        eps = 1e-8

        all_stats = []
        for t in range(T):
            z = patch_seq[:, t]  # [B, N, D]

            # Center and normalize per patch
            z_c = z - z.mean(dim=2, keepdim=True)
            z_s = z_c.std(dim=2, keepdim=True, unbiased=True) + eps
            z_n = z_c / z_s

            # Correlation matrix
            rho = torch.bmm(z_n, z_n.transpose(1, 2)) / (D - 1)
            rho = torch.clamp(rho, -1 + eps, 1 - eps)

            # MI matrix
            M = -0.5 * torch.log(1.0 - rho ** 2 + eps)
            M[:, range(N), range(N)] = 0.0

            # Off-diagonal statistics
            off_mask = ~torch.eye(N, dtype=torch.bool, device=M.device)
            off = M[:, off_mask].reshape(B, N * (N - 1))

            mu = off.mean(dim=1)
            sg = off.std(dim=1, unbiased=True).clamp(min=eps)
            fr = torch.norm(M.reshape(B, -1), dim=1)
            sk = ((off - mu.unsqueeze(1)) ** 3).mean(dim=1) / (sg ** 3 + eps)
            ku = ((off - mu.unsqueeze(1)) ** 4).mean(dim=1) / (sg ** 4 + eps) - 3

            all_stats.append(torch.stack([mu, sg, fr, sk, ku], dim=1))

        # Average over T frames
        stats = torch.stack(all_stats, dim=1).mean(dim=1)  # [B, 5]
        return stats

    # ── MI-F: temporal-frequency mutual information ────────────────────────

    def compute_frequency_mi(self, cls_seq: torch.Tensor) -> torch.Tensor:
        """Compute temporal-frequency features via FFT along time axis.

        For each feature dimension of the CLS token, compute the temporal
        FFT, then extract energy in low/mid/high frequency bands.

        Hypothesis: real videos have structured temporal frequency
        patterns (motion coherence), while fake videos have flatter or
        irregular frequency spectra.

        Args:
            cls_seq: [B, T, D]

        Returns:
            feats: [B, 6]  [E_low, E_mid, E_high, ratio_low, ratio_high, entropy]
        """
        B, T, D = cls_seq.shape

        # FFT along time dimension (dim=1)
        z_fft = torch.fft.rfft(cls_seq.float(), dim=1)  # [B, T//2+1, D]
        magnitude = torch.abs(z_fft)                      # [B, T//2+1, D]

        # Frequency band boundaries
        n_freq = magnitude.shape[1]
        low_end   = max(1, n_freq // 4)
        mid_end   = max(2, n_freq // 2)

        # Energy per band (sum over frequency + feature dimensions)
        E_low  = magnitude[:, :low_end].mean(dim=(1, 2))
        E_mid  = magnitude[:, low_end:mid_end].mean(dim=(1, 2))
        E_high = magnitude[:, mid_end:].mean(dim=(1, 2))

        total = E_low + E_mid + E_high + 1e-8
        ratio_low  = E_low / total
        ratio_high = E_high / total

        # Spectral entropy (how "flat" the spectrum is)
        mag_norm = magnitude / (magnitude.sum(dim=1, keepdim=True) + 1e-8)
        spec_entropy = -(mag_norm * torch.log(mag_norm + 1e-8)).sum(dim=(1, 2))

        return torch.stack([E_low, E_mid, E_high, ratio_low, ratio_high,
                            spec_entropy], dim=1)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, clip_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []

        if self.use_temporal:
            feats.append(self.compute_temporal_mi(clip_features['cls_seq']))

        if self.use_spatial:
            feats.append(self.compute_spatial_mi(clip_features['patch_seq']))

        if self.use_frequency:
            feats.append(self.compute_frequency_mi(clip_features['cls_seq']))

        x = torch.cat(feats, dim=1)
        return self.classifier(x)

    def extract_features(self, clip_features: Dict[str, torch.Tensor]
                         ) -> torch.Tensor:
        """Extract features without classification (for fusion)."""
        feats = []
        if self.use_temporal:
            feats.append(self.compute_temporal_mi(clip_features['cls_seq']))
        if self.use_spatial:
            feats.append(self.compute_spatial_mi(clip_features['patch_seq']))
        if self.use_frequency:
            feats.append(self.compute_frequency_mi(clip_features['cls_seq']))
        return torch.cat(feats, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GT-D: Graph Topology Deepfake Detector
# ═══════════════════════════════════════════════════════════════════════════════

class GTDetector(nn.Module):
    """Video deepfake detector based on graph topology analysis.

    Ablation variants (controlled by flags):
      Temporal Graph : T frame CLS nodes → Laplacian spectrum
      Spatial Graph  : N patch nodes per frame, averaged over T
      Full Graph     : spatio-temporal (T×N nodes)

    Hypothesis
    ----------
      E_G^real < E_G^fake   — real videos have smoother frame transitions
      Λ_real  ≠ Λ_fake      — Laplacian spectrum differs
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 128,
                 use_temporal: bool = True,
                 use_spatial:  bool = False,
                 use_full:     bool = False):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_spatial  = use_spatial
        self.use_full     = use_full
        self.sigma = nn.Parameter(torch.tensor(1.0))

        # Each graph variant contributes: smoothness(1) + f_low(4) +
        # f_high(4) + gap(1) + entropy(1) = 11 features
        in_dim = 0
        self._n_per_graph = 11
        if use_temporal:
            in_dim += self._n_per_graph
        if use_spatial:
            in_dim += self._n_per_graph
        if use_full:
            in_dim += self._n_per_graph
        assert in_dim > 0, "At least one graph variant required"

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def _graph_features(self, A: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute spectral features for a single graph variant.

        Args:
            A: [B, N, N] adjacency
            z: [B, N, D] node features

        Returns:
            feat: [B, 11]  smoothness + 4 low + 4 high + gap + entropy
        """
        B, N, _ = A.shape
        eps = 1e-8

        # Graph smoothness: E_G = Σ A_{ij} ||z_i - z_j||²
        dist_sq = torch.cdist(z, z) ** 2
        E_G = (A * dist_sq).sum(dim=(1, 2)) / max(N * (N - 1), 1)

        # Laplacian spectrum
        L = normalized_laplacian(A)
        spec = laplacian_spectrum_features(L)

        return torch.cat([
            E_G.unsqueeze(1),
            spec['f_low'],
            spec['f_high'],
            spec['gap'].unsqueeze(1),
            spec['entropy'].unsqueeze(1),
        ], dim=1)

    def _temporal_features(self, cls_seq: torch.Tensor) -> torch.Tensor:
        """Temporal graph: T frame CLS tokens as nodes."""
        A = frame_graph_adjacency(cls_seq, self.sigma)  # [B, T, T]
        return self._graph_features(A, cls_seq)

    def _spatial_features(self, patch_seq: torch.Tensor) -> torch.Tensor:
        """Spatial graph: N patch nodes, averaged over T frames."""
        B, T, N, D = patch_seq.shape
        all_feats = []
        for t in range(T):
            z = patch_seq[:, t]  # [B, N, D]
            A = patch_cosine_adjacency(z)
            all_feats.append(self._graph_features(A, z))
        return torch.stack(all_feats, dim=1).mean(dim=1)  # [B, 11]

    def _full_features(self, cls_seq: torch.Tensor,
                       patch_seq: torch.Tensor) -> torch.Tensor:
        """Spatio-temporal graph: T frames with N patches each.

        Builds a T×N node graph. For computational efficiency (T=8,
        N=256 → 2048 nodes), uses approximations:
          - Spatial edges within each frame (N×N per frame, T blocks)
          - Temporal edges connect same patch across adjacent frames
        Then computes Laplacian spectrum on the resulting block matrix.
        """
        B, T, N, D = patch_seq.shape
        eps = 1e-8

        # ── Build block-diagonal spatio-temporal Laplacian ──────────────
        # Approximate: compute temporal graph + average spatial per frame
        # Full eigendecomposition on 2048×2048 is too expensive

        # Temporal: frame-level graph
        A_t = frame_graph_adjacency(cls_seq, self.sigma)  # [B, T, T]

        # Spatial per frame: average eigenvalues
        spatial_eigvals = []
        spatial_smoothness = []
        for t in range(T):
            A_s = patch_cosine_adjacency(patch_seq[:, t])  # [B, N, N]
            dist_sq = torch.cdist(patch_seq[:, t], patch_seq[:, t]) ** 2
            E_G_s = (A_s * dist_sq).sum(dim=(1, 2)) / max(N * (N - 1), 1)
            spatial_smoothness.append(E_G_s)
            L_s = normalized_laplacian(A_s)
            spatial_eigvals.append(torch.linalg.eigvalsh(L_s))

        # Average spatial features across frames
        spatial_smoothness = torch.stack(spatial_smoothness, dim=1).mean(dim=1)  # [B]
        spatial_eigvals = torch.stack(spatial_eigvals, dim=2).mean(dim=2)        # [B, N]

        # Combine: temporal Laplacian features + spatial statistics
        L_t = normalized_laplacian(A_t)
        temporal_spec = laplacian_spectrum_features(L_t)

        # Spatial spectrum statistics
        s_f_low  = spatial_eigvals[:, :4]
        s_f_high = spatial_eigvals[:, N - 4:] if N >= 4 else spatial_eigvals[:, :4]
        s_gap    = spatial_eigvals[:, 1] - spatial_eigvals[:, 0]

        return torch.cat([
            spatial_smoothness.unsqueeze(1),    # 1
            s_f_low,                             # 4
            s_f_high,                            # 4
            s_gap.unsqueeze(1),                  # 1
            temporal_spec['entropy'].unsqueeze(1), # 1
        ], dim=1)

    def forward(self, clip_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []

        if self.use_temporal:
            feats.append(self._temporal_features(clip_features['cls_seq']))

        if self.use_spatial:
            feats.append(self._spatial_features(clip_features['patch_seq']))

        if self.use_full:
            feats.append(self._full_features(clip_features['cls_seq'],
                                             clip_features['patch_seq']))

        x = torch.cat(feats, dim=1)
        return self.classifier(x)

    def extract_features(self, clip_features: Dict[str, torch.Tensor]
                         ) -> torch.Tensor:
        """Extract features without classification (for fusion)."""
        feats = []
        if self.use_temporal:
            feats.append(self._temporal_features(clip_features['cls_seq']))
        if self.use_spatial:
            feats.append(self._spatial_features(clip_features['patch_seq']))
        if self.use_full:
            feats.append(self._full_features(clip_features['cls_seq'],
                                             clip_features['patch_seq']))
        return torch.cat(feats, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GNN-D: Graph Neural Network Deepfake Detector
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalGCNConv(nn.Module):
    """Graph convolution on temporal graph (T frame nodes)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D], A: [B, T, T] → [B, T, out_dim]."""
        D = A.sum(dim=2, keepdim=True) + 1e-8
        A_norm = A / D
        return torch.bmm(A_norm, x @ self.W)


class TemporalGATConv(nn.Module):
    """Single-head graph attention on temporal graph."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, A_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D], A_mask: [B, T, T] (binary or weighted) → [B, T, out_dim]."""
        B, T, _ = x.shape
        Wh = x @ self.W  # [B, T, out_dim]

        # Attention coefficients
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, T, -1)  # [B, T, T, out_dim]
        Wh_j = Wh.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, T, out_dim]
        Wh_cat = torch.cat([Wh_i, Wh_j], dim=-1)       # [B, T, T, 2*out_dim]
        e = F.leaky_relu(Wh_cat @ self.a, 0.2).squeeze(-1)  # [B, T, T]

        # Mask by adjacency
        e = e.masked_fill(A_mask <= 1e-8, -1e9)
        alpha = F.softmax(e, dim=-1)  # [B, T, T]

        return torch.bmm(alpha, Wh)  # [B, T, out_dim]


class STGCNConv(nn.Module):
    """Spatio-temporal graph convolution.

    Separate spatial (within-frame) and temporal (cross-frame, same-patch)
    message passing. Pure PyTorch.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_s = nn.Parameter(torch.empty(in_dim, out_dim))
        self.W_t = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W_s)
        nn.init.xavier_uniform_(self.W_t)

    def forward(self, x: torch.Tensor, A_s: torch.Tensor,
                A_t: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, D], A_s: [B, T, N, N], A_t: [B, T-1, N, N]."""
        B, T, N, _ = x.shape
        eps = 1e-8

        # Spatial message passing
        D_s = A_s.sum(dim=3, keepdim=True) + eps
        out_s = torch.einsum('btij,btjd->btid', A_s / D_s, x) @ self.W_s

        # Temporal message passing
        out_t = torch.zeros_like(out_s)
        for t in range(T - 1):
            D_t = A_t[:, t].sum(dim=2, keepdim=True) + eps
            out_t[:, t + 1] = torch.bmm(A_t[:, t] / D_t, x[:, t]) @ self.W_t

        return out_s + out_t


class GNNDetector(nn.Module):
    """GNN-based video deepfake detector.

    Ablation variants:
      MLP     : temporal mean pool → MLP (no graph)
      GCN     : 2-layer GCN on temporal graph (T frame nodes)
      GAT     : 2-layer GAT on temporal graph
      ST-GCN  : 2-layer spatio-temporal GCN (T×N nodes)

    Hypothesis
    ----------
      GNNs can learn structural anomalies in graph topology that
      hand-crafted features miss.
    """

    VARIANTS = ['mlp', 'gcn', 'gat', 'stgcn']

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256,
                 out_dim: int = 128, variant: str = 'gcn',
                 dropout: float = 0.3):
        super().__init__()
        assert variant in self.VARIANTS, f'variant must be one of {self.VARIANTS}'
        self.variant = variant

        if variant == 'mlp':
            # No graph: temporal pool → MLP
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(out_dim, 2),
            )

        elif variant == 'gcn':
            self.conv1 = TemporalGCNConv(in_dim, hidden_dim)
            self.conv2 = TemporalGCNConv(hidden_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 2),
            )

        elif variant == 'gat':
            self.conv1 = TemporalGATConv(in_dim, hidden_dim)
            self.conv2 = TemporalGATConv(hidden_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 2),
            )

        elif variant == 'stgcn':
            self.conv1 = STGCNConv(in_dim, hidden_dim)
            self.conv2 = STGCNConv(hidden_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(out_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 2),
            )

    def forward(self, clip_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        cls_seq   = clip_features['cls_seq']    # [B, T, D]
        patch_seq = clip_features.get('patch_seq')  # [B, T, N, D]

        if self.variant == 'mlp':
            z = cls_seq.mean(dim=1)  # [B, D]
            return self.classifier(z)

        elif self.variant in ('gcn', 'gat'):
            # Build temporal adjacency from frame CLS tokens
            sigma = torch.tensor(1.0, device=cls_seq.device)
            dist_sq = torch.cdist(cls_seq, cls_seq) ** 2
            A = torch.exp(-dist_sq / (sigma ** 2 + 1e-8))
            A[:, range(A.shape[1]), range(A.shape[1])] = 0.0

            h = F.relu(self.conv1(cls_seq, A))
            h = self.dropout(h)
            h = F.relu(self.conv2(h, A))
            h = self.dropout(h)
            z_v = h.mean(dim=1)  # [B, out_dim]
            return self.classifier(z_v)

        elif self.variant == 'stgcn':
            if patch_seq is None:
                raise ValueError('ST-GCN requires patch_seq in clip_features')
            B, T, N, D = patch_seq.shape
            eps = 1e-8

            # Spatial adjacency (cosine per frame)
            flat = patch_seq.reshape(B * T, N, D)
            flat_n = F.normalize(flat, dim=2)
            A_s = torch.bmm(flat_n, flat_n.transpose(1, 2))
            A_s = F.relu(A_s)
            A_s[:, range(N), range(N)] = 0.0
            A_s = A_s.reshape(B, T, N, N)

            # Temporal adjacency (same-patch cosine across frames)
            patch_n = F.normalize(patch_seq, dim=3)
            A_t = torch.zeros(B, T - 1, N, N, device=patch_seq.device)
            for t in range(T - 1):
                sim = (patch_n[:, t] * patch_n[:, t + 1]).sum(dim=2)
                A_t[:, t, range(N), range(N)] = F.relu(sim)

            h = F.relu(self.conv1(patch_seq, A_s, A_t))
            h = self.dropout(h)
            h = F.relu(self.conv2(h, A_s, A_t))
            h = self.dropout(h)
            z_v = h.mean(dim=(1, 2))  # [B, out_dim]
            return self.classifier(z_v)

    def extract_features(self, clip_features: Dict[str, torch.Tensor]
                         ) -> torch.Tensor:
        """Extract graph embedding before classifier (for fusion)."""
        cls_seq   = clip_features['cls_seq']
        patch_seq = clip_features.get('patch_seq')

        if self.variant == 'mlp':
            return cls_seq.mean(dim=1)

        elif self.variant in ('gcn', 'gat'):
            sigma = torch.tensor(1.0, device=cls_seq.device)
            dist_sq = torch.cdist(cls_seq, cls_seq) ** 2
            A = torch.exp(-dist_sq / (sigma ** 2 + 1e-8))
            A[:, range(A.shape[1]), range(A.shape[1])] = 0.0

            h = F.relu(self.conv1(cls_seq, A))
            h = self.dropout(h)
            h = F.relu(self.conv2(h, A))
            h = self.dropout(h)
            return h.mean(dim=1)

        elif self.variant == 'stgcn':
            B, T, N, D = patch_seq.shape
            flat = patch_seq.reshape(B * T, N, D)
            flat_n = F.normalize(flat, dim=2)
            A_s = F.relu(torch.bmm(flat_n, flat_n.transpose(1, 2)))
            A_s[:, range(N), range(N)] = 0.0
            A_s = A_s.reshape(B, T, N, N)

            patch_n = F.normalize(patch_seq, dim=3)
            A_t = torch.zeros(B, T - 1, N, N, device=patch_seq.device)
            for t in range(T - 1):
                sim = (patch_n[:, t] * patch_n[:, t + 1]).sum(dim=2)
                A_t[:, t, range(N), range(N)] = F.relu(sim)

            h = F.relu(self.conv1(patch_seq, A_s, A_t))
            h = self.dropout(h)
            h = F.relu(self.conv2(h, A_s, A_t))
            h = self.dropout(h)
            return h.mean(dim=(1, 2))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Fusion Detector
# ═══════════════════════════════════════════════════════════════════════════════

class FusionDetector(nn.Module):
    """Fuse features from two detectors → shared classifier.

    Variants:
      MI+GT      : MIDetector features + GTDetector features
      GT+GNN     : GTDetector features + GNNDetector (GCN) features
      MI+GT+GNN  : all three

    Each sub-detector extracts features (no classification), then
    features are concatenated and fed to a shared classifier.
    """

    def __init__(self, detectors: Dict[str, nn.Module], hidden_dim: int = 128):
        """
        Args:
            detectors: {'mi': MIDetector, 'gt': GTDetector, 'gnn': GNNDetector}
        """
        super().__init__()
        self.detectors = nn.ModuleDict(detectors)

        # Determine input dimension from detectors
        in_dim = 0
        with torch.no_grad():
            dummy_cls = torch.zeros(1, 8, 1024)
            dummy_patch = torch.zeros(1, 8, 256, 1024)
            dummy = {'cls_seq': dummy_cls, 'patch_seq': dummy_patch}
            for d in detectors.values():
                feat = d.extract_features(dummy)
                in_dim += feat.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, clip_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [d.extract_features(clip_features) for d in
                 self.detectors.values()]
        x = torch.cat(feats, dim=1)
        return self.classifier(x)
