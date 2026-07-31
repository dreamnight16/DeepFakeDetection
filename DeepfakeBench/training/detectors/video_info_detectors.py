"""
Video-Level Information Graph Detectors for Deepfake Detection
===============================================================

Three independent detector modules operating on video clips (T frames):

  1. MIDetector   — Temporal Mutual Information (MI-D)
  2. GTDetector   — Graph Topology (GT-D)
  3. GNNDetector  — Graph Neural Network (GNN-D)

Each detector has its own theoretical hypothesis, feature extraction,
and classifier. Designed for independent ablation — components can be
enabled/disabled to measure per-module contribution.

Theory
------
  MI-D:  I_real(z_t, z_{t+k}) > I_fake(z_t, z_{t+k})
         Deepfake generation disrupts temporal information dependency
         between frames, measurable via pairwise MI decay.

  GT-D:  E_G^real < E_G^fake   (graph smoothness)
         Real videos exhibit smoother frame-to-frame transitions
         in CLIP feature space, reflected in Laplacian spectral properties.

  GNN-D: P(real|G_st) ≠ P(fake|G_st)
         A GNN can learn to detect structural anomalies in the
         spatio-temporal patch graph that manual features miss.

All three operate on frozen CLIP ViT features.
No interference with the existing EffortDetector training pipeline.

Author: personal experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def pairwise_correlation(z: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Pearson correlation matrix.

    Args:
        z: [B, T, D] normalized frame features

    Returns:
        corr: [B, T, T] correlation matrix
    """
    B, T, D = z.shape
    z_centered = z - z.mean(dim=2, keepdim=True)
    z_std = z_centered.std(dim=2, keepdim=True, unbiased=True) + 1e-8
    z_norm = z_centered / z_std
    corr = torch.bmm(z_norm, z_norm.transpose(1, 2)) / (D - 1)
    return corr


def frame_graph_adjacency(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Build frame-level graph: A_{ij} = exp(-||z_i - z_j||² / σ²).

    Args:
        z:     [B, T, D] frame features
        sigma: scalar or [B] — bandwidth parameter

    Returns:
        A: [B, T, T] adjacency (no self-loops)
    """
    B, T, _ = z.shape
    dist_sq = torch.cdist(z, z) ** 2  # [B, T, T]
    A = torch.exp(-dist_sq / (sigma.view(-1, 1, 1) ** 2 + 1e-8))
    A[:, range(T), range(T)] = 0.0
    return A


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MI-D: Mutual Information Deepfake Detector
# ═══════════════════════════════════════════════════════════════════════════════

class MIDetector(nn.Module):
    """Video deepfake detector based on temporal mutual information.

    Hypothesis
    ----------
        I_real(z_t; z_{t+k}) > I_fake(z_t; z_{t+k})

    Real videos exhibit stable temporal information dependency
    (physics: continuous motion). Generated videos, even when
    frame-wise realistic, break this dependency because each
    frame is independently sampled from a latent distribution.

    Features
    --------
      (a) Temporal MI (MI-T):
          MI_k = −½ log(1 − ρ_k²) for lag k ∈ {1, 2, ..., K}
          where ρ_k = mean correlation between frames t and t+k.
          Statistics: mean MI_k, std MI_k, and decay rate.

      (b) Spatial MI (MI-S):
          Frame-averaged patch MI matrix statistics.
          (reuses MutualInformationAnalyzer from info_graph_theory)

      (c) Frequency MI (MI-F): not yet implemented (placeholder).

    Ablation controls
    -----------------
      use_temporal:  bool — enable temporal MI features
      use_spatial:   bool — enable spatial patch MI features
      use_frequency: bool — enable frequency-domain MI (TODO)
    """

    def __init__(self, feature_dim: int = 1024, max_lag: int = 4,
                 hidden_dim: int = 128, use_temporal: bool = True,
                 use_spatial: bool = False, use_frequency: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_lag = max_lag
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency

        # Feature dimensions
        in_dim = 0
        if use_temporal:
            in_dim += max_lag * 2          # mean + std per lag
        if use_spatial:
            in_dim += 5                    # MI stats (mean, std, frob, skew, kurt)
        if use_frequency:
            in_dim += 0                    # placeholder

        assert in_dim > 0, "At least one MI component must be enabled"

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def compute_temporal_mi(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Compute temporal MI features from frame sequence.

        Args:
            z_seq: [B, T, D] frame CLS tokens (T frames per clip)

        Returns:
            mi_feat: [B, max_lag * 2] — [mean_1, std_1, ..., mean_K, std_K]
        """
        B, T, D = z_seq.shape
        eps = 1e-8

        # Standardize per frame
        z_norm = (z_seq - z_seq.mean(dim=2, keepdim=True)) / \
                 (z_seq.std(dim=2, keepdim=True, unbiased=True) + eps)

        # Pairwise correlation matrix
        corr = torch.bmm(z_norm, z_norm.transpose(1, 2)) / (D - 1)  # [B, T, T]

        feats = []
        for lag in range(1, min(self.max_lag + 1, T)):
            # Extract diagonal at offset 'lag': corr[b, t, t+lag]
            lag_corr = torch.diagonal(corr, offset=lag, dim1=1, dim2=2)  # [B, T-lag]
            # Gaussian MI: I = −½ log(1 − ρ²)
            mi_lag = -0.5 * torch.log(1.0 - lag_corr.clamp(-1 + eps, 1 - eps) ** 2 + eps)
            feats.append(mi_lag.mean(dim=1))   # mean MI at this lag
            feats.append(mi_lag.std(dim=1))    # std MI at this lag

        # Pad if T is smaller than max_lag+1
        while len(feats) < self.max_lag * 2:
            feats.append(torch.zeros(B, device=z_seq.device))
            feats.append(torch.zeros(B, device=z_seq.device))

        return torch.stack(feats, dim=1)  # [B, max_lag * 2]

    def forward(self, clip_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            clip_features: dict with:
                cls_seq:   [B, T, D]  frame CLS tokens
                mi_stats:  [B, 5]     spatial MI stats (optional)

        Returns:
            logits: [B, 2]
        """
        feats = []

        if self.use_temporal:
            z_seq = clip_features['cls_seq']  # [B, T, D]
            mi_t = self.compute_temporal_mi(z_seq)
            feats.append(mi_t)

        if self.use_spatial:
            mi_s = clip_features.get('mi_stats')
            if mi_s is not None:
                feats.append(mi_s)

        x = torch.cat(feats, dim=1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GT-D: Graph Topology Deepfake Detector
# ═══════════════════════════════════════════════════════════════════════════════

class GTDetector(nn.Module):
    """Video deepfake detector based on graph topology analysis.

    Hypothesis
    ----------
        E_G^real < E_G^fake    (graph smoothness)

    Real videos have temporally smooth frame transitions in CLIP
    feature space. This manifests as:
      - Lower graph smoothness energy E_G = Σ A_{ij} ||z_i − z_j||²
      - Different Laplacian eigenvalue distribution
      - Lower von Neumann graph entropy

    Features
    --------
      Graph built from T frame nodes:
        A_{ij} = exp(−||z_i − z_j||² / σ²),  σ = learnable

      (a) Graph smoothness: E_G = Σ A_{ij} ||z_i − z_j||²
      (b) Laplacian eigenvalues: sorted λ_1..λ_T
      (c) Graph entropy: H(G) = −Σ λ_i(ρ_L) log λ_i(ρ_L)

    Ablation controls
    -----------------
      use_smoothness: bool — include E_G
      use_spectrum:   bool — include Laplacian eigenvalues
      use_entropy:    bool — include von Neumann entropy
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 128,
                 use_smoothness: bool = True, use_spectrum: bool = True,
                 use_entropy: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_smoothness = use_smoothness
        self.use_spectrum = use_spectrum
        self.use_entropy = use_entropy
        self.sigma = nn.Parameter(torch.tensor(1.0))

        # Determine input dimension
        in_dim = 0
        if use_smoothness:
            in_dim += 1   # scalar E_G
        if use_spectrum:
            in_dim += 8   # first 8 eigenvalues (T=8 frames → 8 eigenvalues)
        if use_entropy:
            in_dim += 1   # scalar H(G)

        assert in_dim > 0, "At least one topology component must be enabled"

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def compute_features(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Compute graph topology features from frame sequence.

        Args:
            z_seq: [B, T, D] frame CLS tokens

        Returns:
            feat: [B, in_dim] graph topology features
        """
        B, T, D = z_seq.shape
        eps = 1e-8
        feats = []

        # Build frame graph
        A = frame_graph_adjacency(z_seq, self.sigma)  # [B, T, T]

        if self.use_smoothness:
            # E_G = Σ A_{ij} ||z_i − z_j||²  (graph Dirichlet energy)
            dist_sq = torch.cdist(z_seq, z_seq) ** 2  # [B, T, T]
            E_G = (A * dist_sq).sum(dim=(1, 2)) / (T * (T - 1) + eps)  # [B]
            feats.append(E_G.unsqueeze(1))

        if self.use_spectrum or self.use_entropy:
            # Normalized Laplacian
            D = A.sum(dim=2) + eps  # [B, T]
            D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(D))
            I = torch.eye(T, device=z_seq.device).unsqueeze(0)
            L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt  # [B, T, T]
            eigvals = torch.linalg.eigvalsh(L_norm)    # [B, T], ascending

        if self.use_spectrum:
            feats.append(eigvals)  # [B, T] — all eigenvalues

        if self.use_entropy:
            # von Neumann entropy: H(G) = −Σ λ_i log λ_i, λ_i of ρ_L = L/Tr(L)
            trace_L = torch.diagonal(L_norm, dim1=1, dim2=2).sum(dim=1, keepdim=True)
            rho = L_norm / (trace_L.unsqueeze(2) + eps)
            rho_eig = torch.linalg.eigvalsh(rho).clamp(min=eps)
            H_G = -(rho_eig * torch.log(rho_eig)).sum(dim=1)  # [B]
            feats.append(H_G.unsqueeze(1))

        return torch.cat(feats, dim=1)

    def forward(self, clip_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            clip_features: dict with cls_seq [B, T, D]

        Returns:
            logits: [B, 2]
        """
        z_seq = clip_features['cls_seq']  # [B, T, D]
        feat = self.compute_features(z_seq)
        return self.classifier(feat)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GNN-D: Graph Neural Network Deepfake Detector
# ═══════════════════════════════════════════════════════════════════════════════

class STGCNConv(nn.Module):
    """Spatio-temporal graph convolution layer.

    Operates on a spatio-temporal graph where nodes are (frame, patch) pairs.
    Message passing separates spatial (within-frame) and temporal
    (cross-frame, same-patch) edges.

    Pure PyTorch — no torch_geometric dependency.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_s = nn.Parameter(torch.empty(in_dim, out_dim))  # spatial
        self.W_t = nn.Parameter(torch.empty(in_dim, out_dim))  # temporal
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_s)
        nn.init.xavier_uniform_(self.W_t)

    def forward(self, x: torch.Tensor, A_s: torch.Tensor,
                A_t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:   [B, T, N, in_dim]  node features
            A_s: [B, T, N, N]       spatial adjacency (within-frame)
            A_t: [B, T-1, N, N]     temporal adjacency (cross-frame)
                                     A_t[b, t, i, j]: edge from (t,j) to (t+1,i)

        Returns:
            out: [B, T, N, out_dim]
        """
        B, T, N, _ = x.shape
        eps = 1e-8

        # Spatial message passing (within each frame)
        D_s = A_s.sum(dim=3, keepdim=True) + eps  # [B, T, N, 1]
        A_s_norm = A_s / D_s
        msg_s = torch.einsum('btij,btjd->btid', A_s_norm, x)  # [B, T, N, in_dim]
        out_s = msg_s @ self.W_s  # [B, T, N, out_dim]

        # Temporal message passing (same patch across adjacent frames)
        out_t = torch.zeros_like(out_s)
        for t in range(T - 1):
            # From frame t to t+1
            D_t = A_t[:, t].sum(dim=2, keepdim=True) + eps  # [B, N, 1]
            A_t_norm = A_t[:, t] / D_t
            msg_t = torch.bmm(A_t_norm, x[:, t])  # [B, N, in_dim]
            out_t[:, t + 1] = msg_t @ self.W_t  # [B, N, out_dim]

        return out_s + out_t


class GNNDetector(nn.Module):
    """Spatio-temporal GNN for video deepfake detection.

    Hypothesis
    ----------
        P_real(G_st) ≠ P_fake(G_st)

    A GNN operating on the spatio-temporal patch graph can learn to
    detect structural anomalies that manual features (MI, spectrum) miss.
    The graph captures both:
      - Spatial: patch-to-patch information dependency within each frame
      - Temporal: same-patch evolution across consecutive frames

    Architecture
    ------------
      STGCNConv(768→256) → ReLU → Dropout(0.3)
      STGCNConv(256→128) → ReLU → Dropout(0.3)
      Global mean pooling → [B, 128]
      MLP(128→64→2)

    Ablation controls
    -----------------
      use_spatial:  bool — include spatial message passing
      use_temporal: bool — include temporal message passing
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256,
                 out_dim: int = 128, dropout: float = 0.3,
                 use_spatial: bool = True, use_temporal: bool = True):
        super().__init__()
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal

        self.conv1 = STGCNConv(in_dim, hidden_dim)
        self.conv2 = STGCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def build_spatial_adjacency(self, patches: torch.Tensor) -> torch.Tensor:
        """Build spatial adjacency from patch tokens (cosine similarity).

        Args:
            patches: [B, T, N, D] patch tokens

        Returns:
            A_s: [B, T, N, N] spatial adjacency per frame
        """
        B, T, N, D = patches.shape
        # Flatten batch+time for efficiency
        p = patches.reshape(B * T, N, D)
        p_norm = F.normalize(p, dim=2)
        A = torch.bmm(p_norm, p_norm.transpose(1, 2))  # [B*T, N, N]
        A = F.relu(A)
        A[:, range(N), range(N)] = 0.0
        return A.reshape(B, T, N, N)

    def build_temporal_adjacency(self, patches: torch.Tensor) -> torch.Tensor:
        """Build temporal adjacency: connect same patch across adjacent frames.

        Simple approach: use cosine similarity between patch i at
        frame t and patch i at frame t+1.

        Args:
            patches: [B, T, N, D]

        Returns:
            A_t: [B, T-1, N, N] temporal adjacency (diagonal-dominant)
        """
        B, T, N, D = patches.shape
        p_norm = F.normalize(patches, dim=3)  # [B, T, N, D]

        A_t = torch.zeros(B, T - 1, N, N, device=patches.device)
        for t in range(T - 1):
            # Similarity between frame t and t+1 for each patch
            sim = torch.einsum('bnd,bnd->bn', p_norm[:, t], p_norm[:, t + 1])
            # Diagonal temporal edges (same patch)
            A_t[:, t, range(N), range(N)] = F.relu(sim)

        return A_t

    def forward(self, clip_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            clip_features: dict with:
                patch_seq: [B, T, N, D]  patch tokens per frame

        Returns:
            logits: [B, 2]
        """
        x = clip_features['patch_seq']  # [B, T, N, D]

        # Build adjacency matrices
        A_s = self.build_spatial_adjacency(x) if self.use_spatial else \
            torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[2],
                       device=x.device)
        A_t = self.build_temporal_adjacency(x) if self.use_temporal else \
            torch.zeros(x.shape[0], x.shape[1] - 1, x.shape[2], x.shape[2],
                       device=x.device)

        # Two-layer ST-GCN
        h = F.relu(self.conv1(x, A_s, A_t))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, A_s, A_t))
        h = self.dropout(h)

        # Global mean pooling over (T, N)
        z_v = h.mean(dim=(1, 2))  # [B, out_dim]

        return self.classifier(z_v)
