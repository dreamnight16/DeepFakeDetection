"""
Information Graph Theory for Deepfake Detection
================================================

Implements three theoretical frameworks for analyzing CLIP token-space
information structure in real vs. fake images:

  1. Mutual Information Theory  — MI matrix between patch tokens
  2. Graph Spectral Theory       — Laplacian spectrum of information graph
  3. Graph Neural Network        — GCN on information graph for embeddings

All modules operate on frozen CLIP ViT patch tokens (CLS excluded).
No interference with the existing EffortDetector training pipeline.

Theory reference
----------------
  P_r(G) ≠ P_f(G)

  Deepfake generators can approximate pixel-level distributions but cannot
  fully replicate the information dependency structure and graph topology
  that arise from physical image formation (light + geometry + camera).

  Pipeline:
    Image → CLIP tokens → MI matrix → Information Graph
         → Graph Spectrum / GNN → Deepfake Discrimination

Author: personal experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def top_k_adjacency(M: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Sparsify adjacency matrix to top-k edges per node.

    Row-normalizes M first, then keeps top-k entries per row.
    Original edge weights are preserved (not binarized).

    Args:
        M: [B, N, N] dense adjacency matrix (MI values)
        k: number of edges to keep per node

    Returns:
        A: [B, N, N] sparse adjacency with top-k edges per node,
           original weights preserved
    """
    B, N, _ = M.shape
    device = M.device

    # Row-wise normalization
    D = M.sum(dim=2, keepdim=True) + 1e-8
    M_norm = M / D

    # Top-k indices per row
    _, top_idx = torch.topk(M_norm, k=k, dim=2)  # [B, N, k]

    # Create mask
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, k)
    row_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, -1, k)
    mask = torch.zeros(B, N, N, device=device)
    mask[batch_idx, row_idx, top_idx] = 1.0

    # Preserve original MI weights
    A = M * mask

    return A


def build_cosine_adjacency(tokens: torch.Tensor) -> torch.Tensor:
    """Build cosine similarity adjacency matrix from raw CLIP token embeddings.

    A_{ij} = max(0, cos(h_i, h_j))  for i ≠ j;  A_{ii} = 0.

    Args:
        tokens: [B, N, D] raw CLIP patch token embeddings

    Returns:
        A: [B, N, N] cosine adjacency (ReLU thresholded, no self-loops)
    """
    B, N, _ = tokens.shape

    # L2 normalize along feature dimension
    tokens_norm = F.normalize(tokens, p=2, dim=2)  # [B, N, D]

    # Cosine similarity
    A = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # [B, N, N]

    # ReLU threshold — only positive correlations
    A = F.relu(A)

    # Zero diagonal
    A[:, torch.arange(N), torch.arange(N)] = 0.0

    return A


def build_random_adjacency(
    N: int,
    density: float = 0.05,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Build random adjacency matrix (sanity-check baseline).

    Args:
        N: number of nodes (patches)
        density: edge density
        device: torch device

    Returns:
        A: [1, N, N] random symmetric adjacency
    """
    mask = torch.rand(N, N, device=device) < density
    rand_weights = torch.rand(N, N, device=device) * mask.float()
    # Symmetrize and zero diagonal
    rand_weights = (rand_weights + rand_weights.T) / 2.0
    rand_weights.fill_diagonal_(0.0)
    return rand_weights.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Mutual Information Analyzer
# ═══════════════════════════════════════════════════════════════════════════════

class MutualInformationAnalyzer:
    """Computes pairwise MI matrix between CLIP patch tokens.

    Theory
    ------
    Given PCA-reduced token vectors z_i ∈ R^d (d=32), the mutual
    information under a Gaussian assumption is:

        I(z_i; z_j) ≈ −½ log(1 − ρ_{ij}²)

    where ρ_{ij} is the Pearson correlation coefficient computed
    across the d PCA dimensions.

    Diagonal entries M_{ii} are set to 0 — self-entropy is not
    informative for pairwise structure.

    Hypothesis
    ----------
        I_r(x_i; x_j) ≠ I_f(x_i; x_j)

    Real images (physics-based formation) exhibit stable inter-region
    information dependencies. Generated images (learned P_f ≈ P_r)
    fail to fully replicate higher-order dependencies, producing a
    different MI matrix.

    Feature extraction from M:
      - Statistics: mean, std, Frobenius norm, skewness, kurtosis
      - Singular values (SVD): matrix rank structure
      - Eigenvalues of symmetric part: dominant modes
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute_mi_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute MI matrix from PCA-reduced token features.

        Args:
            z: [B, N, d] PCA-reduced patch tokens (CLS excluded).
               B = batch_size, N = num_patches (196), d = PCA dim (32).

        Returns:
            M: [B, N, N] mutual information matrix, M_{ii} = 0.
        """
        B, N, d = z.shape

        # Center each token vector across PCA dimensions
        z_centered = z - z.mean(dim=2, keepdim=True)  # [B, N, d]

        # Covariance matrix per sample
        # C_{ij} = (z_i - μ_i)·(z_j - μ_j)^T / (d−1)
        C = torch.bmm(z_centered, z_centered.transpose(1, 2)) / (d - 1)  # [B, N, N]

        # Standard deviations: σ_i = sqrt(C_{ii})
        sigma_sq = torch.diagonal(C, dim1=1, dim2=2)  # [B, N]
        sigma = torch.sqrt(sigma_sq.clamp(min=self.eps))  # [B, N]

        # Pearson correlation: ρ_{ij} = C_{ij} / (σ_i · σ_j)
        rho = C / (sigma.unsqueeze(2) * sigma.unsqueeze(1) + self.eps)

        # Clamp for numerical stability
        rho = torch.clamp(rho, min=-1.0 + self.eps, max=1.0 - self.eps)

        # Gaussian MI: I = −½ log(1 − ρ²)
        rho_sq = rho ** 2
        M = -0.5 * torch.log(1.0 - rho_sq + self.eps)

        # Zero diagonal — self-information is entropy, not structural
        M[:, torch.arange(N), torch.arange(N)] = 0.0

        return M

    def extract_features(self, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract discriminative features from the MI matrix.

        Args:
            M: [B, N, N] MI matrix (diagonal = 0).

        Returns:
            dict:
                mi_matrix: [B, N, N]  original MI matrix
                mi_stats:  [B, 5]     mean, std, ‖M‖_F, skew, kurtosis
                mi_svd:    [B, 20]    top-20 singular values of M
                mi_eigen:  [B, 10]    top-10 eigenvalues of (M+M^T)/2
        """
        B, N, _ = M.shape
        eps = self.eps

        # ── Off-diagonal statistics ───────────────────────────────────────
        off_mask = ~torch.eye(N, dtype=torch.bool, device=M.device)
        off_diag = M[:, off_mask].reshape(B, N * (N - 1))

        f_mean = off_diag.mean(dim=1)
        f_std = off_diag.std(dim=1, unbiased=True).clamp(min=eps)
        f_frob = torch.norm(M.reshape(B, -1), dim=1)

        # Higher moments (off-diagonal)
        delta = off_diag - f_mean.unsqueeze(1)
        f_skew = (delta ** 3).mean(dim=1) / (f_std ** 3 + eps)
        f_kurt = (delta ** 4).mean(dim=1) / (f_std ** 4 + eps) - 3.0  # excess

        f_mi_stats = torch.stack([f_mean, f_std, f_frob, f_skew, f_kurt], dim=1)

        # ── SVD: singular values capture matrix rank structure ─────────────
        _, S, _ = torch.linalg.svd(M.float(), full_matrices=False)
        f_mi_svd = S[:, :20]  # [B, 20]

        # ── Eigenvalues of symmetric part ──────────────────────────────────
        M_sym = (M + M.transpose(1, 2)) / 2.0
        eigvals = torch.linalg.eigvalsh(M_sym)  # [B, N], sorted ascending
        f_mi_eigen = eigvals[:, -10:]  # [B, 10] — largest eigenvalues

        return {
            'mi_matrix': M,
            'mi_stats':  f_mi_stats,
            'mi_svd':    f_mi_svd,
            'mi_eigen':  f_mi_eigen,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Graph Spectral Analyzer
# ═══════════════════════════════════════════════════════════════════════════════

class GraphSpectralAnalyzer:
    """Analyzes spectral properties of the information graph.

    From an adjacency matrix A (constructed via MI or cosine similarity):

      1. Normalized graph Laplacian:
           L_norm = I − D^{−½} A D^{−½},   D_{ii} = Σ_j A_{ij}

      2. Eigenvalue spectrum:
           0 = λ₁ ≤ λ₂ ≤ ... ≤ λ_N ≤ 2

      3. Spectral features:
           - Low-frequency band:  λ₁, ..., λ₁₀   (global structure)
           - High-frequency band: λ_{N−10}, ..., λ_N  (local anomalies)
           - Fiedler gap:         λ₂ − λ₁  (graph connectivity)
           - Algebraic connectivity: λ₂

      4. von Neumann graph entropy:
             H(G) = −Tr(ρ_L log ρ_L),   ρ_L = L / Tr(L)
                  = −Σ λ_i(ρ_L) log λ_i(ρ_L)

    Hypothesis
    ----------
        Λ_r ≠ Λ_f

    Since generation alters region relationships (A_r ≠ A_f),
    the Laplacian eigenspectrum differs between real and fake images.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        """Compute normalized graph Laplacian.

        L_norm = I − D^{−½} A D^{−½}

        where D_{ii} = Σ_j A_{ij} + ε (prevents division by zero
        for isolated nodes).

        Args:
            A: [B, N, N] adjacency matrix.

        Returns:
            L_norm: [B, N, N] normalized Laplacian.
        """
        B, N, _ = A.shape

        # Degree matrix with epsilon for isolated nodes
        D = A.sum(dim=2) + self.eps  # [B, N]
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(D))  # [B, N, N]

        I = torch.eye(N, device=A.device).unsqueeze(0)  # [1, N, N]
        L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt

        return L_norm

    def compute_spectrum(self, L: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Laplacian eigenvalue spectrum and derived features.

        Args:
            L: [B, N, N] normalized Laplacian.

        Returns:
            dict:
                eigvals:      [B, N]  all eigenvalues (sorted ascending)
                f_low:        [B, 10] λ₁..λ₁₀  (low frequency)
                f_high:       [B, 10] λ_{N-10}..λ_N (high frequency)
                spectral_gap: [B]     λ₂ − λ₁
                alg_conn:     [B]     λ₂ (Fiedler value)
        """
        B, N, _ = L.shape

        eigvals = torch.linalg.eigvalsh(L)  # [B, N], ascending

        # Low-frequency eigenvalues (global structure)
        f_low = eigvals[:, :10]  # [B, 10]

        # High-frequency eigenvalues (local anomalies)
        f_high = eigvals[:, N - 10:]  # [B, 10]

        # Fiedler gap (spectral gap)
        spectral_gap = eigvals[:, 1] - eigvals[:, 0]  # [B]

        # Algebraic connectivity = λ₂ (Fiedler value)
        alg_conn = eigvals[:, 1]  # [B]

        return {
            'eigvals':      eigvals,
            'f_low':        f_low,
            'f_high':       f_high,
            'spectral_gap': spectral_gap,
            'alg_conn':     alg_conn,
        }

    def von_neumann_entropy(self, L: torch.Tensor) -> torch.Tensor:
        """Compute von Neumann graph entropy.

        H(G) = −Tr(ρ_L log ρ_L) = −Σ λ_i log λ_i

        where ρ_L = L / Tr(L) is the density matrix of the graph.

        This measures the structural complexity of the graph —
        higher entropy indicates more uniformly distributed
        connectivity structure.

        Args:
            L: [B, N, N] Laplacian matrix.

        Returns:
            H_G: [B] von Neumann entropy per sample.
        """
        B, N, _ = L.shape

        trace_L = torch.diagonal(L, dim1=1, dim2=2).sum(dim=1, keepdim=True)  # [B]
        rho = L / (trace_L.unsqueeze(2) + self.eps)  # [B, N, N]

        rho_eigvals = torch.linalg.eigvalsh(rho)  # [B, N]
        rho_eigvals = torch.clamp(rho_eigvals, min=self.eps)

        H_G = -(rho_eigvals * torch.log(rho_eigvals)).sum(dim=1)  # [B]

        return H_G

    def extract_features(self, L: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full spectral feature extraction from Laplacian.

        Returns:
            dict:
                f_spec:       [B, 23] concatenated [f_low(10), f_high(10),
                                                  spectral_gap(1), alg_conn(1),
                                                  entropy(1)]
                f_low:        [B, 10]
                f_high:       [B, 10]
                spectral_gap: [B]
                alg_conn:     [B]
                entropy:      [B]
        """
        spec = self.compute_spectrum(L)
        H_G = self.von_neumann_entropy(L)

        f_spec = torch.cat([
            spec['f_low'],
            spec['f_high'],
            spec['spectral_gap'].unsqueeze(1),
            spec['alg_conn'].unsqueeze(1),
            H_G.unsqueeze(1),
        ], dim=1)  # [B, 23]

        return {
            **spec,
            'entropy': H_G,
            'f_spec':  f_spec,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Information Graph Neural Network
# ═══════════════════════════════════════════════════════════════════════════════

class GCNConv(nn.Module):
    """Graph convolution layer — pure PyTorch, no torch_geometric dependency.

    Message passing with mean aggregation and edge weights:

        h'_i = W · ( Σ_{j∈N(i)} A_{ij} h_j  /  Σ_{j∈N(i)} A_{ij} )

    In matrix form:
        H' = D^{−1} A H W

    Edge weights A_{ij} are preserved (not binarized) because MI
    strength is itself informative.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, in_dim]  node features
            A: [B, N, N]       adjacency (sparse top-k, with edge weights)

        Returns:
            out: [B, N, out_dim] updated node features
        """
        # Row-normalize: D^{−1} A  (mean aggregation)
        D = A.sum(dim=2, keepdim=True) + 1e-8  # [B, N, 1]
        A_norm = A / D

        # Message passing: H' = A_norm H W
        support = x @ self.W  # [B, N, out_dim]
        out = torch.bmm(A_norm, support)  # [B, N, out_dim]

        return out


class InfoGraphGNN(nn.Module):
    """Information Graph Neural Network.

    Architecture:
        GCNConv(768 → 256) → ReLU → Dropout(0.2)
        GCNConv(256 → 128) → ReLU → Dropout(0.2)
        Global Mean Pooling → z_G ∈ R^{128}

    NO classifier head — z_G is extracted and fed into the same
    Logistic Regression as all other features for fair comparison.

    The graph is built from MI-based adjacency with top-k sparsification,
    ensuring the GNN operates on information-theoretic relationships
    rather than raw semantic similarity.

    Node features: raw CLIP patch tokens (768-dim)
    Edge weights:   MI values from top-k sparsified adjacency
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, N, 768] raw CLIP patch tokens (before PCA)
            A: [B, N, N]   sparse MI adjacency (top-k per node)

        Returns:
            z_G: [B, 128] graph-level embedding via mean pooling
        """
        h = F.relu(self.conv1(x, A))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, A))
        h = self.dropout(h)

        # Global mean pooling
        z_G = h.mean(dim=1)  # [B, 128]

        return z_G
