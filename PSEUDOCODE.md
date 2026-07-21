# Effort AIGI Detection — Full Training Pipeline (Pseudo-code)

> Reflects the current implementation as of 2026-07-21.
> Covers: data sampling → mixup augmentation → model forward → loss computation → backward → evaluation → inference.

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ | Training set, $\mathbf{x}_i \in \mathbb{R}^{3 \times H \times W}$, $y_i \in \{0,1\}$ |
| $y=0$ | Real image |
| $y=1$ | AI-generated (fake) image |
| $f_\theta(\cdot)$ | Frozen CLIP ViT-L/14 + LoRA adapters → $\mathbb{R}^{1024}$ |
| $g_\phi(\cdot)$ | LoRA-augmented linear classifier → $\mathbb{R}^{2}$ |
| $\mathbf{z} = f_\theta(\mathbf{x})$ | Feature vector |
| $\mathbf{p} = g_\phi(\mathbf{z})$ | Logits |
| $\hat{y} = \operatorname{softmax}(\mathbf{p})_1$ | Predicted probability of fake |
| $B$ | Batch size |
| $\lambda \sim \operatorname{Beta}(\alpha, \alpha)$ | Mixing coefficient |
| $\gamma$ | Asymmetry exponent ($\gamma \ge 1$) |

---

## 2. Data Sampling

### 2.1 Standard Sampling (default)

```
For each epoch:
    Shuffle all indices 0..N-1
    Yield consecutive chunks of size B
    # No guarantee of per-batch class balance
```

### 2.2 BalanceBatchSampler (optional, `use_balance_batch_sampler: true`)

```
Algorithm: BalanceBatchSampler
────────────────────────────────
Input:  labels[0..N-1]        ∈ {0, 1, ..., C-1}
        batch_size_per_class   ∈ ℤ⁺        // samples per class per batch
        shuffle                ∈ {T, F}

Output: batches[0..M-1]       // each batch: exactly batch_size_per_class per class

1.  For each class c ∈ {0, ..., C-1}:
        class_indices[c] ← {i | labels[i] = c}

2.  min_len ← min( |class_indices[c]| for all c )
    M ← ⌊min_len / batch_size_per_class⌋           // num batches

3.  If M = 0: ERROR("not enough samples")

4.  If shuffle:
        For each class c: randomly permute class_indices[c]

5.  batches ← []
    For step ← 0 to M-1:
        batch ← []
        For each class c in sorted order:
            start ← step × batch_size_per_class
            end   ← start + batch_size_per_class
            batch.extend( class_indices[c][start : end] )
        batches.append(batch)

6.  If shuffle: randomly permute batches

7.  Return iterator over batches
```

**Integration:** When `use_balance_batch_sampler=true` and DDP is off:
- `batch_size_per_class` auto-derived as `train_batchSize // num_classes` (or set explicitly)
- DataLoader uses `batch_sampler=BalanceBatchSampler(...)` (no `batch_size`/`shuffle` args)
- Effective batch size = `num_classes × batch_size_per_class`
- Unused tail samples are dropped per epoch

---

## 3. Frequency-Domain Decomposition

```
Algorithm: DecomposeFFT
────────────────────────
Input:  x ∈ ℝ^{B×C×H×W}, cutoff τ ∈ (0, 1)

1.  Build circular mask (post-fftshift coordinates):
        r_max ← τ × min(H, W) / 2
        M_low[u,v]  ← 1  if dist((u,v), (H/2, W/2)) ≤ r_max  else 0
        M_high[u,v] ← 1 - M_low[u,v]

2.  For each image:
        X ← fftshift( FFT2D(x) )
        x_low  ← Re( IFFT2D( ifftshift( X ⊙ M_low  ) ) )
        x_high ← Re( IFFT2D( ifftshift( X ⊙ M_high ) ) )
        # By construction: x ≈ x_low + x_high

3.  Return (x_low, x_high)
```

**HF Blend** (preserve anchor structure, mix texture):
```
hf_blend(x1_low, x1_high, x2_high, λ):
    mixed ← x1_low + λ·x1_high + (1-λ)·x2_high
    clamp mixed to [min(x1), max(x1)]          // suppress FFT ringing
    return mixed
```

**LF Blend** (preserve anchor texture, mix structure):
```
lf_blend(x1_low, x1_high, x2_low, λ):
    mixed ← λ·x1_low + (1-λ)·x2_low + x1_high
    clamp mixed to [min(x1), max(x1)]
    return mixed
```

**YCbCr Option:** Convert RGB → YCbCr (BT.601) before FFT, blend in YCbCr space, convert back.

---

## 4. Mixup Variants

### 4.1 Asymmetric Mixup (`mixup_mode: "original"`)

```
Algorithm: AsymmetricMixup
───────────────────────────
Input:  batch {(x_i, y_i)}_{i=1}^B,  α, γ,  mix_domain D

1.  λ ~ Beta(α, α)
2.  π ← random permutation of {1..B}

3.  // Image blending
    If D = "rgb":
        For i = 1 to B:
            x̃_i ← λ·x_i + (1-λ)·x_{π(i)}              // pixel-space
    Else (D ∈ {hf, lf, ycbcr_hf, ycbcr_lf}):
        (x_low, x_high) ← DecomposeFFT({x_i}, cutoff)
        If D mixes high-freq:
            x̃_i ← hf_blend(x_i^low, x_i^high, x_{π(i)}^high, λ)
        Else (mixes low-freq):
            x̃_i ← lf_blend(x_i^low, x_i^high, x_{π(i)}^low, λ)

    // Q1 anchor rule: all cross-class pairs anchored on real
    For all pairs where (y_i=1, y_{π(i)}=0):  // fake+real
        Swap anchor ← real: x̃_i ← blend(real=anchor, fake=partner)

4.  // Label computation
    For i = 1 to B:
        If y_i = y_{π(i)}:                           // same class
            ỹ_i ← λ·y_i + (1-λ)·y_{π(i)}            // standard mixup
        Else:                                        // cross-class (real+fake)
            ỹ_i ← 1 - λ^γ                            // asymmetric, real-based

5.  Return {(x̃_i, ỹ_i)}_{i=1}^B
```

### 4.2 Hardest-K Mixup (`mixup_mode: "hardest_k"`)

```
Algorithm: HardestKMixup
─────────────────────────
Input:  batch {(x_i, y_i)}_{i=1}^B,  α, γ, K,  mix_domain D,  selection S

1.  λ ~ Beta(α, α)                                    // shared across pair types
2.  π ← random permutation of {1..B}

3.  Partition pairs:
        RR = {i | y_i=0 ∧ y_{π(i)}=0}                 // real+real
        FF = {i | y_i=1 ∧ y_{π(i)}=1}                 // fake+fake
        RF = {i | y_i≠y_{π(i)}}                        // cross-class (Q1: all real-anchored)

4.  // RR pairs: pixel-space, label = 0
    x̃_rr ← λ·x_RR + (1-λ)·x_{π(RR)}
    ỹ_rr ← 0_{|RR|}

5.  // FF pairs: pixel-space, label = 1
    x̃_ff ← λ·x_FF + (1-λ)·x_{π(FF)}
    ỹ_ff ← 1_{|FF|}

6.  // RF pairs: K-candidate selection
    R_anchor ← real positions from RF               // {r_1, ..., r_{n_rf}}
    F_pool   ← {i | y_i = 1}                        // all fake indices
    K_eff    ← min(K, |F_pool|)

    For each real anchor r_m (m = 1..n_rf):
        Sample {f_k}_{k=1}^{K_eff} from F_pool without replacement

        For k = 1 to K_eff:
            λ_{k,m} ~ Beta(α, α)                    // independent per candidate
            ỹ_{k,m} ← 1 - λ_{k,m}^γ

            If pixel-space:
                x̃_{k,m} ← λ_{k,m}·x_{r_m} + (1-λ_{k,m})·x_{f_k}
            Else (frequency):
                x̃_{k,m} ← hf_blend(x_{r_m}^low, x_{r_m}^high, x_{f_k}^high, λ_{k,m})
                           // or LF counterpart

        // Candidate selection
        If S = "hardest":
            Forward x̃_{k,m} through model (no_grad) → p_{k,m}
            ℓ_{k,m} ← -[ ỹ_{k,m}·log σ(p_{k,m})_1 + (1-ỹ_{k,m})·log σ(p_{k,m})_0 ]
            k* ← argmax_k ℓ_{k,m}
        Else if S = "random":
            k* ~ Uniform{1..K_eff}
        Else if S = "mean":
            Keep all K_eff candidates (aggregate loss later)

        Retain (x̃_{k*,m}, ỹ_{k*,m})

7.  // Combine all pair types
    x̃ ← concat[x̃_rr, x̃_ff, {x̃_rf}]
    ỹ ← concat[ỹ_rr, ỹ_ff, {ỹ_rf}]
    ỹ_hard ← concat[0_{|RR|}, 1_{|FF|}, 0_{n_rf}]    // hard labels for margin loss

8.  Return {(x̃_i, ỹ_i, ỹ_i^hard)}
```

**Mean-selection loss aggregation:**
```
L_batch = (1 / (|RR|+|FF|+n_rf)) × (
    Σ_{i∈RR} ℓ_i  +  Σ_{i∈FF} ℓ_i  +  Σ_{r=1}^{n_rf} (1/K)·Σ_{k=1}^K ℓ_{r,k}
)
```

### 4.3 Laplacian Pyramid Residual Mixup (`mixup_mode: "lap_pyramid"`)

```
Algorithm: LapPyramidMixup
───────────────────────────
Input:  batch {(x_i, y_i)}_{i=1}^B,  α, γ,  K_pyr (levels),  ω (weights)

1.  λ ~ Beta(α, α)
2.  π ← random permutation; partition into RR, FF, RF (Q1 merged)

3.  // RR, FF: pixel-space mixup as in Hardest-K, labels 0, 1

4.  // RF pairs: Laplacian pyramid mixing
    For each real anchor r with fake partner f = π(r):

        // (a) Build Gaussian pyramids (5-tap binomial [1,4,6,4,1]/16 + ↓2)
        G_0^r ← x_r,  G_0^f ← x_f
        For ℓ = 0 to K_pyr-1:
            G_{ℓ+1} ← pyr_down(G_ℓ)

        // (b) Build Laplacian pyramids (L_ℓ = G_ℓ - pyr_up(G_{ℓ+1}))
        For ℓ = 0 to K_pyr-1:
            L_ℓ^r ← G_ℓ^r - pyr_up(G_{ℓ+1}^r)
            L_ℓ^f ← G_ℓ^f - pyr_up(G_{ℓ+1}^f)

        // (c) Fake injection strength
        q ← 1 - λ

        // (d) Mix residual bands
        For ℓ = 0 to K_pyr-1:
            L_ℓ^mix ← (1-q)·L_ℓ^r + q·L_ℓ^f

        // (e) Fake evidence from residual energy
        For ℓ = 0 to K_pyr-1:
            E_ℓ^r ← ||L_ℓ^r||_F^2
            E_ℓ^f ← ||L_ℓ^f||_F^2

        e_f ← Σ_ℓ ω_ℓ·q²·E_ℓ^f  /  ( Σ_ℓ ω_ℓ[(1-q)²·E_ℓ^r + q²·E_ℓ^f] + ε )
            // ε = 1e-8

        // (f) Reconstruct: coarse (real) + mixed residuals
        x̃ ← G_{K_pyr}^r
        For ℓ = K_pyr-1 downto 0:
            x̃ ← pyr_up(x̃) + L_ℓ^mix
        Clamp x̃ to [min(x_r), max(x_r)]

        // (g) Energy-grounded soft label
        ỹ ← 1 - (1 - e_f)^γ
            // q→0 ⇒ e_f→0 ⇒ ỹ→0 (real)
            // q→1 ⇒ e_f→1 ⇒ ỹ→1 (fake)

5.  ỹ_hard ← concat[0_{|RR|}, 1_{|FF|}, 0_{n_rf}]

6.  Return {(x̃_i, ỹ_i, ỹ_i^hard)}
```

---

## 5. Model Architecture

```
Model: EffortDetector
──────────────────────
Backbone: CLIP ViT-L/14 (frozen)
    - All q_proj, k_proj, v_proj, out_proj ← replaced with LoRA wrappers
    - LoRA rank r = 4, α = 16, dropout = 0
    - Only lora_A, lora_B parameters are trainable

Classifier Head: LoRA-augmented Linear
    - Input: 1024, Output: 2
    - LoRA rank r = 2, α = 8
    - p = W·z + b + (α/r)·(z·A)·B        // W,b frozen; A,B trainable

Forward pass:
    z ← ViT-L/14_LoRA(x).pooler_output     // [B, 1024]
    p ← head(z)                              // [B, 2]
    ŷ ← softmax(p)[:, 1]                     // [B], fake probability
    Return {cls: p, prob: ŷ, feat: z}
```

---

## 6. Loss Functions

### 6.1 Soft-Label Cross-Entropy (primary)

```
When mixup is active (label_soft present):

    For each sample i with logits p_i and soft label ỹ_i ∈ [0,1]:
        log_probs ← log_softmax(p_i)                  // [2]
        ℓ_i ← -( ỹ_i·log_probs[1] + (1-ỹ_i)·log_probs[0] )

    If mixup_selection = "mean":
        // rr losses + (K·n_rf rf losses averaged per anchor) + ff losses
        rr_losses ← ℓ[real indices][:n_rr]
        rf_losses ← mean( ℓ[real indices][n_rr:].reshape(K, n_rf), dim=0 )  // [n_rf]
        ff_losses ← ℓ[fake indices]
        L_CE ← mean(concat[rr_losses, rf_losses, ff_losses])
    Else:
        L_CE ← mean(ℓ_i)

When mixup is off:
    L_CE ← CrossEntropyLoss(p, y)                    // standard hard-label CE
```

### 6.2 Asymmetric Center Loss (auxiliary, optional)

```
Algorithm: AsymmetricCenterLoss
────────────────────────────────
Input:  features z ∈ ℝ^{B×1024},  hard labels y^hard ∈ {0,1}^B
Params: learnable center c ∈ ℝ^{1024},  margin m = 0.5

1.  ẑ ← normalize(z, dim=1)              // L2-normalized features
2.  ĉ ← normalize(c, dim=0)               // L2-normalized center
3.  d ← ||ẑ - ĉ||_2                       // [B], ∈ [0, 2]

4.  For each sample i:
        If y_i^hard = 0 (real):            loss_i ← d_i²
        If y_i^hard = 1 (fake):            loss_i ← max(0, m - d_i)²

5.  L_center ← mean(loss_i)
```

### 6.3 Combined Loss

```
L_overall = 
    | L_CE                            if margin_loss_mode = "off"
    | L_CE + w · L_center             if margin_loss_mode = "add"    (w = 1.0)
    | L_center                        if margin_loss_mode = "replace"
```

**Per-class decomposition** (for PCGrad optimizer):
```
loss_dict = {
    overall:    L_overall,
    real_loss:  CrossEntropyLoss(p[real],  y[real]),
    fake_loss:  CrossEntropyLoss(p[fake], y[fake]),
    margin_loss: L_center (if active)
}
```

---

## 7. Training Loop

```
Algorithm: TrainEpoch
──────────────────────
Input:  train_loader,  model (f_θ, g_ϕ),  optimizer,  scheduler,
        mixup config,  margin loss config,
        test_loaders,  epoch,  current best metrics

1.  model.train()

2.  For each batch from train_loader:
        data ← batch.to(cuda)

        // ── Step 1: Mixup augmentation (training only) ──
        If use_mixup:
            If mixup_mode = "original":
                data ← AsymmetricMixup(data, α, γ, mix_domain)
            Else If mixup_mode = "hardest_k":
                data ← HardestKMixup(model, data, K, α, γ, mix_domain, selection)
            Else If mixup_mode = "lap_pyramid":
                data ← LapPyramidMixup(data, α, γ, K_pyr)
        // Else: keep original images + hard labels

        // ── Step 2: Forward ──
        pred ← model(data)                       // {cls, prob, feat}
        losses ← model.get_losses(data, pred)    // {overall, real_loss, fake_loss, ...}

        // ── Step 3: Backward ──
        optimizer.zero_grad()
        If optimizer is PCGrad:
            optimizer.pc_backward([losses.real_loss, losses.fake_loss])
        Else:
            losses.overall.backward()
        optimizer.step()

        // ── Step 4: Metrics logging (every 300 iters) ──
        Log train loss + metrics (acc, auc, eer, ap) to TensorBoard

        // ── Step 5: Periodic evaluation (every T steps) ──
        If step % T = 0 and test_loaders exist:
            For each test_set:
                metrics ← Evaluate(model, test_loader)
                Update best checkpoint if AUC improved
                Log to TensorBoard

3.  scheduler.step()                              // if configured

4.  If SWA enabled and epoch > swa_start:
        swa_model.update_parameters(model)
```

---

## 8. Evaluation & Inference

### 8.1 Evaluation (no mixup)

```
Algorithm: Evaluate
────────────────────
Input:  model,  test_loader

1.  model.eval()
2.  predictions ← [],  labels ← [],  features ← []

3.  For each batch:
        data ← batch.to(cuda)
        data.label ← (data.label ≠ 0).long()      // binarize

        pred ← model(data, inference=True)         // {cls, prob, feat}

        labels.extend(data.label)
        predictions.extend(pred.prob)
        features.extend(pred.feat)

4.  Compute metrics:
        AUC, EER, Accuracy, AP + per-class accuracy (real_acc, fake_acc)

5.  Return metrics
```

### 8.2 TAA Inference (multi-crop, texture-aware)

```
Algorithm: TextureAwareInference
─────────────────────────────────
Input:  image I,  model,  N crops,  γ_t,  β

// During dataset __getitem__ (test mode, multi_crop=True):
//   C_0 ← full image I (resized)
//   C_1..C_{N-1} ← selected patches (texture-based or random, resized)
//   t_j ← Laplacian variance score for each patch (t_0 = 0 sentinel)

1.  For j = 0 to N-1:
        z_j ← f_θ(C_j)
        p_j ← g_φ(z_j)
        s_j ← softmax(p_j)[1]                      // per-crop fake probability

2.  If texture scores {t_j} available:
        w_j ← t_j^γ_t / Σ_k t_k^γ_t                // power-law normalized weights
        ŷ ← β·s_0 + (1-β)·Σ_{j=1}^{N-1} w_j·s_j   // weighted ensemble
    Else:
        j* ← argmax_j |s_j - 0.5|                   // max-confidence selection
        ŷ ← s_{j*}

3.  Return ŷ
```

### 8.3 Adaptive Threshold (OWTTT)

```
Algorithm: ComputeAdaptiveThreshold
────────────────────────────────────
Input:  prediction_queue (sliding window of recent predictions)

If |queue| < 32: return 0.5

For th in [0.00, 0.01, ..., 0.99]:
    mask ← (queue ≥ th)
    w1 ← |mask| / |queue|,   w0 ← 1 - w1
    If w1=0 or w0=0: continue
    v0 ← Var(queue[~mask])
    v1 ← Var(queue[mask])
    min_gap ← min(|queue - th|)
    crit ← w0·v0 + w1·v1 - gap_weight · min_gap
    Keep th with minimum crit

Return best_th
// Used to compute acc_adaptive during testing
```

---

## 9. Hyperparameter Summary

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Mixup alpha | α | 1.0 | Beta distribution shape |
| Mixup gamma | γ | 5.0 | Asymmetry exponent |
| Mixup K | K | 1 | Fake candidates per real anchor |
| Mixup selection | S | hardest | {hardest, random, mean} |
| Mixup domain | D | rgb | {rgb, hf, lf, ycbcr_hf, ycbcr_lf} |
| FFT cutoff | τ | 0.125 | Frequency cutoff fraction |
| Pyramid levels | K_pyr | 3 | Laplacian pyramid depth |
| Margin loss mode | — | off | {off, add, replace} |
| Margin m | m | 0.5 | Minimum fake-center distance |
| Margin weight | w | 1.0 | Center loss weight when add |
| LoRA rank (attn) | r_attn | 4 | ViT attention LoRA rank |
| LoRA rank (head) | r_head | 2 | Classifier LoRA rank |
| LoRA α (attn) | α_lora | 16 | Attention LoRA scaling |
| LoRA α (head) | α_lora | 8 | Head LoRA scaling |
| Texture gamma | γ_t | 1.5 | Power for texture attention |
| Fusion weight | β | 0.5 | Full-image weight in TAA |
| Balance batch | — | false | Enable BalanceBatchSampler |
| Batch per class | — | auto | Samples per class per batch |

---

## 10. Reproducibility Checklist

1. CLIP ViT-L/14 backbone: pretrained weights loaded, **all non-LoRA params frozen**.
2. LoRA injected at **all** ViT attention blocks' q_proj, k_proj, v_proj, out_proj.
3. Mixup applied per-batch, before forward, **training only**.
4. Q1 anchor rule: all cross-class pairs use real image as structural anchor.
5. FFT decomposition uses `fftshift` for centered frequency mask.
6. FFT-reconstructed images clamped to source range to suppress ringing.
7. Laplacian pyramid: 5-tap binomial kernel `[1,4,6,4,1]/16` (Burt & Adelson, 1983).
8. Soft-label CE uses `log_softmax` (not sigmoid) for numerical stability.
9. Center loss: features and center are L2-normalized before distance.
10. No mixup during validation/testing.
11. BalanceBatchSampler: `batch_size_per_class` auto-derived from `train_batchSize` by default.
12. Inference: crop predictions fused by TAA or max-confidence.
