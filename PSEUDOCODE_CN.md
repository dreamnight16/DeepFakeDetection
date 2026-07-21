# Effort AIGI 检测框架 — 完整训练流程（伪代码）

> 反映截至 2026-07-21 的当前实现。
> 涵盖：数据采样 → Mixup 增强 → 模型前向 → 损失计算 → 反向传播 → 评估 → 推理。

---

## 1. 符号表

| 符号 | 含义 |
|------|------|
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ | 训练集，$\mathbf{x}_i \in \mathbb{R}^{3 \times H \times W}$，$y_i \in \{0,1\}$ |
| $y=0$ | 真实图像 (Real) |
| $y=1$ | AI 生成图像 (Fake) |
| $f_\theta(\cdot)$ | 冻结 CLIP ViT-L/14 + LoRA 适配器 → $\mathbb{R}^{1024}$ |
| $g_\phi(\cdot)$ | LoRA 增强线性分类头 → $\mathbb{R}^{2}$ |
| $\mathbf{z} = f_\theta(\mathbf{x})$ | 特征向量 |
| $\mathbf{p} = g_\phi(\mathbf{z})$ | 分类 logits |
| $\hat{y} = \operatorname{softmax}(\mathbf{p})_1$ | 预测为 Fake 的概率 |
| $B$ | 批次大小 |
| $\lambda \sim \operatorname{Beta}(\alpha, \alpha)$ | 混合系数 |
| $\gamma$ | 不对称指数（$\gamma \ge 1$，越大标签越偏向 Fake） |

---

## 2. 数据采样

### 2.1 标准采样（默认）

```
每轮 epoch:
    打乱全部索引 0..N-1
    按步长 B 依次切分输出
    # 不保证每批类别均衡
```

### 2.2 BalanceBatchSampler（可选，`use_balance_batch_sampler: true`）

```
算法：BalanceBatchSampler（均衡批次采样器）
────────────────────────────────────────────
输入:  labels[0..N-1]        ∈ {0, 1, ..., C-1}
        batch_size_per_class  ∈ ℤ⁺        // 每类每批样本数
        shuffle               ∈ {T, F}

输出: batches[0..M-1]       // 每个批次：每类恰好 batch_size_per_class 个样本

1.  对每个类别 c ∈ {0, ..., C-1}:
        class_indices[c] ← {i | labels[i] = c}

2.  min_len ← min( |class_indices[c]| 对所有 c )
    M ← ⌊min_len / batch_size_per_class⌋           // 批次数

3.  若 M = 0: 报错（"样本不足"）

4.  若 shuffle 为真:
        对每个类别 c: 随机打乱 class_indices[c]

5.  batches ← []
    对 step ← 0 到 M-1:
        batch ← []
        对每个类别 c（按排序顺序）:
            start ← step × batch_size_per_class
            end   ← start + batch_size_per_class
            batch.extend( class_indices[c][start : end] )
        batches.append(batch)

6.  若 shuffle 为真: 随机打乱 batches

7.  返回 batches 的迭代器
```

**集成方式：** 当 `use_balance_batch_sampler=true` 且非 DDP 时：
- `batch_size_per_class` 自动从 `train_batchSize // 类别数` 推导（也可显式指定）
- DataLoader 使用 `batch_sampler=BalanceBatchSampler(...)`（不传 `batch_size`/`shuffle`）
- 实际批次大小 = `类别数 × batch_size_per_class`
- 每轮丢弃不足一个完整批次的尾部样本

---

## 3. 频域分解

```
算法：DecomposeFFT（FFT 频域分解）
──────────────────────────────────
输入:  x ∈ ℝ^{B×C×H×W},  截止频率 τ ∈ (0, 1)

1.  构建圆形掩码（fftshift 坐标系）:
        r_max ← τ × min(H, W) / 2
        M_low[u,v]  ← 1  若 dist((u,v), (H/2, W/2)) ≤ r_max  否则 0
        M_high[u,v] ← 1 − M_low[u,v]

2.  对每张图像:
        X ← fftshift( FFT2D(x) )
        x_low  ← Re( IFFT2D( ifftshift( X ⊙ M_low  ) ) )
        x_high ← Re( IFFT2D( ifftshift( X ⊙ M_high ) ) )
        # 构造保证: x ≈ x_low + x_high

3.  返回 (x_low, x_high)
```

**HF 混合**（保留锚点结构，混合纹理）:
```
hf_blend(x1_low, x1_high, x2_high, λ):
    mixed ← x1_low + λ·x1_high + (1−λ)·x2_high
    裁剪 mixed 至 [min(x1), max(x1)]          // 抑制 FFT 振铃伪影
    返回 mixed
```

**LF 混合**（保留锚点纹理，混合结构）:
```
lf_blend(x1_low, x1_high, x2_low, λ):
    mixed ← λ·x1_low + (1−λ)·x2_low + x1_high
    裁剪 mixed 至 [min(x1), max(x1)]
    返回 mixed
```

**YCbCr 选项：** 混合前将 RGB 转 YCbCr (BT.601)，在 YCbCr 空间混合后转回 RGB。

---

## 4. Mixup 变体

### 4.1 非对称 Mixup（`mixup_mode: "original"`）

```
算法：AsymmetricMixup（非对称 Mixup）
─────────────────────────────────────
输入:  批次 {(x_i, y_i)}_{i=1}^B,  α, γ,  混合域 D

1.  λ ~ Beta(α, α)
2.  π ← {1..B} 的随机排列

3.  // 图像混合
    若 D = "rgb":
        对 i = 1 到 B:
            x̃_i ← λ·x_i + (1−λ)·x_{π(i)}              // 像素空间混合
    否则（D ∈ {hf, lf, ycbcr_hf, ycbcr_lf}）:
        (x_low, x_high) ← DecomposeFFT({x_i}, cutoff=τ)
        若混合高频:
            x̃_i ← hf_blend(x_i^low, x_i^high, x_{π(i)}^high, λ)
        否则（混合低频）:
            x̃_i ← lf_blend(x_i^low, x_i^high, x_{π(i)}^low, λ)

    // Q1 锚定规则：所有跨类配对以真图为锚
    对所有 (y_i=1, y_{π(i)}=0) 的配对:    // fake+real
        交换锚点 ← 真图: x̃_i ← blend(锚点=真图, 伙伴=假图)

4.  // 标签计算（与混合域无关）
    对 i = 1 到 B:
        若 y_i = y_{π(i)}:                           // 同类别
            ỹ_i ← λ·y_i + (1−λ)·y_{π(i)}            // 标准 Mixup 标签
        否则:                                         // 跨类别（Real+Fake）
            ỹ_i ← 1 − λ^γ                            // 非对称标签，锚定于真图

5.  返回 {(x̃_i, ỹ_i)}_{i=1}^B
```

### 4.2 最难-K Mixup（`mixup_mode: "hardest_k"`）

```
算法：HardestKMixup（最难-K Mixup）
───────────────────────────────────
输入:  批次 {(x_i, y_i)}_{i=1}^B,  α, γ, K,  混合域 D,  选择策略 S

1.  λ ~ Beta(α, α)                                    // 所有配对类型共享
2.  π ← {1..B} 的随机排列

3.  按配对类型分区:
        RR = {i | y_i=0 ∧ y_{π(i)}=0}                 // Real+Real
        FF = {i | y_i=1 ∧ y_{π(i)}=1}                 // Fake+Fake
        RF = {i | y_i≠y_{π(i)}}                        // 跨类别（Q1：统一以真图为锚）

4.  // RR 对：像素空间混合，标签 = 0
    x̃_rr ← λ·x_RR + (1−λ)·x_{π(RR)}
    ỹ_rr ← 0_{|RR|}

5.  // FF 对：像素空间混合，标签 = 1
    x̃_ff ← λ·x_FF + (1−λ)·x_{π(FF)}
    ỹ_ff ← 1_{|FF|}

6.  // RF 对：K-候选采样
    R_anchor ← RF 中的真实样本位置                    // {r_1, ..., r_{n_rf}}
    F_pool   ← {i | y_i = 1}                          // 所有 Fake 索引
    K_eff    ← min(K, |F_pool|)

    对每个真实锚点 r_m（m = 1..n_rf）:
        从 F_pool 无放回采样 {f_k}_{k=1}^{K_eff}

        对 k = 1 到 K_eff:
            λ_{k,m} ~ Beta(α, α)                      // 每个候选独立采样
            ỹ_{k,m} ← 1 − λ_{k,m}^γ

            若像素空间:
                x̃_{k,m} ← λ_{k,m}·x_{r_m} + (1−λ_{k,m})·x_{f_k}
            否则（频域）:
                x̃_{k,m} ← hf_blend(x_{r_m}^low, x_{r_m}^high, x_{f_k}^high, λ_{k,m})
                           // 或 LF 对应版本

        // 候选选择
        若 S = "hardest":
            前向传播 x̃_{k,m}（无梯度） → p_{k,m}
            ℓ_{k,m} ← −[ ỹ_{k,m}·log σ(p_{k,m})_1 + (1−ỹ_{k,m})·log σ(p_{k,m})_0 ]
            k* ← argmax_k ℓ_{k,m}                     // 选损失最大的候选
        否则若 S = "random":
            k* ~ Uniform{1..K_eff}
        否则若 S = "mean":
            保留全部 K_eff 个候选（损失后续聚合）

        保留 (x̃_{k*,m}, ỹ_{k*,m})

7.  // 拼接所有配对类型
    x̃ ← concat[x̃_rr, x̃_ff, {x̃_rf}]
    ỹ ← concat[ỹ_rr, ỹ_ff, {ỹ_rf}]
    ỹ_hard ← concat[0_{|RR|}, 1_{|FF|}, 0_{n_rf}]    // 硬标签（供间隔损失使用）

8.  返回 {(x̃_i, ỹ_i, ỹ_i^hard)}
```

**均值模式 (mean) 的损失聚合：**
```
L_batch = (1 / (|RR|+|FF|+n_rf)) × (
    Σ_{i∈RR} ℓ_i  +  Σ_{i∈FF} ℓ_i  +  Σ_{r=1}^{n_rf} (1/K)·Σ_{k=1}^K ℓ_{r,k}
)
```

### 4.3 拉普拉斯金字塔残差 Mixup（`mixup_mode: "lap_pyramid"`）

```
算法：LapPyramidMixup（拉普拉斯金字塔残差 Mixup）
─────────────────────────────────────────────────
输入:  批次 {(x_i, y_i)}_{i=1}^B,  α, γ,  K_pyr（金字塔层数）,  ω（重要性权重）

1.  λ ~ Beta(α, α)
2.  π ← 随机排列；按配对类型分区 RR, FF, RF（Q1 合并 FR 入 RF）

3.  // RR、FF：像素空间 Mixup（同 Hardest-K），标签分别为 0 和 1

4.  // RF 对：拉普拉斯金字塔混合
    对每个真实锚点 r 与对应 Fake 伙伴 f = π(r):

        // (a) 构建高斯金字塔（5 抽头二项式核 [1,4,6,4,1]/16 + ↓2）
        G_0^r ← x_r,  G_0^f ← x_f
        对 ℓ = 0 到 K_pyr−1:
            G_{ℓ+1} ← pyr_down(G_ℓ)

        // (b) 构建拉普拉斯金字塔（L_ℓ = G_ℓ − pyr_up(G_{ℓ+1})）
        对 ℓ = 0 到 K_pyr−1:
            L_ℓ^r ← G_ℓ^r − pyr_up(G_{ℓ+1}^r)
            L_ℓ^f ← G_ℓ^f − pyr_up(G_{ℓ+1}^f)

        // (c) Fake 注入强度
        q ← 1 − λ

        // (d) 逐层混合拉普拉斯残差
        对 ℓ = 0 到 K_pyr−1:
            L_ℓ^mix ← (1−q)·L_ℓ^r + q·L_ℓ^f

        // (e) 由残差能量推导 Fake 证据度
        对 ℓ = 0 到 K_pyr−1:
            E_ℓ^r ← ||L_ℓ^r||_F^2
            E_ℓ^f ← ||L_ℓ^f||_F^2

        e_f ← Σ_ℓ ω_ℓ·q²·E_ℓ^f  /  ( Σ_ℓ ω_ℓ[(1−q)²·E_ℓ^r + q²·E_ℓ^f] + ε )
            // ε = 1e−8

        // (f) 重建：真实粗结构 + 混合残差
        x̃ ← G_{K_pyr}^r
        对 ℓ = K_pyr−1 downto 0:
            x̃ ← pyr_up(x̃) + L_ℓ^mix
        裁剪 x̃ 至 [min(x_r), max(x_r)]

        // (g) 能量锚定软标签
        ỹ ← 1 − (1 − e_f)^γ
            // q→0 ⇒ e_f→0 ⇒ ỹ→0（Real）
            // q→1 ⇒ e_f→1 ⇒ ỹ→1（Fake）

5.  ỹ_hard ← concat[0_{|RR|}, 1_{|FF|}, 0_{n_rf}]

6.  返回 {(x̃_i, ỹ_i, ỹ_i^hard)}
```

**设计动机：** 与 Asymmetric/Hardest-K Mixup 使用固定公式 ỹ = 1−λ^γ 不同，
拉普拉斯金字塔残差 Mixup 的软标签 ỹ = 1−(1−e_f)^γ 由实际注入图像的
Fake 残差能量占比 e_f 决定，使标签与图像内容的实际变化程度保持一致。

---

## 5. 模型架构

```
模型：EffortDetector
────────────────────
骨干网络: CLIP ViT-L/14（冻结）
    - 所有 q_proj, k_proj, v_proj, out_proj ← 替换为 LoRA 封装层
    - LoRA 秩 r = 4, α = 16, dropout = 0
    - 仅 lora_A, lora_B 参数可训练

分类头: LoRA 增强线性层
    - 输入: 1024, 输出: 2
    - LoRA 秩 r = 2, α = 8
    - p = W·z + b + (α/r)·(z·A)·B          // W,b 冻结; A,B 可训练

前向传播:
    z ← ViT-L/14_LoRA(x).pooler_output       // [B, 1024]
    p ← head(z)                                // [B, 2]
    ŷ ← softmax(p)[:, 1]                       // [B], Fake 概率
    返回 {cls: p, prob: ŷ, feat: z}
```

---

## 6. 损失函数

### 6.1 软标签交叉熵（主损失）

```
Mixup 激活时（存在 label_soft）:

    对每个样本 i（logits p_i，软标签 ỹ_i ∈ [0,1]）:
        log_probs ← log_softmax(p_i)                  // [2]
        ℓ_i ← −( ỹ_i·log_probs[1] + (1−ỹ_i)·log_probs[0] )

    若 mixup_selection = "mean":
        // rr 损失 + (K·n_rf 个 rf 损失按锚点取均值) + ff 损失
        rr_losses ← ℓ[real 索引][:n_rr]
        rf_losses ← mean( ℓ[real 索引][n_rr:].reshape(K, n_rf), dim=0 )  // [n_rf]
        ff_losses ← ℓ[fake 索引]
        L_CE ← mean(concat[rr_losses, rf_losses, ff_losses])
    否则:
        L_CE ← mean(ℓ_i)

Mixup 关闭时:
    L_CE ← CrossEntropyLoss(p, y)                    // 标准硬标签交叉熵
```

### 6.2 非对称中心损失（辅助，可选）

```
算法：AsymmetricCenterLoss（非对称中心损失）
────────────────────────────────────────────
输入:  特征 z ∈ ℝ^{B×1024},  硬标签 y^hard ∈ {0,1}^B
参数:  可学习中心 c ∈ ℝ^{1024},  间隔 m = 0.5

1.  ẑ ← normalize(z, dim=1)              // L2 归一化特征
2.  ĉ ← normalize(c, dim=0)              // L2 归一化中心
3.  d ← ||ẑ − ĉ||_2                       // [B], ∈ [0, 2]

4.  对每个样本 i:
        若 y_i^hard = 0 (Real):           loss_i ← d_i²
        若 y_i^hard = 1 (Fake):           loss_i ← max(0, m − d_i)²

5.  L_center ← mean(loss_i)

直觉：Real 特征聚集于 c 周围（最小化 d²）；Fake 特征距 c 不得小于 m（仅在 d < m 时惩罚）。
```

### 6.3 组合损失

```
L_overall = 
    | L_CE                            若 margin_loss_mode = "off"
    | L_CE + w · L_center             若 margin_loss_mode = "add"     (w = 1.0)
    | L_center                        若 margin_loss_mode = "replace"
```

**按类别分解**（供 PCGrad 优化器使用）:
```
loss_dict = {
    overall:      L_overall,
    real_loss:    CrossEntropyLoss(p[real],  y[real]),
    fake_loss:    CrossEntropyLoss(p[fake], y[fake]),
    margin_loss:  L_center（若启用）
}
```

---

## 7. 训练主循环

```
算法：TrainEpoch（每轮训练）
────────────────────────────
输入:  train_loader,  model (f_θ, g_ϕ),  optimizer,  scheduler,
        mixup 配置,  间隔损失配置,
        test_loaders,  epoch,  当前最佳指标

1.  model.train()

2.  对 train_loader 中的每个批次:
        data ← batch.to(cuda)

        // ── 步骤 1: Mixup 增强（仅训练时）──
        若 use_mixup 为真:
            若 mixup_mode = "original":
                data ← AsymmetricMixup(data, α, γ, mix_domain)
            否则若 mixup_mode = "hardest_k":
                data ← HardestKMixup(model, data, K, α, γ, mix_domain, selection)
            否则若 mixup_mode = "lap_pyramid":
                data ← LapPyramidMixup(data, α, γ, K_pyr)
        // 否则：保留原始图像 + 硬标签

        // ── 步骤 2: 前向传播 ──
        pred ← model(data)                       // {cls, prob, feat}
        losses ← model.get_losses(data, pred)    // {overall, real_loss, fake_loss, ...}

        // ── 步骤 3: 反向传播 ──
        optimizer.zero_grad()
        若 optimizer 为 PCGrad:
            optimizer.pc_backward([losses.real_loss, losses.fake_loss])
        否则:
            losses.overall.backward()
        optimizer.step()

        // ── 步骤 4: 指标记录（每 300 次迭代）──
        将训练损失 + 指标（acc, auc, eer, ap）写入 TensorBoard

        // ── 步骤 5: 定期评估（每 T 步）──
        若 step % T = 0 且 test_loaders 存在:
            对每个 test_set:
                metrics ← Evaluate(model, test_loader)
                若 AUC 改善则更新最佳检查点
                写入 TensorBoard

3.  scheduler.step()                              // 若配置了调度器

4.  若启用 SWA 且 epoch > swa_start:
        swa_model.update_parameters(model)
```

---

## 8. 评估与推理

### 8.1 评估（无 Mixup）

```
算法：Evaluate（评估）
──────────────────────
输入:  model,  test_loader

1.  model.eval()
2.  predictions ← [],  labels ← [],  features ← []

3.  对每个批次:
        data ← batch.to(cuda)
        data.label ← (data.label ≠ 0).long()      // 二值化

        pred ← model(data, inference=True)         // {cls, prob, feat}

        labels.extend(data.label)
        predictions.extend(pred.prob)
        features.extend(pred.feat)

4.  计算指标:
        AUC, EER, Accuracy, AP + 各类别准确率 (real_acc, fake_acc)

5.  返回 metrics
```

### 8.2 TAA 推理（多裁剪纹理感知加权集成）

```
算法：TextureAwareInference（纹理感知推理）
───────────────────────────────────────────
输入:  图像 I,  model,  N 个裁剪,  γ_t（纹理 Gamma）,  β（融合权重）

// 在 dataset __getitem__ 中（测试模式, multi_crop=True）:
//   C_0 ← 完整图像 I（缩放至模型输入尺寸）
//   C_1..C_{N-1} ← 选取的图像块（基于纹理分数或随机，缩放）
//   t_j ← 每个图像块的 Laplacian 方差分数（t_0 = 0 哨兵值）

1.  对 j = 0 到 N−1:
        z_j ← f_θ(C_j)
        p_j ← g_φ(z_j)
        s_j ← softmax(p_j)[1]                       // 逐裁剪 Fake 概率

2.  若纹理分数 {t_j} 可用:
        w_j ← t_j^γ_t / Σ_k t_k^γ_t                  // 幂律归一化注意力权重
        ŷ ← β·s_0 + (1−β)·Σ_{j=1}^{N-1} w_j·s_j     // 加权集成
    否则:
        j* ← argmax_j |s_j − 0.5|                     // 最大置信度选择
        ŷ ← s_{j*}

3.  返回 ŷ
```

### 8.3 自适应阈值（OWTTT）

```
算法：ComputeAdaptiveThreshold（自适应阈值）
────────────────────────────────────────────
输入:  prediction_queue（近期预测的滑动窗口）

若 |queue| < 32: 返回 0.5

对 th ∈ [0.00, 0.01, ..., 0.99]:
    mask ← (queue ≥ th)
    w1 ← |mask| / |queue|,   w0 ← 1 − w1
    若 w1=0 或 w0=0: 跳过
    v0 ← Var(queue[~mask])
    v1 ← Var(queue[mask])
    min_gap ← min(|queue − th|)
    crit ← w0·v0 + w1·v1 − gap_weight · min_gap
    保留 crit 最小的 th

返回 best_th
// 用于测试阶段计算 acc_adaptive
```

---

## 9. 超参数汇总

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| Mixup α | α | 1.0 | Beta 分布形状参数 |
| Mixup γ | γ | 5.0 | 不对称指数，γ>1 使标签偏向 Fake |
| Mixup K | K | 1 | 每个真实锚点的 Fake 候选数 |
| 选择策略 | S | hardest | {hardest, random, mean} |
| 混合域 | D | rgb | {rgb, hf, lf, ycbcr_hf, ycbcr_lf} |
| FFT 截止频率 | τ | 0.125 | 频率截止比例（Nyquist 半径的分数） |
| 金字塔层数 | K_pyr | 3 | 拉普拉斯金字塔深度 |
| 间隔损失模式 | — | off | {off, add, replace} |
| 间隔 m | m | 0.5 | Fake 特征距中心的最小距离 |
| 间隔损失权重 | w | 1.0 | 模式为 add 时的系数 |
| LoRA 秩（注意力） | r_attn | 4 | ViT 注意力投影层的 LoRA 秩 |
| LoRA 秩（分类头） | r_head | 2 | 分类头的 LoRA 秩 |
| LoRA α（注意力） | α_lora | 16 | 注意力层 LoRA 缩放因子 |
| LoRA α（分类头） | α_lora | 8 | 分类头 LoRA 缩放因子 |
| 纹理 Gamma | γ_t | 1.5 | 纹理注意力权重的幂指数 |
| 融合权重 | β | 0.5 | 完整图像在 TAA 融合中的权重 |
| 均衡批次 | — | false | 是否启用 BalanceBatchSampler |
| 每类批样本数 | — | 自动 | 每类每批样本数（默认从 train_batchSize 推导） |

---

## 10. 可复现性检查清单

1. CLIP ViT-L/14 骨干网络加载预训练权重，**所有非 LoRA 参数冻结**。
2. LoRA 适配器注入到**全部** ViT 注意力块的 q_proj, k_proj, v_proj, out_proj。
3. Mixup 在每批次、前向传播前应用，**仅训练阶段**。
4. Q1 锚定规则：所有跨类配对以真实图像为结构锚点。
5. FFT 分解使用 `fftshift` 确保频率掩码居中。
6. FFT 重建图像裁剪至源值范围以抑制振铃伪影。
7. 拉普拉斯金字塔使用 5 抽头二项式核 `[1,4,6,4,1]/16`（Burt & Adelson, 1983）。
8. 软标签交叉熵使用 `log_softmax`（非 sigmoid）保证数值稳定性。
9. 中心损失中，特征与中心在距离计算前进行 L2 归一化。
10. 验证/测试阶段不应用任何 Mixup。
11. BalanceBatchSampler：`batch_size_per_class` 默认从 `train_batchSize` 自动推导。
12. 推理阶段：裁剪级预测通过 TAA 或最大置信度融合。
