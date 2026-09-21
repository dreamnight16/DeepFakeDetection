# G25：在 CLIP ViT 高层插入证据 token

## 实验约定

目标：保留 RGB + CLIP ViT-L/14 + LoRA + pooler CLS 分类路径，通过少量高层 token 收集伪造证据，检验是否提供跨域补充信息。既有结果尚未证明 LFEQ 优于 B0；G25 是待验证假设，不是性能结论。

默认在 **Block 20 输入**处追加 K=4 个可学习 token（block 从 0 编号，共 24 层；新增 token 经过 20–23 四层）。原 CLS embedding 保留预训练值并解冻。新增 token 不改原位置编码，不重复插入，不添加 Transformer、投影层或新的 LayerNorm；它们复用原高层注意力、FFN，以及冻结的 `post_layernorm`。CLS 使用原独立线性头，每个新增位置另有一个独立 `Linear(1024,2)`；普通 patch token 保持特征用途。

数据、增强、LoRA、优化器和评估沿用当前 B0 协议。FF++ 训练，Celeb-DF-v2 **帧级 auc** 选点；最终报告七个跨域集及 FF++ 的 `video_auc`。`AUC_cross` 专指 Celeb-DF-v2 与 DFDC 均值，七集均值另列。固定 seed=1024、v1 sampler real ratio=0.30，关闭 mixup、频率替换和额外 margin/rank loss。`nEpochs=10` 沿用既有循环，实际为 epoch 0–10，不能写成恰好 10 次 epoch。

## 独立对照

| 组别 | CLS 看新增 token | patch 看新增 token | 证据监督 |
|---|---|---|---|
| B0 | 无新增 token，CLS 冻结 | 同原模型 | 仅 CLS CE |
| C0 | 无新增 token，CLS 解冻 | 同原模型 | 仅 CLS CE |
| M00 / A00 | 否 | 否 | max / all |
| M10 / A10 | 是 | 否 | max / all |
| M01 / A01 | 否 | 是 | max / all |
| M11 / A11 | 是 | 是 | max / all |

新增 token 在所有组都可看 CLS、patch 和新增 token。mask 的行是 query，列是 key；仅改变原 token 到新增 token 的边。M/A 内部四组只改变 mask，两种监督之间只改变证据 CE 的归约方式。

**屏蔽直接注意力不等于隔离间接路径。** 例如 10 组中，新增 token 可先影响 CLS，再由 CLS 在下一层影响 patch。只有 00 组能在相同权重、无随机 dropout 的前向中保持所有原 token 输出；训练时共享 LoRA 和解冻 CLS 仍会受证据损失影响，因此不保证训练后等同 B0。

## 损失和分数

- max：`L = CE(CLS,y) + lambda_e * CE(E[argmax P_fake],y) + lambda_d * diversity`。
- all：证据项改成 `mean_k CE(E[k],y)`，不随 K 放大；其余相同。
- `lambda_e=1`，`lambda_d=0.01`。diversity 沿用 LFEQ 的证据注意力图两两 cosine 相似度，取最后一个 block、跨 attention head 平均后的“证据 query → patch key”切片，排除 CLS 和证据 key。
- 两组均用 `0.5 * P_fake(CLS) + 0.5 * max_k P_fake(E[k])` 评分；选择 token 不使用真值标签。all 不同时改成平均评分，以保持单因素比较。
- max 每个位置虽有独立头，但当次未被选中的头没有直接证据 CE 梯度；all 每个头都接受 CE。两者都保留同样的注意力多样性约束。
- 训练指标与最终评分一致；启用 5D 多 crop 时使用原 argmax-confidence TAA，当前配置默认 `multi_crop: false`。

相比 B0，K=4 增加 4,096 个 token 参数和 8,200 个分类头参数；另解冻已有 1,024 个 CLS 参数，共增加 **13,320 个可训练参数**。不计原有 LoRA 和 CLS 头。注意序列变长仍增加高层计算量；参数少不等于零计算开销。

## 验证和结果边界

2026-09-21 本地验证：G25 的 33 项 CPU 测试通过，G22/G23 的 8 项既有回归测试通过。环境为 Python 3.12、torch 2.5.1+cpu、transformers 4.44.2、loralib 0.1.2。

覆盖 mask 方向和实际 attention weights、00 前向不变性、原 CLS/新增 token/早晚层 LoRA 梯度、独立头、max/all loss、5D TAA、配置透传、严格 state_dict 重载及失败产物。模型测试使用随机初始化的小型真实 CLIP；检测器外壳测试替换了预训练加载和数据/指标依赖。**未验证真实预训练 ViT-L/14、完整数据加载或服务器训练，尚无 G25 的 AUC 结果。**

## 运行

命令均在 `DeepfakeBench` 目录运行，使用现有可运行 G18/G19 的服务器环境。

```bash
# 先查看十组配置，不加载 torch、不训练
python experiments/run_g25.py --dry_run

# CPU 小型 CLIP，无需下载预训练权重或数据
python -m pytest tests/test_g25.py tests/test_g25_runner.py -q

# 十组顺序独立训练和评估
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_g25.py --n_epochs 10 --seed 1024 > nohup_G25.log 2>&1 &

# 或按损失拆到两块卡，目录自动隔离
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_g25.py --arms B0 C0 M00 M10 M01 M11 > nohup_G25_max.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python experiments/run_g25.py --arms A00 A10 A01 A11 > nohup_G25_all.log 2>&1 &
```

每次运行创建时间戳目录，每组保存完整 train/eval 配置和结果；testall 概率与图写到各组自己的 `testall_artifacts`，并透传当前 seed。失败写入状态并使入口返回非零。`--insert_layer 20 --num_tokens 4` 可调整位置/数量，首轮保持默认。`--clip_pretrained_path` 可显式指定服务器 CLIP 目录。不要将同一 checkpoint 的重复评估算成独立种子。

CLIP 层接口参考 [Hugging Face CLIP 官方实现](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/clip/modeling_clip.py)。服务器先运行上面的 CPU 测试，确认实际 transformers 版本兼容；G25 需要返回 attention weights，使用 eager attention，兼容旧版 CLIPSdpaAttention 在末层返回权重时回退 eager 的路径，不支持 FlashAttention。
