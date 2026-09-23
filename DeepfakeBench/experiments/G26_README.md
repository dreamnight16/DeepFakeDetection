# G26：自主取证 token、非对称监督与不确定时辅助判断

## 方案与默认设置

保留 RGB → CLIP ViT-L/14 + LoRA → 原 `pooler_output` 分类头。新增证据 token，复用原层的注意力、FFN 与 post-LN，不额外增加 Transformer。

为与 G25 的 4 token / Block 20 对照，默认运行 **K={4,8} × 插入层={16,20} 的 2×2 矩阵，加一组 B0**。层编号从 0 开始，指该 block 的输入处。

| 组别 | 证据 token 数 | 插入层 | 经过后缀 | 用途 |
| --- | --- | --- | --- | --- |
| B0 | 0 | — | — | 原模型基线 |
| K4L20 | 4 | Block 20 | 最后 4 层 | 与 G25 对齐 token 数和插入层 |
| K8L20 | 8 | Block 20 | 最后 4 层 | 相对 K4L20，只增加 token 数 |
| K4L16 | 4 | Block 16 | 最后 8 层 | 相对 K4L20，只提前插入位置 |
| K8L16 | 8 | Block 16 | 最后 8 层 | 检验更多 token 与更长后缀的组合 |

四组的监督、梯度隔离、门控、温度、损失权重、seed、数据与训练预算完全相同。比较 K8L20−K4L20、K8L16−K4L16 可看数量影响；K4L16−K4L20、K8L16−K8L20 可看插入位置影响。交互差分为 `(K8L16−K4L16)−(K8L20−K4L20)`。增加 K 也增加参数量，提前插入也增加计算，不能将收益直接解释为“注意力分工更好”。

K4L20 优先与历史 G25 的 M00/A00 比较：结构尺寸和单向读取 mask 对齐，仍同时改变原 CLS 冻结、辅助梯度、损失、diversity 与评分。因此它是整套方法对照，不能据此单独证明门控或某种损失有效。固定相同 readout 的分支诊断可以帮助分析，但不能消除全部训练差异。单 seed 只能作初步比较。

之前的 8/18 不纳入默认矩阵，避免再扩大实验；为保持已有命令可复现，`--arms G26` 仍是可配置单组入口，未指定形状时沿用 8/18。命名矩阵组的形状固定，不允许用全局 `--num_tokens/--insert_layer` 暗中覆盖。

每个证据 token 可以自由读取原 CLS、所有 patch 和其他证据 token，保留原 patch 的位置编码；没有规则网格划分、局部窗口、全图覆盖、互斥或注意力多样性损失。不同查询可以关注不连续区域，也可以重叠。每个证据 token 有独立 `Linear(1024,2)` 分类头。K=4 新增 **12,296** 个参数，K=8 新增 **24,592** 个参数。

证据查询并非有像素标注的区域检测器；深层 patch 已混入上下文。没有约束能够自动保证查询分工、伪造定位或跨域互补性。

## 主分支保护

| 部分 | 前向读取 / 参数更新 |
| --- | --- |
| 原 CLS、patch | 不读取新增 token（固定 `read_only/00` mask） |
| 原 CLIP CLS embedding | 保持 B0 的冻结状态 |
| 原 ViT 冻结参数 | 保持冻结 |
| 原 LoRA、原分类头 | 只接受原 CLS CE |
| 新增 token、独立证据头 | 只接受辅助证据损失 |

训练时沿用 G25v2 的隔离函数：插入点特征 detach，以 detach 的原层参数重算插入点后的层，保留对新增 token 输入的梯度。仅 detach 特征不足以隔离共享 LoRA；也不能用 `no_grad` 包住辅助路径，否则 token 无法学习。推理无梯度时仅一次带 mask 的前向。原层 dropout 必须为零；不支持 FlashAttention。

新增部分初始化使用独立 RNG 上下文，恢复原 CPU RNG 状态，避免新增初始化改变后续 sampler/数据增强的随机序列。训练不对融合分数施加损失，也不按门控筛选辅助训练样本。

在相同原参数、输入、零 dropout、相同优化器状态且无跨参数全局梯度变换的条件下，原分支前向与更新应与 B0 等价（允许浮点误差）。这不等于保证整个服务器训练逐位一致。两组按各自主评分选检查点，选中步数可能不同，因此所选检查点的 CLS-only 也未必等于 B0 最佳检查点。

## 真图全部真、假图至少一个假

令第 k 个证据头的两类 logits 之差为 `z_k = fake_logit - real_logit`，使用归一化平滑最大值：

```text
s = tau * (logsumexp(z / tau) - log(K))     tau = 0.5
p_evidence = sigmoid(s)
L_real = mean_k softplus(z_k)
L_fake = softplus(-s)
L = mean_batch [ CE(original_CLS_logits, y) + lambda * L_evidence(y) ]
lambda = 1.0
```

真图要求所有证据查询判真；假图允许一个强查询支持判假，不要求其余查询都假。real 项对 K 取平均，batch 按原 sampler 组成取平均，不额外进行类别平衡。

不使用乘积式存在概率，也不直接取 hard max 进行监督。所有 `z=0` 时 `s=0`，不会因 K 增加抬高初始伪造概率；fake 分支在该点的梯度绝对值总和为 0.5，与 K 无关。`max(z)-tau*log(K) <= s <= max(z)`，K 与温度仍会影响单个强查询所需的幅度。

这是存在式语义的平滑代理，不是独立事件的逻辑 OR 概率。相比 hard max，有限 logits 下各查询可以获得梯度，但悬殊分数仍会造成梯度集中，不能宣称已经解决 G12 的全部训练退化。注意力也可能重复，需结合保存的逐查询分数和后续注意力分析判断。

## 主分支不确定时才融合

```text
p = original_CLS fake probability
w = alpha * clamp(1 - abs(p - 0.5) / width, 0, 1)
p_final = (1 - w) * p + w * p_evidence
width = 0.2, alpha = 0.5
```

默认主概率在 `(0.3,0.7)` 内才启用辅助；在 0.5 处辅助权重最大为 0.5，区间外严格返回原概率。门控无可训练参数，既不接收标签，也不自动选择测试集上的最优设置。

**默认门控未经校准**：它只使用概率距 0.5 的距离作为不确定性的代理。参数可在独立验证协议下事先确定，不得依据 DFDC 等最终测试集调参。原模型高置信度错误不会触发；保留原输出也不能保证融合提升。

默认关闭 multi-crop。模型支持 5D 推理时，三个 readout 统一使用原 CLS 最有把握的 crop，避免辅助分数改变原 crop 选择；这与 G25 各 readout 独立选 crop 有区别。

## 运行与产物

默认五次独立训练：`B0`、`K4L20`、`K8L20`、`K4L16`、`K8L16`。不改变 G25/G25v2 的代码、运行入口及产物。

在 `DeepfakeBench` 目录执行：

```bash
# 不导入 torch、不加载数据
python experiments/run_g26.py --dry_run

# CPU 随机小模型合约验证，无需预训练权重
python -m pytest tests/test_g26.py tests/test_g26_runner.py -q

# 服务器正式训练：B0 + 四组 G26
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_g26.py --n_epochs 10 --seed 1024 > nohup_G26.log 2>&1 &

# 已有同期基线时，只运行四组 G26
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_g26.py --arms K4L20 K8L20 K4L16 K8L16 --n_epochs 10 --seed 1024 > nohup_G26_matrix.log 2>&1 &

# 预算受限，只比较两端；不能拆开数量与层位置的影响
python experiments/run_g26.py --dry_run --arms K4L20 K8L16

# 自定义单组，必须显式使用 G26；不覆盖命名矩阵的固定形状
python experiments/run_g26.py --dry_run --arms G26 --num_tokens 12 --insert_layer 16
```

默认输出 `experiment_results/g26/seed<seed>_<timestamp>_<pid>/`，每组有独立 train/eval 配置、日志、检查点、结果与 testall 产物；manifest 保存实际参数、源码 SHA256 和运行版本。旧结果不会被覆盖。

数据、增强、sampler、优化器与 G25 的 B0 协议相同：FF++ c23 训练，Celeb-DF-v2 **主评分帧级 auc** 选检查点，最终用原 `testall.py` 报告 `video_auc`。G26 主评分为门控融合。`n_epochs=10` 沿用既有 trainer 的 epoch 0–10，不改训练轮数语义。

默认对 G26 选中的**同一检查点**额外进行：

1. CLS-only 与 evidence-only 的 testall，各自保存独立配置和结果，不重新选检查点。
2. 每个评测集的一次同输入、同前向诊断，输出 `routing_diagnostics.json`：触发率、纠正帧数、改错帧数、净纠正数、触发子集准确率、每个查询按真/假类别的分数均值与标准差。
3. `routing/<dataset>.npz` 保存该次诊断的标签、原分数、证据分数、门控分数、权重与逐查询概率。各数组逐行对应同一前向样本，不能假定它们与另一次随机数据处理后的 testall 分数逐行等价。

这些是**帧级诊断**，阈值固定 0.5，不代替 video_auc。主评估失败或任何所请求诊断失败都标记 `EVAL_FAILED`，`primary_status` 单独保留。`--skip_diagnostics` 显式跳过两种分支 testall 与同前向诊断，可减少评估成本。

评判优先看 `AUC_cross = mean(CDF-v2 video_auc, DFDC video_auc)`、七跨域集均值及 DFDC；同时看辅助触发后纠正是否超过改错。分支结果不计为独立种子，不能按最终测试集挑选输出方式。

## 验证边界

本地使用 CPU 随机初始化的小型真实 CLIP + LoRA，替换预训练加载、数据和指标外壳；覆盖损失梯度、主分支隔离、Adam 更新、门控、配置透传、checkpoint 重载、multi-crop 与失败产物。

2026-09-23 矩阵更新后本地验证：**G26 相关 59 项、G25/G25v2/G22/G23 回归 69 项，共 128 项通过**。Python 3.12.14、torch 2.5.1+cpu、transformers 4.44.2；覆盖四组固定形状和单变量差异、命名组禁止覆盖、自定义入口兼容、命名组诊断与失败状态。梯度隔离与两步 Adam 等价同时覆盖 loralib 和项目自定义 Linear，另覆盖 24 层小宽度 CLIP 的 Block 18 插入、6 层后缀与 SDPA。独立 Python 审查未发现阻断缺陷。未安装 ruff/mypy/black，未将其写成通过。

```bash
python -m pytest tests/test_g26.py tests/test_g26_runner.py tests/test_g25.py tests/test_g25_runner.py tests/test_g25v2.py tests/test_g25v2_runner.py tests/test_g22_g23.py -q
```

尚未运行真实预训练 ViT-L/14、服务器训练或真实数据推理；无 G26 AUC 结果。矩阵组的后 4 层或后 8 层辅助重算的实际显存、耗时与跨域收益需要服务器实测。
