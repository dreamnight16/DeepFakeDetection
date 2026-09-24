# G27：隔离式专家 token 候选方案

目标：在 G26 的最小结构上比较三种专家学习机制，先完成代码与本地合约验证，待算力可用后批量训练。当前没有 G27 真实训练结果，也不假定专家已经形成语义分工。

## 1. 实验与固定协议

默认依次训练六组；各组独立初始化，不接续上一组权重。

| CLI 组名 | 存在式证据监督 | 假图负载均衡 | CLS 难例加权 | 双视图一致性 |
| --- | --- | --- | --- | --- |
| B0 | 无证据分支 | — | — | — |
| G26 | 有 | 0 | 关闭 | 0 |
| E1 | 有 | 0.1 | 关闭 | 0 |
| E2 | 有 | 0 | 开启 | 0 |
| E3 | 有 | 0 | 关闭 | 0.1 |
| Full | 有 | 0.1 | 开启 | 0.1 |

G26 对照使用 `effort_g27` 将新增目标全部关闭，K=4、Block 20；本地测试验证其前向、总损失与同权重 G26 一致。它不是旧 G26 自定义入口默认的 K8L18。E1/E2/E3 各相对本轮 G26 只开启一个机制，Full 用于检查组合收益，不能分解全部高阶交互。

固定架构：RGB → CLIP ViT-L/14 + LoRA；原 CLS embedding 冻结；4 个新增 token 在 Block 20 输入处插入，经过原最后 4 层；每个新增 token 各有一个 `Linear(1024,2)` 头。新增 12,296 个参数，与 G26-K4L20 相同。专家可以读取 CLS、所有 patch 和其他专家；原 CLS/patch 不读取专家，固定 `read_only/00`。不加入额外 Transformer、区域规则、token 互斥或 attention diversity。

原 LoRA 和主分类头仅接收原 CLS CE。专家损失使用 detach 的原特征和原后缀参数，梯度只更新新增 token 和专家头。主分支从头遵循 B0 训练，不是先训练 B0 再冻结 checkpoint。原 CLS 解冻与 attention-mask 扫描不在本次默认矩阵中，它们已由 G25 单独研究。

协议沿用 G25/G26：FF++ c23 训练，Celeb-DF-v2 主评分帧级 `auc` 选点；最终 `testall` 报告 `video_auc`，`AUC_cross` 为 CDF-v2/DFDC 均值，七集均值不含 FF++。seed=1024、v1 sampler 真图占比 .30；关闭 mixup、额外频率输入、全局优化器 wrapper 和 multi-crop。`--n_epochs 10` 保留既有 trainer 的 epoch 0–10 语义，共 11 次 epoch。每组按自己的主评分选点，所选 checkpoint 的 CLS-only 不保证等于 B0 最优 checkpoint。

## 2. 损失的精确定义

令 `z_ik = fake_logit - real_logit`，标签 0 为真、1 为假：

```text
s_i = tau * (logsumexp(z_i / tau) - log(K)),  tau=0.5
L_e(i) = mean_k softplus(z_ik)   if real
         softplus(-s_i)         if fake
q_ik = softmax_k(z_i / T_r),    T_r=1.0
L_b = sum_k (mean_fake(q_ik) - 1/K)^2
u_i = floor + (1-floor) * clamp(1 - abs(detach(p_cls_i)-0.5)/width, 0, 1)
floor=0.2, width=0.2; disabled hard weighting uses u_i=1
L_c(i) = mean_k (sigmoid(z_ik) - sigmoid(z'_ik))^2
L_total = mean_i[CE_cls(i) + lambda_e*u_i*L_e(i) + lambda_c*L_c(i)] + lambda_b*L_b
lambda_e=1.0
```

`T_r` 是诊断/均衡用的路由温度，`tau` 是 MIL 温度，两者独立。均衡只在当前训练 batch 的假图上计算，无假图时返回可微零；不跨 batch 保存队列，也不跨设备汇总。当前 runner 按既有单进程训练使用。若以后改 DDP，负载均衡将变成每个 rank 本地 batch 的统计，需要另行决定全局定义。

真图训练所有专家为真；假图允许一个或多个专家支持假，不要求其余专家为真，也不强制“恰好一部分”判假。归一化平滑最大值是存在式目标的代理，不是逻辑 OR 概率。

负载均衡允许所有专家都输出相同分数，不能保证专门化。`q` 表示相对责任：所有专家都判真时仍可能产生一个最大 q，不能当作检测到伪造的概率。MIL 提供假图分类梯度，均衡只加批次占比约束。

难例权重只乘 `L_e`，不乘均衡或一致性；也不按权重总和重新归一化，因此 E2 同时改变样本侧重与辅助 CE 总量，不能单独归因于难例选择。高置信度错误仍保留 floor 权重；概率距 .5 是未校准的不确定性代理。

第二视图对输入 batch 每张图固定作轻微光度变换：在像素空间 `x'=.9*x+.1*spatial_mean(x)+.02`，在标准化输入上按配置 `std` 等价实现。不裁剪、不翻转、不截断到 [0,1]，不消耗 RNG；这是明确的候选不变性，并非证明所有伪造痕迹都被保留。两个视图的专家分数均接收一致性梯度。第二视图前缀无梯度、后缀参数 detach，保留 token 梯度，不产生第二个 CLS CE。

只有 E3/Full 在训练时增加一次全 batch 辅助前向，不增加额外可训练参数；验证/推理仅一个视图，`loss_consistency=0`，因此验证 total loss 与训练 total loss 不同。选点依据主评分 AUC。当前未实现降频/半 batch 一致性，以免增加消融变量。

## 3. 推理与诊断

推理完全沿用 G26：

```text
p_e = sigmoid(s)
w = alpha * clamp(1 - abs(p_cls-.5)/gate_width, 0, 1)
p_final = (1-w)*p_cls + w*p_e
alpha=.5, gate_width=.2
```

主概率在 (0.3,0.7) 外直接保留；门控参数未拟合，不在测试集上挑选。训练不对融合分数反传。新增 token 不能影响原分支前向；辅助损失不能更新原参数。但辅助融合仍可能把正确结果改错，不保证 AUC 增益。

每组选择同一个 checkpoint 后默认额外保存 CLS-only 与 evidence-only 的八集 testall；不分别挑选 checkpoint。还做一次同输入的帧级诊断，保存：

- `routing_diagnostics.json`：触发率、纠正/改错/净纠正数、分支准确率、真/假各专家分数均值与标准差、CLS 高置信度错误数。
- 假图 soft load、最大值专家占比（并列平分）、并列率、逐图路由熵、总体负载熵、两种熵之差。均匀且完全相同的专家，其熵差为 0。
- 专家概率 Pearson 相关矩阵；恒定列的相关性未定义，写为 JSON null，不当作低相关。
- `routing/<dataset>.npz`：同一前向逐行对应的 labels、CLS/evidence/fused 概率、门控权重、逐专家概率、原始 log odds 和软责任 q。
- `routing/<dataset>_attention.npz`：每集默认前 16 帧的末层专家→patch attention（head 平均）、同输入标准化图片、帧索引及标签。是相同 tensor 的额外确定性前向，保留分配给 CLS/专家的注意力质量，不对 patch 部分重新归一化。深层 patch 已混合上下文，不能把这些图当成伪造定位真值。

这些帧级诊断阈值固定 `p>.5`，不能代替视频 AUC；也不能与另一次带随机数据处理的 testall 输出按行强行配对。attention 只是有界样本检查，不代表整个数据分布。不预先把 token 命名为“眼睛专家”等语义角色。

## 4. 运行

在 `DeepfakeBench` 目录、现有项目服务器 Python 环境执行：

```bash
# 无 torch、无数据、无预训练权重也能检查六组设置
python experiments/run_g27.py --dry_run

# 六组依次训练并评估，输出不会覆盖旧批次
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_g27.py --n_epochs 10 --seed 1024 > nohup_G27.log 2>&1 &

# 只跑部分组；如复用旧对照，须核对协议与预算且不能算新种子
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_g27.py --arms E1 E2 E3 Full --n_epochs 10 --seed 1024 > nohup_G27_candidates.log 2>&1 &

# 节约评估成本：保留分支评分和路由诊断，关闭 attention 快照
python experiments/run_g27.py --arms E1 --attention_samples 0

# 显式跳过全部额外诊断；仅保留主评分评估
python experiments/run_g27.py --arms E1 --skip_diagnostics

# 第二阶段示例：选定机制后再分别改 K 或插入层；先 dry-run
python experiments/run_g27.py --dry_run --arms E1 --num_tokens 8 --insert_layer 20
python experiments/run_g27.py --dry_run --arms E1 --num_tokens 4 --insert_layer 16
python experiments/run_g27.py --dry_run --arms E1 --num_tokens 8 --insert_layer 16
```

`--num_tokens/--insert_layer` 统一作用于本次所有非 B0 组，默认始终 K4L20。E1 为示例，不代表预先认定优胜。先看预先声明的验证/开发协议决定后续实验；DFDC 等最终测试集只做报告，不能反复用于挑损失、结构或融合参数。确认收益还需后续多种子重复。

结果位于 `experiment_results/g27/seed<seed>_<timestamp>_<pid>/`。manifest 记录实际命令参数、六组设置、源码 SHA256、Python/torch/transformers 版本；每组保存 train/eval 配置、checkpoint 路径、result、testall 及额外诊断。某一组失败后继续其他组；训练失败标记 TRAIN_FAILED，主评分/所请求诊断失败标记 EVAL_FAILED，同时保留 primary_status。任何失败使批量入口最终返回非零。

## 5. 本地验证与边界

```bash
python -m pytest tests/test_g27.py tests/test_g27_runner.py -q
python -m pytest tests/test_g27.py tests/test_g27_runner.py tests/test_g26.py tests/test_g26_runner.py tests/test_g25.py tests/test_g25_runner.py tests/test_g25v2.py tests/test_g25v2_runner.py tests/test_g22_g23.py -q
```

本地测试使用 CPU 随机初始化的小型真实 CLIP + LoRA，替换预训练加载、数据及指标外壳。覆盖均衡边界、detach 权重、光度变换、所有新损失共同反传后的原分支梯度与两步 Adam 等价（loralib / 项目自定义 Linear）、G26 关闭新增目标的数值等价、checkpoint 重载、multi-crop、eval 不产生第二视图、六组单变量配置、testall 参数透传、失败状态及同输入诊断。

2026-09-24 验证：Python 3.12.14、torch 2.5.1+cpu、transformers 4.44.2；G27 专项 **45 项通过**，G25/G25v2/G26/G22/G23 既有回归 **128 项通过**。独立 MLE 审查复跑 G27 45 项通过，未发现阻断缺陷。还覆盖 mocked 批量 main 的单组失败后继续与 manifest 源码哈希记录；未运行真实批量训练。未测覆盖率，不将用例通过数写成覆盖率。

原分支保护的条件沿用 G26：相同原参数、输入与优化器状态，backbone/adapter dropout=0，无跨参数全局梯度变换；允许浮点误差。CPU 合约测试不证明服务器训练逐位一致。实际 ViT-L/14 的额外显存/耗时、增强是否保留判别信息、专家是否分工及泛化收益，都要服务器实测。本次不启动真实数据训练或推理。
