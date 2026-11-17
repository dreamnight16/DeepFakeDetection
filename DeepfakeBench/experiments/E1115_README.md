# E1115：采样、真实约束度量、少量真实适配与强基线对照

本实验按完整矩阵运行，不设置 E1111 的 S0/D1/S1 开发集性能通过门槛。预先固定全部配置，完整输出成功、失败和数据不足的项目，不按测试结果挑选“最佳配置”后覆盖主方法。数据可读性、矩阵定义及分割隔离仍是执行所需条件。

入口为 `experiments/run_e1115.py`。默认完成 7 个训练配置，并对每个成功 checkpoint 执行统一八库的度量、适配和基线实验。所有训练配置独立从原 CLIP 预训练初始化。默认 seed=1024；目标子集的三次重复是采样重复，不冒充三个训练种子。

## 运行

在服务器的 DeepfakeBench 目录、原项目训练环境中运行：

```bash
# 打印完整配置矩阵；不训练、不读取图像、不创建实验输出。
python experiments/run_e1115.py --dry_run

# 完整实验：7 组训练，以及每组的完整度量/适配/基线矩阵。
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1115.py \
  --rgb_root /mnt/data2/effort/data \
  --dataset_json_folder "$PWD/preprocessing/dataset_json" \
  --clip_pretrained_path "$PWD/training/models--openai--clip-vit-large-patch14" \
  > nohup_E1115.log 2>&1 &
```

采用原训练入口 `training/train.py`、Trainer、Adam、预处理、数据增强和 checkpoint 选择接口。默认 FF++ c23，224 输入，batch 32，学习率 2e-4，权重衰减 5e-4，epoch 0..10 共 11 轮，原 rank-4 LoRA 和完整分类头。无 Mixup。训练验证仍按原 Celeb-DF-v2 frame AUC 选 checkpoint；它是训练接口的一部分，不是启动完整矩阵前的开发实验。

默认路径沿用 E1111 的 `/mnt/data2` 迁移规则。使用 `CUDA_VISIBLE_DEVICES` 选择物理 GPU；完整训练的进程内设备固定 `cuda:0`。配置一次运行只使用一张 GPU，训练及后验矩阵顺序执行。可显式设置 `--seed` 运行其他种子；每次生成独立目录，不覆盖历史实验。

分开运行仍然执行相应组的完整注册矩阵：

```bash
# 全部采样训练配置及原分类头完整八库评测。
CUDA_VISIBLE_DEVICES=0 python experiments/run_e1115.py --groups sampling

# 在已有成功 B0 上运行完整后验对照；不再训练骨干。
CUDA_VISIBLE_DEVICES=0 python experiments/run_e1115.py \
  --groups geometry adaptation baseline \
  --base_checkpoint /path/to/B0/ckpt_best.pth \
  --base_config /path/to/B0/train_config.json

# 也支持 E1111B/B0 或既有 E0924 的 base_run 约定。
CUDA_VISIBLE_DEVICES=0 python experiments/run_e1115.py \
  --groups geometry adaptation baseline --base_run /path/to/completed/B0/run
```

`--train_arms` 可显式选择训练行，默认包含全部七行。`--budgets 10 30 100` 指目标真实拟合视频数，另需 `--calibration_videos 30` 个真实视频。`--adaptation_repeats 3` 控制目标划分重复次数。任何非默认值写入结果协议。

## 训练矩阵与变量控制

| 训练行 | 批次真假比例 | 分类损失 | 训练取帧 | 步数 |
|---|---|---|---|---|
| B0_NATIVE | 原配置 10 真 / 22 假 | 整批 CE | 原索引前 8 张 | 原采样器 |
| B0_CONTROL | 10 / 22 | 整批 CE | 同原取帧 | 固定预算采样器 |
| CLASS_BALANCED | 10 / 22 | 0.5 × (真类均值 CE + 假类均值 CE) | 同原取帧 | 同预算 |
| REAL50 | 16 / 16 | 整批 CE | 同原取帧 | 同预算 |
| REAL70 | 22 / 10 | 整批 CE | 同原取帧 | 同预算 |
| FRAME_COVERAGE | 10 / 22 | 整批 CE | 每轮分层重取 8 帧 | 同预算 |
| FULL_REAL70_COVERAGE | 22 / 10 | 整批 CE | 每轮分层重取 8 帧 | 同预算 |

主训练配置预先指定为 FULL_REAL70_COVERAGE；并不声明其效果优于其他行。B0_CONTROL 单独控制更换采样器造成的随机顺序差异。

每轮步数根据原始 0.3 比例采样器在实际训练清单上的长度确定，然后对所有固定预算配置使用同一个长度。每类无放回打乱，耗尽后重新打乱循环，避免提高真实比例后训练步数随之减少。包含 B0_NATIVE 时，显式 `--steps_per_epoch` 必须等于原长度。

FRAME_COVERAGE 按数字帧号排序，将可用帧等分为最多 8 个位置区间，每区间取一帧；选择由种子、视频 ID、epoch 决定。每视频样本数保持不变。刷新发生在创建当轮 DataLoader 迭代器之前，workers 不持久化，因此子进程读取的是新清单。批次顺序、帧选择与全局增强 RNG 分离。固定预算行在 `logs/e1115_sampling.jsonl` 保存每轮步数、真假抽取次数和帧清单摘要。

历史采样和取帧逻辑只在 E1115 专用开关打开时改变。图像读取失败时，E1115 报错，不随机替换样本。分类均衡损失仅用于训练；验证保留普通 CE。训练/验证损失的数值定义分别记录，不能直接当成同一种目标比较。

## 数学定义

所有统计计算使用 float64，缓存的图像特征使用 float32。图像特征为冻结当前 checkpoint 的最后一层 256 个 patch token 的均值，维度 d=1024。原分类头继续使用 pooler_output。两种读出分别保留，不把其差异归因成单一评分公式的效果。

**真实统计。** 对每个真实视频先取帧均值，再对视频等权求中心 μ。总协方差为

`S = mean_video(mean_frame((z - μ)(z - μ)^T))`，

`Σ = (1 - τ) S + τ trace(S)/d I`，固定 τ=0.1。它包含视频内和视频间的变化；同视频的多帧不会作为独立视频重复加权。

**伪造方向。** 设 B 为配对差的二阶矩，Dc 为真实帧 JPEG 再压缩差的二阶矩。记 `N(X)=X/(trace(X)/d)`，零压缩矩阵的 N 定义为零。新方法求解

`B u = ν D u`，其中 `D = N(Dc) + η N(Σ) + 1e-6 I`。

这是在固定 `u^T D u=1` 的约束下最大化 `u^T B u` 的广义 Rayleigh 商。默认 η=1、r=2。真实变化直接进入方向选择的分母，而不只作为可相消的坐标白化。Dc 由同一真实帧的原图与 JPEG quality=15/30/40/50 特征差估计。

保留前 r 个正特征方向 U；正谱不足时该配置明确为 UNAVAILABLE，不补随机方向。默认谱权重 `a_j=sqrt(ν_j)/mean_k(sqrt(ν_k))`，保留相对大小并固定均值为 1；等权与 99% 硬截尾另行消融，避免 E1111 中前几个权重全被截成相同数值却仍被解释为不同方向权重。

**完整新评分。** 令 `C=U^T Σ U`、`q=C^(-1/2) U^T(z-μ)`，

`s_F = sum_j(a_j q_j^2)/r`，`s_R = (z-μ)^T Σ^(-1)(z-μ)/d`。

默认 FULL 只使用 s_F；JOINT 对照使用 `s_R + 2 s_F`。投影白化和均值为 1 的谱权重使 s_F 在所拟合真实协方差下的期望尺度约为 1，与 s_R 相当；不能将新方向项继续除以 d，否则 r=2 时会人为缩小约 512 倍。NO_PROJECT_NORMALIZATION 单独移除 C 的归一化。

VM_LEGACY 保留 E1111 视频均值二阶矩、`Dc+epsilon I` 分母、原 99% 截尾权重、两项均除 d 及 γ=2 的定义。其默认参数固定为前轮来源选定参数，不重新进行 S0/D1/S1 选参；与既有 `e1111.metric.score` 的数值等价性有直接测试。

## 视频信息消融

独立取帧部分使用原 FF++ 内容参考谱系，最多 32 个候选帧。每个来源、每种方法等权。

- VIDEO_MEAN：两段视频各自平均后相减，统计 `mean(δ)mean(δ)^T`。
- FULL / BLOCK2 / BLOCK8：分别将真假各自按时间顺序划成 4/2/8 个等数量片段，片段特征均值相减，再统计片段差的二阶矩。帧不足时使用可形成的片段数并报告实际数量；这属于弱时间对应，不称严格同帧。
- MEAN_ONLY / VARIATION_ONLY：使用同一 4 段差分的均值项和中心化变化项，严格满足 `B_block4 = B_mean4 + B_variation4`。全视频均值与等权分段均值在片段长度不等时可不同，因此分别保留。
- MATCHED_MEAN / MATCHED_BLOCK / MATCHED_FRAME / MATCHED_VARIATION：先在原始索引中求真实及四种伪造的共同帧号，再共同选择最多 32 帧。这四行使用相同来源和帧集合，逐来源/方法等权，满足 `B_frame = B_matched_mean + B_matched_variation`。

MATCHED 使用的是同原始保存帧号、同官方谱系的既有人脸裁剪图片；没有原视频共享重裁剪，不宣称已消除裁剪/身份混杂。无足够共同帧的来源不混入 MATCHED 主矩阵，完整清单和覆盖率写入 source_statistics；这四行与独立分段主方法之间的样本范围差别明确记录。

完整几何矩阵共 22 行：FULL、VM_LEGACY、NO_REAL_CONSTRAINT、NO_COMPRESSION、NO_PROJECT_NORMALIZATION、EQUAL_SPECTRUM、CAPPED_SPECTRUM、REAL_ONLY、JOINT、MEAN_ONLY、VARIATION_ONLY、VIDEO_MEAN、BLOCK2、BLOCK8、RANK4、RANK8、ETA0.1、ETA10 和四个 MATCHED 对照。单因素行保持其他 FULL 参数固定。

## 少量可信真实适配

默认每个目标库使用拟合预算 n=10/30/100，另取 30 个真实视频校准阈值，重复三次确定性划分。实际可信真实数据预算为 40/60/130 个视频，不能只报告 n。

每次重复先预留该数据集可实现的最大拟合预算池；较小 n 从池内嵌套取样。所有 n 使用同一校准集与同一测试集合，较小预算未使用的预留视频也不加入测试。每库至少保留 20 个测试真实视频；无法满足的预算明确记录 unavailable，绝不降低预算或将拟合/校准视频放回测试凑数。默认按完整视频 ID 隔离，未声明跨身份或原始内容家族完全隔离。

固定来源 U、谱权重及网络，在 FULL 和 VM_LEGACY 上分别比较：

| 模式 | 中心 | 协方差 | 阈值 |
|---|---|---|---|
| threshold | 来源 μ | 来源 Σ | 目标校准真实视频 |
| mean | 目标拟合 μT | 来源 Σ | 目标校准真实视频 |
| cov | 来源 μ | 0.5 Σ + 0.5 ΣT | 目标校准真实视频 |
| mean_cov | 目标拟合 μT | 0.5 Σ + 0.5 ΣT | 目标校准真实视频 |

ΣT 仍按目标真实视频等权、τ=.1 收缩估计，方向 U 不重新拟合。FULL 在适配后更新投影坐标的 C；VM_LEGACY 更新其真实距离项。这是明确的受监督“可信目标真实样本”适配，不称为无标签适配。

每个目标测试子集同时评估原分类头的来源阈值、目标阈值，以及来源 FULL。可用原 B0 锚点时也重评该锚点，所有比较使用同一测试子集。目标伪造标签仅用于最终指标，未用于拟合、选 γ、选权重或定阈值。

## 利用强基线

每个 checkpoint 的原始分类头预测单独保存为 B0；不更改其权重。FULL 能量以来源拟合真实帧的经验 CDF 映射到有界分数，固定融合为

`p_mix = (1 - w) p_B0 + w F_source_real(s_FULL)`，`w ∈ {0, 0.05, 0.1, 0.25}`。

w=0 返回原概率的精确副本，不经额外数值变换；有逐行相等测试。非零 w 是否有效由结果判断，不作跨域无损保证。全部权重一起报告，不在测试集挑一个后宣称保证不损伤基线。

默认先导出 B0_NATIVE。后续训练行还额外输出 ANCHOR_B0 及 ANCHOR_PLUS_FULL_*：将原始强基线的冻结概率与当前候选的 FULL 分数融合，而不是只与候选自身分类头融合。两边校准/测试的标签、路径和视频 ID 必须逐行相等。若显式运行子矩阵时省略 B0_NATIVE，则仅有该 checkpoint 自身 B0 对照；不会自动寻找一个未经核实的历史锚点。

固定阈值比较报告 corrected_videos 和 harmed_videos。它们是相对于该比较传入的 B0 阈值预测的变化计数；实际 FPR/TPR 同时报告，不能用纠正数单独证明整体提升。

## 评测与输出

统一复用原 `get_data_loader(test)`、`strict_loader`、`readout_metrics`，测试固定原生最多 8 帧，保留原索引取帧协议。一个骨干前向同时导出原分类概率和 patch 特征。逐帧原始分数先算 AUC，再按完整路径视频 ID 取帧均值算视频 AUC；旧 basename 指标单列，六库均值不使用它。

测试库为 WDF、FFIW、CDF-v2、DFD、DFDC、DFDCP、DeeperForensics 和 FF++。七库均值排除 FF++；独立六库再排除参与 checkpoint 选择的 CDF-v2。目标适配的 held-out 子集结果与完整八库结果通过 scope 区分，均值缺少所需库时为空，不以部分成功库冒充完整均值。

来源阈值使用 FF++ train 内独立于度量拟合的 theta 来源、原生 8 帧评分。因为骨干训练见过这些来源，标作经验校准，不宣称整个流程的独立 split-conformal 保证。目标阈值用独立目标真实校准视频，分位下标 `ceil((n+1)(1-alpha))`，α=.10/.05；样本不足以支持该阶统计量时明确标记。目标覆盖性质仍依赖校准与测试真实视频的代表性/可交换性。报告实际视频 FPR、TPR、Clopper–Pearson 95% 区间。energy>.5 的帧 accuracy 仅保留作接口参考，不与分类概率阈值准确率混作主要结论。

输出位于 `experiment_results/E1115/seed.../`：

- `all_results.json`：全流程状态、参数、源码摘要和各训练/后验输出路径。
- `summary.csv` / `metrics.csv`：汇总所有骨干的配置均值、逐库指标。
- `training/<arm>/`：训练/评测配置、原训练日志和 checkpoint。
- `source/e1115_source_manifest.json`：来源分割、独立候选与共同帧清单。
- `posthoc/<arm>/native8/`：原生预测、浮点特征 memmap、行标识及摘要。
- `posthoc/<arm>/source_features/`：来源特征、压缩矩阵和校准预测。
- `posthoc/<arm>/comparisons/`：全部模型、逐帧分数、目标分割、适配分数、JSON/CSV。

完整流程需要服务器上的原预处理图片、CLIP 权重及项目训练依赖。原始图片和 checkpoint 不上传外部服务。本地单元/集成测试使用合成特征，验证公式与编排，不代表完成了真实数据训练或测得了效果。

## 代码定位与验证

`training/e1115/sampling.py` 管固定步数和每轮取帧；`assets.py` 管索引审计与特征导出；`statistics.py` 管视频/片段/逐帧二阶矩；`math.py` 管方向求解、评分和适配；`evaluation.py` 复用 AUC 并补阈值指标；`suite.py` 执行完整矩阵。对历史代码仅增加专用开关：训练入口、训练取帧/严格读图、类别均值损失。

```bash
python -m pytest -q tests/test_e1115_core.py tests/test_e1115_protocol.py tests/test_e1115_integration.py
python -m ruff check training/e1115 experiments/run_e1115.py tests/test_e1115*.py --ignore E402
```

2026-11-17 本地验证：E1115 新测试覆盖数学分解、方向约束、固定步数采样、动态取帧、目标三方隔离、完整八库合成数据矩阵、原分类损失梯度、实际 DataLoader 接入、旧评分公式等价、原 B0 零融合精确保留、冻结训练配置的跨库标签及根目录汇总。另运行既有 E1111 视频均值和统一评测回归。真实数据训练尚需在数据与依赖齐全的服务器上执行。

数学依据与相关工作：广义 Rayleigh 商对应对称正定广义特征问题；协方差采用显式固定强度收缩，[协方差收缩说明](https://scikit-learn.org/stable/modules/covariance.html)。真实概念与伪造信号分别建模可参考 [RealID](https://www.ijcai.org/proceedings/2025/177)；冻结分类输出与特征分数组合可参考 [ViM](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html)，本实现不是这些论文的完整复现。分布漂移下校准的条件可参考 [Conformal Prediction Under Covariate Shift](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)；本实验未套用未经估计的密度比，也不声称对任意漂移有保证。
