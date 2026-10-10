# E1010：局部伪造监督与辅助 token 的完整对照

在 `DeepfakeBench` 目录、现有服务器环境中运行：

```bash
python experiments/run_e1010.py --dry_run

CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1010.py \
  > nohup_E1010.log 2>&1 &
```

默认先从原 CLIP 预训练权重训练 B0，完成 B0 评估后冻结它，再运行 75 个辅助配置。
需要现有数据和 CLIP 预训练权重，不再要求保留旧 E0924 检查点。
CLIP 位置不同可加 `--clip_pretrained_path /path/to/clip`。

已有 B0 时仍可加 `--base_run /path/to/E0924/run`，或指向本次新 E1010 运行目录；
也可用 `--base_checkpoint /path/to/b0.pth --base_config /path/to/train_config.json`。
显式提供源目录或模型时只复用该 B0，源不存在、配置不匹配时直接报错，不偷偷重训。

## G0：重新训练 B0 并查看复现指标

只复跑 B0，不运行后续 75 组：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1010.py --baseline_only \
  > nohup_E1010_B0.log 2>&1 &
tail -f nohup_E1010_B0.log
```

训练直接复用 E0924 的 G26_B0 构建与训练入口：FF++ c23、224 输入、seed=1024，
训练所有 CLIP block 的 LoRA 及普通线性分类头，原 CLIP 参数与 CLS embedding 冻结。
保留原数据增强，关闭 mixup、margin、频率输入、texture crop、rank loss；
Adam lr=2e-4、weight decay=5e-4、v1 sampler real ratio=0.30、batch size=32。
按 CDF-v2 帧 AUC 选最好 checkpoint。`--base_n_epochs 10` 沿原循环执行 epoch 0–10，
实际 11 轮；它与辅助分支的 `--n_epochs 10`（10 轮）是两个独立预算。
`--base_sampler_real_ratio`、`--base_batch_size` 可显式调整 B0 条件，默认保留原设置。
辅助的 lr、weight decay、batch 和采样比例不会改动 B0 训练条件。

B0 训练完成后先用原评估入口输出八域历史 basename 视频 AUC，
控制台直接显示七域均值（含 CDF、不含 FF++）、六独立域均值、AUC_cross 和 G。
保存位置为 `G0/training/reproduction.json`；原训练/评估配置、结果与模型日志也位于
`G0/training/`。原训练增强配置完整保留，后续 E1010 严格导出才关闭增强。

接着 `G0/result.json` 保存严格成对导出的 full-path 视频 AUC 与 legacy 指标，
两种口径分别记录。核对历史成绩时使用相同数据集和同一视频分组口径。
若没有旧结果参考，报告标为 `REFERENCE_NOT_PROVIDED`，只展示新成绩，
不自动声称数值或权重复现成功；实际是否复现仍需服务器训练结果与旧成绩比较。
训练或历史评估失败会记录状态并停止后续辅助实验。

看完 B0 后，用同一新模型运行后续组：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1010.py \
  --base_run ./experiment_results/E1010/seed... \
  > nohup_E1010.log 2>&1 &
```

## 要验证什么

在同一种辅助 token 结构内，比较是否加入局部伪造似然监督，以及是否使用
“真图全部证据为真、假图至少存在假证据”的不对称监督。
局部目标只作用于辅助分支的可训练 patch 投影，梯度不进入 B0。
分别报告 B0、辅助预测和固定融合，检查局部监督是否改善补充证据。

该实验不能单独证明历史联合训练中的下降就是缺少局部监督造成的。
这里 B0 已经冻结，最终融合仍可能降低效果；“B0 不变”和“融合有效”分别检查。

## G 分组

G0 为原协议 B0 的训练/复用和评估。其余组默认有 75 个配置：72 组辅助训练，加 3 组固定原型诊断。

| 组 | 结构来源与读出 | 配置数 |
| --- | --- | ---: |
| G1 | G18 LFEQ：L1 决策/证据各 0.5；L2 决策读出；L3 证据读出 | 3 × 6 = 18 |
| G2 | G25 晚层添加 token：四种注意力连接 × max/all 原监督 | 4 × 2 × 6 = 48 |
| G3 | G30 独立交叉注意力辅助解码器 | 1 × 6 = 6 |
| G4 | 全部假图 patch / 随机 K 个 / 遮挡贡献最高 K 个构建全局伪造原型 | 3（无需训练） |

G1–G3 每种结构覆盖相同的六种监督组合：

| 后缀 | patch 对比损失 | 证据 token 监督 |
| --- | --- | --- |
| PLAIN | 无 | 原 max/all 分类监督 |
| UNIFORM | 等权对比 | 原 max/all 分类监督 |
| FL | 伪造似然加权对比 | 原 max/all 分类监督 |
| MIL | 无 | 真图全部证据真、假图存在假证据 |
| UNIFORM_MIL | 等权对比 | 不对称监督 |
| FL_MIL | 伪造似然加权对比 | 不对称监督 |

G1 保留 G18 的独立决策 token、证据 token、self/cross-attention、FFN、
两分类头和原损失组成。默认 K=8、hidden=256、8 heads、2 blocks、dropout=0.1，
读取 B0 最终 `last_hidden_state` 的 patch。G18 原实现用 LFEQ 替代 B0 读出；
E1010 将完整 LFEQ 放在辅助分支，保留 B0 原分数。因此它是 G18 结构的隔离对照，
不是历史 G18 训练结果的重现。

G2 默认 K=4，在零起始 block 20 的输入处添加 token。完整 B0 正向不添加 token；
独立副路读取捕获的 `[CLS, patches]`，通过初始为恒等映射的可训练 patch 投影，
添加 token，再调用 B0 的冻结晚层算子。新增 token 和独立分类头保持可训练。
G2 的晚层维度、head 数和层数继承 B0；`hidden_dim`、`num_heads`、`depth` 仅调整 G1/G3 的独立解码器。
注意力编码第一位是副路 CLS 能否读新增 token，第二位是副路 patch 能否读新增 token：
00=read_only、10=cls_only、01=patch_only、11=full。它们均不改变 B0 原正向。
M=max、A=all；二者均用最高假概率证据评分，保留 G25 的注意力多样性约束。

MIL 替代原 max/all 证据损失，因此同一注意力连接的 M/A MIL 配置具有相同目标。
这些重复配置用于覆盖历史组名，不能当作独立方法或独立随机种子证据。
G18 L1/L2/L3 的训练目标也相同，但读出与选点分数不同。

G3 默认 K=4、读取 block 20 输入、hidden=256、4 heads、2 blocks、无 dropout。
复用 G30 的解码器机制，不引入 Full 的 balance、hard weighting、第二视图或学习路由。
PLAIN 是 max 监督对照，MIL 组复用 G26/G30 的不对称证据监督；各组使用同一固定读出。

只运行选定配置：

```bash
python -u experiments/run_e1010.py --base_run /path/to/E0924/run \
  --arms G1_L1_PLAIN G1_L1_FL G1_L1_MIL G1_L1_FL_MIL

python -u experiments/run_e1010.py --base_run /path/to/E0924/run \
  --arms G2_M00_FL_MIL G2_A11_FL_MIL G3_FL_MIL
```

## 局部伪造分析与不对称目标

真实和伪造图的辅助 patch 特征分别按 batch 求均值作为原型。patch 的相对伪造似然为
`delta = cosine(patch, fake_prototype) - cosine(patch, real_prototype)`。
FL 使用 `sigmoid(delta / 0.1)`，每张假图的权重归一到均值 1，并停止权重的梯度；
UNIFORM 使用权重 1。对比温度为 0.1、损失系数为 0.14。

为限制 patch 两两比较的显存，每张图默认均匀选取最多 16 个 patch。
假图 patch 作 anchor，其他假图的 patch 作正样本，真实图 patch 作负样本；
分母包括采样池中除 anchor 自身之外的所有 patch。同图 patch 不作为正样本。
它是基于图像标签的弱监督，假图正样本仍可能包含未篡改区域。
等权与 FL 组使用完全相同的采样和对比目标，仅权重不同。

这参考 QTFP 的原型差分与软权重思想，不宣称严格复现论文公式（6）。
来源：[QTFP 项目](https://banishedknight.github.io/CVPR_QTFP/)。
权重不是真实局部篡改标注，也不能仅凭高分或注意力图声称定位准确。
开启对比学习时，每个 batch 必须包含至少 1 张真图和 2 张假图，否则报错；
不得把没有正样本的零损失计作有效训练。

MIL 对真图压低所有证据 token 的假 log odds；对假图使用温度 0.5 的
归一化 log-sum-exp 存在性目标。它允许其余 token 对假图输出真实，
不强制假图有固定比例的真实区域。约束作用于聚合后的证据槽位，不等价于 patch 定位标注。

## G4：利用整图判别知识筛选伪造原型

单独运行三种原型对照，不训练 G1–G3：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1010.py \
  --arms G4_ALL G4_RANDOM G4_TOPK \
  > nohup_E1010_G4.log 2>&1 &
```

没有显式 B0 来源时，此命令先训练 B0，再运行三个 G4 对照。
已有 B0 可加 `--base_run` 避免重新训练。
G4 从 FF++ **train** 元信息中，按路径排序后使用固定 seed 分别打乱真/假帧，
默认每类最多选 32 个唯一帧。三种方法共用这一个池，真实原型均为全部真实 patch 的均值。
ALL 指这个训练子集的全部假图 patch，不是全 FF++ 训练集；RANDOM 和 TOPK 每张假图各选 K=16 个。
构建全局共享原型后固定不变，不更新 B0、辅助 token 或分类头，不做迭代自训练。

TOPK 先计算原 B0 的真假 logit 差 `m = fake_logit - real_logit`。
根据实际 CLIP patch embedding 的网格，在输入图像上逐 patch 遮挡：
用该图每个颜色通道的空间均值替换对应像素，贡献为 `m(original) - m(occluded)`。
提取未遮挡图像的最终 patch 特征，仅将贡献最大的 K 个假图 patch 加入伪造原型。
负贡献也参与排名，不强行解释为正证据；输出选中 patch 中贡献非正的比例。

默认一次最多推理 16 张遮挡变体，可用 `--occlusion_batch_size` 调整显存预算。
224/14 对应 P=256，32 张假训练帧需要额外 8192 次遮挡图像推理；分批只减少内存峰值，
不减少图像推理数量。ALL/RANDOM 单独运行时不会计算遮挡归因，测试阶段也不计算归因。

三种原型采用完全相同的推理公式：各 patch 与假/真原型的 cosine 相似度差，
经温度 0.1 的 sigmoid，再将分数最高的 16 个 patch 取均值。
分别报告原 B0、prototype-only、相同固定 gate 融合后的分数。
`--prototype_top_k` 控制训练筛选 K，`--prototype_pool_top_k` 控制测试聚合 K；
两者超过实际 patch 数时直接报错，不静默截断。

`--prototype_frames_per_class` 调整训练帧预算，`--prototype_temperature` 调整固定读出温度。
这些参数在运行前确定，不根据终评集挑选。G4 不受辅助 epochs、head 数或 balanced sampler 的约束。
它检验的是筛选原型的判别效用，不能证明选中 patch 就是真实篡改区域：
整图分类器的捷径和遮挡造成的分布变化都可能影响贡献。

根目录 `G4_selection.json` 记录 split、seed、实际帧数、全部选帧路径和源元信息哈希。
每个 G4 子目录保存绑定 B0 的 `prototype.pth`、原型实际 patch 数、
`selection.npz` 中的贡献/选择索引及原 B0 分数，还有各域成对导出和指标。
原型只从选定训练帧构建；测试标签只用于计算指标。

## 训练与评估边界

辅助训练默认 seed=1024，10 epochs，AdamW lr=1e-4、weight decay=0.01，
FF++ train，balanced sampler real ratio=0.30，batch size 继承 B0 配置。
数据路径、compression、resize、normalization 和 B0 架构继承原保存配置；
严格读取单张 RGB 帧，无新增随机增强。每个配置重置同一 seed、独立初始化、独立保存。
同一结构的六组参数数量和推理方式相同，只改变损失设置。

B0 全参数冻结，始终 eval；辅助输入 detach，晚层副路使用冻结参数的功能调用。
辅助检查点仅保存辅助权重，绑定 B0 文件 SHA256 和完整配置。
每轮验证和最终导出检查 B0 分数、log odds、标签、帧路径和完整视频路径逐位一致，
训练前后检查 B0 参数状态及源文件哈希。漂移立即中止。

所有组用相同的固定 gate：B0 在 0.5 附近宽度 0.2 的区域才融合，辅助权重最多 0.5。
不训练额外决策模型，也不根据终评结果更换融合规则。
G1–G3 的辅助 checkpoint 按每轮 CDF-v2 的 gated frame AUC 选择；G4 无选点过程。
复用的 B0 与辅助选点都可能用过 CDF，因此仍按 E1010 同一口径，
独立域平均只包括 WDF、FFIW、DeepFakeDetection、DFDC、DFDCP、DeeperForensics-1.0。
CDF 和 FF++ 单独报告。主视频 AUC 按完整视频路径汇总帧均值，旧 basename 指标另存，
不能直接拿新主指标与旧 G18/G25 的 basename 分数相减。

每组报告三个读出的 frame/video AUC、真实图 FPR、纠错/伤害帧数，以及六个独立域均值。
默认覆盖全部配置不等于多 seed；可靠的增益仍需要独立随机种子验证。

## 输出与本地验证

输出位于新建的 `experiment_results/E1010/seed.../`，不覆盖旧实验。
根目录保存 manifest、运行配置、B0 缓存和 `all_results.json`；
每组保存 settings、独立辅助 checkpoint、history、成对 NPZ 与 result。
manifest 记录 B0、相关源码、数据元信息和运行环境，识别未提交的源码变化。

```bash
python experiments/run_e1010.py --preflight
python -m pytest tests/test_e1010.py tests/test_e1010_baseline.py tests/test_e1010_prototypes.py tests/test_e1010_runner.py -q
```

`--dry_run` 不导入模型、不加载权重或数据；`--preflight` 检查配置、
数据元信息和采样要求，复用模式还检查 B0 文件身份；新训练模式不会提前训练 B0。
预检不代表图片、GPU、checkpoint 加载或训练环境验证成功。
本地测试使用随机初始化的小型真实 CLIP，检查辅助训练、梯度隔离、严格重载和成对导出；
通过 CPU 合约测试不代表真实数据上的训练或 G4 原型评估已经完成，也不代表获得了 AUC 增益。
