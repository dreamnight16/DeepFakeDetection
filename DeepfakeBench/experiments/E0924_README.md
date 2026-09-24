# E0924：待算力恢复后的统一实验批次

本批次把 G26、G27、G28、G29 组成一条可顺序运行、失败留痕、可续跑的流程。默认 seed=1024，主模型 `n_epochs=10`（原 trainer 实际 epoch 0–10）。当前交付代码、合约测试和源码包，尚无本批次真实训练结果。

## 包含哪些实验

| 顺序 | 实验 | 作用 |
| --- | --- | --- |
| 1 | G26/B0 | 共享 RGB + CLIP-LoRA 基线 |
| 2–5 | G26/K4L20、K8L20、K4L16、K8L16 | token 数量与插入层的 2×2 矩阵 |
| 6–9 | G27/E1、E2、E3、Full | 负载均衡、难例加权、光度一致性、三者组合 |
| 后处理 | G28_linear、G28_mlp | 模型状态输入，线性/16维 MLP 决策器 |
| 后处理 | G29_freq、G29_color、G29_both | 在 G28_mlp 上加频域、色域、两类统计 |

共 **9 次主模型训练 + 5 次小决策器 CPU 拟合**。默认训练依次进行，使用命令指定的 GPU。两项去重：G27/B0 复用 G26/B0；G27/G26 对照复用 G26/K4L20。这是协议等价的共享对照，不能统计成新的独立种子。

G25 已有结果不重复跑；G25v2 保留原独立入口，不纳入本次讨论形成的 G26–G29 默认矩阵。G27 的结构扫描留待后续验证后另开批次，避免当前训练量膨胀。

G28/G29 **预先固定 G27-Full** 为感知模型来源，严格冻结同一个 checkpoint。不依据 DFDC 或其他测试结果选择哪个 G27 组进入决策阶段。此处的 G28/G29 为本地小决策器，不调用外部 Jev，不把 FFT 或 YCbCr 图再送入 CLIP。

## 服务器启动

源码压缩包是现有 `Effort-AIGI-Detection-main` 仓库的 source overlay，并非完整 Python 环境。`manifest.json` 记录打包时基准 commit 及文件 SHA256。先在服务器备份工作区/核对差异，再将包内 `DeepfakeBench/` 的文件合入现有仓库；不要覆盖自己未保存的改动。包内不含数据集 JSON、图片、预训练权重、实验 checkpoint、密钥或 G21 未提交修改。

在服务器仓库的 **DeepfakeBench 目录**执行：

```bash
# 只列计划：不导入 torch、不读取图片、不启动训练
python experiments/run_e0924.py --dry_run

# 只做元数据预检：需要八个数据集 JSON，但不需要 GPU/图片推理
python experiments/run_e0924.py --preflight

# 正式执行全部实验；此命令留待算力可用时运行
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_e0924.py --n_epochs 10 --seed 1024 > nohup_E0924.log 2>&1 &

# 如服务器路径不同，明确指定并贯穿训练与评估子进程
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_e0924.py \
  --dataset_json_folder /absolute/path/to/dataset_json \
  --rgb_root /absolute/path/to/data \
  --clip_pretrained_path /absolute/path/to/clip-vit-large-patch14 \
  --n_epochs 10 --seed 1024 > nohup_E0924.log 2>&1 &

# 查看日志
tail -f nohup_E0924.log
```

输出根目录：`experiment_results/E0924/seed1024_<时间>_<pid>/`。默认额外执行 G26/G27 的分支诊断，会增加推理时间；`--attention_samples 0` 仅关闭 G27 的 attention 样本，`--skip_diagnostics` 跳过 G26/G27 的全部额外诊断，但 G28/G29 所需的特征导出仍执行。

需要配置现有项目训练环境、CLIP 权重和八个数据集：WDF、FFIW、Celeb-DF-v2、DeepFakeDetection、DFDC、DFDCP、DeeperForensics-1.0、FaceForensics++。沿用项目 RGB 单裁剪、224 分辨率、每视频8帧、FF++ c23、v1 sampler 真图占比 .30。数据路径通过显式 CLI 传给训练、主评估、readout 子进程；其他实验未启用 `e0924_protocol` 时保留原行为。

## 决策层数据隔离

主模型仅使用 FF++ train 训练，CDF-v2 选择 checkpoint。决策器使用 **FF++ 官方 val**，预检校验 train/val/test 的文件名数字 ID 和帧路径无交集。没有 val 时直接失败，不回退到 train、FF++ test 或 DFDC。

对 val 视频名里的全部数字 ID 做连通分量分组，跨伪造方法统一分组；按 seed 的稳定哈希将分量划为 70% fit、30% holdout。它是保守的潜在同源分组，不声称已验证源/目标配对或人物身份。fit 用于拟合，holdout 仅报告，不选择步数、结构或阈值。本地现有 FF++ c23 元数据划出 **490 个 fit 视频、210 个 holdout 视频**；这不是已读到真实图片的证明。

每个视频使用元数据前8帧，匹配当前项目同设置的帧选择行为。导出复用项目 RGB 加载、bicubic resize、标准化，关闭随机增强与多裁剪。读帧失败直接终止导出，禁止原 Dataset 中“随机换另一帧”的容错，从而保证 path/label/score 对齐。色频统计基于模型看到的同一个 resize 后 crop，经逆标准化恢复 RGB；不利用标签构造特征。

## G28/G29 的具体实现

原方案的 `TRUST_CLS` 与 `KEEP_CLS` 有相同输出，本批次实现为二分类 `trust_expert`，置信度不足时保留 CLS。只用 fit 集中 CLS 与 expert 在阈值 `p>.5` 上分歧的样本拟合，目标为 expert 是否正确；二分类分歧时恰好一方正确。若 fit 无分歧则明确失败，不伪造可训练结果。

决策器输入模型状态包括 CLS logit 差、CLS/expert 概率、逐专家 logit 差、最高分、前两名差、软责任熵和两分支分歧。G28_linear 为一个线性层，G28_mlp 为 `输入→16→tanh→1`。G29 三组均使用同样 16 维 MLP，仅改变输入统计。

- 频域6维：去均值亮度的低/中/高频能量占比、高低频比的对数、径向频谱斜率、高频水平/垂直能量差。全二维 FFT，半径按对角 Nyquist 归一化，分界 .25、.65。
- 色域12维：Y/Cb/Cr 均值及标准差、Cb/Cr 高频比例、RGB 局部残差相关、亮度与两个色度通道的边缘相关、接近0或1的像素比例。色域使用文中明确的浮点 BT.601 风格转换，不把它称为可靠伪造检测器。

标准化均值/方差只拟合 fit 集，标准化值裁剪到 [-10,10]；每个分歧视频总权重相同。Adam lr=.01、weight_decay=.001、固定300步；`--router_steps/--router_lr` 可在新批次预先指定。测试标签只参与指标计算，不进入 `score_router`。

推理：两分支一致时保留 CLS；分歧且 `P(trust_expert)>.7` 时切换到 expert 连续概率，其余保留 CLS。`.7` 为预先固定、未经校准的阈值，可以用 `--reject_threshold` 在新批次预设，但不能按最终测试集选择。此处是分支切换，不是原 G26/G27 的 .5 上限软融合；可能改变概率排序，必须实际报告 AUC。

每组同时报告同一特征导出上的 CLS、evidence、D0 固定门控作为对照。官方决策训练集之外的测试数据不会参与拟合、标准化、迭代数或阈值选择。CDF-v2 已参与主模型选点，其最终数值不能单独作为未见域泛化证据。

## 指标与产物

主模型依旧使用原 testall/video_auc。决策评估在一次冻结模型导出的逐帧分数上离线重算，保存两种视频分组：`video_auc` 按项目历史倒数第二级目录名分组，`video_auc_fullpath` 按完整父目录分组，并报告旧口径的命名碰撞数。两种 `AUC_cross` 与七集均值分别保存，不混用。决策层的 CLS 与 D0 对照来自同一导出输入，避免把不同数据处理路径的结果强行做配对比较。

Oracle **仅报告固定阈值下 `CLS正确 OR expert正确` 的帧准确率上限**。不产生可部署预测、不报告 Oracle AUC，不把它当作 AUC 上界；它也不保证真实决策器能学到这种选择。

```text
E0924/<run>/
  manifest.json                  计划、源码/全部数据集JSON签名、参数
  calibration_partition.json     fit/holdout视频及分量
  runtime.json                   Python/torch/transformers版本
  status.json                    本批次总状态
  training_results.json          所有主模型最新结果/重试目录指针
  training/<family_arm>/         原runner配置、日志、checkpoint与诊断
  attempts/<family_arm_time>/    失败后重试，保留旧目录
  decision_exports/              fit、holdout、各测试集NPZ及校验和
  decisions/                     五个router JSON、holdout指标、逐帧分数、results.json
```

路由器 JSON 包含权重、标准化参数、固定阈值、源 checkpoint SHA256，可以离线重建。导出的 NPZ 保存 label、path、完整 video_id、各分支分数、专家 logit 和色频统计。可复用这些小特征文件做 CPU 后处理，不需再次训练或前向完整 CLIP。

## 中断、失败和续跑

```bash
# 参数与首次运行保持一致；把路径替换成日志中打印的完整run目录
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_e0924.py \
  --resume /absolute/path/to/experiment_results/E0924/seed1024_<time>_<pid> \
  --n_epochs 10 --seed 1024 > nohup_E0924_resume.log 2>&1 &

# 仅主模型阶段，或之后仅补决策阶段
python experiments/run_e0924.py --stages train
python experiments/run_e0924.py --resume /absolute/path/to/run --stages decisions
```

续跑要求源码、实际协议参数、元数据根路径与全部数据集 JSON 哈希相同。成功且 checkpoint 存在的训练组会跳过；失败/中断组写入新的 attempts 目录，保留其他成功重试指针。不会覆盖用户旧训练目录。G27-Full 主评估失败时，决策阶段记录失败，不自动切换到其他 checkpoint。完整导出带哈希才允许复用，半成品导出另存后重建。

`--stages train` 成功只表示请求的训练阶段完成，决策阶段尚未执行。阶段结果任一失败，退出码非零。元数据缺失/不合规在主训练启动前返回2。当前只对元数据/代码及导出的checkpoint/特征缓存做哈希；未对全部图片或默认 CLIP 预训练目录做哈希。若原图片或预训练权重改变，必须新建批次，不能据同路径声称实验相同。

## 本地验证与打包

```bash
python -m pytest tests/test_e0924.py tests/test_g27.py tests/test_g27_runner.py \
  tests/test_g26.py tests/test_g26_runner.py tests/test_g25.py tests/test_g25_runner.py \
  tests/test_g25v2.py tests/test_g25v2_runner.py tests/test_g22_g23.py -q

# 生成可搬运源码覆盖包；拒绝覆盖已有同名zip
python experiments/make_e0924_bundle.py --output /absolute/path/E0924_source.zip
```

本地验证使用随机小 CLIP、合成图片和 mocked 训练/子进程，真实验证了损失梯度、路由器拟合/重载、数值统计、划分不交叠、实际子进程命令透传、失败继续与缓存校验；它不等同于真实 CLIP-L/14 训练/数据加载通过。预检只校验 JSON，GPU、图片可读性、所有服务器依赖及实际额外显存耗时留待服务器确认。独立审查曾发现并修复适配器缺接口、数据路径被默认配置覆盖及重试指针丢失问题。

2026-09-24 本地验证结果：**191 项通过**（E0924 18 项、G27 45 项、此前相关回归128项），Python 3.12.14、torch 2.5.1+cpu、transformers 4.44.2。独立 MLE 复审确认修复后代码交付通过。无 GPU 训练、无真实图片前向、无本批次 AUC 结果。

当前本地完整预检明确缺少 **WDF.json、FFIW.json、DeeperForensics-1.0.json** 并返回2；其余已有 JSON 不补猜。服务器须在 `--dataset_json_folder` 指定目录补齐八份元数据后重新预检。此前 490/210 的数字仅来自现有 FF++ 元数据划分检查。
