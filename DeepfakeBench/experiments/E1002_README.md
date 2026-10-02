# E1002：四条结构路线与统一 B0 对照

在 DeepfakeBench 目录执行：

```bash
nohup python -u experiments/run_e1002.py \
  --base_run ./experiment_results/E0924/seed1024_20260930_110618_353460_2771190 \
  > nohup_E1002.log 2>&1 &
```

```bash
tail -f nohup_E1002.log
python experiments/run_e1002.py --dry_run
python experiments/run_e1002.py --base_run /path/to/E0924/run --preflight
```

默认 **13 组独立训练，各 10 epochs、seed=1024**，直接复用 E0924/G26_B0。原 checkpoint 和 E0924/E1001 输出不覆盖。现有 `effort` B0 是普通 LoRA + CLIP ViT-L/14 + 全量分类头；本实验探索受约束矩阵更新，不假称已有 B0 是原论文 Effort 的正交分解实现。

| 路线 | Arm | 实际训练内容 |
| --- | --- | --- |
| 特征读出 | CLS_HEAD | 冻结 B0 CLS，训练 64 维瓶颈二分类器 |
| 特征读出 | PATCH_MEAN | 冻结最终 patch 均值，训练相同瓶颈分类器 |
| 二阶关系 | COV_GLOBAL | patch 投影到最多 64 维，中心化协方差、稳定符号平方根与范数归一化，上三角读出 |
| 二阶关系 | COV_REGIONAL | 2×2 区域协方差，共享投影/瓶颈后按区域顺序拼接，保留粗粒度位置 |
| 矩阵适配 | MATRIX_LORA | 独立 B0 副本，最后 4 层 Q/K/V/out 新增 rank4 更新，训练副本分类头 |
| 矩阵适配 | MATRIX_SVD_TAIL | 相同新更新与预算，左右投影到原有效权重前 32 个奇异方向的正交补 |
| 矩阵适配 | MATRIX_SVFT | 固定原有效权重的奇异向量，训练 4 条循环稀疏系数带；这里 rank 表示带数，不是更新矩阵秩 |
| 环境解耦 | ENV_CONTROL | 伪造/环境两个编码器；真假 CE + 已知光度干预类型 CE，解耦权重 0 |
| 环境解耦 | ENV_ORTH | 相同结构/干预，加 0.1×样本余弦平方及批次交叉协方差约束 |
| 视频时序 | TEMP_MEAN | 按有效帧平均已投影的 CLS 特征，训练视频分类头 |
| 视频时序 | TEMP_DIFF | 特征均值、连续已采样帧的绝对/平方差统计，训练瓶颈分类头 |
| 视频时序 | TEMP_TCN | 两层带掩码的一维时序卷积与有效帧池化 |
| 视频时序 | TEMP_SSM | 输入依赖步长/B/C、稳定负对角 A、门控的纯 Torch 状态空间读出 |

所有组均读取原始 RGB 和 B0 resize/normalization，不随机增强。环境干预仅为原图、亮度 ±0.02、对比度 0.9 四类；环境标签来自实际施加的干预。没有人物身份监督，不能把这两组称为已验证的身份解耦。两个编码器的正交/低交叉协方差也不等于统计独立。

矩阵组冻结副本中原来的 ViT/已拟合 LoRA，新增更新从零开始，保留原 projection 的计算顺序；只有新更新与副本分类头训练。它们是从同一个已训练 B0 出发的继续适配实验，**不是 Effort/SVFT 的完整复现**。SVD_TAIL 保留单个投影的主奇异方向，不保证整个网络语义不变。

SSM 是轻量状态空间实验，**不是官方 Mamba**。时序输入按数字帧编号排序、每视频最多 8 个现有采样帧，短视频补零并用 mask 排除；记录帧路径/编号。连续采样帧不一定是原视频相邻帧，差分不按物理时间归一化。时序概率广播到各有效帧便于统一导出，其 frame AUC 是广播视频分数的统计，不表示逐帧定位能力。

## 训练与选择

- FF++ official train 训练，official val 的完整路径 **video AUC** 选 checkpoint；val 使用沿用 E0924 校准划分中的全部 fit+holdout 记录，这里两部分都用于选择与阈值估计，不能再称 holdout 为独立验证。原有 filename source-ID 分离检查保留。
- 默认 AdamW：读出/环境/时序 lr=1e-4，矩阵 lr=1e-5，weight decay=0.01，梯度范数裁剪 1。无随机图像增强，balanced real ratio=0.3。
- 单帧 batch size 继承 B0，时序 `--clip_batch_size 4`，避免把 32 个视频一次作为 256 张图输入。每组记录实际参数量、更新步数与耗时；相同 epoch 不等于相同样本或参数预算。
- 全部测试集都只报告最终选定 checkpoint，不根据测试结果挑 arm。B0 过去已经用 CDF-v2 选点，因此 CDF 仍不能作为整个方法完全独立的未见域证据。
- 同时报固定 0.5 下的帧/视频 FPR、FNR 和视频 AUC/AP；另在 val 真实视频上估计 5% FPR 的阈值后冻结，报告测试 FPR/TPR。平分数及有限样本可使 val FPR 低于目标，目标阈值不保证域外 FPR 仍为 5%。
- 冻结读出组持续验证 B0 state/file SHA256 和相同输入/批次上的逐位分数。矩阵组的增强输出允许改变，原 B0 对照不变。时序 B0 单独缓存同一视频帧/批次下的分数以避免跨批次浮点口径差异。

## 常用参数

数据位置迁移：`--dataset_json_folder`、`--rgb_root`、`--clip_pretrained_path`。也支持 `--base_checkpoint /path/to/B0.pth --base_config /path/to/train_config.json`。预检只验证原 checkpoint/配置及全部元数据，不验证图片、CUDA 或完整训练环境。

只跑二阶读出：

```bash
python -u experiments/run_e1002.py --base_run /path/to/E0924/run \
  --arms CLS_HEAD PATCH_MEAN COV_GLOBAL COV_REGIONAL
```

三种子：

```bash
python -u experiments/run_e1002.py --base_run /path/to/E0924/run \
  --seeds 1024 2048 4096
```

短流程核验（结果明确记录训练截断，不是正式结果）：

```bash
python -u experiments/run_e1002.py --base_run /path/to/E0924/run \
  --n_epochs 1 --max_train_batches 2
```

## 输出与验证

每次写到 `experiment_results/E1002/seed...`：

- `manifest.json`、`runtime_config.json`、`calibration_partition.json`：协议、源码/元数据/B0 哈希、运行环境与隔离检查。
- `B0_frame/`、`B0_temporal/`：按各自相同输入和批次生成的独立原始分数；未请求时序组则不生成 temporal 缓存。
- `<arm>/best.pth`、`history.json`、`result.json`、`exports/*.npz`：独立 checkpoint、选点过程、指标与标签/帧/视频身份。NPZ 保存 `cls_prob`、`global_log_odds` 和增强 `score`；可用帧路径解析原始数字帧编号。
- `all_results.json`：全部成功/失败状态。普通 arm 失败会继续后续独立 arm，整体返回非零；任何 B0 漂移立即中止。结果/checkpoint 用临时文件替换，防止正式文件写入一半。

矩阵 checkpoint 包含独立副本的完整冻结权重和固定 SVD 基底，磁盘开销高于小读出 checkpoint；不会将它们写入 B0 文件。

```bash
python -m pytest tests/test_e1002.py tests/test_e1002_heads.py \
  tests/test_e1002_matrix.py tests/test_e1002_temporal.py -q
python -m pytest -q
```

本地 tiny CLIP 流程覆盖全部 13 组训练、FF++ val 选点、checkpoint 重载和八集导出，不下载预训练模型。真实数据/CUDA 性能只有运行服务器实验后才能判断。

研究来源：[Gram-Net](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Global_Texture_Enhancement_for_Fake_Face_Detection_in_the_Wild_CVPR_2020_paper.html)、[Effort](https://arxiv.org/abs/2411.15633)、[SVFT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html)、[音频双粒度解耦](https://arxiv.org/abs/2606.16532)、[BiCrossMamba-ST](https://www.isca-archive.org/interspeech_2025/elkheir25_interspeech.html)。这些是结构动机，不能代替同协议复现或实验证据。

2026-10-02 本地验证：原测试环境完整测试集 **383 passed**；新版 PyTorch/Transformers 环境 E1002 测试 **128 passed**。包括全部 13 arm 的真实 tiny CLIP 训练/选点/重载/导出、临时 LMDB 生命周期与失败继续流程。独立代码审查通过。未运行服务器真实数据或 CUDA 训练。
