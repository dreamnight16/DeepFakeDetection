# E1001 / G30：冻结 B0 的独立证据与决策 token

在 `DeepfakeBench` 目录，使用服务器上已有 E0924 的 B0：

```bash
nohup python -u experiments/run_e1001.py \
  --base_run ./experiment_results/E0924/seed1024_20260930_110618_353460_2771190 \
  > nohup_E1001.log 2>&1 &
```

默认运行 **G30/K1L20、G30/K4L20、G30/Full 三次辅助训练，以及 linear、MLP、decision token 三次 CPU 决策拟合**。B0 直接复用，不重训。Full 是决定拟合的固定来源，不根据测试结果挑选来源。

```bash
python experiments/run_e1001.py --dry_run
python experiments/run_e1001.py --base_run /path/to/E0924/run --preflight
tail -f nohup_E1001.log
```

也可显式提供原始 B0 checkpoint 和对应的 JSON 配置：

```bash
python -u experiments/run_e1001.py \
  --base_checkpoint /path/to/G26_B0/logs/effort_.../test/avg/ckpt_best.pth \
  --base_config /path/to/G26_B0/train_config.json
```

数据迁移时使用 `--dataset_json_folder`、`--rgb_root`、`--clip_pretrained_path`。默认数据位置、预处理和 LoRA 类型继承 B0 配置；加载 B0 state_dict 时严格匹配。未找到原始 B0 就报错，不回退到 G27 或重新训练。`--preflight` 仅检查 checkpoint、配置、元数据和分区，不验证图片、GPU 或训练依赖。

B0 的训练配置不包含 WDF、FFIW 等测试标签，E1001 仅从项目 `test_config.yaml` 补齐缺失的 `label_dict` 项，不覆盖模型架构或预处理，也保留已有 fake 子类编号。已有标签的 real/fake 含义若与测试配置冲突，或选定元数据中有未知标签，预检立即报错。预检覆盖全部测试集的 test split，以及 FF++ 的 train/val/test。

## 结构与隔离约束

```text
原始图像 → 原始 B0 ViT + LoRA → 原始分类头 → B0 原始分数
                    │
                    └─ 只读捕获第 L 个 block 的输入 patch，detach
                         → 独立 cross-attention/FFN + evidence tokens + heads
                         → evidence 分数 / 固定 gate 分数

缓存的 B0 分数 + evidence 分数及各 query log odds
  → 独立 linear / MLP / decision token → 增强分数
```

- ViT、LoRA、B0 分类头全部冻结，始终 `eval`，原始序列从不追加辅助 token。
- 证据解码器拥有自己的投影、attention、FFN、token 和分类头。读取 B0 特征时强制 detach，优化器只接收该解码器参数。
- `memory_layer` 是零起始 block 编号，与 E0924 的插入层编号一致，但 G30 读取该 block **输入**，不复用其后 B0 block 来处理辅助序列。L20 不是额外 20 层。
- 决策 token 读取已导出的标量特征，将各维映射为 feature tokens，再用一个独立 query 读取它们。拟合过程中不加载图像模型。输入边界同样强制 detach。
- `auxiliary_enabled=False` 直接调用并返回原始 B0 的输出。
- `auxiliary_best.pth` 只保存证据解码器，绑定原始 B0 文件 SHA256；决策 artifact 另存，并绑定 B0 与证据 checkpoint 两个 SHA256。
- 辅助训练前后检查 B0 state_dict 哈希；每轮 CDF 验证及最终所有导出均与预先缓存的原始 B0 逐位比较分数、log odds、标签、帧及视频身份，任何漂移都中止运行。

这里保证的是 **B0 不受辅助训练影响**。固定融合或学习决策仍可能让增强分数的 AUC 下降；没有数学保证它们超过 B0。报告分别展示 B0、evidence、gated 和各决策读出。

## 实验设置

| Arm | evidence tokens | 读取 block | 辅助目标 |
| --- | ---: | ---: | --- |
| K1L20 | 1 | 20 | MIL |
| K4L20 | 4 | 20 | MIL |
| K8L20（可选） | 8 | 20 | MIL |
| K4L16（可选） | 4 | 16 | MIL |
| Full | 4 | 20 | MIL + 0.1 balance + hard weighting + 0.1 view consistency |

证据解码器默认维度 256、4 heads、2 blocks，无 dropout；AdamW，lr=1e-4，weight decay=0.01，10 epochs。训练 FF++ train，严格读图，沿用 B0 resize/normalization，单帧无随机增强；balanced sampler 的 real ratio=0.3，batch size 继承 B0。Full 的第二视图为固定 contrast=0.9、brightness=0.02。MIL 与 Full 的机制不是已证实的性能改进，仍需要对照。

辅助 checkpoint 按每个 epoch 的 CDF-v2 **gated frame AUC** 选择，同一 checkpoint 导出所有分支。CDF 已参与选择，因此不能把它当作独立的未见域证据。训练新 decoder 和冻结现成 B0 与 E0924 从头训练整个 LoRA 分支的设置不同；直接跨实验差值不能单独归因于某个 loss。

决策拟合复用 E0924 的 FF++ official val 按 filename source-ID 连通分量划分的 70/30 fit/holdout。只在 fit 上估计标准化和监督 disagreement，按视频等权；300 steps、Adam lr=0.01、weight decay=0.001。拒绝阈值预先固定为 0.7；仅两分支分类不一致且信任 evidence 超过阈值时切换。linear、16-unit tanh MLP 和 decision token 使用相同输入与预算。没有 disagreement 时记录 `SKIPPED_NO_DISAGREEMENT`，不伪装成已拟合成功。

主 `video_auc` 按**完整视频路径**聚合帧均值，另存 `video_auc_legacy` 复现旧 basename 口径。与 E0924 对比时使用一致口径；不能把新主指标直接减去旧 basename 指标。

扩展到五组辅助训练：

```bash
python -u experiments/run_e1001.py --base_run /path/to/E0924/run \
  --arms K1L20 K4L20 K8L20 K4L16 Full
```

仅验证一个证据模块，不拟合决策：

```bash
python -u experiments/run_e1001.py --base_run /path/to/E0924/run \
  --arms K4L20 --skip_decisions
```

## 结果与验证

输出保存到新建的 `experiment_results/E1001/seed...`，不覆盖 E0924：

- `manifest.json`：B0 文件与 config、源码和各元数据 SHA256、环境版本、前后 state 哈希。
- `B0/`：加入任何辅助模块前的原始分数和八个测试集指标。
- `G30_<arm>/auxiliary_best.pth`、`history.json`、`exports/*.npz`、`result.json`：独立证据 checkpoint、训练/选择过程、成对分数和一致性证据。
- `decisions/`：三个独立决策 artifact、holdout 与各测试集指标及分数。
- `all_results.json`：每组状态和全部指标。NPZ、JSON 与证据 checkpoint 使用临时文件后 rename，避免正常中断留下半个正式导出。

本地契约测试使用随机初始化的小 CLIP，不下载预训练模型；检查 eager/SDPA、K=1/4/8、LoRA、训练与 checkpoint reload，以及 cache-only 的决策过程。通过这些测试不能代替服务器 GPU 上的真实实验或证明增强 AUC 提升。

2026-10-01 本地验证：PyTorch 2.5.1 / Transformers 4.44.2 下完整测试集 **255 passed**；PyTorch 2.9.1 / Transformers 4.57.3 下新增 G30/E1001 测试 **34 passed**。两种 LoRA 后端（项目 custom 与 loralib）均验证；流程测试替换外部应用的模型/数据入口，内部运行真实 tiny CLIP、优化器、严格导出与 checkpoint 重载，并用实际临时 LMDB 检查环境生命周期。预检回归测试使用项目真实 train YAML 和各测试集对应的标签/元数据结构，覆盖 WDF/FFIW 缺失标签的启动错误。未运行服务器真实数据或 CUDA 训练。

```bash
python -m pytest tests/test_g30.py tests/test_e1001.py -q
```

如测试环境没有 `lmdb`，仅 LMDB 两个流程测试会跳过；普通 RGB 流程照常验证。决策拟合失败独立记为 `FAILED`，保留成功的证据训练及其他决策结果，并返回非零退出码。
