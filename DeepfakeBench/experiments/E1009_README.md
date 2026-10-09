# E1009：冻结 LoRA，训练 CLS、证据 token 和分类头

在 `DeepfakeBench` 目录、现有服务器训练环境中运行：

```bash
# 查看 28 组配置，不加载模型、不训练
python experiments/run_e1009.py --dry_run

# 顺序完成 28 组训练和评估
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_e1009.py > nohup_E1009.log 2>&1 &
```

CLIP 权重位置不同，可加 `--clip_pretrained_path /path/to/clip`。数据位置继续使用现有训练环境的配置。

## 做什么

从原 CLIP 预训练权重重新开始，检验冻结 LoRA 后，只学习原 CLS、新增证据 token 和分类头的效果。每组独立初始化、独立训练；不加载已训练 B0、G25 或 G25v2 检查点。

主组保留原 LoRA 模块，但冻结 `lora_A`、`lora_B`。初始化时 `lora_B=0`，所以 LoRA 的增量为零，训练期间也保持零；它不参与参数更新。保留模块便于沿用原模型结构和严格检查点加载。

所有组都训练原 CLS embedding、4 个新增 token、原 CLS 分类头及新增 token 的独立分类头。区别仅为额外开放哪些 backbone 参数：

| 方法 | 额外更新的 backbone 参数 | 覆盖设置 | 两版合计 |
|---|---|---|---:|
| TOKENS（主组） | 无 | M00/M10/M01/M11、A00/A10/A01/A11 | 16 |
| LATE_LORA | 最后 4 个 block 的 LoRA | M01、M11 | 4 |
| LAYERNORM | 全部 LayerNorm 的 weight、bias，含前后归一化 | M01、M11 | 4 |
| ALL_LORA | 原全部 24 层的 LoRA | M01、M11 | 4 |

`G25` 使用原联合反传；`G25V2` 隔离证据损失对原参数的梯度。主组仍训练原 CLS，因此两版仍有区别：G25 的证据损失可以更新 CLS，G25v2 的证据损失不能更新 CLS。

M/A 分别表示 max/all 证据监督；注意力编码第一位表示 CLS 能否直接读新增 token，第二位表示 patch 能否直接读新增 token。新增 token 均可读原 CLS、patch 和新增 token。

组名示例：`G25_TOKENS_M01`、`G25V2_TOKENS_A11`、`G25_LATE_LORA_M01`、`G25V2_LAYERNORM_M11`。可指定子集：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run_e1009.py \
  --arms G25_TOKENS_M01 G25V2_TOKENS_M01
```

## 保持相同的实验条件

沿用 G25/G25v2 的数据、预处理、增强和训练协议：FF++ c23 训练，224 输入，seed=1024，v1 sampler real ratio=0.30，Adam lr=0.0002、weight_decay=0.0005。`nEpochs=10` 沿用原循环，实际执行 epoch 0–10。

Block 20 输入处加入 K=4 个 token。分类损失、证据损失和多样性约束保持原设置：证据权重 1、多样性权重 0.01；固定评分为 `0.5 * CLS + 0.5 * max-token`。仅改变可更新的参数范围，不调整学习率、损失或融合权重。

仍由 Celeb-DF-v2 的融合分数**帧级 AUC**选择检查点；每组在同一个检查点上报告 fused、CLS-only 和 evidence-only。后两种只用于诊断，不重新选点、不根据终评结果选择评分方法。`--skip_readout_diagnostics` 可仅做融合评估。

最终评估 WDF、FFIW、Celeb-DF-v2、DeepFakeDetection、DFDC、DFDCP、DeeperForensics-1.0 和 FF++。七集平均排除 FF++、包含参与选点的 CDF-v2；`AUC_cross` 仍仅指 CDF-v2 与 DFDC 均值。`video_auc` 沿用历史 G25 的 basename 视频分组，不能直接与 E1001 的 full-path 指标相减。

## 产物与验证

输出位于 `experiment_results/E1009/seed.../`。保存每组训练/评估配置、检查点、分支评估和结果，根目录保存源文件 SHA-256、Git 版本及 `all_results.json`。现有训练日志记录实际可训练参数名和数量。训练或评估失败会记录状态，入口返回非零。

```bash
python -m pytest tests/test_e1009.py tests/test_e1009_runner.py \
  tests/test_e1009_integration.py tests/test_g25.py tests/test_g25_runner.py \
  tests/test_g25v2.py tests/test_g25v2_runner.py -q
```

测试使用随机初始化的小型真实 CLIP 与 LoRA，无需下载模型或读取训练数据；验证冻结参数在 Adam 更新后不变、token 梯度保留、其他方法只更新指定参数、两版损失/评分及旧检查点结构兼容。CPU 合约测试通过不代表服务器训练已完成，也不代表有 AUC 结果。

2026-10-09 本地全套 `pytest tests -q`：682 通过、3 跳过；跳过项为其他实验依赖 LMDB 的测试。环境为 Python 3.12、torch 2.5.1、transformers 4.44.2、loralib 0.1.2。本机缺少完整 CLIP 权重和 RGB 数据，尚未启动 E1009 正式训练。
