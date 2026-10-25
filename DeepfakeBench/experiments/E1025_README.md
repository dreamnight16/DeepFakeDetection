# E1025：软梯度分配与 nnPU 消融

两条实验线分别运行，不将软分配与 nnPU 叠加。默认共 16 组，每组独立从 CLIP 预训练初始化，不复用其他组的 detector checkpoint。

| 实验线 | 默认组别 | 训练目标 |
| --- | --- | --- |
| B0 | 原单 LoRA，rank 4 | 原来的整批平均 CE |
| SOFT | R2F2、R4F4、R2F6、R6F2，各取 β=0、0.25、1，共 12 组 | 两类分别取 CE 均值；按 β 分配 LoRA 梯度 |
| PU | 原单 LoRA，prior=0.1、0.3、0.5，共 3 组 | 用给定的 nnPU 公式替换训练 CE |

## 软分配

沿用 E1021 的 `W0 + WR + WF`。两支 LoRA 始终共同前向，每支缩放为 4。真实支接收 `∇(Lreal + β·Lfake)`，生成支接收 `∇(β·Lreal + Lfake)`，共享分类头接收 `∇(Lreal + Lfake)`。这里是每步确定性梯度加权，没有随机路由。

β=0 是同结构硬分配对照，β=1 是同结构完整梯度对照。它们的损失都是 `mean(real CE) + mean(fake CE)`，与 B0 的 `mean(batch CE)` 不同；默认 batch 为 10 real / 22 fake，不能把相对 B0 的差异全部归因于双分支结构。比较同 rank 的不同 β 才是在隔离梯度分配。

## nnPU

严格采用所给公式，`0=Real`、`1=Generated`，令 `s=logit_real-logit_fake`：

```text
positive_risk = prior * mean(softplus(-s_real))
negative_risk = mean(softplus(s_generated)) - prior * mean(softplus(s_real))
loss = positive_risk + clamp(negative_risk, min=0)
```

这是直接截断版本；不额外加入作者训练算法中的负风险反向修正、CE 混合或样本重标注。负风险小于零时，整个生成子批对该损失的梯度归零。`prior` 表示 U 集合中真实类的先验，与 `sampler_real_ratio` 无关；将已标注伪造样本作为 U 是本实验的启发式设定，prior 的真实混合假设未经验证。[nnPU 论文](https://proceedings.neurips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf)、[作者实现](https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py)。

训练 batch 必须同时含两类，沿用 balanced sampler。验证 batch 可以只有一类，因此验证时 `overall` 记录普通 CE；`real_loss/fake_loss` 始终是 CE 诊断值。训练 `overall` 是 nnPU，不能与验证 CE 直接比较。checkpoint 仍按 Celeb-DF-v2 frame AUC 选择。

## 共同条件

只训练对应 LoRA 和完整的原普通二分类头；原 CLIP 参数、CLS embedding 冻结。不增加 token、损失项、梯度裁剪或数据增强技巧。沿用 FF++ c23、224 输入、8/8 帧、batch 32、real ratio 0.3、seed 1024、Adam `lr=2e-4 / wd=5e-4`、epoch 0..10（11 轮），保留 B0 原训练增强，关闭 Mixup、多裁剪、自适应阈值。使用单 GPU 普通 Adam。

## 运行

在 `DeepfakeBench` 目录运行：

```bash
# 打印全部 16 组；不导入 ML 依赖、不创建结果目录
python experiments/run_e1025.py --dry_run

# 完整实验，按顺序训练和评估
python experiments/run_e1025.py --clip_pretrained_path /path/to/clip

# 分开运行两条实验线；B0 单独训练一次即可
python experiments/run_e1025.py --arms B0
python experiments/run_e1025.py --arms SOFT
python experiments/run_e1025.py --arms PU

# 自定义消融取值或只运行某个组合
python experiments/run_e1025.py --arms SOFT --ranks R4F4 R2F6 --betas 0 0.1 0.25 0.5 1
python experiments/run_e1025.py --arms SOFT --ranks R4F4 --betas 0.25
python experiments/run_e1025.py --arms PU --priors 0.1 0.2 0.3 0.4 0.5
```

各次运行均可指定 `--clip_pretrained_path`、`--dataset_json_folder`、`--rgb_root`、`--output_dir`、`--batch_size`、`--sampler_real_ratio`、`--seed`、`--n_epochs`；比较组保持条件一致。每次调用创建独立运行目录，不会自动跳过或复用历史组，重新执行同一命令会重新训练。

输出位于 `experiment_results/E1025/seed.../`，包含各组训练/评估配置、checkpoint 路径与 SHA256、逐帧 NPZ、`result.json`、总表 `all_results.json`、源码与数据元信息 SHA256。组名如 `R4F4_B0.25`、`PU_P0.3`。训练失败记为 `TRAIN_FAILED`，评估失败记为 `EVAL_FAILED`，任何组失败时整体返回非零。

`video_auc`/`video_auc_fullpath` 按完整父目录 video_id 聚合帧分数；旧 basename 结果单列 `video_auc_legacy`。七库均值含用于选模的 Celeb-DF-v2；独立六库均值为 DFD、DFDC、DFDCP、DeeperForensics-1.0、WDF、FFIW，排除 Celeb-DF-v2 和训练源 FF++。严格导出遇到坏图直接失败，不随机换图。

本地测试使用 CPU 上的随机微型 CLIP，验证公式、各层梯度、训练入口、配置复建与计算图释放，不下载预训练权重。真实数据训练、CUDA 显存占用和 AUC 结果仍须服务器运行后确认。
