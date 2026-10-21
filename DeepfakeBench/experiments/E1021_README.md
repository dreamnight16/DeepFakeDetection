# E1021：真假双 LoRA

在 B0 的每层注意力 `q/k/v/out` 投影中，用两个可独立设 rank 的分支替代原 LoRA：`W0 + WR + WF`。两分支在训练和推理时始终参与前向；真实样本只更新 WR，伪造样本只更新 WF，同时保留两分支对前层输入的梯度。分类头仍是共享的普通二分类头，整个头参与训练；原始 CLIP（包括 CLS）冻结。各 rank 的缩放固定为 4，沿用 B0 的 `alpha/r=16/4`。

双分支损失固定为 `mean(real CE) + mean(fake CE)`，两类都更新共享头。B0 保留原来的 `mean(batch CE)`，因此双分支同时改变了分支结构与类别损失权重；默认 batch 中 10 real / 22 fake，不能把与 B0 的差异全部归因于结构。

默认顺序运行 B0、R2F2、R4F4、R2F6、R6F2。每组从 CLIP 预训练重新初始化，不使用其他组 checkpoint。沿用 FF++ c23、8/8 帧、batch 32、real ratio 0.3、seed 1024、Adam `lr=2e-4 / wd=5e-4` 和 epoch 0..10；保留 B0 训练增强，禁用 Mixup、多裁剪和自适应阈值。仍按原 trainer 验证时点的 Celeb-DF-v2 frame AUC 选择 checkpoint。

```bash
# 仅打印方案，不导入 ML 依赖、不创建输出文件
python experiments/run_e1021.py --dry_run

# 完整默认实验
python experiments/run_e1021.py --clip_pretrained_path /path/to/clip

# 只运行指定组
python experiments/run_e1021.py --arms B0 R2F6

# 自定义 rank
python experiments/run_e1021.py --arms CUSTOM --real_rank 3 --fake_rank 5
```

可指定 `--dataset_json_folder`、`--rgb_root`、`--output_dir`、`--batch_size`、`--seed`、`--n_epochs` 和 `--sampler_real_ratio`；改变默认条件时，需与对照保持一致。

每组保存训练/评估配置、选中 checkpoint 的路径与 SHA256、严格逐帧 NPZ 和结果。`video_auc` 与 `video_auc_fullpath` 均按完整父目录 video_id 平均帧分数；`video_auc_legacy` 单列旧 basename 口径。逐帧导出沿用原 RGB 加载和归一化，图片读取失败直接记为 `EVAL_FAILED`，不随机替换。七库均值包含用于选模的 Celeb-DF-v2；独立六库均值排除 Celeb-DF-v2 和训练源 FF++。运行目录保存参数、源码/元数据 SHA256 与各组状态；任何组失败时进程返回非零。

显存短检：`python experiments/check_e1021.py --steps 10`。使用相同 batch、分辨率、rank、损失、Adam 和实际 `Trainer.train_step`，保留真实日志记录器的损失引用；仅使用固定的随机输入，不读取数据、不保存 checkpoint 或实验结果。逐批打印 CUDA 实际分配量，结束时汇报训练峰值（不含模型构建阶段），预热后分配量增长超过 16 MiB 即报错退出。这个短检检验计算图释放和显存占用，不能代替真实数据训练、验证和磁盘空间检查。已完成的 B0 应保留；后续可用 `--arms R2F2 R4F4 R2F6 R6F2` 只检查或训练剩余四组。

反向传播的最后一次 `autograd.grad` 必须覆盖所有可训练路径以释放骨干网络的中间张量，再仅取其中分类头的梯度；真假 LoRA 仍保留前两次按类别计算的梯度。只对分类头求最后一次梯度，会使日志记录器持有此前保留的骨干计算图，造成跨 batch 显存累积。
