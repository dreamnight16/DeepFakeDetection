# G22 / G23 交叉注意力消融

两组实验均沿用 G19-A：FF++ 训练、Celeb-DF-v2 选点、K=8、hidden=256、
depth=2、heads=8、dropout=0.1、mean readout、单一交叉熵、seed=1024。
不使用 hard argmax、fusion 或 diversity loss。

## G22：查询最后一个 ViT block 的输入

当前 G19 使用 `last_hidden_state`，它已经是最后一个 ViT block 的输出。
因此，若“倒数第一层”仍解释为该输出，实验会与 G19-A 重复。G22 将其操作化为：

```text
CLIP hidden_states[-2][:, 1:, :]
```

即最后一个 encoder block 的输入（倒数第二个隐藏状态，去除 CLS token）。除此之外，
查询模块与 G19-A 保持一致。模型注册名为 `effort_g22_last_block_input`。

运行：

```bash
python experiments/run_g22.py --n_epochs 10 --seed 1024
```

默认根目录：`experiment_results/g22_last_block_input/`。每次运行会创建独立的
`seed<seed>_<timestamp>/` 子目录，避免重跑时误用旧 checkpoint。

## G23：cross-only 轻量查询块

G23 仍查询最终层 patch token，仅将每个完整查询块

```text
self-attention -> cross-attention -> FFN
```

替换为：

```text
cross-attention
```

K、hidden、depth、heads 与 readout 均不改变。查询模块参数量由 2,373,634 降到
794,114，减少 66.54%；这里仅统计新增 query/readout 模块，不含共享 CLIP-LoRA
backbone。模型注册名为 `effort_g23_cross_only`。

运行：

```bash
python experiments/run_g23.py --n_epochs 10 --seed 1024
```

默认根目录：`experiment_results/g23_cross_only/`，同样按 seed 与时间建立独立子目录。
只有八个测试集均返回有限的 `video_auc` 才写入 `status=OK`；缺失指标会写入
`EVAL_FAILED` 并使启动脚本返回非零退出码。

## 验证

服务器安装 PyTorch 后执行：

```bash
python DeepfakeBench/tests/test_g22_g23.py
```

测试覆盖层选择、缺失 hidden states 时 fail-closed、输出形状、参数量、结构约束及
反向梯度。本地代码环境没有 PyTorch/pytest，因此当前只能执行静态编译和源码契约检查；
不能把本地结果视为真实模型前向或训练通过。
