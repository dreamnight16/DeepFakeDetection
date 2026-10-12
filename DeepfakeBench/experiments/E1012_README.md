# E1012：从预训练重新开始的 LoRA 与 token 联合训练

在服务器现有 `DeepfakeBench` 目录运行：

```bash
python experiments/run_e1012.py --dry_run

CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1012.py \
  > nohup_E1012.log 2>&1 &
tail -f nohup_E1012.log
```

每组从同一 CLIP 预训练权重重新初始化，不继续训练 E1010 的 B0。
CLIP 路径不同可加 `--clip_pretrained_path /path/to/clip`。
默认先独立训练 G0/B0，再运行 123 个对照：120 个训练配置和 3 个固定原型诊断。
全程在新的 `experiment_results/E1012/seed.../` 下保存结果，不覆盖旧运行。

## 训练条件

使用原 G18/G25/G25v2 的基础协议：FF++ c23、224 输入、每视频 8 帧、
seed=1024、batch=32、v1 balanced sampler real ratio=0.30、原数据增强。
Adam lr=2e-4、weight decay=5e-4；mixup、margin、频率输入、texture crop、rank loss 关闭。
`--n_epochs 10` 执行 epoch 0–10，共 11 轮，按 CDF-v2 的主读出帧 AUC 选点。

“不冻结 B0”指 LoRA 和对应分类模块参与训练。CLIP 原始参数仍冻结，不做全参数微调。
完整模型权重随 checkpoint 保存；每组独立初始化和选点，不共享训练后的权重。
E1010 的冻结辅助阶段使用 AdamW、无随机增强、10 轮；这里同时恢复了历史训练条件。
因此 E1012 与 E1010 的差值不能仅归因于解除冻结，应优先比较 E1012 内部同结构的监督对照。

## 配置范围与梯度

| 组 | 结构与训练范围 | 配置数 |
| --- | --- | ---: |
| G0 | 原 B0，训练 LoRA 和普通分类头，原 CLS embedding 冻结 | 1 |
| G1 | 原 G18 LFEQ；LoRA 与 LFEQ 联合训练，原 CLS 和旧分类头冻结 | 18 |
| G2_G25 | 四种 mask × max/all；LoRA、CLS、分类头、patch 投影、token 和证据头联合训练 | 48 |
| G2_G25V2 | 同结构；原路径继续训练，隔离证据损失对 LoRA/CLS 的梯度 | 48 |
| G3 | LoRA、原分类头与独立交叉注意力解码器联合训练 | 6 |
| G4 | 同一个 G0/B0 上的全部、随机、遮挡贡献最高 patch 原型，不训练模型 | 3 |

G1 的 L1/L2/L3 分别读取决策与最高假概率证据各 0.5、仅决策、仅证据。
使用 LFEQ 原读出，不再与独立 B0 作外层 gate 融合。三者训练目标相同，读出和选点不同。

G2 在零起始 block 20 的主序列插入 4 个 token。
00/10/01/11 控制 CLS/patch 是否读取新增 token。
G25 的所有目标可反传到 LoRA/CLS；G25v2 的全局分类目标仍更新原路径，
证据、MIL、多样性及局部对比目标通过 detach 前缀和冻结参数的晚层重放隔离。
推理使用同一主路，评分为 0.5 × CLS 概率 + 0.5 × 最高假概率证据。

G2 保留 E1010 的恒等初始化 patch 投影，让隔离模式下的局部损失能够训练有效模块。
投影同时参与主路和证据重放；六种监督共用该结构。
这符合 G25/G25v2 的梯度方式，但包含原 G25 没有的投影，不是原模型的严格复现。
注意力多样性系数仍为 0.01；需要注意力权重的层调用使用 eager，调用后恢复原后端。
没有通过关闭多样性损失绕过原 G2 的 SDPA 报错。

G3 默认读取 block 20 输入、4 个 query、hidden=256、4 heads、2 blocks。
主路与解码器都参与训练，最终仍使用 E1010 的固定 gate：仅 B0/CLS 分数在
0.3–0.7 内融合，辅助权重最高 0.5。这里训练后的 CLS 分数不等于独立 G0/B0。

## 六种监督

| 后缀 | 局部 patch 对比 | 证据监督 |
| --- | --- | --- |
| PLAIN | 无 | 原 max/all 分类 |
| UNIFORM | 等权 | 原 max/all 分类 |
| FL | 伪造似然加权 | 原 max/all 分类 |
| MIL | 无 | 真图所有证据真、假图存在假证据 |
| UNIFORM_MIL | 等权 | MIL |
| FL_MIL | 伪造似然加权 | MIL |

沿用 E1010 的局部目标：batch 真假 patch 均值原型差生成停止梯度的软权重，
每假图权重均值归一为 1；每图最多均匀采样 16 个 patch，温度 0.1、系数 0.14。
MIL 温度为 0.5。它们是图像标签产生的弱监督，不等于局部篡改标注。
训练局部对比需要每 batch 至少 1 真、2 假；验证不计算训练局部对比项，允许单类 batch。
同 mask 的 M/A MIL 目标重复，不能当成独立方法或多 seed 证据。

只运行选定配置，例如：

```bash
python -u experiments/run_e1012.py --arms \
  G1_L1_PLAIN G1_L1_FL_MIL \
  G2_G25_M00_FL_MIL G2_G25V2_M00_FL_MIL G3_FL_MIL
```

G4 使用本次独立 G0，不从各联合模型构建原型。三个方法共享 FF++ train 的
32 真、32 假训练帧池，测试不计算遮挡贡献，不更新模型，不加入迭代自训练。

## 评估与结果

所有读出使用同一个选定 checkpoint，不分别选点。
主指标是完整视频路径内帧分数均值的 video AUC，历史 basename 指标另存。
六独立域均值等权包括 WDF、FFIW、DeepFakeDetection、DFDC、DFDCP、DeeperForensics-1.0。
CDF 用于选点，FF++ 为训练同域，二者单独报告。

保存每组配置、完整 checkpoint 路径、成对样本导出和结果；manifest 记录源码、
元信息与训练条件。纠错/误伤相对独立 G0/B0 的同路径预测计算，阈值为 0.5。
训练后的 CLS 读出与独立 B0 分开记录，不要求它们分数不变。
G4 复用 E1010 原型入口，子运行记录位于 `G4_diagnostics/`，由 E1012 汇总。

## 本地验证

```bash
python -m pytest tests/test_e1012.py tests/test_e1012_runner.py \
  tests/test_e1012_integration.py tests/test_e1010.py -q
```

本地 tiny CLIP 测试只验证训练、梯度、重载和入口契约，不代表服务器 GPU 实验已完成。
