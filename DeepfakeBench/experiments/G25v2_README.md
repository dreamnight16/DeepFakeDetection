# G25v2：保持追加 token 结构，隔离证据损失对原参数的梯度

## 从 G25 结果出发

G25 seed=1024：B0 的 AUC_cross 为 92.7642%，七集均值为 93.6633%；最佳新增组分别只有 91.9162%（A11）和 93.2069%（M01）。仅解冻 CLS 的 C0 也下降。8 个新增组全部在第一次验证（epoch 0、step 286）选到最佳点，后续验证明显退化。

这些结果说明当前联合训练配方没有达成整体改进，但还不能确认是梯度冲突、证据分支本身还是 0.5 融合造成退化。v2 只改变一个训练因素，并补充同检查点的分支诊断，不将假设写成已证实原因。

## 保留的设计与唯一默认改动

保留原 CLS 可训练、Block 20 输入处追加 K=4 个 token、原 ViT+LoRA 高层注意力/FFN、每个新增位置独立分类头。00/10/01/11 四种 mask、max/all 两种监督及 B0/C0 对照均保留。普通 patch 不新增分类头。

默认仍是原学习率 0.0002、原 CLS 学习率、证据 CE 权重 1、多样性权重 0.01、0.5 CLS + 0.5 max-evidence 评分。数据、增强、sampler、seed、选点和评估口径沿用 G25。**没有同时改 token 数、层位置、学习率、损失权重或融合权重。**

唯一默认训练改动：

| 参数 | 旧 G25 的梯度来源 | G25v2 的梯度来源 |
| --- | --- | --- |
| 原 ViT LoRA | CLS CE + 证据 CE + diversity | 仅 CLS CE |
| 原 CLS embedding | CLS CE + 证据 CE + diversity | 仅 CLS CE，仍可训练 |
| 原 CLS 分类头 | CLS CE | CLS CE |
| 新增 token | 证据 CE/diversity，以及 mask 允许时的 CLS CE | 同左 |
| 每个新增 token 的独立头 | 对应证据 CE | 同左 |
| 原冻结 ViT 参数 | 不更新 | 不更新 |

实现上，主路径正常计算并反传 CLS 损失；辅助路径对插入点前的特征 detach，同时用 **detach 的原层参数**重算插入点之后的高层。证据损失从这条辅助路径反传，仍可穿过固定算子更新新增 token，但不能更新原参数。仅 detach patch 特征是不够的，因为共享投影参数仍可能接收梯度。

使用 [PyTorch functional_call](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html) 临时传入 detach 参数，不复制模型权重、不增加注册参数、不更改原参数 requires_grad 标志。要求 PyTorch >= 2.0；依赖延迟导入，不影响旧 detector 的导入。原 G25 的 CLIP/LoRA dropout 为零，v2 对非零 dropout 明确报错，避免重算路径与主路径数值不一致。

训练时多算一次高层后缀（默认 4 个 block），前 20 层只算一次；推理仍只算一次。**新增参数量相对 G25 为零，但训练计算和激活占用增加。** 实际耗时/显存必须在服务器测量。

## “减少干扰”的准确边界

- 相同权重、输入、零 dropout 下，旧版与 v2 的前向分数和损失数值相同，变化在反向路径。
- 原参数梯度等于“同一带 token 前向，仅反传 CLS CE”的梯度。
- 00 下原 token 看不到新增 token，可以进一步验证原参数的更新不受证据损失影响；测试覆盖连续两步 Adam。
- 10/01/11 允许新增 token 改变原 CLS/patch 的前向值。即使隔离辅助损失，CLS CE 仍可能因这些前向变化产生不同梯度。不能称为与 B0 完全相同的训练。
- 原 CLS 按用户要求仍可训练，C0 的退化因素并未被偷偷取消。这一轮优先检验证据梯度的影响，不能保证解决全部退化。
- 默认融合仍为 0.5；原分支保留不保证融合性能提高。

## 运行方式

命令在 `DeepfakeBench` 目录执行。原 `run_g25.py`、原模型和已有结果保持原样，新模型注册名是 `effort_g25v2`，默认产物在独立 `experiment_results/g25v2`。

```bash
# 查看全部 10 组，不导入 ML 依赖、不训练
python experiments/run_g25v2.py --dry_run

# 先做 CPU 合约测试，无需模型下载或数据
python -m pytest tests/test_g25.py tests/test_g25_runner.py tests/test_g25v2.py tests/test_g25v2_runner.py -q

# 10 组独立训练；每个有 token 的组附带同检查点的三种评分诊断
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_g25v2.py --n_epochs 10 --seed 1024 > nohup_G25v2.log 2>&1 &

# 先只验证 00 梯度隔离，保留两种监督
CUDA_VISIBLE_DEVICES=0 nohup python experiments/run_g25v2.py --arms M00 A00 --seed 1024 > nohup_G25v2_00.log 2>&1 &

# 同实现的联合梯度对照，其他参数相同
python experiments/run_g25v2.py --arms M00 A00 --aux_grad_mode joint
```

`--skip_readout_diagnostics` 只跑主融合评估，可减少评估开销。默认每个有 token 的组会在同一个按 CDF-v2 **融合分数帧级 auc** 选出的检查点上，另用 `testall.py` 评估 CLS-only 与 evidence-only；各模式目录隔离。诊断不会重新选检查点、不会改变训练损失，也不根据测试集自动挑选评分方式。分支诊断失败时总状态为 EVAL_FAILED，primary_status 单独记录主评分是否成功。

### 直接诊断已有 G25 检查点

v2 没有新增权重键，可严格读取旧 G25 state_dict。提供该检查点原来的 eval_config.json，恢复它的 K、mask、层位置、损失和 seed，不能只猜默认值。脚本同时核对同目录 result.json 的组名、seed、结构参数和 checkpoint 相对路径；迁移数据时保留整个 arm 目录（含 logs）。这可防止误选另一组路径；旧结果没有权重内容哈希，不能检测原路径下的文件被替换。以下路径需替换成服务器对应组的真实路径：

```bash
python experiments/run_g25v2.py --arms A11 \
  --checkpoint /path/to/A11/logs/effort_g25_run/test/avg/ckpt_best.pth \
  --source_config /path/to/A11/eval_config.json
```

此模式只评估，不训练；原 G25 的训练模式会记录为 joint。保存 fused/cls/evidence 三种结果，不将它们计为三个种子。当前默认 multi_crop=false；若另行开启多 crop，每种评分沿用自身 argmax-confidence 的 TAA，因此分支比较还包含 crop 选择差异。

## 验证与决策

2026-09-23 本地结果：G25v2 核心 18 项、运行入口 10 项、旧 G25 33 项、G22/G23 8 项，共 **69 项 CPU 测试通过**。环境为 Python 3.12、torch 2.5.1+cpu、transformers 4.44.2、loralib 0.1.2。检测器测试只替换预训练加载及数据/指标外壳，内部执行真实小型 CLIP 与 LoRA；未运行服务器真实权重推理或训练。

CPU 测试应覆盖：四种 mask × 两种监督的前向等价和原参数梯度等价、aux-only 不产生原参数梯度、新 token/独立头可训练、00 两步 Adam 原参数更新等价、joint 对照旧版梯度等价、参数集合不变、严格旧 checkpoint 加载、训练与推理评分一致、5D TAA、配置透传、分支产物隔离及失败状态。

先看 CLS-only 是否恢复、证据-only 是否有独立排序能力，以及融合是否受益；再比较主预设评分的 AUC_cross、七集均值与 DFDC。不会仅凭 CDF-v2 改善宣布跨域提升，也不会依据最终测试集调整融合权重。当前实现只是验证“辅助梯度是否是退化来源”的可运行调整，尚无 v2 正式训练结果。
