# G21：同源配对的稳健视频排序

## 范围与当前状态

G21 独立实现 PIVR，**A/B/C/D 一次完整实现、一次命令顺序执行，不设置阶段门槛**。

按最新要求，测试方法与 G18/G19 保持一致。正式训练和运行验证均应在服务器进行。本次本机交付采用静态语法检查及代码审查；下面的服务器张量、恢复和集成测试尚未执行，不能称为已通过，也没有 G21 性能结果。

## 实验矩阵

| 臂 | 基础目标 | 配对目标 | 配对关系 |
|---|---|---|---|
| A | 视频级 CE | 关闭 | 使用相同配对数据，只不加关系监督 |
| B | 视频级 CE | 完整窗口 mean | 同源对应 |
| C | 视频级 CE | 完整窗口 mean | 同方法内无固定点置换，只改损失索引 |
| D | 视频级 CE | 完整窗口 group-softmax | 同源对应 |

默认 λ=1、margin=0、τ=0.5。A 跳过配对项计算，避免 `0*NaN`。四臂共用初始化、视频/帧计划、增强、完整窗口、前向次数与优化器更新预算。

每个微批次：一个伪造方法、4 对不同来源视频、每视频2帧、reference/JPEG两个视图，共32张前向输入。每个完整窗口包含四个 FF++ 方法，单遍128张；所有臂都执行两遍，因此每个 optimizer update 为 **256 张前向、128 张反向输入**。默认6000次窗口更新。这不是旧训练器的“batch=32、10 epoch”，训练成本必须按真实计数比较。

A 是当前配对/视频训练流程的匹配基线，不能与 G18/B0 当作完全相同的训练配置；B/C/D 对 A 的比较用于隔离新目标的作用。

## 与以前保持一致的测试设置

| 项目 | G21 设置 |
|---|---|
| 训练来源 | FF++ train，DF/F2F/FS/NT 四方法全部保留 |
| 验证来源 | Celeb-DF-v2，复用旧 `DeepfakeAbstractBaseDataset(mode='test')` |
| 最佳检查点 | 按旧 `auc`（帧级）选点；日志同时记录所选点的 video_auc |
| 最终入口 | **直接调用原 `testall.py → training/test.py`** |
| 测试集 | WDF、FFIW、CDF-v2、DFD、DFDC、DFDCP、DeeperForensics、FF++ |
| 取帧与预处理 | 原 Dataset 实现、`frame_num.test=8`、c23、224、原 CLIP 归一化 |
| 多 crop / 动态阈值 | false / false，与 G18/G19 默认一致 |
| 指标与视频分组 | 原 `metrics.utils.get_test_metrics`，包括历史视频分组方式 |
| 汇总 | 原八集 average 保留；A_cross=CDF/DFDC均值；mean7不含FF++ |

**没有切换为留一方法验证、均匀全视频取帧、新视频 ID 分组或新的 `video_auc` 选点。**已有方案中这些改进不进入 G21 默认实验，避免换口径后难以横向对比。CDF 既用于选点又参与最终测试的历史限制也保留并显式记录。

原测试文件仅增加可选输出/路径参数。未传入时保留原默认行为：

- `testall.py` / `training/test.py`：可选 `artifact_dir`、`dataset_json_folder`、`data_root`。
- 原 Dataset：可选 `rgb_root_override`，未设置仍使用原来的数据路径。
- 原 EffortDetector：可选 `clip_pretrained_path`，未设置仍使用原模型路径。
- **未修改原取帧算法、原指标函数和预测融合规则。**

G21 测试使用独立的概率文件和图片目录，避免多个实验写同名 `/tmp/effort_probs_*.npy` 或 `prob_density.png`。原 `testall.py` 即使只对某个数据集失败、整体返回0，G21也会检查 warning 和全部八集指标，拒绝将部分测试写为成功。

## 模块与数据流

```text
experiments/verify_g21_pairing.py
    → 图像内容核验 A/B 候选 + 时间线兼容性 → pair_map.json（含逐视频排除与审计）

experiments/build_g21_manifest.py
    → 经核验的映射规则 / pair_map + 原 FF++ JSON → pairs_train.jsonl

experiments/run_g21.py run
    → preflight（不加载模型）
    → seed/arm 独立进程
        → training/g21/data.py       配对计划与共享增强
        → 原 EffortDetector          RGB/LoRA/pooler/Linear
        → losses.py + replay.py      完整窗口两遍梯度
        → legacy.py                 CDF 原数据集、原指标选点
        → best_testall.pth           原始 state_dict，旧 test.py 可直接加载
        → 原 testall.py              八数据集测试
    → all_results.json / aggregate.json
```

`training/g21/evaluation.py` 中的完整 ID 评测仅用于合成夹具测试；**不是正式 G21 测试入口**。

## 服务器准备

以下命令均在服务器 `Effort-AIGI-Detection-main/DeepfakeBench` 目录执行，沿用已经能运行 G18/G19 的 Python/CUDA 环境。

```bash
# 只添加测试工具；不要把服务器的 CUDA torch 换成 CPU 版
python -m pip install -r requirements-g21-tests.txt

# 合成张量/轻量模型验收，不下载或加载真实 CLIP
python experiments/smoke_test_g21.py

# 可选：覆盖率报告。目标 >=80%，当前尚无服务器报告。
python -m pytest tests/g21 --cov=training/g21 --cov-report=term-missing
```

需要 Python 3.10+、PyTorch 2.2+、numpy、OpenCV、PyYAML 及原 DeepfakeBench/transformers/loralib 依赖。两遍回放和检查点读取使用 `weights_only=True`，只读取张量与基础类型。

### 构建真实的同源映射

```bash
python experiments/build_g21_manifest.py template --output /path/to/g21_pair_map.json
```

模板默认全部未验证。对每个方法填入以下字段，**核验前不要设为 true**：

```json
{
  "verified": true,
  "reference_component": 0,
  "same_original_index_verified": true,
  "evidence": "填写实际使用的生成元信息、对应规则来源及核验记录"
}
```

`reference_component=0` 只是字段示例，不是本实现对 FF++ 命名的默认判断。`001_870` 的真实参考究竟是哪一侧，需要确认保留背景/时间线的原视频；不能仅凭编号位置猜测。

同样，只有确认原始帧号对应时，才允许 `same_original_index_verified=true`。不同视频列表长度或抽样位置不一致时，构建器取已验证帧号的交集，不按列表位置 zip。

不满足同帧号对应的个例可在 `overrides` 下显式提供：

```json
{
  "FF-F2F/001_870": {
    "verified": true,
    "reference_component": 0,
    "same_original_index_verified": false,
    "evidence": "填写该视频时间映射的核验依据",
    "frame_matches": [[0, 0], [14, 19], [29, 38]]
  }
}
```

上述帧对应也仅演示格式，不代表实际正确映射。映射需单调、一对一，且所有索引在 JSON 中存在；每对不足2帧将记录排除原因。若某方法没有足够4个不同来源，preflight失败。

不需要某个视频（或整个方法）时，在规则或 `overrides` 条目上写显式排除标记。**没有标记且 `verified != true` 仍然直接抛错**，排除是显式选择而非默认行为：

```json
{
  "FF-DF/001_870": {
    "excluded": true,
    "exclude_reason": "manual_review",
    "exclude_detail": {"note": "背景与两侧均不符，留待人工核对"}
  }
}
```

`exclude_reason` 必须是非空字符串。构建器记录该条排除后继续处理其余条目。按方法排除会让该方法无配对，随后触发"某训练方法无有效配对"的硬失败——这是刻意的：G21 不允许静默丢掉一个方法。

```bash
python experiments/build_g21_manifest.py build \
  --dataset-json preprocessing/dataset_json/FaceForensics++.json \
  --pair-map /path/to/g21_pair_map.json \
  --output /path/to/g21_manifest
```

输出 `pairs_train.jsonl`、`manifest_report.json`，包含四方法配对数量、排除原因及来源文件哈希。`manifest_report.json` 的 `funnel` 给出每方法与总体的 见过的伪造视频 / 成对 / 排除（按原因码）计数；`insufficient_frame_intersection` 的排除另带 `real_available`/`fake_available`/`intersection` 计数。原测试 JSON 不转换、不改写。

### 用图像内容自动核验 A/B 候选

FF++ 文件名 `<target>_<source>` 的官方顺序只是候选提示：抽取帧的索引集合是纯视频长度指纹，在真实视频中高度混淆，不能单独用于核验。本实现因此以图像内容为主要证据，长度/网格信号仅降级为候选提示与时间线兼容性检查，二者不一致时作为一等输出报告，而不是隐藏。

```bash
python experiments/verify_g21_pairing.py \
  --dataset-json preprocessing/dataset_json/FaceForensics++.json \
  --data-root /path/to/FF++/images \
  --output /path/to/g21_pairing_v1 \
  --require-usable
```

第一次务必带 `--require-usable`：不带时即使某方法一个可用配对都没有，进程仍返回 0，只有构建 manifest 时才会以一句难懂的错误暴露出来。`--output` 目录**不允许已存在**（`FileExistsError`），所以每次改阈值重跑都要换一个新目录名，并把 `pair_map.json` 与 `pairing_audit.json` 一起留作 provenance。

**证据必须落在构建器真正会用的帧上。** 每个距离都在"伪造帧与候选帧**原始帧号完全相同**"的交集内测量——`build_ffpp` 把 `same_original_index_verified` 展开成 `[[i, i] …]`，所以只有这个交集能为该断言作证。近似对齐（归一化时间、邻帧替代）一律不用：设计明确禁止断言一个证据并未建立的帧号对应关系。某候选在该帧号没有帧时，该行留空而不是拿邻帧顶替。

只比较排除伪造掩码后的可用背景区域（掩码默认向外膨胀 `--mask-dilation 8` 像素，224 尺度），灰度化、缩放到 224、以 sigma=1 高斯模糊容忍亚像素错位，取平均绝对差。掩码缺失或不可读时回退到中心人脸框并在审计中记录模式；可用像素比例低于 `--eligible-floor`（默认 0.10）的帧记为不可用，不计入平均。

两种判定模式：

- **head_to_head**：两个候选与伪造视频共有 ≥ `--min-verification-frames`（默认 8）个同号帧时，在**同一批帧**上正面比较。要求中位数与去极值均值指向同一候选，且 `--win-rate-min`（默认 0.9）、`--margin-min`（默认 0.5，`r=(d_loser-d_winner)/(d_loser+1e-6)`）与 `d_winner <= --max-winner-distance`（默认 12.0，0..255 尺度）同时满足。绝对上限防止两个候选都很远时 margin 仍显得决定性（例如 d_A=100、d_B=180 仍有 r=0.29）。
- **single_side_confirmation**：共有帧不足 8 帧时，两侧各自在自己的同号交集上判定。**但两侧仍然共有的那些帧依然具有决定性**：确认侧必须在这些共有帧上不被对方压过（`shared_agreement`）。没有这一条时，只要真正的内容来源恰好可测帧更少，一个"近邻但错误"的场景就会胜出——这是本设计早期版本的真实缺陷。仅当两侧**一帧共有都没有**（真实索引中占 3/2877）时才退化为只按绝对上限判定，因此该上限是 4.0 而不是 12.0。两侧都成立时判 `ambiguous_content` 而不是挑一个。

`margin_ratio` 只在两个候选**测于完全相同的帧集**时才计算：跨不同支撑比中位数会在支撑更大的一侧显得决定性，方向可能是反的。共有帧数记录在 `shared_pairs`，共有帧指向哪一侧记录在 `shared_agreement`（0=A、1=B、-1=平手、null=无共有帧）。

无论哪种模式，**任何决定都不得建立在少于 8 个可测帧之上**：某些帧不可读时，中位数不再是无支撑的噪声，而是直接判 `unusable_frames`，不会把胜利判给证据更少的一方。

最后与内容赢家做时间线 Jaccard 检查（`--timeline-jaccard-min` 默认 0.9）；不匹配的视频以 `timeline_mismatch` 排除，不用邻居/时间规整规则强行对齐。同时记录 `timeline_coverage = |伪造帧 ∩ 赢家帧| / |伪造帧|`，用来区分"真正不相交的时间线"与"网格嵌套、赢家多抽了帧"这两种都被 Jaccard 拒绝的情形。单个条目格式错误（帧号重复、无法解析等）只排除该条目并记录，不会中断整轮扫描；**帧号列表的顺序不参与校验**——FF++ 索引里有 69 个真视频的帧号是有效升序的乱序排列，而帧号是按值寻址的，拒绝它们会白白损失 184/2877（6.4%）个训练键，因此改为接受并在 `frames_out_of_order` 中记录。

输出：

- `pair_map.json`：可直接交给 `build_g21_manifest.py build`。verified 的视频写入逐视频 override；其余写入显式排除标记（`excluded`/`exclude_reason`），构建器据此记录排除并继续，不会静默丢弃方法。
- `pairing_audit.json`：逐视频记录 key、方法、所选分量、判定模式、d_A/d_B、中位数与去极值均值、win_rate、margin_ratio、可用像素比例、掩码模式、估计长度、两侧同号交集大小、实际测量帧数、是否确认、共有帧数与共有帧指向（`shared_pairs`/`shared_agreement`）、帧号是否为乱序存储、时间线 Jaccard 与 coverage 与判定；并给出 total/verified/ambiguous/weak margin/not closer/timeline mismatch/insufficient intersection/other/usable pairs 漏斗（按方法与总体）。`distributions` 只统计 **verified** 的键：`d_winner_verified`（被接受参考的距离）与 `d_rejected_on_verified`（同一批视频上被否候选的距离）可直接对比，`margin_ratio_verified` 同理；被拒绝的视频不进入任一分布，否则会把"接受的参考"和"否掉的候选"混进同一个直方图。`distinct_references` 给出每方法实际用到的不同参考视频数。`warnings` 给出单条记录看不见的全局征兆：掩码大范围不可用（可能是极性反了）、整个运行没有可读掩码、以及**共有帧为 0 因而只靠绝对上限判定的 verified 键数**。

命令行同时打印上述漏斗、warnings 与分布摘要，便于先看数据再选阈值。`--require-usable` 使任一方法没有任何可用配对、或不同内容来源数低于 `--min-sources`（默认 4，preflight 的 microbatch 要求）时返回非零。相同输入产生逐字节一致的输出，不写入时间戳；任何未 verified 或未通过时间线检查的条目都不会产生配对。

> 阈值仍是未经真实图像校准的保守默认值；**先跑一次，读 `pairing_audit.json` 的 `distributions` 再定阈值**。但在真实 FF++ 索引上先量过判定模式的分布（只读索引，不含图像）：2877 个训练键里 head_to_head 只有 64 个（2.2%），2808 个走 single_side_confirmation（其余 5 个引用的真视频不在训练集索引里）。这 2808 个里，2805 个键两侧仍有 1..7 个共有帧——共有帧规则覆盖 97.5% 的键——只有 3 个键两侧共有帧为 0，即单侧绝对上限真正单独承担判定的只有 3 个键。因此实际强度由共有帧规则与 4.0 上限共同决定，而不是由 head_to_head 决定。

### 填入 G21 配置

```bash
cp training/config/g21.yaml training/config/g21.local.yaml
```

填写 `paths.data_root`、`paths.clip_pretrained_path`、`paths.train_manifest`。`paths.dataset_json_folder` 指向之前 G18/G19 使用的同一套索引，并包含全部八个测试数据集。相对路径相对于配置文件本身解析。

默认 GPU 为进程内 `cuda:0`。如需用物理 GPU1，设置 `CUDA_VISIBLE_DEVICES=1`，仍保留 `device: cuda:0`。

```bash
python experiments/run_g21.py preflight \
  --config training/config/g21.local.yaml \
  --output /path/to/g21_preflight.json
```

preflight 只检查元数据、图像可读性、原测试配置及权重文件，**不加载检测模型**。它会核验源视频/路径别名、训练与评测的重复内容、完整数据集以及有效配对数。CDF同时用于验证和测试属于有意保持的历史协议，不误报为两套独立验证数据。

## 一次运行 A/B/C/D

通过服务器验收并确定映射后：

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/run_g21.py run \
  --config training/config/g21.local.yaml \
  --arms A B C D --seeds 1024 \
  --run-dir /path/to/experiment_results/g21/run01
```

此命令顺序完成每臂训练和原 testall 八集测试，没有阶段开关。每臂为独立子进程，使用相同解释器；失败记录保留，其余臂继续执行，最终有任意失败则返回非零。

后台运行：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python experiments/run_g21.py run \
  --config training/config/g21.local.yaml --arms A B C D --seeds 1024 \
  --run-dir /path/to/experiment_results/g21/run01 \
  > /path/to/g21_runner.log 2>&1 < /dev/null &
```

查看外层日志及 `worker_1024_A.log` 等独立 worker 日志。前台可用 Ctrl+C；后台只对该 runner/worker 的已确认 PID 发 SIGINT，不使用按进程名批量终止。

### 恢复同一次运行

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/run_g21.py run \
  --config training/config/g21.local.yaml --arms A B C D --seeds 1024 \
  --run-dir /path/to/experiment_results/g21/run01 --resume
```

恢复要求配置、清单/数据、源码、臂、种子及初始化一致。有完整 last 的中断臂从已完成的 optimizer update 恢复；完成的臂跳过，未开始的臂正常启动。没有 last 的失败臂需使用新的 run-dir 重启指定臂。

原子 `.running.lock` 防止两个进程同时写一个臂。正常退出会移除锁；若机器宕机留下锁，先确认其中 PID/主机对应的旧任务已经结束，再仅移除该臂的锁文件后恢复。

last 中包含优化器、随机状态及已选 best 的权重快照，保证“新 best 已写出但 last 尚未更新”的崩溃情形可恢复。代价是 last 约需两份模型权重的磁盘空间，请为四臂预留空间。

### 只评测某个检查点，或重评以前的 B0

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/run_g21.py evaluate \
  --config training/config/g21.local.yaml \
  --checkpoint /path/to/old_or_g21_checkpoint.pth \
  --output /path/to/isolated_retest
```

同样调用旧 testall 八数据集流程。G21 的 `best_testall.pth` 也可直接交给原 `testall.py --weights_path`，无需 G21 模型注册类。

## 输出与比较

```text
run01/
  run_plan.json                  # 协议/配置/数据/源码摘要
  preflight.json
  configs/1024_A.json ...
  worker_1024_A.log ...
  seed_1024/A/                   # B/C/D 同构
    config.resolved.json
    provenance.json
    train.jsonl
    validation.jsonl
    data_plan_digests.jsonl
    checkpoints/
      best.pth                   # G21 完整检查点
      last.pth                   # 可恢复状态及 best 快照
      best_testall.pth            # 与旧评测兼容的原始 state_dict
    evaluation/<attempt_id>/
      test_config.resolved.yaml
      testall.log
      testall_metrics.json
      artifacts/effort_probs_<dataset>.npy
      artifacts/prob_density.png
    result.json
  all_results.json               # 保留 exp_name/testall/ckpt/status 风格
  aggregate.json                 # 多种子均值/std；单种子 std=null
```

`testall` 字段的 acc、auc、video_auc 来自原脚本。论文/实验比较仍以 **video_auc** 为主；`validation_auc` 是选点用的帧指标，不能当成最终主指标。

```bash
python experiments/run_g21.py aggregate --run-dir /path/to/experiment_results/g21/run01
```

汇总会检查同种子各臂的初始化、输入/源码摘要及实际计算预算是否一致，拒绝混合不同条件的结果。缺集不重定义同名平均数，失败臂不会写成 OK。

## 服务器验收重点

测试文件位于 `tests/g21`，包含：未验证映射拒绝、划分与身份别名、确定性数据计划、同输入不同臂一致、视频分数数值稳定、公共 logit 平移不变、真实/伪造双侧梯度、带 dropout 的两遍/完整图梯度对照、更新边界恢复、日志截断恢复、内容配对（同号交集证据、背景距离、掩码回退、不可读帧不得翻转胜负、候选短于抽样窗口不崩溃、共有帧否决单侧确认、共有帧在支撑更大的一侧面前胜出、乱序帧号按值接受且不被位置读错、重复帧号仍拒绝、掩码极性告警、时间线排除、逐字节确定性、CLI 退出码与来源下限），注入真值端到端恢复（合成的 A/B 两侧候选分别经 `pair_map.json` → `build_g21_manifest.py` 落到正确参考来源，并核验审计字段与漏斗分区），以及旧 testall 完整性检查。

建议依次确认：

1. `smoke_test_g21.py` 输出 ALL PASS，记录实际测试数。
2. preflight 输出 PREFLIGHT_OK；未把未经核验的映射直接设为 true。
3. 在单独输出目录中使用短运行配置核验真实 CLIP 的 LoRA/head 梯度、显存、best/last及旧 testall加载。短跑不要混入正式结果。
4. 恢复正式6000次窗口预算后再运行完整矩阵。

这是一套实现与运行验收顺序，不是分阶段的方法实验。当前代码不能被表述为已获得 SOTA；它只提供可复现、可比较的 G21 实验流程。
