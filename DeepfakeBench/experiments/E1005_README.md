# E1005：G31–G36 服务器运行

在 `DeepfakeBench` 目录、原训练环境中执行。下列命令复用 E0924 的 **G26_B0**，不覆盖原 checkpoint。配对 receipt 必须先按下一节审计，保存为 `pair_audit_verified.json`。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1005.py \
  --base_run ./experiment_results/E0924/seed1024_20260930_110618_353460_2771190 \
  --pair_audit ./experiment_results/E1005/audit/pair_audit_verified.json \
  --seed 1024 > nohup_E1005.log 2>&1 &
```

```bash
tail -f nohup_E1005.log
```

`--base_run` 指向 **E0924 run 根目录**，runner 从其 `training_results.json` 或 `training/G26_B0/result.json` 找到成功的 B0 checkpoint 和 `train_config.json`；不要传 G26/G27 增强模型或把参数直接指向 `training/G26_B0`。路径迁移时也可用 `--base_checkpoint /path/to/B0.pth --base_config /path/to/train_config.json`，与 `--base_run` 二选一。数据/预训练位置通过 `--dataset_json_folder`、`--rgb_root`、`--clip_pretrained_path` 覆盖。

当前实现已具备训练、选点、隔离检查、最终导出和复现入口；已通过 Torch2.5.1/Transformers4.44.2 与 Torch2.9.1/Transformers4.57.3 的 CPU tiny CLIP/fixture 验证及独立代码审查。**尚未正式运行服务器真实数据/GPU 训练，也未证实超过 B0**；两版本 CPU 验证不等于真实 CUDA 性能验证。

## 先审计，再使用配对/空间资产

不加载 B0 或模型，先产生元数据候选和未核实 receipt 模板：

```bash
python experiments/run_e1005.py --audit_metadata \
  --dataset_json_folder ./preprocessing/dataset_json
```

输出到 `experiment_results/E1005/audit/`：`input_manifests.json`、`pair_candidates.json`、`pair_audit_template.json`。`--audit_metadata` 可列出缺失的外域 JSON；这只能形成候选审计。正式运行需要完整 FF++、CDF-v2 和历史六域元数据及可读图片。

模板里的 `time_verified=false`、`target_face_verified=false` 不能批量改成 true 来跳过核查。对保留条目逐项核实 target/source 谱系、同一时刻和实际目标人脸，记录非空 `evidence`、唯一 `receipt_id`，仅保留至少两个已核实公共索引，另存 `pair_audit_verified.json`。未确认的条目从已核实 receipt 中排除；公共帧编号或相同文件名不构成时间/人脸证据。配对 episode 必须包含四种伪造方法和四个不同 target 来源，实际匹配失败会报错。

同时核对 B0 配置/文件和全部元数据：

```bash
python experiments/run_e1005.py \
  --base_run ./experiment_results/E0924/seed1024_20260930_110618_353460_2771190 \
  --pair_audit ./experiment_results/E1005/audit/pair_audit_verified.json \
  --preflight
```

`--preflight` 验证原 B0 来源和元数据/receipt 身份；不加载模型，不验证图片、CUDA 或完整训练环境。`METADATA_OK` 不表示 GPU 训练就绪，也不表示配对已由程序独立证明。

G36 的 mask/原生分辨率另需 `--asset_audit /path/to/asset_audit.json`。schema 为 `schema_version=1`、非空 `receipt_id`，以及按原 RGB canonical path 索引的 `registered_masks` / `native_crops` 映射。每个条目含资产 `path` 和证据 `evidence`；mask 需 `coordinates_verified`，native crop 需 `same_frame_verified`、`crop_verified`、`native_verified`。这些标志也必须来自实际审计。

运行时会解码资产、核对尺寸/身份并保存 `asset_audit_result.json`。mask 必须与训练 RGB 注册，全为空的 fake mask 集合不合格；native crop 原像素边长须至少为配置的 448，且覆盖已核实训练帧、开发集及完整历史六域。推理不读取测试 GT mask。缺资产时阻止对应臂，不能用零 mask 或插值图冒充原生细节；`G36_PIXEL_224` 不要求空间资产，但仍要求已核实配对及 G32 leader。

## G 编号与最多 50 个槽位

| 家族 | 设计别名 | 槽位数 | 实际作用 |
| --- | --- | ---: | --- |
| G31 | A1–A6 | 6 | 原头/原 LoRA/联合、重置头、新残差、新残差＋头 |
| G32 | B：H/L/J × V/S/M/MK/MG/MN/MKGN | 21 | 更新范围 × 视频/配对/保留/困难风险监督 |
| G33 | R1–R3 | 3 | 同协议冷启动 B0、视频 B0、GenD 配方控制 |
| G34 | C1–C6 | 6 | 从 G32 开发 leader 派生的机制消融 |
| G35 | D1–D8 | 8 | 局部适配、邻接/位置/激活控制、预训练表示 delta |
| G36 | E1–E6 | 6 | 像素取证、mask/边界/打乱监督、插值/原生分辨率 |

完整 ID：

- G31：`G31_HEAD`、`G31_LORA`、`G31_JOINT`、`G31_RESET_HEAD`、`G31_NEW_RESIDUAL`、`G31_RESIDUAL_HEAD`。
- G32：`G32_{H,L,J}_{V,S,M,MK,MG,MN,MKGN}`，例如 `G32_J_MKGN`。
- G33：`G33_MATCHED_B0`、`G33_VIDEO_B0`、`G33_GEND`。
- G34：`G34_SHUFFLE`、`G34_INDEPENDENT_NUISANCE`、`G34_NO_KEEP`、`G34_NO_METHOD_RISK`、`G34_NO_NUISANCE_RISK`、`G34_FRAME_OBJECTIVE`。
- G35：`G35_MID_LOCAL_SWIGLU`、`G35_SHUFFLED_NEIGHBORS`、`G35_LATE_LOCAL`、`G35_MID_GEGLU`、`G35_PRETRAINED_LINEAR_DELTA`、`G35_B0_LINEAR_DELTA`、`G35_PRETRAINED_LN_DELTA`、`G35_LN_METRIC_DELTA`。
- G36：`G36_PIXEL_224`、`G36_TAMPER_MASK`、`G36_BOUNDARY`、`G36_SHUFFLED_MASK`、`G36_INTERPOLATED_448`、`G36_NATIVE_448`。

省略 `--arms` 按 G31→G36 顺序请求全部目录槽位；可传家族、完整 G ID 或原设计别名，例如 `--arms G31 G32 G33`。G34/G35/G36 不会自动补跑 G32，需本轮此前已产生 G32 leader 或同一 run 已有合格结果。G34 只消融 leader 实际含有的因素，缺少因素记 `NOT_APPLICABLE`；同训练签名记 `REUSED` 并引用已有 checkpoint。缺配对、leader 或空间资产记 `NOT_ELIGIBLE_*`。**50 是目录上限，不是 50 次独立训练承诺**；相同 seed 的冠军复现另计。

仅查看目录与协议，不加载训练依赖：

```bash
python experiments/run_e1005.py --dry_run
```

## 预算、评测与复现

默认 warm-start 追加 `--steps 5750`、有效图像 batch=32，配对 episode 固定每视频两个有效公共时间点、两视图；`--train_frames 8` 不会将配对 episode 扩成八帧。评测清单从完整已提取元数据数字排序、均匀选最多八帧，短视频不重复帧凑数；同输入 B0 重新推理，不能沿用历史 93.653% 作为新采样门槛。

G33 总预算默认等于已审计 B0 选中更新数＋追加预算。runner 尝试从确实保存该 checkpoint 的训练日志解析零起始 step，并加一转换为完成更新数；无法核实时记录 unknown。仅在已有可信证据时传 `--base_selected_steps`；`--cold_steps` 是显式预算，不能独自证明与 B0 总曝光匹配。未知/不足预算的结果不会通过正式突破预算门槛。

训练损失只读 FF++ train。保存 S1=源域 frame AUC、S2=源域 video AUC、S3=CDF-v2 开发 video AUC、S4=最终 step；S3 选 checkpoint 和冠军。CDF-v2 已参与旧 B0 选点，本轮仍是开发集，不计入历史六域回归均值。六域固定为 WDF、FFIW、DeepFakeDetection、DFDC、DFDCP、DeeperForensics-1.0，冠军锁定后统一导出；它们已反复查看，不能称全新确认集。普通基线的固定面板比较不能改变开发冠军。

默认 `--repeat_runs 1` 对冠军按相同 `seed=1024`、配置、数据清单、来源和预算重新训练一次；不是多种子扫描。复现同时报告逐位一致、最大预测差、selected step、macro AUC 差；数值门槛为 macro 差≤0.05pp、单域差≤0.1pp，相同 seed 本身不保证逐位一致。`--repeat_runs 0` 省略复现，不支持已复现声明。默认启用确定性算法；`--allow_nondeterministic` 会进入身份记录。

配对 bootstrap 默认 2,000 次，在每个固定域内分别重采样真实/伪造**视频**，同一抽样同时用于 B0/候选，再计算六域 macro 差。当前没有来源成组抽样，明确采用**视频独立近似**；同源视频关联、多次研究选择及跨 seed 方差不能由该区间覆盖，帧和真假排序对不作为独立样本。

`evaluation.json` 区分相对 B0 提升、普通基线以上的增量及误报成本。预定 AUC 门槛为 mean6≥+0.5pp、至少四域不下降、单域最多下降1pp、配对 CI95 下界>0，再检查完整预算与相同配置复现；新方法增量还需冠军相对最强预声明普通基线的 macro 差严格>0且对应 CI95 下界>0。普通 recipe 自身成为冠军时不宣称新模块贡献。若冠军使用 native 输入，普通控制保持固定训练轨迹，先在相同 native-derived global224 的 CDF 开发输入上按 S3 重选 checkpoint，锁定后才导出六域。源域目标5% FPR 校准后，外域相对 B0 的 FPR 增幅≤2pp 是单独门槛。设计细节见 [E1005_DESIGN.md](E1005_DESIGN.md)。

开发运行可加 `--no_evaluation`：训练/开发选点并写 `development_leader.json`，不导出历史六域分数、不作最终突破判定。小 `--steps` 可核验流程，不能当正式预算结果。

## 日志、输出与恢复

每次新运行写到 `experiment_results/E1005/seed1024_<时间>_<pid>/`：

- 终端/`nohup_E1005.log`：启动、模型加载、进度及错误；`run.log` 追加已 flush 的进度行，`progress.json` 原子替换当前阶段、G ID、数据集、step、预算、耗时等。
- `manifest.json`、`runtime_config.json`：B0/config/code/metadata/receipt/运行环境身份及原 B0 state/file 隔离结果；`experiment_catalog.json`、`input_manifests.json`、`pair_candidates.json`、`verified_pairs.json`、`asset_audit_result.json`：目录和审计依据。
- `B0_uniform/`、`B0_legacy_prefix8/`，合格原生资产时另有 `B0_native/`：绑定输入/哈希的独立基线缓存。原生臂与普通控制统一使用该 native crop 导出的 global224 输入。
- `<G_ID>/checkpoints/step*.pth`、`history.json`、`result.json`：可训练 state/新增 buffer 快照、四种选点和实际参数/步数/训练耗时；最终评估子集另有 `exports/*.npz`。
- `champion_lock.json`、`all_results.json`、`analysis.csv`、最终 `evaluation.json`、`reproductions/`：冠军、全部状态、六域比较、bootstrap 和复现。

普通 arm 失败会记录原因并继续，最后返回非零；B0 漂移立即中止。缺少前置条件可返回零，但整体状态是 `PARTIAL_PREREQUISITES`；不要把 shell 成功或 `COMPLETE_ELIGIBLE_ARMS` 解读为全部 50 槽位均完成训练，查看 `coverage` 和各 arm 的 `status`。

恢复同一 run，例如：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1005.py \
  --resume ./experiment_results/E1005/seed1024_<原运行目录> \
  --pair_audit ./experiment_results/E1005/audit/pair_audit_verified.json \
  --seed 1024 > nohup_E1005_resume.log 2>&1 &
```

替换为真实 run 目录，并重复原运行使用过的 `--arms`、数据路径、receipt、训练参数和 asset 参数。`--resume` 可从 manifest 恢复 B0/config 路径；不会自动还原全部 CLI 覆盖。源码、配置、环境、元数据/receipt、B0 或训练身份改变时拒绝复用。

**恢复仅复用已完成且签名/checkpoint 哈希一致的 `OK`/`REUSED` arm 和基线缓存；中断/失败 arm 从 step=0 重跑。** 当前快照不保存 optimizer/RNG 中途状态，不支持断点续训。已锁定冠军不能因后来结果而更换。
