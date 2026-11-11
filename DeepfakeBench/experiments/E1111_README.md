# E1111：冻结特征度量与 CE 兼容 PU 正则

本机仅编写代码、阅读源码与静态检查；**数据审计、特征提取、运行测试、训练及评估全部留在服务器执行**。当前没有 E1111 实验指标，也没有认定 P-D7 已通过。

| 分支 | 基础 | 本轮改变 |
|---|---|---|
| E1111A | v4.4.3、冻结 B0 | 严格配对审计、平衡统计、逐折嵌套选参、后验度量与归因对照 |
| E1111B | E1025 单 LoRA B0/PU | 保留整批 CE，增加可缩小到零的非负 PU 风险正则 |

两个分支互不依赖，结果目录分别为 experiment_results/E1111/E1111A、experiment_results/E1111/E1111B。每个阶段用新目录，不覆盖历史实验。原 E1025 runner/损失和已有 G21 配对改动保留。

## E1111A：预注册执行规则

2026-11-11 在查看实验结果前确定：

1. FF++ 官方 train 的719个真实内容源按排序后 seed=1024 打乱，依次分为 fit/rad/sel/beta/theta = 395/72/58/57/137；官方 val 单列开发验证。实际划分保存为 splits.json。
2. D_sel_FF++ 仅指这58个 FF++ train 源。CDF-v2 real train 的辅助诊断样本不参与 D1、τR、最终λm或配对差估计；CDF-v2 train 作为开发评估源使用。
3. 每个真实源按数字帧位排序，等距选最多32个不同候选帧；不走旧 loader 的前8帧路径。缺图、坏图、未验证映射单列，禁止复制帧补齐。
4. 配对需验证真实内容源、原始帧位对应和共同裁剪。A_B 命名只作候选，不自动认定第一个 ID 就是内容源。P-D7 要求 N4≥30、N4≥0.5×395（至少198）、Ncell≥0.9；未通过不进入 S0/S1。
5. 只用完整四方法源，每格统一 K_pair=min(n_v)。主 B、Dm、五项分解共用清单和权重。审计同时报告 K_pair、有效帧丢弃比例、共同帧丢弃比例、缺失格和剔除原因。
6. μR/ΣR/Dc/Dtemporal 仍用全部 D_fit 真实源。Dc 的 Q 为所有有效(真实源,候选帧,JPEG质量)组合，质量固定15/30/40/50，每组合等权；提取时遇到新坏图直接失败。
7. τR∈{0.1,0.3,0.5}：最小化 D_sel_FF++ 真实视频等权的高斯 NLL，使用与后续白化相同的特征值地板；相差≤1e-12时取小τR。
8. 原 B 方案 r∈{2,4,8}、λm∈{0,0.1,0.3,1}，λt固定0。外层留m，内层再留j，只用另两法重估 B、Dm、归一化Dm、Dnuis、L、V。最大化三折平均 J=mean(rho_cap/(r/d))；相差≤1e-12时先取小r，再取小λm。外层用其余三法重新拟合后评估m。
9. 四个外层秩取较小中位候选。固定最终r后，在四个方法留一折重新计算同一J来选择最终λm，再用四法重拟合；不得回写 D1 外层指标。D1 要求至少三折捕获率>1.5r/d，四折平均的源簇 bootstrap 双侧95%区间下界>平均r/d，1000次、seed1024。
10. Fisher 降级采用**专用r=1**：两方法内层的Σmethod秩最多1，不沿用 B 的r网格。每个内外层重估 Fisher 分子、残差分母与子空间，重新完成全部D1，阈值仍按该折r/d计算。所有秩不得超过数值正谱秩。Fisher 通过不改写原D7失败，不恢复完整机制主张；之后仍须随机谱归因。
11. D1 零分母停止S0；D2/D8 零分母标为未定义、不给通过结论，分别阻止完整机制/R4继续结论。D3当前仅描述、R3关闭，未来启动阈值不参与本轮通过/失败。
12. S0后扫描γ∈{0,0.25,0.5,1,2,4,8,inf}。用 D_sel_FF++ 真实视频的临时conformal阈值，在α=0.10下最大化伪造视频TPR，并列取小γ；固定后才读取D_theta。四个检测器及各对照分别在D_theta上按相同FPR预算定阈值。

**统计边界**：划分按内容参考源，不保证 A_B 的两个谱系源均不跨集合；审计保存 cross_split_other_lineage。D1 是方法迁移诊断，不能称为方法×双谱系源独立验证。同一 D_sel 源的不同留出方法共同重采样。S1 对 FF++ val 按 A_B 两个源ID的连通簇共同抽样；CDF-v2 缺原始谱系映射时使用视频代理簇，区间边界随结果记录。

### 服务器步骤

在 DeepfakeBench 目录执行。填写已有服务器路径；B0配置使用保存的完整 train_config.json，不要只用 detector YAML。

    export DB_DATA_ROOT=/path/to/preprocessed_rgb
    export DB_JSON=/path/to/dataset_json
    export DB_CLIP=/path/to/clip-vit-large-patch14
    export DB_B0=/path/to/B0.pth
    export DB_B0_CONFIG=/path/to/B0/train_config.json
    export E1111_OUT=experiment_results/E1111/E1111A

    # 1. 索引盘点：不读取图像，不把索引数量当成已验证配对数量
    python experiments/run_e1111a.py inventory \
      --dataset-json "$DB_JSON/FaceForensics++.json" \
      --output-dir "$E1111_OUT/inventory"

    # 2. 先复用 G21 内容配对工具，不能只凭命名猜内容源
    python experiments/verify_g21_pairing.py \
      --dataset-json "$DB_JSON/FaceForensics++.json" \
      --data-root "$DB_DATA_ROOT" --output "$E1111_OUT/content_pairing"

原 G21 映射支持方法级 rules 和逐视频 overrides。A支另要求每条生效规则有 crop_alignment_verified=true 和非空字符串 crop_alignment_evidence，记录共同裁剪的预处理依据；源映射 verified/evidence、帧对齐验证仍需保留。只有确有依据时才补裁剪字段，背景内容对应不能自动证明共同裁剪。

    # 使用核验并补齐裁剪依据后的映射文件
    python experiments/run_e1111a.py audit \
      --dataset-json "$DB_JSON/FaceForensics++.json" \
      --data-root "$DB_DATA_ROOT" --pair-map /path/to/verified_pair_map.json \
      --output-dir "$E1111_OUT/audit"

    # 3. 只有 P_D7_pass=true 才继续；失败不降低门槛
    python experiments/run_e1111a.py extract \
      --audit "$E1111_OUT/audit/audit.json" \
      --base-config "$DB_B0_CONFIG" --checkpoint "$DB_B0" \
      --clip-pretrained-path "$DB_CLIP" --batch-size 8 --d8 \
      --output-dir "$E1111_OUT/features"

    # 4. S0：不读 D_theta 或测试集
    python experiments/run_e1111a.py s0 \
      --audit "$E1111_OUT/audit/audit.json" --cache "$E1111_OUT/features/features.npz" \
      --output-dir "$E1111_OUT/s0"

    # 5. S0/新 Fisher D1 通过后选gamma，再在D_theta定阈值
    python experiments/run_e1111a.py calibrate \
      --audit "$E1111_OUT/audit/audit.json" --cache "$E1111_OUT/features/features.npz" \
      --metric "$E1111_OUT/s0/metric.npz" \
      --base-config "$DB_B0_CONFIG" --checkpoint "$DB_B0" \
      --clip-pretrained-path "$DB_CLIP" --output-dir "$E1111_OUT/calibrated"

S0报告保存五项重构误差、各ρ、D1逐折选参表/拟合方法、bootstrap、D2/D8、有效秩和数值恢复记录。失败状态返回非零，不以NaN继续。D4的gap诊断本次标记 NOT_RUN，不自动引入多中心；它不是S0通过门槛。

### S1归因与后续评估

三个对照：C_random为随机正交方向+谱置换；C_shuffle将真实参考源按固定随机错排重建差分；C_real使用真实同源跨帧差。保持最终真实白化、Dnuis和r不变，仅替换方向来源；对照为与D_F同型的子空间能量，各自校准阈值。正谱秩不足时明确省略，不补任意方向。Fisher的真实方法均值分子不可用，标未定义；继续条件仍要求独立随机谱比较。

    python experiments/run_e1111a.py export \
      --dataset-json "$DB_JSON/FaceForensics++.json" --dataset-id FaceForensics++ --split val \
      --data-root "$DB_DATA_ROOT" --base-config "$DB_B0_CONFIG" \
      --checkpoint "$DB_B0" --clip-pretrained-path "$DB_CLIP" \
      --output-dir "$E1111_OUT/dev_ff"
    python experiments/run_e1111a.py export \
      --dataset-json "$DB_JSON/Celeb-DF-v2.json" --dataset-id Celeb-DF-v2 --split train \
      --data-root "$DB_DATA_ROOT" --base-config "$DB_B0_CONFIG" \
      --checkpoint "$DB_B0" --clip-pretrained-path "$DB_CLIP" \
      --output-dir "$E1111_OUT/dev_cdf"
    python experiments/run_e1111a.py score \
      --features "$E1111_OUT/dev_ff/features.npz" \
      --metric "$E1111_OUT/calibrated/calibrated_metric.npz" \
      --output-dir "$E1111_OUT/dev_ff_scores"
    python experiments/run_e1111a.py score \
      --features "$E1111_OUT/dev_cdf/features.npz" \
      --metric "$E1111_OUT/calibrated/calibrated_metric.npz" \
      --output-dir "$E1111_OUT/dev_cdf_scores"
    python experiments/run_e1111a.py attribution \
      --evaluations "$E1111_OUT/dev_ff_scores" "$E1111_OUT/dev_cdf_scores" \
      --output-dir "$E1111_OUT/attribution"

S1对两个开发源等权平均TPR，配对、按源簇及数据集分层bootstrap1000次。通过要求 D_J-max(D_R,D_F) 的单侧95%下界>0.01，且随机谱未达到D_F的90%。不接受测试集结果作为归因输入。

S2同时报告TPR(γ)曲线、内点/端点最优和独立D1。**ρcap不含γ，不会随γ改善**，因此不实现原决策文档中将其要求为γ单调函数的矛盾判据。S3身份混杂仍需ArcFace/匹配子集证据；结果保持 NOT_EVALUATED，不能据S1自动宣称完整机制。

完成S3/S4研究判定并冻结后，可用同一export/score接口评估 DeepFakeDetection、DFDC、DFDCP、UADFV 的官方test，CDF-v2 test单列seen-source。主流程不自动运行盲测。n<50只报AUC；50≤n<100仅报α=0.10；n≥100报0.10/0.05，FPR均带Clopper–Pearson区间。DFDCP无假样本时仅覆盖率/FPR。历史选点checkpoint的结果标为开发性；正式主张仍需干净checkpoint。

## E1111B：CE + 小权重 PU 正则

实现类 nnPUCEWithReg(prior=0.3, lam=0.5) 保持所给公式：

    CE = mean(batch CE)
    negative_risk = -mean(log p1[labels==1]) + prior * mean(log p1[labels==0])
    loss = CE + lam * clamp(negative_risk, min=0)

lam=0直接返回CE，值与梯度均一致。非零lam的训练批必须同时含两类；风险被截断仍保留CE梯度。验证批可只有一类，日志保持普通CE。prior是U内的真实类先验设定，与batch real ratio不同；将已标注伪造作为U属于给定的启发式正则。

默认四组 lam={0,0.01,0.05,0.1}、prior固定0.3。CE_L0为同代码路径CE对照，不重复增加B0。各组从同一CLIP预训练独立初始化：单LoRA rank4、完整原二分类头、seed1024、FF++ c23、8/8帧、batch32、real ratio0.3、Adam lr2e-4/wd5e-4、epoch0..10。沿用E1025增强、Celeb-DF-v2 frame-AUC选点与完整路径视频聚合，关闭Mixup，不叠加SOFT路由。

    python experiments/run_e1111b.py --dry_run
    python experiments/run_e1111b.py \
      --prior 0.3 --lams 0 0.01 0.05 0.1 \
      --dataset_json_folder "$DB_JSON" --rgb_root "$DB_DATA_ROOT" \
      --clip_pretrained_path "$DB_CLIP" \
      --output_dir experiment_results/E1111/E1111B

要保留0.5对照可显式添加 --lams 0 0.01 0.05 0.1 0.5。不要据盲测回调lam。输出配置、源码/索引/权重摘要、选中checkpoint、逐帧NPZ、逐库完整路径video AUC、七库/独立六库均值及失败状态。CDF-v2参与选点，不计入独立六库。

## nohup 后台运行

在服务器原环境的 DeepfakeBench 目录执行；数据、索引和 CLIP 路径沿用前次实验。先运行下节测试，再启动训练。本机不执行这些命令。

    mkdir -p logs
    E1111_STAMP=$(date +%Y%m%d_%H%M%S)
    nohup env CUDA_VISIBLE_DEVICES=0 python -u experiments/run_e1111b.py \
      --prior 0.3 --lams 0 0.01 0.05 0.1 \
      --dataset_json_folder "$PWD/preprocessing/dataset_json" \
      --rgb_root /home/user1/effort/data \
      --clip_pretrained_path "$PWD/training/models--openai--clip-vit-large-patch14" \
      --output_dir experiment_results/E1111/E1111B \
      > "logs/E1111B_${E1111_STAMP}.log" 2>&1 < /dev/null &
    echo "E1111B PID=$!"

A 支是冻结特征统计实验，先后台核验配对，之后按审计门槛继续，不从头训练骨干：

    nohup python -u experiments/verify_g21_pairing.py \
      --dataset-json "$PWD/preprocessing/dataset_json/FaceForensics++.json" \
      --data-root /home/user1/effort/data \
      --output "experiment_results/E1111/E1111A/content_pairing_${E1111_STAMP}" \
      --require-usable \
      > "logs/E1111A_pairing_${E1111_STAMP}.log" 2>&1 < /dev/null &
    echo "E1111A pairing PID=$!"

查看日志：tail -f "logs/E1111B_${E1111_STAMP}.log"。A 的配对工具只负责内容/时间对应，仍需补齐共同裁剪依据后，再执行 audit；不得把工具退出成功视为 P-D7 已通过。

## 服务器验证

沿用原服务器依赖环境；未增加新的训练框架或第三方依赖。测试已编写，**本机未执行**：

    python -m pytest -q tests/test_e1111a.py tests/test_e1111b.py
    python -m pytest -q tests/test_e1025_pu.py tests/test_e1025_runner.py \
      tests/test_e1025_integration.py tests/g21/test_builder.py tests/g21/test_pairing.py

第一条覆盖分解恒等式、数据门槛、帧平衡、归一化选参、内外层隔离、Fisher专用秩、零分母、视频分组、CE值/梯度、真实Trainer步与checkpoint重载。第二条检查E1025/G21原路径回归。随后先运行A的inventory/audit，确认实际数据，再决定是否进入后续统计与训练。
