# E1111：冻结特征度量与 CE 兼容 PU 正则

本机仅编写代码、阅读源码与静态检查；**数据审计、特征提取、运行测试、训练及评估全部留在服务器执行**。当前没有 E1111 实验指标，也没有认定 P-D7 已通过。

| 分支 | 基础 | 本轮改变 |
|---|---|---|
| E1111A | v4.4.3、冻结 B0 | 同源同帧配对、平衡统计、逐折嵌套选参、后验度量与归因对照 |
| E1111B | E1025 单 LoRA B0/PU | 保留整批 CE，增加可缩小到零的非负 PU 风险正则 |

A、B共用原有DeepfakeBench预处理图片，原版B0作为共同对照；结果目录分别为 experiment_results/E1111/E1111A、experiment_results/E1111/E1111B。每个阶段用新目录，不覆盖历史实验。原 E1025 runner/损失和已有 G21 配对改动保留。


## 搬迁到 /mnt/data2 后的路径

当前迁移仅将 /home/user1 前缀替换为 /mnt/data2，其余相对目录保持不变。E1111默认图片根目录为 /mnt/data2/effort/data；索引默认取当前DeepfakeBench/preprocessing/dataset_json，CLIP取当前DeepfakeBench/training/models--openai--clip-vit-large-patch14。也支持同名CLI参数显式覆盖；旧B0结果中的checkpoint、数据目录及E0924的artifact_dir会按同一前缀规则读取新位置，不修改原结果文件。

在搬迁后的DeepfakeBench目录执行：

    nohup python -u experiments/run_e1111.py --gpu_a 0 --gpu_b 1 \
      --rgb_root /mnt/data2/effort/data \
      --dataset_json_folder "$PWD/preprocessing/dataset_json" \
      --clip_pretrained_path "$PWD/training/models--openai--clip-vit-large-patch14" \
      > nohup_E1111.log 2>&1 &

B0训练前会构造与原管线相同的Dataset，只读取其实际采样清单，检查训练及评估文件是否存在；不迭代loader、不调用getitem、不解码图片、不改变清单、采样或增强。错误根目录或缺失文件会在训练前停止，避免旧loader的随机换样本分支掩盖路径错误。这个预检不等于图片解码完整性审计。

先检查日志里报错样本的新位置是否存在。例如：

    ls /mnt/data2/effort/data/FaceForensics++/original_sequences/youtube/c23/frames/113/081.png

若存在，只需修正读取根目录，无须重跑preprocess；若不存在，再检查预处理输出是否搬全、索引是否与本次抽帧/人脸检测输出匹配。只有实际缺少预处理产物时才补跑对应数据，不因旧路径报错直接重做全部预处理。索引若仍包含/home/user1的绝对路径，预检会单独提示更新索引前缀，不把它误判为缺少原始图片。

已经出现缺图随机替代的错误训练应停止，修复后启动新run；不要把该run的checkpoint当成原版B0对照继续使用。

## 一条命令运行全部实验

在原服务器环境、DeepfakeBench目录中，使用总入口 run_e1111.py：

    # 一张卡：原版B0 → A → B，串行执行
    nohup python -u experiments/run_e1111.py --gpu_a 0 --gpu_b 0 \
      > nohup_E1111.log 2>&1 &

    # 两张卡：先在GPU1重训一次原版B0；完成后A用GPU0、B用GPU1并行
    nohup python -u experiments/run_e1111.py --gpu_a 0 --gpu_b 1 \
      > nohup_E1111.log 2>&1 &

以上两条选一条执行，不要同时提交。--gpu_a/--gpu_b为物理CUDA卡号，总入口会覆盖继承的CUDA_VISIBLE_DEVICES，为每个子进程单独指定卡；因此无需在命令外再设置单卡掩码。每个子进程的LOCAL_RANK固定0、WORLD_SIZE固定1，A子进程内使用逻辑cuda:0。

如果已经完成了前面的新B0训练，传入那次包含all_results.json的运行目录即可复用，不再重训：

    nohup python -u experiments/run_e1111.py --gpu_a 0 --gpu_b 1 \
      --base_run ./experiment_results/E1111/E1111B/seed1024_已完成的运行目录 \
      > nohup_E1111.log 2>&1 &

总入口只接收成功的原版effort+CE B0，校验checkpoint SHA及种子、epoch、batch、real ratio、c23和8/8帧设置。B阶段固定传--arms PUCE，只运行CE_L0和三个小权重配置，原版B0只训练一次；A明确传入该次B0目录，不会自动回到历史E0924权重。需要覆盖epoch、batch或数据路径时，在总入口传同名训练参数，同时用于新B0和B；复用B0时若训练条件不一致会在启动前报出差异。两个分支同时继承B0的实际索引目录、RGB根目录及CLIP路径；索引逐文件SHA与B0 manifest核对，RGB/CLIP覆盖与B0不一致时拒绝，避免两支使用不同数据或预训练源。旧配置省略的默认路径仅在原加载代码SHA一致时从其源码读取，不加载模型。

A不再要求原视频、dlib权重或共享重裁剪，直接读取与B0/B相同RGB根目录中的预处理图片。其他数据/CLIP参数沿用现有入口；P-D7、统计选参、B0/B训练条件均保持不变。总入口仅执行服务器开发实验，不自动启动正式盲测。

总输出目录为 experiment_results/E1111/all/seed1024_本次时间戳/，包含B0.log、A.log、B.log和总all_results.json；每个分支保留自己的完整结果目录。看总调度日志用 tail -f nohup_E1111.log，具体训练日志路径在启动时打印。A因数据门槛或统计判据停止时，B仍继续；B0失败时两支均不启动。停止总进程时会终止其正在运行的子进程组。

    # 仅在服务器检查调度计划，不加载模型
    python experiments/run_e1111.py --gpu_a 0 --gpu_b 1 --dry_run

## 按原实验方式直接启动

在服务器原环境的 DeepfakeBench 目录中。**只重新训练最基础B0对照**：

    CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1111b.py --arms B0 \
      > nohup_E1111_B0.log 2>&1 &

这组使用原 effort 检测器：CLIP ViT-L/14、原单LoRA rank4、CLS/pooler原二分类头、整批平均CE。从CLIP预训练重新初始化，不加载历史detector checkpoint。沿用E1025 B0的FF++ c23、8/8帧、batch32、real ratio0.3、seed1024、Adam lr2e-4/wd5e-4、原增强和CDF-v2 frame-AUC选点。结果同时保留完整路径和历史basename的视频AUC口径；它是独立原版B0，不用新损失的CE_L0替代。

需要手动分别启动时，以下A使用已指定的B0，B默认运行新B0和PUCE四组（共五组）；共用新B0并自动分卡优先使用上面的总入口：

    CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1111a.py \
      --base_run ./experiment_results/E0924/seed1024_20260930_110618_353460_2771190 \
      > nohup_E1111A.log 2>&1 &

    CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/run_e1111b.py \
      > nohup_E1111B.log 2>&1 &

A复用已有B0，默认入口自动执行：索引盘点/路径视图 → FF++官方谱系与原始帧号配对 → P-D7审计 → 特征提取 → S0 → 校准 → 两个开发源导出/归因。无需人工制作配对文件或修改“已验证”字段。它不会从头训练骨干，也不会自动运行盲测。B直接沿用E1025训练链路，默认依次运行独立原版B0、CE_L0和三个小权重正则配置。

看日志：tail -f nohup_E1111A.log 或 tail -f nohup_E1111B.log。A的all_results.json保存当前阶段和停止原因；每次启动创建独立运行目录。P-D7等科研判据不通过时如实停止，不修改门槛。

A的拟合、D_theta校准和开发/测试评测均读取同一套现有预处理图片；不生成新的裁剪目录，也不改变原JSON或PNG。原视频已删除不影响这一路径。对于绝对路径或带./的索引，程序仅生成指向同一文件的内部相对路径视图，保持帧列表顺序和样本成员不变；不会借此换图或放宽配对条件。

训练结束后，终端会打印该次运行目录下的all_results.json。A可将**该次新运行目录**作为 --base_run；程序只接受成功的原生B0、读取B0/train_config.json并校验checkpoint SHA，不会误取PUCE权重。早期示例中的E0924路径仅用于旧B0复现；需要新B0时改用新运行目录。B0历史对照仍使用8帧评估；A的32帧统计结果单列，不与其直接混算。

## E1111A：预注册执行规则

2026-11-11 在查看实验结果前确定：

1. FF++ 官方 train 的719个真实内容源按排序后 seed=1024 打乱，依次分为 fit/rad/sel/beta/theta = 395/72/58/57/137；官方 val 单列开发验证。实际划分保存为 splits.json。
2. D_sel_FF++ 仅指这58个 FF++ train 源。CDF-v2 real train 的辅助诊断样本不参与 D1、τR、最终λm或配对差估计；CDF-v2 train 作为开发评估源使用。
3. 每个真实源按数字帧位排序，等距选最多32个不同候选帧；不走旧 loader 的前8帧路径。缺图、坏图、未验证映射单列，禁止复制帧补齐。
4. 直接使用原有独立对齐的PNG。按[FF++官方命名约定](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)将 target_source 的第一个ID作为参考真实源；本仓库预处理保存原始 cnt_frame 作为PNG帧号，仅取真假相同数字帧号的交集，不按列表位置或最近邻配对。每格仍须至少2个有效共同帧，再检查四方法共同帧；不经过G21背景相似度的8帧接受门槛。此核验范围是官方谱系与保存帧号，不声明像素配准或共享仿射。P-D7仍要求N4≥30、N4≥0.5×395（至少198）、Ncell≥0.9，未通过不进入S0/S1。裁剪误差可能混入配对差，但五项经验分解恒等式及其统一权重不变。
5. 只用完整四方法源，每格统一 K_pair=min(n_v)。主 B、Dm、五项分解共用清单和权重。审计同时报告 K_pair、有效帧丢弃比例、共同帧丢弃比例、缺失格和剔除原因。
6. μR/ΣR/Dc/Dtemporal 仍用全部 D_fit 真实源。Dc 的 Q 为所有有效(真实源,候选帧,JPEG质量)组合，质量固定15/30/40/50，每组合等权；提取时遇到新坏图直接失败。
7. τR∈{0.1,0.3,0.5}：最小化 D_sel_FF++ 真实视频等权的高斯 NLL，使用与后续白化相同的特征值地板；相差≤1e-12时取小τR。
8. 原 B 方案 r∈{2,4,8}、λm∈{0,0.1,0.3,1}，λt固定0。外层留m，内层再留j，只用另两法重估 B、Dm、归一化Dm、Dnuis、L、V。最大化三折平均 J=mean(rho_cap/(r/d))；相差≤1e-12时先取小r，再取小λm。外层用其余三法重新拟合后评估m。
9. 四个外层秩取较小中位候选。固定最终r后，在四个方法留一折重新计算同一J来选择最终λm，再用四法重拟合；不得回写 D1 外层指标。D1 要求至少三折捕获率>1.5r/d，四折平均的源簇 bootstrap 双侧95%区间下界>平均r/d，1000次、seed1024。
10. Fisher 降级采用**专用r=1**：两方法内层的Σmethod秩最多1，不沿用 B 的r网格。每个内外层重估 Fisher 分子、残差分母与子空间，重新完成全部D1，阈值仍按该折r/d计算。所有秩不得超过数值正谱秩。Fisher 通过不改写原D7失败，不恢复完整机制主张；之后仍须随机谱归因。
11. D1 零分母停止S0；D2/D8 零分母标为未定义、不给通过结论，分别阻止完整机制/R4继续结论。D3当前仅描述、R3关闭，未来启动阈值不参与本轮通过/失败。
12. S0后扫描γ∈{0,0.25,0.5,1,2,4,8,inf}。用 D_sel_FF++ 真实视频的临时conformal阈值，在α=0.10下最大化伪造视频TPR，并列取小γ；固定后才读取D_theta。四个检测器及各对照分别在D_theta上按相同FPR预算定阈值。

**统计边界**：划分按内容参考源，不保证 A_B 的两个谱系源均不跨集合；审计保存 cross_split_other_lineage。D1 是方法迁移诊断，不能称为方法×双谱系源独立验证。同一 D_sel 源的不同留出方法共同重采样。S1 对 FF++ val 按 A_B 两个源ID的连通簇共同抽样；CDF-v2 缺原始谱系映射时使用视频代理簇，区间边界随结果记录。

### 按阶段调试（正常启动无需逐条执行）

在 DeepfakeBench 目录执行。填写已有服务器路径；B0配置使用保存的完整 train_config.json，不要只用 detector YAML。

    export DB_DATA_ROOT=/path/to/preprocessed_rgb
    export DB_JSON=/path/to/dataset_json
    export DB_CLIP=/path/to/clip-vit-large-patch14
    export DB_B0=/path/to/B0.pth
    export DB_B0_CONFIG=/path/to/B0/train_config.json
    export E1111_OUT=experiment_results/E1111/E1111A

    # 1. 索引盘点：不读取图像，不把索引数量当成已验证配对数量
    python experiments/run_e1111a.py inventory \
      --dataset-json "$DB_JSON/FaceForensics++.json" --data-root "$DB_DATA_ROOT" \
      --output-dir "$E1111_OUT/inventory"

    # 2. 按官方目标源和相同原始帧号生成映射，尚不表示P-D7通过
    python experiments/run_e1111a.py index-pairing \
      --dataset-json "$E1111_OUT/inventory/dataset.json" \
      --data-root "$DB_DATA_ROOT" --output-dir "$E1111_OUT/index_pairing"

映射与后续图片审计使用同一份索引快照，只选择相同原始帧号，不生成裁剪文件。旧G21内容核验工具保留独立使用，其像素相似度接受结果不再作为A的默认可用样本门槛。

    python experiments/run_e1111a.py audit \
      --dataset-json "$E1111_OUT/inventory/dataset.json" \
      --data-root "$DB_DATA_ROOT" \
      --pair-map "$E1111_OUT/index_pairing/pair_map.json" \
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

默认先运行独立原版B0，再运行四组lam={0,0.01,0.05,0.1}、prior固定0.3。B0使用effort原CE实现；CE_L0保留为新损失接口的等价性校验，两者同seed，不当作独立种子重复。各组从同一CLIP预训练独立初始化：单LoRA rank4、完整原二分类头、seed1024、FF++ c23、8/8帧、batch32、real ratio0.3、Adam lr2e-4/wd5e-4、epoch0..10。沿用E1025增强、Celeb-DF-v2 frame-AUC选点与完整路径视频聚合，关闭Mixup，不叠加SOFT路由。

    # 只训练原版B0
    python experiments/run_e1111b.py --arms B0
    # 只训练正则分支；默认包含新B0
    python experiments/run_e1111b.py --arms PUCE
    python experiments/run_e1111b.py --dry_run
    python experiments/run_e1111b.py \
      --prior 0.3 --lams 0 0.01 0.05 0.1 \
      --dataset_json_folder "$DB_JSON" --rgb_root "$DB_DATA_ROOT" \
      --clip_pretrained_path "$DB_CLIP" \
      --output_dir experiment_results/E1111/E1111B

要保留0.5对照可显式添加 --lams 0 0.01 0.05 0.1 0.5。不要据盲测回调lam。输出配置、源码/索引/权重摘要、选中checkpoint、逐帧NPZ、逐库完整路径video AUC、七库/独立六库均值及失败状态。CDF-v2参与选点，不计入独立六库。

## 服务器验证

沿用原服务器依赖环境；未增加新的训练框架或第三方依赖。测试已编写，**本机未执行**：

    python -m pytest -q tests/test_e1111a.py tests/test_e1111b.py tests/test_e1111_shared_crop.py tests/test_e1111_launcher.py tests/test_e1111_paths.py
    python -m pytest -q tests/test_e1025_pu.py tests/test_e1025_runner.py \
      tests/test_e1025_integration.py tests/g21/test_builder.py tests/g21/test_pairing.py

第一条覆盖分解恒等式、数据门槛、帧平衡、归一化选参、内外层隔离、Fisher专用秩、零分母、视频分组、CE值/梯度、真实Trainer步与checkpoint重载。第二条检查E1025/G21原路径回归。随后先运行A的inventory/audit，确认实际数据，再决定是否进入后续统计与训练。

统一输入测试覆盖：没有任何MP4或关键点权重时可解析A输入；默认runner不调用共享裁剪；绝对路径与相对路径引用同一PNG且不重写像素；两个共同帧按数字原帧号配对；缺目标源时不改用第二个ID；少于2帧、非法命名及其他索引快照均被拒绝；未验证的外部映射仍不能进入配对；P-D7失败仍停止。历史共享裁剪工具及独立prepare子命令保留供旧实验使用，不在默认A流程中。

总调度器测试通过模拟子进程检查一次原版B0、同卡串行/异卡并行、CUDA设备隔离、复用基线、A停止后B继续、缺失结果拒绝成功和中断清理。本机未执行这些运行测试。

2026-11-13数据协议调整：用户确认A与标准DeepfakeBench共用既有预处理帧。取消的是原视频共享重裁剪条件，不降低同源同帧验证、P-D7、平衡权重或统计阈值。A的裁剪输入与此前严格方案不同，旧A统计缓存按协议/源码摘要失效；B0/B模型及训练、历史8帧指标不受影响。A既定32帧候选采样、本轮评分和选点规则保持不变；对外结果仍须按同一取帧与指标口径比较。

2026-11-14配对入口修复：标准PNG路径误用了旧G21至少8帧的背景核验接受集合。A改为官方谱系与精确保存帧号配对，映射绑定索引摘要，图片可读性及四方法完整度由原审计继续检查。旧运行的N4=1来自额外核验筛选，不能据此判断官方索引的真实可配对量；修复后仍须审计实际N4/Ncell/K_pair，不降低P-D7或改变B0/B。此变更在协议及源码摘要中记录，旧A缓存不可混用。
