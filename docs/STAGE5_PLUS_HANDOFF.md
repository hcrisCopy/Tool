# 第 5 阶段以后：实验与产物交接说明

## 1. 交接目标

本阶段从已经完成的数据改造、模型相关标签和 residual probe 出发，交付三条可独立运行的流水线：

- 第 5 阶段：抽取 FFN/SwiGLU 中间激活，定位 `NONE/A/B/C` 类别相关神经元；
- 第 6 阶段：用 target mask 与逐层等数量 random mask 做端到端因果验证；
- 第 7-8 阶段：构造成功工具轨迹，训练 target/random/dense MLP LoRA，并在 scoped/full 两种工具菜单下评估。

研究定位、相关工作和允许的论文主张见[研究定位文档](RESEARCH_POSITIONING.md)。原始统计阶段的完成项和 partial 行为 checkpoint 见[统计阶段报告](../reports/stages/STAGE_STATISTICS_QWEN3_4B.md)。

## 2. 当前证据与结论边界

### 已完成的前置证据

Qwen3-4B-Instruct-2507 的 full-menu residual probe 已得到：

| 任务 | Accuracy | Balanced Accuracy | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| tool vs NONE | 0.8542 | - | - | 0.9239 |
| NONE/A/B/C | 0.8569 | 0.7840 | 0.8085 | 0.9633 OVR |

四动作多数类基线为 0.5267。needed-only A/B/C 在 environment-held-out 下的 mean Accuracy/Balanced Accuracy/Macro-F1 为 0.7005/0.6714/0.6117。现有结果支持“固定 When2Tool 协议下存在可解码结构”，但不证明 FFN 神经元具有因果性，也不证明信号与题目主题完全解耦。

### 第 5 阶段以后应回答的问题

| 假设 | 需要的证据 | 不充分的证据 |
|---|---|---|
| H1：四动作信息进入 FFN intermediate | 冻结 train 选择后，test 上 MLP 特征明显超过 majority/prior baseline | 只在 train 上分开 |
| H2：存在类别选择性的因果子空间 | target 对目标 recall 的破坏超过同层同量 random，且 off-target 损伤较小 | 只有 final accuracy 普遍下降 |
| H3：该子空间可被训练利用 | target LoRA 在同数据、同步数对照下优于 random/dense，并改善端到端指标 | 只比未训练 base 好 |

第 5 阶段已在远程完成正式运行，结果和审计见本文第 8 节。第 6-8 阶段在本次交接中只准备代码和冻结矩阵，未启动正式计算。

## 3. 代码与大文件边界

```text
<PROJECT_ROOT>/
├── CallTool_code/                         # Git 仓库
│   ├── when2tool_action/                  # Python 实现
│   ├── scripts/                           # 一键阶段入口
│   ├── tests/                             # 不加载大模型的单元测试
│   ├── docs/                              # 方案、清单和本交接说明
│   └── reports/                           # 小型 Markdown 报告
└── CallTool_data/                         # 不进入 Git
    └── when2tool_precise_shield/
        └── qwen3-4b-instruct-2507/
            └── stages/
                ├── 05_probing/
                │   ├── activations/
                │   ├── discovery/
                │   ├── logs/
                │   └── manifests/
                ├── 06_causal/
                │   ├── conditions/
                │   ├── logs/
                │   └── manifests/
                ├── 07_training/
                │   ├── sft/
                │   ├── adapters/
                │   ├── logs/
                │   └── manifests/
                └── 08_evaluation/
                    ├── outputs/
                    ├── comparison/
                    ├── logs/
                    └── manifests/
```

模型、activation tensor、逐样本生成、adapter、日志和图均放在数据侧。代码侧禁止保存机器专属绝对路径、访问凭据或自动下载缓存。

每个阶段单独保存 `manifests/runtime_provenance.json`。旧统计阶段的全局 receipt 绑定旧代码提交，不得覆盖；阶段脚本会创建或严格复用本阶段 receipt，并把其 SHA256 与代码 commit 写入全部后续产物。若报告更新使仓库 HEAD 前进，恢复旧 checkpoint 时应检出产物记录的生成 commit，而不是伪造一个新 receipt。

## 4. 冻结的实验矩阵

### 4.1 第 5 阶段：探测

共同设置：

| 参数 | 冻结值 |
|---|---|
| backbone | Qwen3-4B-Instruct-2507 |
| split | train 900；test 2250 |
| menu | full，固定 15 env/33 tools |
| prompt | current + no_reasoning |
| token position | 最后一个有效输入 token |
| activation | `SiLU(gate_proj(x)) * up_proj(x)` |
| shape | `[N, 36, 9728]` |
| 存储 dtype | float16；打分转 float64/float32 |
| selection split | 只使用 train |
| control | one-vs-rest；每个 difficulty 对 target/control 双方按最大可行配额无放回采样 |
| control seed | 42 |

发现矩阵共 9 组：

```text
rho ∈ {0.001, 0.003, 0.005}
activation_variant ∈ {signed, positive, abs}
```

对每个 `rho × variant × class × layer`：

```text
k_l = floor(rho * intermediate_size_l)
T_class = TopK(saliency on class data)
T_control = TopK(saliency on balanced one-vs-rest control)
N_class = T_class - T_control
```

差集后保留真实数量，不补随机神经元。主设置固定为 `rho=0.003 + signed`；其余 8 组仅作 rho/activation-variant 稳健性，不允许在 test 因果结果上择优。

若某个配置使 `k_l=0`，实现会直接报错，不会暗中提升为 1；本项目的 9728 维 FFN 与三档 rho 均不会触发该错误。

由于 full train 的 `NONE=498`，而 `A+B+C=402`，NONE control 不可能无放回取得 498 个样本。实现对每个 difficulty 使用 `q_d=min(target_d, rest_d)`，再从双方各抽 `q_d`；所有池容量、实际配额和双方 task IDs 都写入 manifest，不做有放回重复。

### 4.2 第 6 阶段：因果验证

主矩阵对 primary mask 运行 25 个完整 full-test 条件：

| 条件 | 数量 | 说明 |
|---|---:|---|
| no-mask | 1 | 同一 HF backend 的共享基线 |
| target mask | 4 | 分别 mask `N_A/N_B/N_C/N_NONE` |
| random mask | 20 | 每类 5 seeds，逐层匹配真实差集数量 |

random mask seeds 固定为 `0,1,2,3,4`，并逐层从四类 target union 的补集中采样。mask 覆盖 prompt prefill 与整个 autoregressive decode；每个条件都跑 2250 条 full-menu test，不只跑目标类。

最小主矩阵固定 generation seed 0，共 25 个完整轨迹条件。若资源允许，用完全相同的 25 个条件补 generation seeds 1/2，共 75 个 condition-seed checkpoints；mask seed 与 generation seed 在文件和汇总中是两个独立字段。

必须报告：

- ActionAcc、Macro-F1、Recall_A/B/C/NONE；
- FinalAcc、TotalTC、OverCall、UnderCall、WrongCat；
- invalid/schema/parse failure、mixed-category、exact env/tool 诊断；
- `Recall_target` drop、off-target mean drop 和 final-accuracy drop。

预注册门槛为：

```text
target_recall_drop > mean(random_recall_drop) + std(random_recall_drop)
```

这只是继续训练的门槛，不自动等于统计显著。若 target mask 只降低 final accuracy 而不选择性改变 action/category，第 7 阶段不得以“动作神经元已验证”为前提启动。

### 4.3 第 7 阶段：轨迹和训练

SFT 只使用 900 条 train 派生的数据：

- `gold_action=NONE`：使用 full hard-no-tool 下正确的 direct-answer response，重建到 full/current prompt；
- `gold_action=A/B/C`：使用 scoped/current/no_reasoning 的成功工具轨迹，但把对话尾部移植到重建的 full/current prompt；
- 最终答案错误、首个工具类别错误、INVALID、未 boxed 结束的轨迹全部丢弃并在 manifest 中逐项计数。

所有训练对照共享同一份过滤后 JSONL。主超参：

| 参数 | 值 |
|---|---|
| epochs | 1 |
| learning rate | `1e-5` |
| LoRA rank / alpha | 8 / 16 |
| LoRA dropout | 0 |
| target modules | `gate_proj`, `up_proj` |
| precision | bf16 |
| gradient checkpointing | on |
| max length | 8192；超长直接报错，不截断 |
| global / per-device batch | 8 / 1 |
| scheduler / warmup / weight decay | constant / 0 / 0 |
| seed | 42 |
| loss | assistant response tokens only |
| external logging | none |

训练对照：

| ID | LoRA 行范围 | seeds |
|---|---|---|
| `base` | 不训练 | - |
| `dense_mlp_lora` | gate/up 全部 output rows | 42 |
| `target_neuron_lora` | primary `N_A∪N_B∪N_C∪N_NONE` | 42 |
| `random_neuron_lora` | 逐层同数量补集采样 | 0,1,2 |

LoRA-A 仍是完整可训练矩阵，不能把 mask B 行后的 selected-row 数量冒充真实 trainable parameter count。manifest 必须同时记录 raw trainable parameters 与 effective selected output rows。

### 4.4 第 8 阶段：训练后评估

每个 adapter 在两个 scope 下分别跑 generation seeds `0,1,2`：

- scoped-tools：与 When2Tool 对齐，报告 Accuracy、Total/Avg TC、TCR、TC reduction、Accuracy loss 和 Pareto；
- full-tools：报告 Accuracy、ActionAcc、Macro-F1、ToolNeed-F1、四类 recall、Over/Under/WrongCat 和 mixed-category。

所有方法使用同一份冻结 base-model gold action，禁止为 adapter 重新打标签。比较对象包括 base Default、Prompt-only/Reason-then-Act Pareto、Probe&Prefill threshold sweep、dense/random/target LoRA。

自动汇总只在统一 HF backend 的 base/dense/random/target 之间计算 Pareto、TC reduction、accuracy loss 与 `target - control` delta，并输出 `target_vs_controls.csv/json`。三 generation seeds 的差值属于描述性、探索性 effect size，不写成统计显著。vLLM Prompt/Probe&Prefill 只能进入单列上下文表；未做同后端复现前，不能把它与 HF adapter 的差值解释为纯训练因果效应。

seed panel 只允许严格追加：可以先跑 `[0]` debug，再以 `[0,1,2]` resume；程序会在扩展 manifest 前重验旧 checkpoints，禁止缩减或替换旧 seed。

## 5. 运行前验收

在代码仓库目录执行：

```bash
conda activate ../CallTool_data/conda_envs/calltool_qwen3
python -m pip check
git submodule update --init --recursive
python -m pytest -q
```

运行程序会检查数据、模型、上游 commit、shape、ID 顺序、full-menu SHA 和输出冲突；任何不一致都会直接失败。程序不会自动下载模型或数据，也不会静默改路径。

具体一键命令和单组命令以根 [README](../README.md) 为准。

Stage 5 的输入与输出根已严格分离：`INPUT_RUN_ROOT` 只提供冻结 `data/labels`，`RUN_ROOT` 只决定本次 `stages/05_probing` 输出。默认两者相同以兼容正式运行；复现必须保持正式输入根、改用新的输出根，避免覆盖已有产物。

## 6. 交接验收清单

- [x] 代码仓库提交已推送，远程 worktree clean，submodule commit 正确；
- [x] 基础模型和 When2Tool 路径存在，Stage 5 runtime provenance 已验证；
- [x] 第 5 阶段 train/test activation 均有 manifest、shape、dtype、ID 与输入 SHA；
- [x] 9 组 discovery 产物目录互不覆盖，primary 固定为 `.003_signed`；
- [x] mask JSON 符合 `when2tool-neuron-mask-v1`，每层真实数量与 direction 可审计；
- [ ] 第 6 阶段 25 条件均完整，random 逐层同量且排除 target；
- [ ] target 通过选择性因果门槛后才启动第 7 阶段；
- [ ] SFT retained/dropped 分布完整，四动作均非空；
- [ ] dense/random/target 共享相同数据、步数和超参；
- [ ] 两种 scope 均使用冻结标签和三 generation seeds；
- [x] 当前大 tensor、日志与图片只在数据侧；
- [x] 报告没有把可解码性写成因果性，也没有在 test 上选 rho/variant。

## 7. 已知限制

- 单模型、单 benchmark、single-hop 不能支持跨模型或真实开放工具生态的普适结论。
- A/B/C 与环境语义绑定；现有 set difference 不能完全排除 topic neuron。
- full-menu 固定工具顺序可能引入位置偏差；工具顺序 counterbalancing 尚未进入主矩阵。
- 全程 mask 同时影响决策、工具参数生成与最终回答；必须依靠类别选择性和错误诊断解释。
- 原始 single-run no-tool 标签存在潜在随机性；本阶段通过冻结同一标签保持方法间公平，但没有把它提升为外部真值。

## 8. 第 5 阶段实际结果

### 8.1 运行与硬审计

正式运行日期为 2026-07-22，生成代码为 commit `3f4bf60c1b9e34665fbfcd22ea830950dac2ce4e`，Stage 5 runtime receipt SHA256 为 `0a78f15d5327f90038792213da5a7798896d0e79532ee47d6e4b44541d12ed24`。阶段目录共有 70 个文件、`2,214,392,758` bytes（约 2.06 GiB）。

| split | shape | dtype | tensor bytes | tensor SHA256 | manifest SHA256 |
|---|---|---|---:|---|---|
| train | `[900,36,9728]` | float16 | 630,375,615 | `c35675432b2ad338a95462b80c5d3b6e37edeba601d4a18318a92170b9337db4` | `21f4b9ed592b5b357e0e6de60ce1296a7ccd14e466b598f73240d688e72fbefb` |
| test | `[2250,36,9728]` | float16 | 1,575,937,215 | `712a6cc1f45ed68217c6e0c27e15a7fcf55a511b38408cc3ad548496e62af719` | `dfaee8ff42e258feee6fb6c2b3110a4d3ac575f6925d369caa5940b375e09bbb` |

共同配置 SHA256 为 `0cb3ad38b4ef42b6431a5f41fac8d90d8f8655ca1aa4019313599c4a89a37d8c`，模型配置 SHA256 为 `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`，`down_proj` column norms SHA256 为 `27e1fbcc4960c700ef7855ada7130e2640d4de2b732510a313aa7a6c37ed7d46`。train/test 中每条样本的 menu SHA 都等于冻结 full-menu SHA `9fe32b5541d03e6948982b1669fb0d289325f0124ef204d159c239487766f117`。

9/9 discovery 目录完整，每组恰有 mask JSON、neuron CSV、probe JSON、probe model 和 3 张 PNG。独立硬审计已经通过：

- 9 个 mask 的 canonical JSON SHA 与各自 probe receipt 完全一致；
- 9 组共享同一 selection hashes、evaluation hashes、control sample receipt、生成 commit 和 runtime receipt；
- `selection_split=train` 且 `test_used_for_selection=false`；
- activation tensor、down-norm、ID、task、label、config、mask union SHA 均与 manifest 一致；
- primary 仍是预注册的 `.003_signed`，没有依据 test 指标改选。

Primary 的 one-vs-rest control seed 为 42。实际每侧样本数及 difficulty 分布如下；target 与 control 两侧的分布逐项相同。

| class | target pool / rest pool | actual each side | easy / medium / hard |
|---|---:|---:|---|
| NONE | 498 / 402 | 246 | 77 / 97 / 72 |
| A | 125 / 775 | 125 | 6 / 35 / 84 |
| B | 193 / 707 | 193 | 46 / 47 / 100 |
| C | 84 / 816 | 84 | 25 / 15 / 44 |

### 8.2 九组探针结果

下表两个指标列都按 `Accuracy / Balanced Accuracy / Macro-F1 / AUROC` 排列；类别数量按 `NONE/A/B/C` 排列。9 组是预先定义的设计条件，不是随机重复，不能对其求 mean±std。

| rho | variant | unique / assignments | NONE/A/B/C | binary 四指标 | 四动作四指标 |
|---:|---|---:|---|---|---|
| .001 | signed | 103 / 119 | 17/35/36/31 | .6422/.6231/.5771/.8451 | .5267/.2500/.1725/.9056 |
| .001 | positive | 125 / 154 | 28/43/47/36 | .6067/.5846/.5091/.8627 | .5267/.2500/.1725/.9132 |
| .001 | abs | 101 / 117 | 17/34/35/31 | .6404/.6212/.5746/.8455 | .5267/.2500/.1725/.9056 |
| **.003** | **signed（primary）** | **306 / 375** | **58/110/109/98** | **.8018/.7923/.7917/.8827** | **.6031/.3365/.3101/.9349** |
| .003 | positive | 312 / 371 | 58/108/115/90 | .7920/.7831/.7826/.8887 | .6378/.3759/.3531/.9370 |
| .003 | abs | 306 / 370 | 57/109/106/98 | .8013/.7918/.7911/.8844 | .6076/.3415/.3161/.9361 |
| .005 | signed | 512 / 608 | 99/182/186/141 | .8200/.8121/.8131/.9016 | .7244/.4906/.4901/.9452 |
| .005 | positive | 482 / 580 | 97/176/167/140 | .7933/.7828/.7808/.8965 | .6960/.4426/.4137/.9442 |
| .005 | abs | 497 / 590 | 95/178/181/136 | .8213/.8134/.8145/.9019 | .7187/.4836/.4843/.9448 |

四动作 majority classifier 的 Accuracy/Balanced Accuracy/Macro-F1 为 `.5267/.25/.1725`。Primary 的四动作 Accuracy、Balanced Accuracy 和 Macro-F1 分别只高 `.0764/.0865/.1376`；OVR-AUROC `.9349` 则说明 one-vs-rest 排序信号较强。两者背离意味着不能仅凭 AUROC 宣称路由分类可靠。

与既有 full residual probe 相比，primary sparse FFN 的四动作 Accuracy/Balanced Accuracy/Macro-F1/AUROC 分别低 `.2538/.4475/.4984/.0284`。因此不能写成“306 个神经元保留了大部分动作能力”或“已经找到主要决策位置”。

### 8.3 Primary 神经元结构

每类每层先取 29 个候选，再做 target-control set difference；`survival` 是最终数量除以 `36×29`，不是预设保留率。early/middle/late 固定定义为层 `0-11/12-23/24-35`；normalized entropy 为 `-Σ p_l log(p_l) / log(36)`。

| class | neurons | survival | direction `- / +` | early / middle / late | 数量最高的层 |
|---|---:|---:|---:|---|---|
| NONE | 58 | 5.56% | 24 / 34 | 12.1% / 41.4% / 46.6% | 21-25（各 4） |
| A | 110 | 10.54% | 53 / 57 | 8.2% / 44.5% / 47.3% | 23（9），21/22（各 8） |
| B | 109 | 10.44% | 48 / 61 | 11.0% / 39.4% / 49.5% | 25（8），23（7），20/21/22（各 6） |
| C | 98 | 9.39% | 47 / 51 | 11.2% / 38.8% / 50.0% | 20/25（各 9），22/24/28（各 7） |

全部 375 个 class assignments 合并后，early/middle/late 占比约为 `10.4%/41.1%/48.5%`。各类 normalized layer entropy 为 `.892-.917`，所以正确表述是“后半层偏重、但仍跨层分布”，不是“集中在少数层”。

306 个 unique features 中，241 个只属于一个动作、61 个属于两个动作、4 个属于三个动作，没有同时属于四动作的 feature。类别间 Jaccard：

|  | NONE | A | B | C |
|---|---:|---:|---:|---:|
| NONE | 1 | .0839 | .0570 | .0833 |
| A | .0839 | 1 | .0631 | .1123 |
| B | .0570 | .0631 | 1 | .0248 |
| C | .0833 | .1123 | .0248 | 1 |

这些集合大体区分但并非互斥；低重叠只能提供候选类别选择性，不能代替消融验证。

### 8.4 稳健性与假设判断

- `.001` 的四动作 hard decision 在三种 variant 下都退化到 majority，说明极稀疏定位不稳健。
- `.003` 的四动作 Balanced Accuracy 仅 `.336-.376`；primary 不是同组 test 指标最高的配置，但仍必须保持预注册选择。
- `.005` 的四动作 Balanced Accuracy/Macro-F1 提升到 `.443-.491`；存在明确 feature-budget/rho 依赖，改善不能全部解释为更精准定位。
- Primary 与 `.003_abs` 的 union Jaccard 为 `.9245`，是同 rho 的正向稳定性；与 `.003_positive` 仅 `.0369`，说明对 activation transform 敏感。
- signed 的 primary 与 `.001_signed/.005_signed` union Jaccard 仅 `.0124/.0863`，说明具体 neuron 身份对 rho 高度不稳定。
- 当前没有同维度 random-neuron decoding probe、置信区间或显著性检验，因此不能声称这 306 个 neurons 是唯一、充分、必要或统计显著优于随机的子集。

| 假设 | 当前判断 | 理由 |
|---|---|---|
| H1 | **有限但明确支持** | binary 证据较强；四动作 test hard metrics 有限超过 prior，OVR 排序信号较强，但明显弱于 residual 且依赖 feature budget |
| H2 | **尚不能判断；值得按冻结 primary 进入 Stage 6** | 需要 target 相对同层同量 random 的目标 recall drop、off-target、INVALID/parse、FinalAcc 联合证据 |
| H3 | **尚不能判断** | Stage 7-8 未运行，不能从 probe 结果推断 masked-LoRA 的训练收益 |

Stage 6 的关键区分是：target mask 是否选择性改变动作类别，还是只破坏晚层工具格式/通用生成。若目标 recall 下降同时伴随所有类别、parse/schema、FinalAcc 普遍恶化，只能支持“高影响输出神经元”，不能支持“类别专属路由”。无论 Stage 6 阳性或阴性都应保留：阳性支持局部因果贡献；阴性则与强 residual probe 一起指向分布式或冗余表征。
