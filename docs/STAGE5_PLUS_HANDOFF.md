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

第 5 阶段实际运行结果完成后应填写本文第 8 节。第 6-8 阶段在本次交接中只准备代码和冻结矩阵，不启动正式计算。

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
                │   ├── metrics/
                │   ├── figures/
                │   ├── logs/
                │   └── manifests/
                ├── 07_training/
                │   ├── sft/
                │   ├── adapters/
                │   ├── logs/
                │   └── manifests/
                └── 08_evaluation/
                    ├── outputs/
                    ├── metrics/
                    ├── figures/
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

差集后保留真实数量，不补随机神经元。主设置固定为 `rho=0.003 + signed`；其余 8 组仅作 rho/sign 稳健性，不允许在 test 因果结果上择优。

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

## 6. 交接验收清单

- [ ] 代码仓库 worktree clean，submodule commit 正确；
- [ ] 基础模型和 When2Tool 路径存在，runtime provenance 可验证；
- [ ] 第 5 阶段 train/test activation 均有 manifest、shape、dtype、ID 与输入 SHA；
- [ ] 9 组 discovery 产物目录互不覆盖，primary 固定为 `.003_signed`；
- [ ] mask JSON 符合 `when2tool-neuron-mask-v1`，每层真实数量与 direction 可审计；
- [ ] 第 6 阶段 25 条件均完整，random 逐层同量且排除 target；
- [ ] target 通过选择性因果门槛后才启动第 7 阶段；
- [ ] SFT retained/dropped 分布完整，四动作均非空；
- [ ] dense/random/target 共享相同数据、步数和超参；
- [ ] 两种 scope 均使用冻结标签和三 generation seeds；
- [ ] 大 tensor、逐样本 JSON、adapter、日志与图片只在数据侧；
- [ ] 报告没有把可解码性写成因果性，也没有在 test 上选 rho/variant。

## 7. 已知限制

- 单模型、单 benchmark、single-hop 不能支持跨模型或真实开放工具生态的普适结论。
- A/B/C 与环境语义绑定；现有 set difference 不能完全排除 topic neuron。
- full-menu 固定工具顺序可能引入位置偏差；工具顺序 counterbalancing 尚未进入主矩阵。
- 全程 mask 同时影响决策、工具参数生成与最终回答；必须依靠类别选择性和错误诊断解释。
- 原始 single-run no-tool 标签存在潜在随机性；本阶段通过冻结同一标签保持方法间公平，但没有把它提升为外部真值。

## 8. 第 5 阶段实际结果

本节在远程探测完成后更新。没有产物前不得提前填写支持 H2/H3 的结论。

| 项目 | 状态/结果 |
|---|---|
| train MLP activation | 待运行 |
| test MLP activation | 待运行 |
| 9 组 neuron discovery | 待运行 |
| primary union MLP probe | 待运行 |
| primary layer distribution/overlap | 待运行 |
| 是否支持 H1 | 待运行后判断 |
| 是否支持 H2 | 必须等第 6 阶段，当前不能判断 |
| 是否支持 H3 | 必须等第 7-8 阶段，当前不能判断 |
