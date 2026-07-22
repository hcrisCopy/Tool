# CallTool：能力自适应的工具动作神经元实验

## 项目目标与当前状态

本项目在 When2Tool single-hop 上研究一条完整证据链：Qwen3-4B-Instruct-2507 是否把“无需工具 / 应调用 A、B、C 哪类工具”编码为 `NONE/A/B/C` 四动作；这些信号能否在 SwiGLU FFN 中被定位、被选择性消融，并被 neuron-targeted LoRA 用于改善端到端工具路由。

四动作不是人工静态真值。先固定基础模型在 hard-no-tool 下是否答对：答对记为 `NONE`，答错时再由任务所属环境映射到 `A/B/C`。因此标签表达的是“对这个基础模型而言应该采取什么动作”。

| 阶段 | 状态 | 本次交付 |
|---|---|---|
| 数据、标签、residual probe | 已完成必要前置 | full train/test、冻结标签、binary/四动作 probe |
| 05 FFN 探测 | 远程正式实跑完成（2026-07-22；生成提交 `3f4bf60`） | train-only 选神经元、9 组 mask、MLP probes、图表 |
| 06 因果验证 | 代码与 25 条件矩阵已完成，未正式运行 | no-mask / 4 target / 20 random，原子 checkpoint |
| 07 轨迹与训练 | 代码与固定对照已完成，未正式运行 | target / dense / 3 random masked-LoRA |
| 08 训练后评测 | 代码与 12 格汇总已完成，未正式运行 | 2 scopes × 6 conditions × 3 generation seeds |

已有 residual 特征在 full test 上的结果：binary Accuracy/AUROC 为 `0.8542/0.9239`；四动作 Accuracy/Balanced Accuracy/Macro-F1/OVR-AUROC 为 `0.8569/0.7840/0.8085/0.9633`，多数类基线为 `0.5267`。这只证明可解码，不等于 FFN 因果性。研究主张与最新直接相关工作见[研究定位](docs/RESEARCH_POSITIONING.md)。

第 5 阶段预注册主设置 `.003_signed` 在 held-out test 上得到：binary Accuracy/Balanced Accuracy/Macro-F1/AUROC 为 `0.8018/0.7923/0.7917/0.8827`；四动作对应为 `0.6031/0.3365/0.3101/0.9349`。四动作 hard-decision 指标仅有限超过 majority/prior（Accuracy `0.5267`、Balanced Accuracy `0.25`、Macro-F1 `0.1725`），因此 H1 只获得“有限但明确”的支持；H2/H3 尚未运行，不能判断。

## 目录与大文件边界

所有命令均从 `CallTool_code/` 执行。代码仓库与大文件目录必须保持同级；共享基础模型放在项目目录的同级模型目录。

```text
<WORKSPACE>/
├── Qwen/Qwen3-4B-Instruct-2507/       # 共享基础模型，不进 Git
└── CallTool/
    ├── CallTool_code/                  # 本 Git 仓库
    │   ├── when2tool_action/           # 核心实现与 CLI
    │   ├── scripts/                    # 第 5-8 阶段一键入口
    │   ├── tests/                      # 协议、mask、LoRA、评测测试
    │   ├── third_party/when2tool/      # 固定 commit 的上游子模块
    │   ├── docs/                       # 方案、研究定位、数据与交接文档
    │   └── reports/                    # 小型阶段报告
    └── CallTool_data/                  # 环境、数据、tensor、adapter、日志和图
        ├── When2Tool/
        ├── conda_envs/calltool_qwen3/
        └── when2tool_precise_shield/
            └── qwen3-4b-instruct-2507/
                ├── data/
                ├── labels/qwen3-4b-instruct-2507/
                ├── probes/             # 前置 residual probes
                ├── outputs/            # 前置行为输出
                └── stages/
                    ├── 05_probing/
                    ├── 06_causal/
                    ├── 07_training/
                    └── 08_evaluation/
```

Git 中禁止放模型、数据集、activation tensor、逐样本轨迹、adapter、训练日志或批量图片。脚本使用相对路径，不含个人电脑、平台、服务器或凭据路径；路径错误会直接报错，不会自动下载或换用缓存。

## 环境配置

### 已交付服务器

使用已经准备好的独立环境：

```bash
cd CallTool_code
conda activate ../CallTool_data/conda_envs/calltool_qwen3
python -m pip check
python -m pytest -q
```

若该环境尚未安装本阶段新增依赖，只在这个服务器环境中执行：

```bash
python -m pip install peft==0.17.1
python -m pip check
```

### 新机器从零配置

```bash
conda create --prefix ../CallTool_data/conda_envs/calltool_qwen3 python=3.11 -y
conda activate ../CallTool_data/conda_envs/calltool_qwen3
python -m pip install -r requirements.txt
git submodule update --init --recursive
python -m pytest -q
```

固定主版本为 Python 3.11、CUDA 12.4、PyTorch 2.6.0、Transformers 4.55.2、vLLM 0.8.5、Accelerate 1.13.0、PEFT 0.17.1。不要在个人电脑环境安装这些依赖。

## 数据和权重准备

| 资源 | 放置路径 | 固定来源/版本 | 用途 |
|---|---|---|---|
| When2Tool | `../CallTool_data/When2Tool` | `cesun/When2Tool` revision `4a6d05f2ac8fc366c9fe5760c395ebe0f0320537` | 900 train / 2250 test |
| Qwen3-4B-Instruct-2507 | `../../Qwen/Qwen3-4B-Instruct-2507` | `Qwen/Qwen3-4B-Instruct-2507`；本地文件 SHA 由 provenance 固定 | backbone |
| 处理数据 | `../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507/data` | 前置脚本生成并已交付 | scoped/full single-hop |
| 冻结标签 | 同一 run root 的 `labels/qwen3-4b-instruct-2507` | 基础模型 seed 0 hard-no-tool | 所有后续方法共同 gold action |
| 第 5-8 阶段产物 | 同一 run root 的 `stages/` | 本仓库脚本生成 | 探测、因果、训练、评测 |

程序完全离线加载模型，`local_files_only=True` 且禁止 remote code。详细文件、revision、SHA 与目录含义见[数据清单](docs/data_manifest.md)。

## 第 5 阶段：FFN 探测

### 复现（使用新的 `RUN_ROOT`）

当前正式产物目录为不可覆盖设计；复现时应在持久终端会话中显式指定一个新的输出 `RUN_ROOT`，不要对已有正式目录直接重跑。脚本从 `INPUT_RUN_ROOT` 读取既有 `data/` 与 `labels/`；其默认值是正式 run root，因此下面只改输出根即可：

```bash
RUN_ROOT=../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507-replica \
  bash scripts/run_stage5_probing.sh
```

如果接手方把输入快照放到了另一个相对目录，必须显式同时设置两者；脚本不会复制、下载或猜测输入：

```bash
INPUT_RUN_ROOT=../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507 \
RUN_ROOT=../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507-replica \
  bash scripts/run_stage5_probing.sh
```

若同一个 replica `RUN_ROOT` 下的 train/test activation 已完整生成、只需运行 9 组 CPU 探测：

```bash
RUN_ROOT=../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507-replica \
  STAGE_START=discovery bash scripts/run_stage5_probing.sh
```

默认 batch size 为 1；确认显存余量后可显式改成 2，但不同 batch size 会写入 manifest：

```bash
RUN_ROOT=../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507-replica \
  EXTRACTION_BATCH_SIZE=2 bash scripts/run_stage5_probing.sh
```

抽取严格使用 full menu、`current + no_reasoning`、最后一个有效输入 token，以及每层 `SiLU(gate_proj(x)) * up_proj(x)`；保存 shape 为 train `[900,36,9728]`、test `[2250,36,9728]` 的 float16 tensor。神经元选择只读取 train，所有 9 个 mask 冻结后才打开 test 数据。

固定矩阵：

| 参数 | 值 |
|---|---|
| `rho` | `0.001, 0.003, 0.005` |
| activation variant | `signed, positive, abs` |
| control seed | `42` |
| control | 每个 difficulty 双方最大可行配额、无放回 one-vs-rest |
| top-k | 每层 `floor(rho × 9728)`，随后 target-control set difference，不补位 |
| primary | `rho=0.003 + signed`，其余仅作稳健性 |
| probe | StandardScaler + L2 logistic，`C=0.0001`，train fit / test only evaluation |

### 已完成的正式结果

本次正式运行由 commit `3f4bf60c1b9e34665fbfcd22ea830950dac2ce4e` 生成；阶段 runtime receipt SHA256 为 `0a78f15d5327f90038792213da5a7798896d0e79532ee47d6e4b44541d12ed24`。Stage 5 共 70 个文件、`2,214,392,758` bytes（约 2.06 GiB）；9/9 discovery 目录完整，每组固定 7 个文件。硬审计已验证 activation tensor SHA、9 个 canonical mask SHA、共享 selection/evaluation/control receipt、生成 commit，以及 `selection_split=train`、`test_used_for_selection=false`。

| split | shape / dtype | tensor bytes | tensor SHA256 | manifest SHA256 |
|---|---|---:|---|---|
| train | `[900,36,9728]` / float16 | 630,375,615 | `c35675432b2ad338a95462b80c5d3b6e37edeba601d4a18318a92170b9337db4` | `21f4b9ed592b5b357e0e6de60ce1296a7ccd14e466b598f73240d688e72fbefb` |
| test | `[2250,36,9728]` / float16 | 1,575,937,215 | `712a6cc1f45ed68217c6e0c27e15a7fcf55a511b38408cc3ad548496e62af719` | `dfaee8ff42e258feee6fb6c2b3110a4d3ac575f6925d369caa5940b375e09bbb` |

下表指标顺序均为 `Accuracy / Balanced Accuracy / Macro-F1 / AUROC`；类别数量顺序为 `NONE/A/B/C`。9 组是设计条件，不是随机重复，不能求 mean±std，也不能按 test 结果改选 primary。

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

Primary 的 306 个 unique features 对应 375 个类别 assignments；241 个只属于一类、61 个属于两类、4 个属于三类。类别间 Jaccard 为 `0.0248-0.1123`，集合大体区分但不互斥；按 assignments 汇总的 early/middle/late（层 0-11/12-23/24-35）占比约为 `10.4%/41.1%/48.5%`，应表述为“后半层偏重、仍跨层分布”。`.003_signed` 与同 rho 的 abs union Jaccard 为 `0.9245`，但与 positive 仅 `0.0369`；signed 跨 rho 仅 `0.0124/0.0863`。具体 neuron 身份和四分类能力明显依赖变体与 feature budget，详细结果、关键 SHA 和结论边界见[阶段交接](docs/STAGE5_PLUS_HANDOFF.md)与[数据清单](docs/data_manifest.md)。

关键输出：

```text
stages/05_probing/
├── activations/{train,test}_mlp_lasttoken_fulltools.pt
├── activations/*_manifest.json
├── discovery/rho0.003_signed/
│   ├── tool_action_neurons.json       # 后续唯一 primary mask
│   ├── top_neurons_by_layer.csv
│   ├── probe_results.json
│   ├── probe_model.pt
│   └── *.png
├── manifests/runtime_provenance.json
└── logs/stage5_probing.log
```

## 第 6 阶段：因果验证

本次不启动正式计算。上级接手后先跑最小主矩阵：

```bash
bash scripts/run_stage6_causal.sh
```

这会在相同 HF backend 下运行 25 个条件：1 个 no-mask、4 个 target mask、每类 5 个 random mask；random seeds 为 `0-4`，generation seed 默认为 `0`。若资源允许，再补齐到三个 generation seeds：

```bash
CAUSAL_GENERATION_SEEDS="0 1 2" bash scripts/run_stage6_causal.sh
```

mask 覆盖 prompt prefill 和整个 autoregressive decode。random neurons 在每层匹配真实 target 数量，并从四类 target union 的补集中采样。正式门槛为：

```text
target_recall_drop > random_recall_drop_mean + random_recall_drop_population_sd
```

同时必须检查 off-target recall 与 FinalAcc；只造成通用能力下降不能称为动作神经元。输出在 `stages/06_causal/conditions/`，其中 `summary.json`、两张 target-vs-random 图和全部轨迹 checkpoint 是训练门控依据。该 runner 明确为单进程；多卡分片时必须保持条件 ID、seed、receipt 和最终汇总合同不变。

## 第 7 阶段：SFT 与 masked LoRA

### 1. 先构造统一训练集

```bash
STAGE_START=source bash scripts/run_stage7_training.sh
```

它会生成 seed-0 scoped/current/no-reasoning 成功工具轨迹，再把 assistant/tool 尾部移植到重建的 full/current 33-tool prompt。`NONE` 使用正确的 full hard-no-tool direct answer。错误答案、错误首类、INVALID、非 boxed 结束都丢弃并逐条记入 manifest；所有训练对照使用同一 JSONL。

### 2. 查看因果门槛后训练

单卡 debug/正式顺序运行：

```bash
CAUSAL_GATE_PASSED=1 STAGE_START=training NPROC_PER_NODE=1 \
  bash scripts/run_stage7_training.sh
```

单机八卡：

```bash
CAUSAL_GATE_PASSED=1 STAGE_START=training NPROC_PER_NODE=8 \
  bash scripts/run_stage7_training.sh
```

脚本依次训练 target、dense、random seed `0/1/2` 五个 adapter。固定训练参数：1 epoch、lr `1e-5`、global/per-device batch `8/1`、LoRA rank/alpha/dropout `8/16/0`、`gate_proj + up_proj`、bf16、gradient checkpointing、max length 8192、constant scheduler、zero warmup/weight decay、seed 42、只在 assistant tokens 上计算 loss、无外部 logger。超过 8192 会直接报错，不截断。

训练中途停止时不要重跑已完成且非空的 adapter 目录。用 `TRAIN_CONDITIONS` 明确列出剩余条件，例如 target 已完成后：

```bash
CAUSAL_GATE_PASSED=1 STAGE_START=training NPROC_PER_NODE=8 \
TRAIN_CONDITIONS="dense random0 random1 random2" \
  bash scripts/run_stage7_training.sh
```

可选 ID 只有 `target dense random0 random1 random2`，重复或未知 ID 会直接报错。source 构造同样不静默覆盖：SFT JSONL 已完整生成后应使用 `STAGE_START=training`，不要再次运行 `STAGE_START=source`。

target/random 模式只允许所选 FFN output rows 的 LoRA-B 产生更新；训练后会断言所有非选中行严格为零。manifest 同时报告 raw trainable parameters 和 effective selected rows，不能把二者混写。

## 第 8 阶段：训练后评测

五个 adapter 完整后运行：

```bash
bash scripts/run_stage8_evaluation.sh
```

固定面板为两个 tool scopes × 六个条件（base、target、dense、三个 random）× generation seeds `0/1/2`。每格都使用 HF backend 和同一份基础模型冻结标签；禁止为 adapter 重新打 gold action。中断后可直接用同一命令 `--resume` 语义继续。

只跑评测、不发布 12 格汇总：

```bash
STAGE_START=evaluation EVAL_GENERATION_SEEDS="0" \
  bash scripts/run_stage8_evaluation.sh
```

debug 无误后，用完整 seed 超集严格追加 `1/2`；程序会先重验 seed 0 checkpoint，禁止缩减或替换旧 seed：

```bash
STAGE_START=evaluation EVAL_GENERATION_SEEDS="0 1 2" \
  bash scripts/run_stage8_evaluation.sh
```

三 seeds 与 12 格均齐全后，单独发布横向表和图：

```bash
STAGE_START=summary bash scripts/run_stage8_evaluation.sh
```

`stages/08_evaluation/comparison/` 会包含逐 seed 表、聚合表、机器可读 Pareto 标记、`target_vs_controls.csv/json`、scoped Accuracy-vs-TotalTC Pareto 图和 full action metrics 图。汇总报告 ToolNeed-F1、TCR，以及 target 相对同 backend base 的 TC reduction/accuracy loss和相对 dense/random3 的描述性 delta；这些 delta 是 exploratory effect size，不自动等于显著性。vLLM Probe&Prefill 只作为上下文 baseline，不与 HF adapter 条件混写成直接因果对照。

## 如何判断跑成功

| 阶段 | 必须看到 | 结论边界 |
|---|---|---|
| 05 | **已验收**：两个 activation manifest、9 个完整 discovery 目录、canonical mask/probe/hash 合同全部通过 | H1 获有限但明确支持；仍不能推出因果性 |
| 06 | `panel_status=complete`、25 条件全齐、target/random 选择性比较 | 才能判断 H2 因果性 |
| 07 | SFT retained/dropped 审计、5 个 `train_manifest.json`、非选中行断言 | 只能说明训练正确执行 |
| 08 | 12 格 × 3 seeds、`comparison_summary.json` 和两张图 | 只能在统一 HF backend 内描述性评估 H3/Pareto；不能据此宣称优于 vLLM Probe&Prefill |

每个阶段都有独立 `runtime_provenance.json`。旧统计 receipt 不覆盖；若仓库在生成报告后前进，恢复旧 checkpoint 时应检出产物记录的生成 commit。

## 常见问题

- `runtime provenance ... does not match`：代码提交、依赖、模型或数据变了；不要覆盖 receipt，检出记录的 commit 或新开实验目录。
- `Refusing to overwrite`：目标已有产物；探测用正确的 `STAGE_START`，因果/评测使用内置 resume，训练用 `TRAIN_CONDITIONS` 只启动尚未完成的条件。
- CUDA OOM：第 5 阶段保持 batch 1；训练开启的 bf16 和 gradient checkpointing 不要关闭。
- 缺 PEFT：只在项目服务器 Conda 环境安装固定的 `peft==0.17.1`。
- adapter 与 condition 不匹配：random 目录、`control_id` 和 `random_mask_seed` 会严格交叉验证，不能改名冒充。
- scoped/full 混用：scoped 使用 `tasks_v1_test_category.json`，full 使用 `tasks_v1_test_fulltools_category.json`；gold labels 始终冻结为 base full hard-no-tool 版本。
- 想在 test 上选最优 rho/variant：禁止。primary 已预注册为 `.003_signed`，其余八组只报告稳健性。

## 文档入口

- [第 5 阶段以后交接与阶段总结](docs/STAGE5_PLUS_HANDOFF.md)
- [研究定位、最新相关工作与允许主张](docs/RESEARCH_POSITIONING.md)
- [完整实验方案](docs/EXPERIMENT_PLAN.md)
- [数据、模型与大文件清单](docs/data_manifest.md)
- [前置统计阶段报告](reports/stages/STAGE_STATISTICS_QWEN3_4B.md)
- [报告索引](reports/README.md)
