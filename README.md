# CallTool：工具动作神经元实验

本项目研究一个问题：Qwen3-4B-Instruct-2507 是否会把“无需工具 / 应调用 A、B、C 哪类工具”编码在 SwiGLU FFN 中，以及这些信号能否被因果干预并用于训练。

动作标签为 `NONE/A/B/C`。其中 `NONE` 表示基础模型在 hard-no-tool 条件下已经答对；只有答错的样本才按任务环境映射到 `A/B/C`。因此，这里的标签描述的是“这个基础模型对当前题目应采取什么动作”，不是人工指定的静态类别。

当前进度：Stage 5 探测已完成；Stage 6 因果验证、Stage 7 训练和 Stage 8 评测的代码已准备好，等待接手方运行。

## 目录

- [快速开始](#快速开始)
- [数据和模型](#数据和模型)
- [代码与产物位置](#代码与产物位置)
- [Stage 5：FFN 探测](#stage-5ffn-探测)
- [Stage 6：因果验证](#stage-6因果验证)
- [Stage 7：SFT 与 masked LoRA](#stage-7sft-与-masked-lora)
- [Stage 8：训练后评测](#stage-8训练后评测)
- [验收标准](#验收标准)
- [文档入口](#文档入口)

## 快速开始

所有命令都在 `CallTool_code/` 下执行。只在项目服务器环境中安装依赖，不要在个人电脑上安装。

使用已经准备好的环境：

```bash
conda activate ../CallTool_data/conda_envs/calltool_qwen3
pip install -e .
pytest -q
```

新机器从零配置：

```bash
git clone --recurse-submodules https://github.com/hcrisCopy/Tool.git CallTool_code
cd CallTool_code
conda create --prefix ../CallTool_data/conda_envs/calltool_qwen3 python=3.11 -y
conda activate ../CallTool_data/conda_envs/calltool_qwen3
pip install -r requirements.txt
pip install -e .
pytest -q
```

## 数据和模型

| 资源 | 下载地址 | 放置路径 |
|---|---|---|
| When2Tool 数据集 | [Hugging Face](https://huggingface.co/datasets/cesun/When2Tool) | `../CallTool_data/When2Tool/` |
| Qwen3-4B-Instruct-2507 | [Hugging Face](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `../../Qwen/Qwen3-4B-Instruct-2507/` |
| When2Tool 上游代码 | [GitHub](https://github.com/Trustworthy-ML-Lab/when2tool) | `third_party/when2tool/`，由 Git submodule 管理 |
| 处理后数据与冻结标签 | 由本项目脚本生成；当前服务器已准备好 | `../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507/` |

程序只从本地加载模型和数据。具体版本、文件校验值及目录含义见[数据清单](docs/data_manifest.md)。

## 代码与产物位置

```text
<WORKSPACE>/
├── Qwen/Qwen3-4B-Instruct-2507/       # 模型权重，不进 Git
└── CallTool/
    ├── CallTool_code/                  # 本仓库
    │   ├── when2tool_action/           # 核心实现与命令行模块
    │   ├── scripts/                    # Stage 5–8 的 Python 入口
    │   ├── tests/
    │   ├── docs/
    │   └── reports/
    └── CallTool_data/                  # 数据、tensor、adapter、日志和图
        └── when2tool_precise_shield/qwen3-4b-instruct-2507/
            ├── data/
            ├── labels/qwen3-4b-instruct-2507/
            └── stages/{05_probing,06_causal,07_training,08_evaluation}/
```

每个阶段的入口、核心代码和输出位置如下。

| 阶段 | Python 入口 | 核心代码 | 输出目录 |
|---|---|---|---|
| 5 探测 | `scripts/run_stage5_probing.py` | `when2tool_action/mlp_activations.py`、`when2tool_action/neuron_probing.py` | `stages/05_probing/` |
| 6 因果 | `scripts/run_stage6_causal.py` | `when2tool_action/neuron_ablation.py`、`when2tool_action/hf_agent.py` | `stages/06_causal/conditions/` |
| 7 训练 | `scripts/run_stage7_training.py` | `when2tool_action/sft.py`、`when2tool_action/masked_lora.py` | `stages/07_training/{sft,adapters}/` |
| 8 评测 | `scripts/run_stage8_evaluation.py` | `when2tool_action/adapter_evaluation.py`、`when2tool_action/hf_agent.py` | `stages/08_evaluation/{outputs,comparison}/` |

## Stage 5：FFN 探测

正式结果已经完成，通常不需要重跑。若要复现，请写入新的输出根目录：

```bash
python scripts/run_stage5_probing.py \
  --run-root ../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507-replica
```

固定设置：`rho=0.001/0.003/0.005` × `signed/positive/abs`，control seed 为 `42`，probe `C=1e-4`；预注册主设置为 `.003_signed`。神经元只用 train split 选择，test split 只用于冻结后的评测。

主设置 `.003_signed` 的 test 结果：

| 任务 | Accuracy | Balanced Accuracy | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| 是否需要工具 | 0.8018 | 0.7923 | 0.7917 | 0.8827 |
| `NONE/A/B/C` 四动作 | 0.6031 | 0.3365 | 0.3101 | 0.9349 |

主设置找到 306 个 unique neurons，对应 375 个类别分配；信号以后半层为主，但仍跨层分布。四动作硬分类指标只有限超过多数类基线，因此当前结果只有限但明确地支持“FFN 中存在可探测的动作信号”，还不能证明因果性。

关键产物：

- `stages/05_probing/activations/`：train/test FFN activation tensor 与 manifest。
- `stages/05_probing/discovery/rho0.003_signed/tool_action_neurons.json`：后续实验唯一使用的 primary mask。
- `stages/05_probing/discovery/*/`：9 组 probe 结果、CSV 和图。

## Stage 6：因果验证

运行主实验：

```bash
python scripts/run_stage6_causal.py
```

默认矩阵为 1 个 no-mask、4 个 target mask、20 个分层匹配 random mask；random mask seed 为 `0–4`，主分析 generation seed 为 `0`。如果需要稳健性结果，可再追加 seed `1/2`，具体命令见[阶段交接](docs/STAGE5_PLUS_HANDOFF.md)。

因果门槛是目标类别 recall 降幅大于 random 的均值加总体标准差；同时必须检查 off-target recall、最终答题准确率和解析失败，避免把通用能力损伤误写成选择性因果效应。这个门槛是预注册的效应量筛选，不是统计显著性检验。

关键产物在 `stages/06_causal/conditions/`：每个条件的轨迹、`summary.json/csv`、`per_condition_metrics.csv`，以及 `recall_drop.png`、`target_vs_random.png`、`confusion_before_after.png`。

## Stage 7：SFT 与 masked LoRA

先构造所有训练条件共用的 SFT 数据：

```bash
python scripts/run_stage7_training.py --step prepare
```

确认 Stage 6 通过因果门槛后，再训练五个 adapter：

```bash
python scripts/run_stage7_training.py --step train --causal-gate-passed
```

五个条件为 `target`、`dense`、`random0`、`random1`、`random2`。固定参数：1 epoch、学习率 `1e-5`、global/per-device batch `8/1`、LoRA rank/alpha/dropout `8/16/0`、作用于 `gate_proj + up_proj`、bf16、gradient checkpointing、max length 8192、seed 42，仅在 assistant tokens 上计算 loss。

关键产物：

- `stages/07_training/sft/`：统一训练 JSONL 与筛选 manifest。
- `stages/07_training/adapters/`：五个 adapter 目录及各自的 `train_manifest.json`。

## Stage 8：训练后评测

五个 adapter 完成后运行：

```bash
python scripts/run_stage8_evaluation.py
```

固定面板为 2 个 tool scopes × 6 个条件（base、target、dense、3 个 random）× generation seeds `0/1/2`。所有条件使用同一个 HF backend 和冻结的基础模型标签。

关键产物：

- `stages/08_evaluation/outputs/`：每格的轨迹和指标。
- `stages/08_evaluation/comparison/`：`comparison_summary.*`、`target_vs_controls.*`、`scoped_accuracy_vs_totaltc.png`、`full_action_metrics.png`。

这里的结果用于判断 target adapter 能否改善准确率—工具调用成本权衡；不能把 HF adapter 结果与旧 vLLM Probe&Prefill 写成直接因果对照。

## 验收标准

| 阶段 | 完成标志 | 能得出的结论 |
|---|---|---|
| 5 | 两个 activation manifest、9 个 discovery 目录完整 | 只能判断 FFN 信号是否可探测 |
| 6 | 25 个条件完成，`panel_status=complete` | 才能判断 target mask 是否有选择性因果效应 |
| 7 | SFT 筛选记录、5 个 `train_manifest.json`、非目标行检查通过 | 只能说明训练按设计执行 |
| 8 | 12 格 × 3 seeds 完整，并生成汇总表和两张主图 | 可评估 target adapter 与成本—准确率权衡 |

遇到输入缺失、产物冲突、参数不合法或运行记录不一致时，程序会直接报错，不会静默跳过。

## 文档入口

- [Stage 5 以后实验参数、运行顺序与阶段总结](docs/STAGE5_PLUS_HANDOFF.md)
- [数据、模型、版本和 SHA 清单](docs/data_manifest.md)
- [研究定位、相关工作与允许主张](docs/RESEARCH_POSITIONING.md)
- [完整实验方案（原文保留）](docs/EXPERIMENT_PLAN.md)
- [历史报告索引](reports/README.md)
