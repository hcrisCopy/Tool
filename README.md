# When2Tool Full-Menu 工具决策实验

## 项目简介

本项目基于 Qwen3-4B-Instruct-2507 和 When2Tool，研究模型是否需要调用工具，以及需要调用 `A/B/C` 中哪一类工具。核心改造是让每道题面对相同的完整工具菜单，并把模型行为统一为 `NONE/A/B/C` 四类。

当前仓库包含数据改造、模型相关标签、residual probe、行为评测和统计代码。FFN 神经元探测、因果验证与定向训练的完整设计见 [实验方案](docs/EXPERIMENT_PLAN.md)；实际完成进度和阶段结论见 [阶段报告](reports/stages/STAGE_STATISTICS_QWEN3_4B.md)。

## 目录说明

推荐保持以下相对布局，所有命令均从 `CallTool_code/` 执行：

```text
<WORKSPACE>/
├── Qwen/
│   └── Qwen3-4B-Instruct-2507/       # 共享基础模型
└── CallTool/
    ├── CallTool_code/                # 本 Git 仓库
    │   ├── when2tool_action/         # 核心 Python 代码
    │   ├── scripts/                  # 阶段运行脚本
    │   ├── tests/
    │   ├── third_party/when2tool/    # 固定版本的上游代码
    │   ├── docs/
    │   └── reports/
    └── CallTool_data/                # 数据、权重外产物、日志和图表
```

`CallTool_code` 进入 Git；`CallTool_data`、基础模型、hidden states、运行输出和图表不进入 Git。

## 环境配置

```bash
conda create -n calltool_qwen3 python=3.11 -y
conda activate calltool_qwen3
pip install -r requirements.txt
git submodule update --init --recursive
pytest -q
```

验证环境为 CUDA 12.4、PyTorch 2.6.0、Transformers 4.55.2 和 vLLM 0.8.5；Python 依赖已固定在 `requirements.txt`。

## 数据和权重准备

| 资源 | 来源 | 放置路径 | 用途 |
|---|---|---|---|
| When2Tool | [cesun/When2Tool](https://huggingface.co/datasets/cesun/When2Tool) | `../CallTool_data/When2Tool` | 训练集、测试集和工具环境 |
| Qwen3-4B-Instruct-2507 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `../../Qwen/Qwen3-4B-Instruct-2507` | 基础模型 |
| 实验产物 | 阶段脚本生成 | `../CallTool_data/when2tool_precise_shield/` | 标签、probe、轨迹、统计和图表 |

从 Hugging Face 准备新副本：

```bash
mkdir -p ../CallTool_data ../../Qwen

hf download cesun/When2Tool \
  --repo-type dataset \
  --revision 4a6d05f2ac8fc366c9fe5760c395ebe0f0320537 \
  --local-dir ../CallTool_data/When2Tool

hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir ../../Qwen/Qwen3-4B-Instruct-2507
```

接手已有实验时应直接使用交付的数据和模型快照，不要覆盖后重新续跑。精确版本、文件校验和生成数据说明见 [数据清单](docs/data_manifest.md)。

## 运行统计阶段

配置文件：

```text
when2tool_action/configs/qwen3_4b_instruct_2507.yaml
```

从空白产物目录运行完整统计阶段：

```bash
STAGE_START=fresh bash scripts/run_statistics_stage.sh
```

脚本依次完成数据改造、标签生成、hidden states 抽取、probe、行为评测、Probe&Prefill 和统计图表。脚本默认拒绝覆盖已有产物。

当前服务器已有部分 checkpoint，不能按空白目录处理。其完成项、缺失项和精确恢复方式统一记录在 [阶段报告](reports/stages/STAGE_STATISTICS_QWEN3_4B.md)。

## 输出说明

```text
../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507/
├── data/           # category/full-tools 改造数据
├── labels/         # Qwen3-4B-Instruct-2507 的 NONE/A/B/C 标签
├── probes/         # hidden states、probe 权重和探测结果
├── outputs/        # 模型运行轨迹
├── analysis/       # 统计表、summary.json 和关键图表
├── manifests/      # 数据与运行清单
├── logs/           # 运行日志
└── reports/        # 阶段报告副本
```

统计阶段完整结束后，应看到：

```text
analysis/fulltools/summary.json
analysis/scoped_adapted/summary.json
analysis/scoped_original_w2t/summary.json
```

## 文档入口

- [实验方案](docs/EXPERIMENT_PLAN.md)：研究问题、探测、因果验证和训练设计。
- [数据清单](docs/data_manifest.md)：数据/模型来源、版本和相对路径。
- [阶段报告](reports/stages/STAGE_STATISTICS_QWEN3_4B.md)：当前进度、结果、问题和恢复说明。
- [报告索引](reports/README.md)：当前报告与旧方案归档。

## 常见问题

- **路径错误**：保持 `CallTool_code`、`CallTool_data` 和 `Qwen` 的相对布局，或修改 YAML 后重新开始对应实验。
- **已有输出时拒绝运行**：不要直接覆盖；按阶段报告中的 checkpoint 方式恢复。
- **缺少模型或数据**：程序会直接报错，不会自动下载或改用缓存中的其他资源。
