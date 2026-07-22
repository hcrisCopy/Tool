# CallTool：When2Tool 工具决策实验

## 项目简介

本项目基于 Qwen3-4B-Instruct-2507 和 When2Tool，研究模型“是否需要调用工具”以及“应调用 A/B/C 中哪类工具”。当前 Qwen3-4B 统计阶段已按要求暂停，完成项、结果和恢复方式见[阶段报告](reports/stages/STAGE_STATISTICS_QWEN3_4B.md)。

## 目录说明

```text
<WORKSPACE>/
├── Qwen/
│   └── Qwen3-4B-Instruct-2507/   # 共享基础模型
└── CallTool/
    ├── CallTool_code/                         # 本 Git 仓库
    │   ├── when2tool_action/                  # 核心 Python 包
    │   │   ├── configs/                      # 模型、路径和实验协议配置
    │   │   └── scripts/                      # 数据、标签、probe、评测、统计 CLI
    │   ├── scripts/                           # 整阶段 Bash 调度脚本
    │   ├── tests/                             # 单元测试与协议篡改测试
    │   ├── third_party/when2tool/             # 固定 commit 的上游子模块
    │   ├── docs/                              # 实验方案、数据清单和服务器说明
    │   └── reports/
    │       ├── stages/                        # 当前方案阶段报告
    │       └── archive/                       # 已废弃方案的历史报告
    └── CallTool_data/                         # 不进入 Git 的数据与运行产物
        ├── When2Tool/                         # 固定 revision 的原始数据集
        ├── conda_envs/                        # 项目独立 Conda 环境
        ├── cache/                             # 可再生安装缓存，不是正式产物
        └── when2tool_precise_shield/
            └── qwen3-4b-instruct-2507/        # 当前模型的主 run root
                ├── data/                      # category/scoped/full-tools 数据
                ├── labels/                    # hard-no-tool 与 A/B/C/NONE 标签
                ├── probes/                    # hidden、probe 权重和探测图表
                ├── outputs/                   # prompt、P&P 与重标行为轨迹
                ├── analysis/                  # 三种协议的统计 CSV/JSON/PNG
                ├── manifests/                 # provenance、审计和交接清单
                ├── logs/                      # setup/labels/hidden/probes/behavior 日志
                └── reports/                   # 阶段报告的数据盘副本
```

所有命令均从 `CallTool_code/` 执行。仓库目录请保持为 `CallTool_code`；代码使用相对路径，不依赖个人电脑或服务器绝对路径。首次克隆时需同时取得仓库中的 When2Tool 子模块，已有服务器目录无需重复操作。

## 环境配置

```bash
conda create --prefix ../CallTool_data/conda_envs/calltool_qwen3 python=3.11 -y
conda activate ../CallTool_data/conda_envs/calltool_qwen3
pip install -r requirements.txt
```

验证环境：CUDA 12.4、PyTorch 2.6.0、Transformers 4.55.2、vLLM 0.8.5。

## 数据和权重准备

| 资源 | 目标路径 | 来源/版本 | 用途 |
|---|---|---|---|
| When2Tool | `../CallTool_data/When2Tool` | `cesun/When2Tool`，revision `4a6d05f2ac8fc366c9fe5760c395ebe0f0320537` | 训练与测试数据 |
| Qwen3-4B-Instruct-2507 | `../../Qwen/Qwen3-4B-Instruct-2507` | 共享模型目录 | 基础模型 |
| original-W2T baseline | `../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507/probes/scoped_original_w2t` | 随数据交付，或由旧 baseline 审计迁移 | 原论文 scoped 对照 |
| 实验产物 | `../CallTool_data/when2tool_precise_shield/` | 脚本生成 | 标签、probe、轨迹、统计和图表 |

仅需重新准备 When2Tool 时：

```bash
hf download cesun/When2Tool \
  --repo-type dataset \
  --revision 4a6d05f2ac8fc366c9fe5760c395ebe0f0320537 \
  --local-dir ../CallTool_data/When2Tool
```

Git 仓库不包含数据、模型和运行产物。接手已有实验时请使用同一次交付的 `CallTool_data/` 与共享模型快照，不要覆盖后续跑。详细版本和校验信息见[数据清单](docs/data_manifest.md)。

## 运行

从空白产物目录运行完整统计阶段：

```bash
STAGE_START=fresh bash scripts/run_statistics_stage.sh
```

`fresh` 运行前必须预置上表的 original-W2T baseline receipt，或按[数据清单](docs/data_manifest.md)显式提供旧 baseline 根目录完成一次性迁移。

当前服务器已有 partial checkpoint，不能按空白目录运行。请按照[阶段报告](reports/stages/STAGE_STATISTICS_QWEN3_4B.md)中的命令恢复；脚本也支持 `STAGE_START=behavior` 和 `STAGE_START=statistics` 两个分段入口。

## 输出说明

各目录用途见上面的完整目录树。完整统计阶段结束后应生成三套 `analysis/*/summary.json`、`manifests/formal_stage_audit.json` 和 `manifests/stage_handoff.json`。当前 partial checkpoint 尚无这些最终文件是正常状态。

## 文档入口

- [实验方案](docs/EXPERIMENT_PLAN.md)：完整研究与实验设计。
- [数据清单](docs/data_manifest.md)：数据、模型、版本和大文件边界。
- [阶段报告](reports/stages/STAGE_STATISTICS_QWEN3_4B.md)：当前进度、结果、清理记录和恢复命令。
- [报告索引](reports/README.md)：当前报告与旧方案归档。

## 常见问题

- 路径错误：保持 `CallTool_code`、`CallTool_data` 和 `Qwen` 的相对布局。
- 已有输出时拒绝运行：不要覆盖，按阶段报告恢复 checkpoint。
- 缺少数据或模型：程序会直接报错，不会自动下载或改用缓存。
- 版本不匹配：使用 `requirements.txt` 和固定的 When2Tool submodule。
