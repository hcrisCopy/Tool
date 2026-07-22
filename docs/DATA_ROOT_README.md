# CallTool_data 说明

本目录只保存不进入 Git 的环境和大型产物；其维护代码位于同级 `../CallTool_code/`。

```text
CallTool_data/
├── README.md
├── When2Tool/                         # 固定上游数据快照
├── cache/pip/                         # 可再生安装缓存，不是正式产物
├── conda_envs/calltool_qwen3/        # 独立运行环境
├── environment.lock.txt              # 本次环境冻结记录
├── hfd.sh                             # 通用资源下载辅助脚本，不含凭据
└── when2tool_precise_shield/
    └── qwen3-4b-instruct-2507/
        ├── data/                             # scoped/full-tools 处理数据及 manifest
        ├── labels/qwen3-4b-instruct-2507/   # 三种标签口径及生成审计
        ├── probes/
        │   ├── fulltools/                   # binary/四动作/逐层/跨环境 probe
        │   ├── scoped/                      # 当前 scoped-adapted binary probe
        │   └── scoped_original_w2t/         # 原始 W2T probe、receipt、audit_source
        ├── outputs/
        │   ├── fulltools/                   # full-menu prompt 与 P&P 轨迹（待生成）
        │   ├── scoped_adapted/              # 当前 scoped prompt 与 P&P 轨迹
        │   └── scoped_original_w2t/         # 严格重标与原始 probe P&P（待生成）
        ├── analysis/
        │   ├── fulltools/                   # full-menu 统计与图表（待生成）
        │   ├── scoped_adapted/              # adapted scoped 统计与图表（待生成）
        │   └── scoped_original_w2t/         # original-W2T 统计与图表（待生成）
        ├── manifests/                       # runtime provenance、formal audit、handoff
        ├── stages/
        │   ├── 05_probing/                 # 已完成：MLP activation、9 组 mask/probe、图和日志
        │   ├── 06_causal/                  # 待运行：25 条件 target/random mask 轨迹与汇总
        │   ├── 07_training/                # 待运行：SFT 审计与 target/dense/random adapters
        │   └── 08_evaluation/              # 待运行：scoped/full 评测、横向表与 Pareto 图
        ├── logs/
        │   ├── setup/                       # 环境安装记录
        │   ├── labels/                      # 标签生成日志
        │   ├── hidden/                      # hidden 抽取日志
        │   ├── probes/                      # probe 训练与评测日志
        │   └── behavior/                    # 行为生成日志
        └── reports/                         # Git 阶段报告的普通文件副本
```

主 run root 为 `when2tool_precise_shield/qwen3-4b-instruct-2507/`。其中：

- `manifests/runtime_provenance.json` 固定行为生成时的代码、配置、GPU、依赖、数据和模型文件 SHA；
- `manifests/formal_stage_audit.json` 与 `manifests/stage_handoff.json` 只在完整正式面板和三套统计都结束后生成；当前暂停状态下不存在是有意的；
- `reports/STAGE_STATISTICS_QWEN3_4B.md` 是 Git 阶段报告的普通文件副本，当前记录 partial checkpoint；
- `stages/05_probing/manifests/runtime_provenance.json` 是已完成 Stage 5 的独立 receipt，SHA256 为 `0a78f15d5327f90038792213da5a7798896d0e79532ee47d6e4b44541d12ed24`；
- `analysis/{fulltools,scoped_adapted,scoped_original_w2t}/` 保存统计 CSV、JSON 与关键 PNG；
- `logs/{setup,labels,hidden,probes,behavior}/` 只用于诊断，不在 stage handoff 的哈希范围内。

第 5-8 阶段分别维护自己的 `manifests/runtime_provenance.json`，不覆盖旧统计阶段的全局 receipt。正式阶段布局与最小成功产物：

| 阶段 | 入口 | 最小完整产物 |
|---|---|---|
| 05 | `../CallTool_code/scripts/run_stage5_probing.sh` | **已完成**：两个 activation manifest、9 个 discovery 目录 |
| 06 | `../CallTool_code/scripts/run_stage6_causal.sh` | 25 条件 seed-0 checkpoints、`summary.json` |
| 07 | `../CallTool_code/scripts/run_stage7_training.sh` | SFT manifest、5 个 adapter `train_manifest.json` |
| 08 | `../CallTool_code/scripts/run_stage8_evaluation.sh` | 12 格三 seeds、`comparison_summary.json` |

阶段脚本会校验并记录 receipt SHA、生成 commit、数据/标签/mask SHA。不要手工改 adapter 名称、移动单个 checkpoint 后继续跑，或为训练后模型重新生成 gold labels。

Stage 5 当前共有 70 个文件、`2,214,392,758` bytes（约 2.06 GiB），生成 commit 为 `3f4bf60c1b9e34665fbfcd22ea830950dac2ce4e`。Primary mask 位于 `stages/05_probing/discovery/rho0.003_signed/tool_action_neurons.json`；完整指标、文件 SHA 和 H1/H2/H3 边界见 `../CallTool_code/docs/STAGE5_PLUS_HANDOFF.md` 与 `../CallTool_code/docs/data_manifest.md`。已有正式目录不可覆盖；复现必须指定新的 `RUN_ROOT`。

不要把这些文件复制进 Git 仓库，也不要用缓存中的同名模型或数据覆盖现有快照。资源来源、revision 与已知 SHA 见 `../CallTool_code/docs/data_manifest.md`。
