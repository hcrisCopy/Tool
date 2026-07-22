# CallTool_data 说明

这里保存不进入 Git 的环境、数据、模型中间量和实验结果；代码在同级 `../CallTool_code/`。

## 目录

```text
CallTool_data/
├── When2Tool/                         # 固定数据快照
├── conda_envs/calltool_qwen3/        # 项目环境
├── environment.lock.txt              # 环境版本
└── when2tool_precise_shield/
    └── qwen3-4b-instruct-2507/
        ├── data/ labels/              # 处理数据与冻结标签
        ├── probes/ outputs/ analysis/ # 前置实验
        ├── manifests/                 # 前置实验 provenance
        └── stages/                    # Stage 5–8 产物
```

主实验根目录是 `when2tool_precise_shield/qwen3-4b-instruct-2507/`。

## 当前状态

| 阶段 | 状态 | 入口 | 输出目录 |
|---|---|---|---|
| Stage 5 探测 | 已完成 | `../CallTool_code/scripts/run_stage5_probing.py` | `stages/05_probing/` |
| Stage 6 因果验证 | 待运行 | `../CallTool_code/scripts/run_stage6_causal.py` | `stages/06_causal/` |
| Stage 7 训练 | 待 Stage 6 通过后运行 | `../CallTool_code/scripts/run_stage7_training.py` | `stages/07_training/` |
| Stage 8 训练后评测 | 待运行 | `../CallTool_code/scripts/run_stage8_evaluation.py` | `stages/08_evaluation/` |

Stage 5 的 primary mask 是 `stages/05_probing/discovery/rho0.003_signed/tool_action_neurons.json`。每个阶段都有独立的 `manifests/runtime_provenance.json`，不要互相覆盖。

## 使用规则

- 不要把本目录的大文件提交到 Git。
- 不要移动单个 checkpoint 后继续训练，也不要为训练后模型重新生成 gold labels。
- 正式目录已有产物时不要原地重跑；复现应使用新的实验根目录。
- 操作命令见 `../CallTool_code/README.md`。
- 完整来源与 SHA 见 `../CallTool_code/docs/data_manifest.md`，实验结果见 `../CallTool_code/docs/STAGE5_PLUS_HANDOFF.md`。
