# CallTool 服务器交接入口

## 目录

```text
CallTool/
├── README.md
├── CallTool_code/   # Git 仓库、配置、测试和文档
└── CallTool_data/   # 环境、数据和大型实验产物
```

共享模型放在同级 `../Qwen/Qwen3-4B-Instruct-2507/`，不在项目内重复保存。

## 当前做到哪里

- 已完成数据改造、冻结标签、residual probes 和 Stage 5 FFN 神经元探测。
- Stage 5 有限但明确地支持“FFN 中存在可探测的动作信号”；它仍然不是因果证据。
- Stage 6–8 的因果验证、masked LoRA 和训练后评测代码已准备，尚未正式运行。

## 接手后先做

```bash
cd CallTool_code
conda activate ../CallTool_data/conda_envs/calltool_qwen3
python -m pip check
python -m pytest -q
```

阶段入口依次为：

```bash
python scripts/run_stage5_probing.py --run-root ../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507-replica
python scripts/run_stage6_causal.py
python scripts/run_stage7_training.py --step prepare
python scripts/run_stage7_training.py --step train --causal-gate-passed
python scripts/run_stage8_evaluation.py --step all
```

Stage 5 的正式目录已有产物，不要原地重跑。上面的命令把复现结果写到 `-replica` 目录；`--start activations` 只抽 activation，`--start discovery` 复用完整 activation 做探测。Stage 8 可用 `--step evaluation|summary` 分步恢复。具体参数和输出目录见 `CallTool_code/README.md`。

## 文档入口

- 操作手册：`CallTool_code/README.md`
- Stage 5–8 结果与实验矩阵：`CallTool_code/docs/STAGE5_PLUS_HANDOFF.md`
- 研究定位与结论边界：`CallTool_code/docs/RESEARCH_POSITIONING.md`
- 数据与 SHA：`CallTool_code/docs/data_manifest.md`
- 旧统计阶段：`CallTool_code/reports/stages/STAGE_STATISTICS_QWEN3_4B.md`
- 数据目录说明：`CallTool_data/README.md`
