# CallTool 服务器交接入口

本目录采用代码与大文件分离布局：

```text
CallTool/
├── README.md
├── CallTool_code/   # Git 仓库、配置、测试与小型文档
└── CallTool_data/   # Conda 环境、When2Tool、hidden、行为输出与分析
```

共享基础模型不在项目内重复保存，固定从本目录的同级路径 `../Qwen/Qwen3-4B-Instruct-2507/` 读取。

当前 Qwen3-4B-Instruct-2507 single-hop 已完成数据改造、冻结模型相关标签以及 full/scoped residual probes。第 5 阶段以后代码现已交付：FFN intermediate `(layer, neuron_idx)` 的 9 组探测、25 条件因果 mask、target/dense/random masked-LoRA，以及 scoped/full 训练后评测。第 5 阶段实跑状态见阶段交接文档；第 6-8 阶段仅准备代码和冻结矩阵，不应在未检查因果门槛时直接训练。

权威入口：

- 操作手册：`CallTool_code/README.md`
- 当前阶段报告：`CallTool_code/reports/stages/STAGE_STATISTICS_QWEN3_4B.md`
- 第 5 阶段以后交接：`CallTool_code/docs/STAGE5_PLUS_HANDOFF.md`
- 研究定位与主张边界：`CallTool_code/docs/RESEARCH_POSITIONING.md`
- 大文件清单：`CallTool_code/docs/data_manifest.md`
- 数据目录说明：`CallTool_data/README.md`
- 生成 provenance：`CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507/manifests/runtime_provenance.json`

旧统计行为面板仍是 partial checkpoint；它不阻塞以冻结 labels 开展 FFN 探测，但最终与 Prompt-only/Probe&Prefill 的完整论文比较仍需补齐。不要把旧统计的全局 runtime receipt 覆盖为新代码提交；第 5-8 阶段各自使用 `stages/<stage>/manifests/runtime_provenance.json`。

从本目录恢复环境与运行测试：

```bash
cd CallTool_code
conda activate ../CallTool_data/conda_envs/calltool_qwen3
python -m pip check
python -m pytest -q
```

Git clone 只包含代码，不能代替 `CallTool_data/` 与共享模型快照。当前 partial 交付先核对 runtime provenance、各产物 manifest/receipt 与阶段报告列出的行为 SHA；完整阶段结束后再按 `stage_handoff.json` 逐文件验证。不要重新创建已经清理的旧 `experiments_2d` 方案树。

第 5-8 阶段的一键入口依次为：

```bash
bash scripts/run_stage5_probing.sh
bash scripts/run_stage6_causal.sh
STAGE_START=source bash scripts/run_stage7_training.sh
CAUSAL_GATE_PASSED=1 STAGE_START=training NPROC_PER_NODE=8 bash scripts/run_stage7_training.sh
bash scripts/run_stage8_evaluation.sh
```

具体参数、恢复方式、输出和成功判据只以 `CallTool_code/README.md` 与阶段交接文档为准。
