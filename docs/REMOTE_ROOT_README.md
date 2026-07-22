# CallTool 服务器交接入口

本目录采用代码与大文件分离布局：

```text
CallTool/
├── README.md
├── CallTool_code/   # Git 仓库、配置、测试与小型文档
└── CallTool_data/   # Conda 环境、When2Tool、hidden、行为输出与分析
```

共享基础模型不在项目内重复保存，固定从本目录的同级路径 `../Qwen/Qwen3-4B-Instruct-2507/` 读取。

当前 Qwen3-4B-Instruct-2507 single-hop 精确统计处于主动暂停状态：数据/标签、full 与 scoped residual probes 已完成；scoped prompt 行为只完成一部分可恢复 checkpoint，完整行为面板、Probe&Prefill 和最终统计尚未完成。GPU 已停止。尚未进入 FFN intermediate component `(layer l, component i)` 的神经元筛选、因果 mask 或 LoRA。

权威入口：

- 操作手册：`CallTool_code/README.md`
- 当前阶段报告：`CallTool_code/reports/stages/STAGE_STATISTICS_QWEN3_4B.md`
- 大文件清单：`CallTool_code/docs/data_manifest.md`
- 数据目录说明：`CallTool_data/README.md`
- 生成 provenance：`CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507/manifests/runtime_provenance.json`

完整 formal-stage 尚未结束，因此当前没有伪造 `formal_stage_audit.json` 或 `stage_handoff.json`；暂停点、已完成 seeds 与恢复提交见阶段报告。

从本目录恢复环境与运行测试：

```bash
cd CallTool_code
conda activate ../CallTool_data/conda_envs/calltool_qwen3
python -m pip check
python -m pytest -q
```

Git clone 只包含代码，不能代替 `CallTool_data/` 与共享模型快照。当前 partial 交付先核对 runtime provenance、各产物 manifest/receipt 与阶段报告列出的行为 SHA；完整阶段结束后再按 `stage_handoff.json` 逐文件验证。不要重新创建已经清理的旧 `experiments_2d` 方案树。
