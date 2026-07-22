# 报告目录

- [stages/STAGE_STATISTICS_QWEN3_4B.md](stages/STAGE_STATISTICS_QWEN3_4B.md)：当前方案的 Qwen3-4B 阶段报告；目前状态为 paused/partial，记录已完成 probe 与可恢复行为 checkpoint，最终统计完成后在原文件继续收口。
- [archive/legacy_onset/STAGE_01_ONSET_QWEN3_4B_SEED0.md](archive/legacy_onset/STAGE_01_ONSET_QWEN3_4B_SEED0.md)：已废弃 onset 方案的历史记录；状态为 archived/superseded，不作为当前结论依据。

第 5 阶段以后的代码、冻结实验矩阵、产物交接、结果状态与结论边界统一维护在 [docs/STAGE5_PLUS_HANDOFF.md](../docs/STAGE5_PLUS_HANDOFF.md)。相关工作与创新主张边界见 [docs/RESEARCH_POSITIONING.md](../docs/RESEARCH_POSITIONING.md)。

大体积 CSV、JSON、PNG、hidden states、probe 权重和运行日志不进入 Git；它们保存在同级 `../CallTool_data/when2tool_precise_shield/<model>/`，并由阶段报告和机器清单引用。
