# 报告索引

当前 Stage 5–8 的进度、结果和运行入口统一见 [`docs/STAGE5_PLUS_HANDOFF.md`](../docs/STAGE5_PLUS_HANDOFF.md)。本目录主要保留前置阶段和旧方案记录。

- [`stages/STAGE_STATISTICS_QWEN3_4B.md`](stages/STAGE_STATISTICS_QWEN3_4B.md)：Stage 5 之前的统计结果，以及尚未完成的旧行为实验 checkpoint。
- [`DAILY_REPORT_2026-07-21.md`](DAILY_REPORT_2026-07-21.md)：2026-07-21 的历史日报，简要记录旧 onset 实验。
- [`archive/legacy_onset/STAGE_01_ONSET_QWEN3_4B_SEED0.md`](archive/legacy_onset/STAGE_01_ONSET_QWEN3_4B_SEED0.md)：已废弃 onset 方案的审计摘要，不作为当前论文结论。

阅读时请注意：旧报告分析的是 residual/hidden 表征，不能证明当前定义下的 FFN 神经元具有因果作用。当前主张边界见 [`docs/RESEARCH_POSITIONING.md`](../docs/RESEARCH_POSITIONING.md)。

大体积数据、权重、图片和日志不进入 Git，统一保存在同级数据目录；具体位置与校验信息见 [`docs/data_manifest.md`](../docs/data_manifest.md)。
