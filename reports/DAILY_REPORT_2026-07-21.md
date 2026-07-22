# 2026-07-21 实验日报（历史记录）

> 这是旧 onset 路线的当天记录。它分析的是最后一个 prompt token 的 residual/hidden 分量，不是当前方案中的 FFN intermediate 神经元，也不是因果结论。

- 模型：Qwen3-4B-Instruct-2507
- 数据集：When2Tool
- 状态：旧方案已归档；当前进度见 [`docs/STAGE5_PLUS_HANDOFF.md`](../docs/STAGE5_PLUS_HANDOFF.md)

## 当天完成

1. 验证模型、数据、独立环境和标签生成流程可以运行。
2. 提取 36 层 hidden states，复现 When2Tool 全层 probe。
3. 用 200 次置换检验分析“是否需要工具”和“A/B/C 工具类型”信号随层数的变化。

## 关键结果

| 结果 | 数值 |
|---|---:|
| Probe train accuracy | 0.9278 |
| Probe test accuracy | 0.8853 |
| Probe test AUROC | 0.9467 |
| Necessity onset | 第 23 层 |
| Necessity 峰值层 | 第 28 层 |
| Necessity 峰值 Z | 44.1159 |
| Necessity `R_peak` | 1.6583 |

工具类型总体曲线在第 20 层达到峰值，但主要由 B 类贡献；A 类只有弱趋势，C 类没有稳定信号。因此不能把第 20 层解释成 A/B/C 共有的统一机制。

## 当时能得出的结论

- “是否需要工具”有强表征，但主要在中后层成熟，不支持“早层已经完成决策”。
- 不同工具类型的表征强度不对称，A/C 仍需更严格的环境控制。
- 这些结果只说明信息可检测，不能证明某个分量具有因果作用。

完整的旧方案审计见 [`archive/legacy_onset/STAGE_01_ONSET_QWEN3_4B_SEED0.md`](archive/legacy_onset/STAGE_01_ONSET_QWEN3_4B_SEED0.md)。
