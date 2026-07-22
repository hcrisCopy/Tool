# 旧 onset 方案审计摘要（已归档）

> 本文记录已废弃的 `experiments_2d/onset` 方案，仅供历史审计，不属于当前 full-menu FFN 实验，也不能作为当前论文结论。

- 日期：2026-07-21
- 模型：Qwen3-4B-Instruct-2507
- 数据：When2Tool train/test = 900/2250
- 状态：完成 seed-0 标签、hidden 提取、全层 probe 和 200 次置换扫描；未做神经元筛选或因果干预

## 1. 旧方案怎么测

旧方案分析最后一个 prompt token 的 residual/hidden 分量，而不是当前定义的 FFN intermediate 神经元。每层写入量为 `r_l = h_l - h_(l-1)`；onset 定义为平滑曲线第一次达到全局峰值 95% 的层，不是第一次显著的层。

necessity 比较各类别的 hard-tool-needed 与 easy-no-tool 样本；type 在 easy/hard 内分别做 A/B/C one-vs-rest。置换检验以 environment 为块，避免把环境身份误当成工具类型。

这比原方案文字中的普通逐样本 shuffle 更严格，但会改变 Z、onset 和 `R_peak`。因此它是有意的协议修正，不能称为逐字复现。

## 2. Probe 基线

| 指标 | 结果 |
|---|---:|
| Train accuracy | 0.9278 |
| Test accuracy | 0.8853 |
| Test AUROC | 0.9467 |

## 3. Necessity onset

| 指标 | 结果 |
|---|---:|
| Onset | 第 23 层 |
| 全局峰值层 | 第 28 层 |
| 峰值 Z | 44.1159 |
| `R_peak` | 1.6583 |
| 预注册早层条件 `onset/36 <= 0.45` | 失败 |

A/B/C necessity 对比的 FWER p 均为 0.004975。结果支持“必要性信息跨多层积累，并在中后段成熟”，不支持“早层已经完成形成”。

## 4. Tool type onset

总体曲线在第 20 层达峰，但主要由 B 类贡献：

| 类别 | Easy FWER p | Hard FWER p | 峰值层 | 峰值 Z |
|---|---:|---:|---:|---:|
| A | 0.3881 | 0.2289 | 15 | 1.4493 |
| B | 0.0448 | 0.0597 | 20 | 5.4765 |
| C | 0.2438 | 0.5622 | 20 | 0.3730 |

B 类有明显峰值但 hard 状态仅接近显著，A/C 均不显著。因此不能宣称 A/B/C 存在统一的 type onset。

## 5. 产物状态与结论边界

旧 `experiments_2d` 数据树已在可复用内容迁移并校验后删除，不应重建。迁移审计信息保存在当前数据树的 `probes/scoped_original_w2t/migration_receipt.json`，源码仍可从 Git 历史追溯。

本归档只说明旧 residual/hidden 表征中存在统计信号。它不能证明某个 FFN 神经元具有因果作用，也不能替代当前 Stage 5–8 的探测、因果验证和训练结果。当前入口见 [`docs/STAGE5_PLUS_HANDOFF.md`](../../../docs/STAGE5_PLUS_HANDOFF.md)。
