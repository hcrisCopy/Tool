# 研究定位与结论边界

## 1. 我们到底在研究什么

项目关心的不是“模型会不会按 schema 调工具”，而是模型内部是否形成了下面这条链：

```text
判断是否需要工具 → 判断需要哪类工具 → 产生正确工具行为
```

任务被统一成四个动作：`NONE`（不需要工具）以及 `A/B/C`（三类工具）。标签由基础模型在 hard-no-tool 条件下能否正确作答，再结合任务类别得到。因此它是**针对当前模型能力的操作性标签**，不是永远不变的人工真值。

本项目要依次回答：

1. 四动作信息能否从 residual 和 FFN intermediate 中读出；
2. 找到的 FFN 子空间是否会选择性影响对应动作；
3. 只训练这个子空间，能否改善端到端工具路由和调用成本。

## 2. 和已有工作的区别

- [When2Tool](https://arxiv.org/abs/2605.09252) 证明工具必要性可从最后输入 token 的全层 residual hidden states 中线性解码，并提出 Probe&Prefill。本项目沿用其数据、环境和主要评测协议。
- [Precise Shield](https://arxiv.org/abs/2604.08881) 提供 FFN saliency、top-k 差集、神经元消融和 LoRA-B 行遮罩的思路。其公开材料不足以逐项复现完整训练设置，所以本项目只能称为 **Precise-Shield-style adaptation**，不能称为官方复现。
- [Tool Calling is Linearly Readable and Steerable in Language Models](https://arxiv.org/abs/2605.07990) 已证明工具身份可以读取和 steering，因此不能再声称“首次发现模型知道该调用哪个工具”。
- [Tool-Cognition-Action](https://arxiv.org/abs/2605.14038) 明确区分 necessity cognition 和实际 action。本文也必须分开报告 gold action 与模型真实生成的 pred action。
- [Heading-Specific Activation Steering](https://arxiv.org/abs/2607.05790) 和 [ASA](https://arxiv.org/abs/2602.04935) 已覆盖部分推理期 steering，但没有完成本项目的“能力自适应四动作标签—类别选择性 FFN 因果验证—稀疏训练”整条链。

如果后续证据成立，本项目可辩护的贡献是：

> 在模型能力自适应标签下，把“是否调用”和“调用哪类工具”统一为四动作；定位类别选择性的 FFN 子空间，用完整工具轨迹验证其因果作用，再检验 targeted LoRA 能否改善准确率—调用成本 Pareto。

创新点在完整证据链，不在某一个单独技术。

## 3. 证据要到什么程度

| 阶段 | 证据 | 允许的结论 |
|---|---|---|
| residual / FFN probe | 冻结 train 选择后在 test 上超过 prior | 特征中存在可解码的四动作结构 |
| target-vs-random 消融 | 目标类 recall 的下降超过同层同量 random，且 off-target 和 FinalAcc 损伤受控 | 该 FFN 子空间对对应动作有局部、选择性的因果贡献 |
| target-vs-random/dense LoRA | 同数据、同训练步数、同评测后 target 更好 | 该子空间可以被训练利用 |

主模型固定为 Qwen3-4B-Instruct-2507，主结果只覆盖 When2Tool single-hop。神经元只在 train split 选择；test 只做冻结后的评测。因果比较必须使用同一 HF backend，不能把 vLLM 与 HF 的差异直接归因于 mask 或训练。

## 4. 必须主动说明的混淆

- **题目主题和动作可能混在一起。** A/B/C 与环境主题相关，候选神经元可能编码数学、检索或代码语义。只有目标 recall 选择性下降、off-target 较稳，才更像动作因果证据。
- **gold action 不等于 pred action。** probe 预测规范标签，不代表模型一定做出该行为。
- **全程 mask 影响不止决策。** 它也可能破坏参数生成和最终答案，所以必须同时看 invalid、parse/schema failure、exact tool/env 和 FinalAcc。
- **固定菜单可能有位置效应。** 未做工具顺序 counterbalancing 前，不能把类别差异全部归因于能力边界。
- **no-tool 标签存在随机性。** 所有方法必须共用同一份冻结标签，不能为训练后模型重新打标签。
- **预注册门槛不等于统计显著。** `target drop > random mean + std` 只决定是否继续训练；没有置信区间或置换检验时，只能说“超过门槛”。

## 5. 当前不能写的结论

- “首次证明 LLM 知道应该调用哪个工具”；
- “找到了普适的工具神经元”；
- “这些神经元只编码动作、不编码题目主题”；
- “已经泛化到多模型、多跳或开放工具生态”；
- “probe 准确率证明了因果机制”。

当前结果与 H1/H2/H3 的实际判断见[阶段交接](STAGE5_PLUS_HANDOFF.md)。
