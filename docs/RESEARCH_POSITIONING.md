# 研究定位与主张边界

## 1. 核心问题

本项目研究的不是模型能否按给定 schema 调用工具，而是一个更细的 knowing-doing 问题：

1. 对当前模型而言，这道题是否真的需要工具；
2. 如果需要，应该选择计算规模、知识边界、执行追踪中的哪类工具；
3. 这种能力自适应的四动作信号是否由类别选择性的 FFN 子空间承载；
4. 只更新该子空间，能否把内部信号更可靠地转成端到端工具行为。

四动作标签 `NONE/A/B/C` 由当前基础模型在 hard-no-tool 条件下的正确性与任务静态类别共同决定。它是模型相关的操作性标签，不是人工声明的永恒真值。

## 2. 与直接相关工作的关系

- [When2Tool](https://arxiv.org/abs/2605.09252) 证明工具必要性可以从最后输入 token 的全层 residual hidden states 线性解码，并用 Probe&Prefill 改善调用成本；[官方代码](https://github.com/Trustworthy-ML-Lab/when2tool)与[数据](https://huggingface.co/datasets/cesun/When2Tool)是本项目的评测基础。
- [Precise Shield](https://arxiv.org/abs/2604.08881) 提供 FFN 中间激活、逐层 saliency、top-k 差集、目标神经元消融和 LoRA-B 行遮罩的思路。论文没有公开足以逐项复现的完整训练超参，关联仓库也没有对应实现，因此本项目应称为 *Precise-Shield-style adaptation*，不能称为官方代码复现。
- [Tool Calling is Linearly Readable and Steerable in Language Models](https://arxiv.org/abs/2605.07990) 已证明工具身份可从 residual representation 读取，并能通过 activation steering 和 patching 改变工具选择。因此“首次发现模型知道该调用哪个工具”不是可成立的创新主张。
- [Model-Adaptive Tool Necessity / Tool-Cognition-Action](https://arxiv.org/abs/2605.14038) 区分 necessity cognition 与实际 action，指出两者间存在 knowing-doing gap；[官方代码](https://github.com/chengez/Tool-Cognition-Action)进一步说明 gold action 与模型生成的 pred action 必须分开报告。
- [Controlling Tool Use with Heading-Specific Activation Steering](https://arxiv.org/abs/2607.05790) 和 [ASA](https://arxiv.org/abs/2602.04935) 已覆盖部分 inference-time 工具调用 steering。它们不等价于能力自适应的四动作 FFN 定位、端到端类别选择性因果消融与稀疏训练。

## 3. 可辩护的创新点

本项目应聚焦以下组合贡献：

> 在模型能力自适应的工具必要性标签下，把“是否调用”和“工具家族”统一为 `NONE/A/B/C` 四动作，定位类别选择性的 FFN 神经元子空间，以完整多轮工具轨迹验证其因果选择性，并检验 neuron-targeted LoRA 能否改善端到端工具路由的准确率-调用成本 Pareto。

创新来自完整证据链，而不是其中任一单独技术：

```text
线性可解码
  -> 行为相关
  -> 类别选择性的因果必要性
  -> 可被稀疏训练利用
  -> 端到端 Pareto 改善
```

## 4. 冻结的实验决策

- 主模型固定为 Qwen3-4B-Instruct-2507，主结果只做 single-hop。
- residual probe 是可解码基线；FFN neuron 严格指每层 SwiGLU intermediate activation 的一个分量 `(layer, neuron_idx)`。
- 神经元选择只使用 train split。test activation 只用于冻结后的 probe/evaluation，不参与 top-k、rho 或 activation variant 选择。
- 主设置预注册为 `rho=0.003, activation_variant=signed`。`rho={0.001,0.003,0.005}` 与 `signed/positive/abs` 的其余组合只作为稳健性结果，不能在 test 上择优。
- 因果主对照为同一 HF backend 下的 no-mask、每类 target mask、逐层等数量 random mask（5 seeds）。vLLM baseline 与 HF mask 条件不直接做数值归因。
- 训练主对照必须共享完全相同的过滤轨迹和训练步数：base、普通 MLP LoRA、random-neuron LoRA、target-neuron LoRA。

## 5. 必须主动报告的混淆

1. **题目语义与动作混淆。** A/B/C 与环境主题绑定，one-vs-rest neuron 可能编码数学、检索、代码语义，而非动作本身。因果结果必须以目标类 recall 的选择性下降为主，并报告 off-target recall 和 final accuracy；只损伤通用解题能力不算成功。
2. **gold 与 pred 不同。** gold action 是规范性的、模型能力相关标签；pred action 是实际生成行为。高 gold-action probe 分数不能直接称为行为决策神经元。
3. **全程 mask 的作用范围更广。** 全生成过程消融可能同时影响工具决定、参数生成和最终回答。必须保留 invalid call、schema/parse failure、exact env/tool 与终止状态诊断。
4. **固定工具菜单存在位置效应。** 当前主实验冻结 33-tool 菜单及其 SHA。未完成工具顺序 counterbalancing 前，不能把类别差异全部归因于能力边界。
5. **单次 no-tool 标签可能不稳定。** 所有后续方法必须冻结同一份基础模型标签；不能为训练后模型重新生成 gold action 并移动评测目标。
6. **target-vs-random 不是统计显著性的充分条件。** `target degradation > random mean + std` 只是预注册门槛。没有配对置信区间或置换检验时，报告应使用“超过门槛”，而不是“显著”。

## 6. 允许与禁止的结论

只有在对应证据完成后才允许写：

- residual/MLP 特征中存在四动作可解码结构；
- 某类 FFN 子空间的消融对该类动作产生强于随机且具有选择性的破坏；
- target-neuron LoRA 在同数据、同参数和同训练步数对照下改善端到端指标。

在当前单模型、单 benchmark 阶段禁止写：

- 首次证明 LLM 知道应该调用哪个工具；
- 找到了普适的“工具神经元”；
- FFN 神经元只编码动作而不编码任务主题；
- 对真实开放工具生态或多跳 agent 已完成泛化；
- 仅凭 probe 准确率证明了因果机制。
