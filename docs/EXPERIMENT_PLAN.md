# When2Tool + Precise Shield 实验方案

> 这是用户冻结的新方案原文的可交接副本，仅移除了个人电脑文件路径。它是设计依据，不是可直接复制命令的执行手册；文中后续章节的草案文件名可能与最终实现不同。当前阶段实际入口、执行契约、已发现的必要修正（例如标签 12 轮、行为 10 轮）及结果以根 README、`scripts/run_statistics_stage.sh`、代码测试和阶段报告为准；尚未完成的 MLP 神经元、因果 mask 与 LoRA 章节仍是后续计划。

## 0. 目标和边界

本实验的核心目标是：在 When2Tool 上证明 LLM 不仅内部学到了“是否需要调用工具”，也学到了“应该调用哪一类工具”，然后仿照 Precise Shield 的神经元级方法，定位 FFN/MLP 中与四类动作相关的神经元，并通过因果 mask 与定向训练证明这些神经元可以带来优于 When2Tool baseline 的效果。

四类动作定义为：

| 动作标签 | 含义 |
|---|---|
| `A` | 调用 Category A 工具：计算规模类 |
| `B` | 调用 Category B 工具：知识边界类 |
| `C` | 调用 Category C 工具：执行追踪类 |
| `NONE` | 不调用工具，直接回答 |

阶段边界：

- 第一阶段只做 single-hop，因为原始 baseline、900/2250 数据划分、all-layer probe 和 Probe&Prefill 都是 single-hop 主实验。
- multi-hop 只作为后续验证，不作为现阶段必须完成的主结果。
- 主模型先用 `qwen3-4b-instruct`，因为它在原仓库 pipeline 中支持完整 baseline，且计算成本适中。若结果成立，再补 `qwen3-1.7b` 或 `llama3.1-8b` 做泛化验证。
- 不做额外新方法设计。神经元打分、mask、训练尽量贴近 Precise Shield；只做适配到 When2Tool 的必要改造。

## 1. 依据

When2Tool 论文与开源实现提供以下基础：

- 数据：15 个 single-hop 环境，3 个类别，每类 5 个环境；single-hop 训练集 900，测试集 2250。
- 原论文的 `tool_necessary` 标签不是人工 hard/easy/medium，而是模型在 no-tool 设置下是否答对：答对为 `0`，答错为 `1`。
- 原论文 probe 使用最后一个输入 token 对应的全层 hidden states，拼接所有层后训练 L2 logistic regression。
- 原 baseline 是 Prompt-only、Reason-then-Act、Probe&Prefill，主要用 accuracy 和 tool calls 的 tradeoff 比较。

Precise Shield 提供以下可模仿方法：

- 在 FFN 中定义神经元激活，使用中间层激活 `h` 而不是 residual hidden state。
- 每层每个神经元的 saliency 由“平均激活强度 × down projection 影响”得到。
- 每层取 top-k，然后用 set difference 去掉通用神经元，得到目标行为相关神经元。
- 因果验证通过 mask 目标神经元，并与等数量 random neurons 对比。
- 训练通过 masked LoRA/gradient masking，只更新目标神经元对应的 FFN 子空间。

参考材料：

- When2Tool PDF：论文标题 *LLM Agents Already Know When to Call Tools — Even Without Reasoning*
- Precise Shield PDF：以 arXiv `2604.08881` 对应版本为准
- When2Tool paper: https://arxiv.org/abs/2605.09252
- When2Tool code: https://github.com/Trustworthy-ML-Lab/when2tool
- When2Tool dataset: https://huggingface.co/datasets/cesun/When2Tool
- Precise Shield paper: https://arxiv.org/abs/2604.08881

## 2. 数据改造

### 2.1 类别映射

作者在论文附录中给出了环境到类别的映射。实验中固定使用如下映射。

| Category | 环境 | 工具意图 |
|---|---|---|
| `A` | `CalculatorEnv` | 算术表达式 |
| `A` | `StatisticsEnv` | 统计量 |
| `A` | `CountingEnv` | 组合、排列、阶乘 |
| `A` | `MatrixEnv` | 矩阵计算 |
| `A` | `PrimeEnv` | 质数、分解 |
| `B` | `RetrieverEnv` | 本地语料检索 |
| `B` | `HistoricalYearEnv` | 年份查询 |
| `B` | `GameRuleEnv` | 游戏规则查询 |
| `B` | `HashEnv` | 哈希算法 |
| `B` | `DecodingEnv` | 编码/解码 |
| `C` | `ListManipulationEnv` | 列表操作 |
| `C` | `DateTimeEnv` | 日期时间计算 |
| `C` | `CodeExecutorEnv` | 代码执行 |
| `C` | `ScheduleEnv` | 日程冲突/空档 |
| `C` | `RegexMatchEnv` | 正则匹配 |

本地检查结果：15 个环境共 33 个工具名，全部唯一。因此可以用 `tool_name -> env_name -> category` 直接判断模型调用了哪类工具，不需要重命名工具。

### 2.2 新增静态标签

先生成静态类别增强数据：

- `data/tasks_v1_train_category.json`
- `data/tasks_v1_test_category.json`

每条样本保留原字段，同时新增：

```json
{
  "category": "A",
  "category_name": "scale",
  "gold_env_name": "CalculatorEnv",
  "gold_tools": ["evaluate_expression", "get_last_result", "clear_last_result"]
}
```

注意：`category` 是数据集静态标签，表示如果该任务需要工具，正确工具类别是什么。它不是 `tool_necessary`。

### 2.3 完整工具列表改造

原始 When2Tool 其实把题目“喂得太明显”了：每道题只给模型对应环境的一小组工具。

比如一道计算题，原数据只给计算器工具；一道检索题，原数据只给检索工具。这样只能测“用不用工具”，很难测“模型知不知道该用哪一类工具”。

所以我们要把实验改成：每道题都给模型全部 33 个工具，让模型自己从 A/B/C 三类工具里选。这样如果它该用检索却用了计算器，就能被统计成“调错工具类别”。

不要改坏原始数据，另外生成两份新数据：

- `data/tasks_v1_train_fulltools_category.json`
- `data/tasks_v1_test_fulltools_category.json`

具体做法：

- 模型看到完整 33 个工具。
- 正确工具所属的环境保留原来的数据，比如 Retriever 的 corpus、GameRule 的 corpus。
- 其他环境也要能接住模型的错误调用，但可以用空参数 `{}` 初始化。
- `gold_env_name` 和 `category` 只给评测程序用，不能写进 prompt 里提示模型。

实现上有两种方式：

| 方式 | 做法 | 是否推荐 |
|---|---|---|
| 直接改数据 | 把每条样本的 `environments` 改成 15 个 env | 不太推荐，容易破坏原始结构 |
| 加 `--tool_scope full` | 原始样本仍只保存正确 env；评测时临时注入全部工具 | 推荐 |

推荐第二种。说白了就是：数据里仍然记住这题真正属于哪个 env，但模型答题时看到的是所有工具。

### 2.4 模型相关标签

`tool_necessary` 必须按模型生成，不能直接用 easy/medium/hard。

对每个模型、split、prompt mode 保存：

- `labels/{model}/train_labels_no_reasoning_fulltools.json`
- `labels/{model}/test_labels_no_reasoning_fulltools.json`

字段：

```json
{
  "id": 11501,
  "difficulty": "easy",
  "category": "A",
  "gold_env_name": "CalculatorEnv",
  "no_tool_correct": 1,
  "tool_necessary": 0,
  "gold_action": "NONE"
}
```

`gold_action` 不是人工直接标出来的，它由两部分合成：

1. 先看这道题属于哪类工具。
   原始数据里每条题都有 `env_name`，比如 `CalculatorEnv`。根据作者给的映射，`CalculatorEnv -> A`，`RetrieverEnv -> B`，`CodeExecutorEnv -> C`。这个得到静态标签 `category`。

2. 再看这道题对当前模型到底需不需要工具。
   按 When2Tool 的做法，把工具禁掉，让模型直接答。如果直接答对，说明这题对这个模型来说不需要工具；如果直接答错，说明需要工具。

最后合成：

- no-tool 答对：`tool_necessary=0`，`gold_action=NONE`
- no-tool 答错：`tool_necessary=1`，`gold_action=category`，也就是 A/B/C

举例：

| 原始 env | category | no-tool 是否答对 | tool_necessary | gold_action |
|---|---|---:|---:|---|
| `CalculatorEnv` | A | 答对 | 0 | `NONE` |
| `CalculatorEnv` | A | 答错 | 1 | `A` |
| `RetrieverEnv` | B | 答错 | 1 | `B` |
| `CodeExecutorEnv` | C | 答错 | 1 | `C` |

所以 `gold_action=NONE` 和 When2Tool 的 `tool_necessary=0` 是同一个标签来源：不给工具也能答对，所以这题本来不需要工具。区别只是我们把原来的二分类标签扩展成四分类：`NONE/A/B/C`。

## 3. 统计实验

统计实验用于证明两个事实：

1. 模型内部/行为上确实有“是否调工具”的信号。
2. 在完整工具列表下，模型对“调哪类工具”也有可统计的偏好，但 A/B/C 不均衡。

### 3.1 运行设置

主统计先只跑 single-hop，也就是 `tasks_v1_test_fulltools_category.json`。这和 When2Tool 主实验对齐，结论最干净。

多跳暂时不混进主统计表。原因是 multi-hop 每题有 3 步工具链，统计口径会变成“每题是否调对类”还是“每一步是否调对类”，容易和 single-hop 指标混在一起。可以后面单独作为附录实验，用 `tasks_v1_multihop_test.json` 另起一组表。

对 single-hop 测试集跑：

- `current + no_reasoning`
- `necessary_tool + no_reasoning`
- `sparse_tool + no_reasoning`
- Probe&Prefill threshold sweep，仍使用完整工具列表

每个 setting 跑 3 次，保持和 When2Tool 一致。

每条样本记录：

```json
{
  "id": 11501,
  "gold_action": "NONE",
  "pred_action": "A",
  "first_tool_name": "evaluate_expression",
  "first_tool_category": "A",
  "tool_calls": 1,
  "final_correct": false,
  "error_type": "over_call"
}
```

先对齐 When2Tool 的口径：

- When2Tool 主指标看的是整条解题轨迹里的工具调用数量，也就是 `total_tool_calls`。
- 它不是只看第一轮，也不是只看第一次工具调用。
- 所以我们的“是否用了工具”也必须按整条轨迹算：只要最终结束前调用过任何工具，就算用了工具。

在这个基础上，`pred_action` 只是为了做 A/B/C 分类而派生出来：

- 如果整条解题轨迹从开始到最终答案都没有任何工具调用：`NONE`
- 如果中途出现过工具调用：取第一次工具调用的类别，即 `A/B/C`

注意：`pred_action=NONE` 不是“第一轮没用工具”。如果第一轮没用工具，但后面又调用了工具，那它仍然算用了工具，`pred_action` 取第一次工具调用的类别。

多次工具调用时：

- 主指标使用第一次工具调用，因为它代表模型第一次决定“我要用哪类工具”。
- 额外记录 `category_sequence`，用于分析模型是否先错类再改类。
- 如果一题里调用了多个不同类别的工具，标记 `mixed_category_calls=1`。
- 工具调用成本仍然用 `total_tool_calls`，和 When2Tool baseline 完全对齐。

### 3.2 错误类型

| 条件 | `error_type` |
|---|---|
| `gold_action=NONE` 且 `pred_action!=NONE` | `over_call` |
| `gold_action in {A,B,C}` 且 `pred_action=NONE` | `under_call` |
| `gold_action in {A,B,C}` 且 `pred_action` 是错误工具类 | `wrong_category` |
| 类别正确但最终答案错 | `correct_category_wrong_answer` |
| 不调工具且最终答案错 | `direct_answer_wrong` |
| 类别和答案都正确 | `success` |

这张错误表直接对应你的观察：

- A 类该调时是否更愿意调：看 `Recall_A = P(pred_action=A | gold_action=A)`。
- B/C 类该调时调得少：看 `under_call_B/C` 和 `wrong_category_B/C`。
- 不该调时是否过度调用：看 `OverCall = P(pred_action!=NONE | gold_action=NONE)`。

### 3.3 统计表和图

必须输出：

1. 数据标签分布表
   `difficulty x category x tool_necessary`

2. 四分类混淆矩阵
   行为 `gold_action`，列为 `pred_action`，类别为 `NONE/A/B/C`。

3. 类别召回表
   对 A/B/C 分别统计 recall、undercall、wrong-category、final accuracy。

4. 不调工具表
   对 `gold_action=NONE` 统计 no-call precision、overcall rate、final accuracy。

5. 可视化
   - 4x4 confusion heatmap
   - per-category recall bar chart
   - error type stacked bar
   - accuracy vs total tool calls tradeoff，沿用 When2Tool 图形风格

### 3.4 A/B/C 类调工具出错与不调工具出错统计

这一部分单独作为论文里的核心统计结果，直接回答“模型知道该不该调工具、以及该调哪类工具吗”。

对 `gold_action in {A,B,C}` 的样本，分别统计：

| 统计项 | 定义 | 解释 |
|---|---|---|
| `call_correct_category` | `pred_action == gold_action` | 该调工具时，第一次工具调用就是正确类别 |
| `under_call` | `pred_action == NONE` | 该调工具但没有调工具 |
| `wrong_category` | `pred_action in {A,B,C} and pred_action != gold_action` | 调了工具，但调错 A/B/C 类别 |
| `correct_category_wrong_answer` | `pred_action == gold_action and final_correct == false` | 工具类别调对，但最终答案错 |
| `needed_final_success` | `pred_action == gold_action and final_correct == true` | 该类工具决策和最终答案都正确 |

按类别输出：

| Gold class | N | Correct category call | Under-call | Wrong category call | Correct category but wrong answer | Final success |
|---|---:|---:|---:|---:|---:|---:|
| A | | | | | | |
| B | | | | | | |
| C | | | | | | |

对 `gold_action=NONE` 的样本，单独统计“不调工具”相关错误：

| 统计项 | 定义 | 解释 |
|---|---|---|
| `no_call_correct` | `pred_action == NONE and final_correct == true` | 不该调工具，模型也没调，且答对 |
| `direct_answer_wrong` | `pred_action == NONE and final_correct == false` | 不该调工具，模型没调，但直接答错 |
| `over_call_A/B/C` | `pred_action in {A,B,C}` | 不该调工具，但模型调用了 A/B/C 某类工具 |
| `over_call_total` | `pred_action != NONE` | 不该调工具时的总过度调用率 |

按 NONE 类输出：

| Gold class | N | No-call correct | Direct answer wrong | Over-call A | Over-call B | Over-call C | Over-call total |
|---|---:|---:|---:|---:|---:|---:|---:|
| NONE | | | | | | | |

### 3.5 多次调用且跨类别调用统计

还要单独记录一种情况：一题里面模型调用了多次工具，而且这些工具不属于同一类。

比如：

```text
第一次调用 CalculatorEnv -> A
第二次调用 RetrieverEnv -> B
第三次又调用 CodeExecutorEnv -> C
```

这种题说明模型不只是“多调用工具”，而是在不同工具类别之间摇摆。统计时给每条样本新增：

```json
{
  "tool_call_categories": ["A", "B", "C"],
  "unique_tool_call_categories": ["A", "B", "C"],
  "n_tool_call_categories": 3,
  "mixed_category_calls": 1
}
```

判定规则：

- 没有工具调用：`tool_call_categories=[]`，`mixed_category_calls=0`
- 调了一次工具：`mixed_category_calls=0`
- 调了多次工具，但全是同一类：`mixed_category_calls=0`
- 调了多次工具，且出现至少两个不同类别：`mixed_category_calls=1`

输出总表：

| Setting | N | Any tool call | Multi-call tasks | Mixed-category tasks | Mixed / all tasks | Mixed / multi-call tasks |
|---|---:|---:|---:|---:|---:|---:|

按 gold action 分组：

| Gold action | N | Multi-call tasks | Mixed-category tasks | Top category sequences |
|---|---:|---:|---:|---|
| NONE | | | | |
| A | | | | |
| B | | | | |
| C | | | | |

`Top category sequences` 记录最常见的跨类别调用路径，例如：

```text
A->B: 18
B->A: 12
C->A: 9
A->B->C: 3
```

这部分写在统计结果里，用来说明模型有没有“工具类别混乱”的问题。如果 mixed-category 很少，说明模型一般第一次选类后不会乱跳；如果很多，说明完整工具列表下的类别选择不稳定。

论文叙述重点：

- 如果 A 类 `Correct category call` 高，而 B/C 类 `Under-call` 高，可以支持“模型对计算规模边界更敏感，但对知识边界/执行追踪边界更保守或更容易低估工具必要性”。
- 如果 B/C 类 `Wrong category call` 高，说明模型不是单纯不愿调用工具，而是完整工具列表下的类别选择能力不足。
- 如果 NONE 类 `Over-call total` 高，说明模型存在过度调用；如果 `Direct answer wrong` 高，则说明 no-tool 标签和实际直接回答能力之间存在模型不稳定性，需要用 3-run mean±std 报告。

## 4. Baseline 复现和比较指标

### 4.1 原始 baseline

必须先复现 When2Tool 原始 pipeline：

```bash
./run_pipeline.sh qwen3-4b-instruct
```

保留原始 scoped-tool 设置结果，作为论文 baseline 对齐。

核心 baseline：

- Prompt-only：`force_tool/current/necessary_tool/sparse_tool/no_tool`
- Reason-then-Act：同五种 prompt mode，但要求先显式判断工具必要性
- Probe&Prefill：binary probe + threshold sweep

### 4.2 是否调工具指标

沿用 When2Tool：

| 指标 | 含义 |
|---|---|
| `Accuracy` | `\boxed{}` 中最终答案是否正确 |
| `Total TC` | 测试集总工具调用数 |
| `Avg TC` | 平均每题工具调用数 |
| `TCR` | `tool_calls / expected_steps` |
| `TC reduction` | 相对 Default 的工具调用减少比例 |
| `Accuracy loss` | 相对 Default 的准确率下降 |
| `Cost per saved call` | 每节省一次调用付出的准确率损失 |

为了和论文图一致，主图画 `Accuracy vs Total TC` 或 `Accuracy vs Avg TC`。更靠左上表示更好。

### 4.3 调哪类工具指标

原论文没有显式“调哪类工具”指标，因此本实验新增但保持简单：

| 指标 | 定义 |
|---|---|
| `ActionAcc` | `pred_action == gold_action` 的四分类准确率 |
| `MacroF1_action` | `NONE/A/B/C` 四类 macro-F1 |
| `ToolNeed_F1` | 二分类 `tool` vs `NONE` 的 F1 |
| `CategoryAcc_needed` | 在 `gold_action in {A,B,C}` 且模型调用工具时，工具类别是否正确 |
| `Recall_A/B/C` | `P(pred_action=c | gold_action=c)` |
| `OverCall` | `P(pred_action!=NONE | gold_action=NONE)` |
| `UnderCall_c` | `P(pred_action=NONE | gold_action=c)` |
| `WrongCat_c` | `P(pred_action notin {NONE,c} | gold_action=c)` |

类别 baseline 定义为：

- 在完整工具列表下运行 When2Tool 原始方法。
- Prompt-only/Reason-then-Act/Probe&Prefill 只控制用不用工具，不直接告诉模型工具类别。
- 模型实际生成的 first tool call 决定 `pred_action`。

这样可以公平检验：原方法是否只能解决“是否调用”，而我们的神经元训练是否进一步提升“调用哪类”。

## 5. FFN 神经元探测

### 5.1 激活抽取位置

和 When2Tool 对齐：

- 输入仍是完整工具列表 prompt。
- 使用最后一个输入 token 的表示。
- 但不再只取 residual hidden state，而是取每层 MLP/FFN 的中间激活。

对 Qwen/Llama 类 SwiGLU MLP：

```text
h_l(x) = act(gate_proj_l(x)) * up_proj_l(x)
y_l(x) = down_proj_l(h_l(x))
a_{l,i}(x) = h_l(x_last)[i]
```

其中 `i` 是第 `l` 层 MLP intermediate dimension 中的神经元。

保存：

- `activations/{model}/train_mlp_lasttoken_fulltools.pt`
- `activations/{model}/test_mlp_lasttoken_fulltools.pt`
- shape: `[n_tasks, n_layers, intermediate_dim]`

### 5.2 四类数据集

使用模型相关 `gold_action` 构造：

```text
D_A    = {x | gold_action = A}
D_B    = {x | gold_action = B}
D_C    = {x | gold_action = C}
D_NONE = {x | gold_action = NONE}
```

为了模仿 Precise Shield 的 harmful vs benign set difference，每个类别 `c` 配一个 control set：

```text
D_ctrl(A)    = balanced_sample(D_B union D_C union D_NONE)
D_ctrl(B)    = balanced_sample(D_A union D_C union D_NONE)
D_ctrl(C)    = balanced_sample(D_A union D_B union D_NONE)
D_ctrl(NONE) = balanced_sample(D_A union D_B union D_C)
```

采样时尽量按 difficulty 分层匹配，避免 top neurons 只反映 easy/hard 难度。

### 5.3 Precise Shield 风格打分

对每层 `l`、每个数据集 `D`、每个 neuron `i`：

```text
mean_act[l,i,D] = mean_{x in D} a_{l,i}(x)
importance[l,i,D] = || mean_act[l,i,D] * W_down_l[:, i] ||_2
saliency[l,i,D] = importance[l,i,D] / (sum_j importance[l,j,D] + eps)
```

每层取 top-k：

```text
k_l = floor(rho * intermediate_dim_l)
T_D[l] = topk_i saliency[l,i,D]
N_c[l] = T_{D_c}[l] \ T_{D_ctrl(c)}[l]
```

`rho` 是 intervention ratio。主实验先用：

```text
rho in {0.001, 0.003, 0.005}
```

如果计算资源紧张，先跑 `rho=0.003`。

### 5.4 激活正负问题

Precise Shield 公式使用平均激活再乘 down projection 并取 L2 norm。由于 SwiGLU intermediate activation 可能有正有负，现阶段按两种方式都跑，主结果以因果验证更强者为准：

1. signed mean，贴近论文：

```text
mean_act = mean(a)
importance = || mean_act * W_down[:, i] ||_2
```

2. positive/absolute 版本，用于稳健性：

```text
mean_act_pos = mean(max(a, 0))
mean_act_abs = mean(abs(a))
```

报告时不把它包装成新方法，只作为“activation sign handling”确认实验。

每个选中 neuron 额外保存方向：

```text
direction[l,i,c] = sign(mean_act[l,i,D_c] - mean_act[l,i,D_ctrl(c)])
```

这个方向用于后续 activation amplification 或解释，不影响 mask 主实验。

### 5.5 输出表和图

每个 `model x rho x activation_variant` 输出：

- `neuron_masks/{model}/tool_action_neurons_rho{rho}_{variant}.json`
- `neuron_tables/{model}/top_neurons_by_layer.csv`
- `figures/{model}/neuron_layer_distribution.png`
- `figures/{model}/category_overlap_jaccard.png`
- `figures/{model}/saliency_heatmap_by_layer.png`

表格字段：

| 字段 | 含义 |
|---|---|
| `class` | A/B/C/NONE |
| `layer` | 层号 |
| `neuron_idx` | MLP neuron index |
| `saliency_class` | 目标类 saliency |
| `saliency_control` | control saliency |
| `mean_act_class` | 目标类平均激活 |
| `mean_act_control` | control 平均激活 |
| `direction` | 激活方向 |
| `down_norm` | `W_down[:,i]` 的 L2 norm |

## 6. 因果验证

### 6.1 Mask 方法

对选出的 `N_c` 神经元，在模型 forward 中 hook MLP intermediate activation：

```text
h_l[..., selected_indices] = 0
```

主实验在生成全过程 mask，而不只是在 prompt prefill 阶段 mask。这样更接近 Precise Shield 的“移除神经元后行为崩塌”验证。

对照组：

- `no_mask`: 原模型
- `target_mask`: mask `N_c`
- `random_mask`: 每层随机选同数量 neuron，重复 5 个 random seeds

### 6.2 验证逻辑

对每个类别单独验证：

| Mask | 预期变化 |
|---|---|
| mask `N_A` | `Recall_A` 明显下降，A 类 under-call 或 wrong-category 上升 |
| mask `N_B` | `Recall_B` 明显下降，B 类 under-call 或 wrong-category 上升 |
| mask `N_C` | `Recall_C` 明显下降，C 类 under-call 或 wrong-category 上升 |
| mask `N_NONE` | `OverCall` 上升，`gold_action=NONE` 的 no-call recall 下降 |

核心判据：

```text
target_mask degradation > random_mask mean degradation + std
```

至少在 `Recall_c`、`ActionAcc` 或 `MacroF1_action` 中成立。

### 6.3 因果验证表

输出：

| Class masked | Setting | ActionAcc | MacroF1 | Recall_A | Recall_B | Recall_C | Recall_NONE | FinalAcc | TotalTC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | no_mask | | | | | | | | |
| A | random_mask | mean±std | | | | | | | |
| A | target_mask | | | | | | | | |

可视化：

- per-class recall drop bar chart
- target vs random degradation plot
- confusion matrix before/after mask

如果 mask `N_c` 后只降低 final accuracy 但不改变 action/category，则说明这些 neuron 更像解题能力 neuron，不是工具决策 neuron，需要调小 `rho` 或加强 set difference。

## 7. 基于神经元探测的训练

### 7.1 训练目标

训练目标不是全面提升模型能力，而是让模型更好地把内部工具决策信号转化为生成行为，尤其提升：

- 是否调用工具的 tradeoff
- B/C 类该调用时的 recall
- 完整工具列表下的正确工具类别选择

### 7.2 训练数据

仿照 When2Tool 附录的 SFT 数据构造，但训练方式仿照 Precise Shield 的神经元定向更新。

对 900 条 single-hop train：

- 若 `gold_action=NONE`：使用 no-tool direct-answer 轨迹作为 target。
- 若 `gold_action in {A,B,C}`：使用原始 scoped-tool 设置下 Default prompt 产生的成功工具轨迹作为 target。
- 如果工具轨迹最终答案错误，或 first tool category 不等于 gold category，则跳过该样本，避免把错误行为写入训练。

训练样本保存：

- `sft_data/{model}/train_action_trajectories.jsonl`

每条样本包含完整 multi-turn messages，但 loss 只打在 assistant response tokens 上。

### 7.3 Masked LoRA

严格模仿 Precise Shield 的思路：只更新被探测出的 FFN neuron 子空间。

对 decoder-only 模型：

- LoRA 只挂在 MLP 的 `gate_proj` 和 `up_proj`。
- 不挂 attention，不挂 embedding，不挂 lm_head。
- 对 LoRA `B` 矩阵做 row mask，使只有 selected neuron 对应行可以更新。
- 非 selected rows 的 gradient 置零。

使用 union mask：

```text
N_all = N_A union N_B union N_C union N_NONE
```

因为最终训练目标是整体四分类动作，而不是只增强某一类。

可选对照：

- `random_neuron_lora`: 每层同数量 random neurons，训练同样步数。
- 只在因果验证不充分时跑，不作为主线必需。

### 7.4 训练超参

先用小规模、可复现实验：

| 参数 | 值 |
|---|---|
| model | `qwen3-4b-instruct` |
| epochs | 1-2 |
| lr | `1e-5` 或 `2e-5` |
| batch size | 根据显存设定 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| precision | bf16 |
| gradient checkpointing | on |
| loss mask | 只训练 assistant tokens |

如果 1 epoch 已超过 baseline，不继续加复杂训练。

### 7.5 训练后评测

训练后必须在两个设置下评测：

1. 原始 scoped-tool 数据
   用于和 When2Tool 原论文 baseline 对齐。

2. full-tools 数据
   用于检验“调哪类工具”。

对比对象：

- Default prompt
- Best Prompt-only / Reason-then-Act
- Probe&Prefill threshold sweep
- Neuron-targeted LoRA

主成功标准：

- 在 `Accuracy vs TotalTC` 图上，Neuron-targeted LoRA 至少有一个点 Pareto 优于 Probe&Prefill。
- 在 full-tools 设置中，`MacroF1_action` 和 B/C recall 高于 Probe&Prefill full-tools baseline。
- `OverCall` 不明显恶化。

## 8. 需要新增或改造的脚本

> 交接提示：第 8--10 节保留的是冻结设计稿中的拟议名称和伪命令，不是当前文件树的可执行入口；其中 `src/*`、仓库内大文件目录和自动下载写法均未采用。实际实现位于 `when2tool_action/`，唯一操作入口见根 README 与 `scripts/run_statistics_stage.sh`，不得照抄本节草案执行。

尽量保持原仓库结构，只加小脚本和少量 flag。

| 脚本 | 作用 |
|---|---|
| `src/prepare_category_fulltools.py` | 添加 `category/gold_env_name`，生成 full-tools 数据 |
| `src/extract_tool_labels.py` | 跑 hard_no_tool，生成模型相关 `tool_necessary/gold_action` |
| `src/run_eval.py --tool_scope full` | 在完整工具列表下评测 |
| `src/collect_action_stats.py` | 生成四分类混淆矩阵和错误类型统计 |
| `src/extract_mlp_activations.py` | hook MLP intermediate，抽最后 token 激活 |
| `src/probe_tool_action_neurons.py` | Precise Shield 风格 saliency/top-k/set-difference |
| `src/run_neuron_ablation.py` | target mask vs random mask 因果验证 |
| `src/train_masked_lora.py` | masked LoRA 训练 |
| `src/plot_action_results.py` | 画 confusion、recall、tradeoff 图 |

`src/screen_layers.py` 可以保留为辅助诊断，但主实验必须所有层 top-k，不只筛一两个层。

## 9. 推荐执行顺序

### Step 1: 复现原始 baseline

```bash
cd when2tool_repo
./run_pipeline.sh qwen3-4b-instruct
```

确认得到：

- `outputs/qwen3-4b-instruct/*`
- `probe_data/qwen3-4b-instruct/*`
- `figures/qwen3-4b-instruct_tradeoff.pdf`

### Step 2: 生成类别和 full-tools 数据

```bash
python src/prepare_category_fulltools.py \
  --train data/tasks_v1_train.json \
  --test data/tasks_v1_test.json \
  --output_dir data
```

如果本地没有 data 文件，先用仓库自动从 HF 加载或运行：

```bash
bash generate_data.sh
```

### Step 3: 生成模型相关 tool_necessary 标签

```bash
python src/extract_tool_labels.py \
  --model_path qwen3-4b-instruct \
  --data_path data/tasks_v1_train_fulltools_category.json \
  --data_path_test data/tasks_v1_test_fulltools_category.json \
  --output_dir labels/qwen3-4b-instruct \
  --tool_scope full
```

### Step 4: 跑 full-tools 统计

```bash
python src/run_eval.py \
  --model_path qwen3-4b-instruct \
  --data_path data/tasks_v1_test_fulltools_category.json \
  --prompt_mode current \
  --reasoning_mode no_reasoning \
  --tool_scope full \
  --record_mode lite \
  --output_path outputs_fulltools/qwen3-4b-instruct/current_no_reasoning.json \
  --n_runs 3

python src/collect_action_stats.py \
  --outputs outputs_fulltools/qwen3-4b-instruct/current_no_reasoning.json \
  --labels labels/qwen3-4b-instruct/test_labels_no_reasoning_fulltools.json \
  --output_dir analysis/qwen3-4b-instruct
```

### Step 5: 抽取 MLP 激活

```bash
python src/extract_mlp_activations.py \
  --model_path qwen3-4b-instruct \
  --data_path data/tasks_v1_train_fulltools_category.json \
  --data_path_test data/tasks_v1_test_fulltools_category.json \
  --labels_dir labels/qwen3-4b-instruct \
  --output_dir activations/qwen3-4b-instruct \
  --tool_scope full
```

### Step 6: 探测四类神经元

```bash
python src/probe_tool_action_neurons.py \
  --model_path qwen3-4b-instruct \
  --activations_dir activations/qwen3-4b-instruct \
  --labels_dir labels/qwen3-4b-instruct \
  --rho 0.003 \
  --activation_variant signed \
  --output_dir neuron_masks/qwen3-4b-instruct
```

再跑：

```bash
--activation_variant positive
--activation_variant abs
```

### Step 7: 因果 mask 验证

```bash
python src/run_neuron_ablation.py \
  --model_path qwen3-4b-instruct \
  --data_path data/tasks_v1_test_fulltools_category.json \
  --labels labels/qwen3-4b-instruct/test_labels_no_reasoning_fulltools.json \
  --mask_path neuron_masks/qwen3-4b-instruct/tool_action_neurons_rho0.003_signed.json \
  --classes A B C NONE \
  --random_seeds 5 \
  --output_dir ablations/qwen3-4b-instruct
```

### Step 8: 训练 masked LoRA

只有在 Step 7 证明 target mask 明显强于 random mask 后再训练。

```bash
python src/train_masked_lora.py \
  --model_path qwen3-4b-instruct \
  --train_data sft_data/qwen3-4b-instruct/train_action_trajectories.jsonl \
  --mask_path neuron_masks/qwen3-4b-instruct/tool_action_neurons_rho0.003_signed.json \
  --output_dir checkpoints/qwen3-4b-instruct-neuron-lora \
  --epochs 1 \
  --lr 1e-5 \
  --lora_rank 8 \
  --lora_alpha 16
```

### Step 9: 训练后评测

```bash
python src/run_eval.py \
  --model_path checkpoints/qwen3-4b-instruct-neuron-lora \
  --data_path data/tasks_v1_test_fulltools_category.json \
  --prompt_mode current \
  --reasoning_mode no_reasoning \
  --tool_scope full \
  --output_path outputs_fulltools/qwen3-4b-instruct-neuron-lora/current_no_reasoning.json \
  --n_runs 3
```

再生成：

- action metrics
- baseline comparison table
- accuracy vs total tool calls plot

## 10. 最终结果表格清单

### 表 1: 数据和标签统计

| Split | Category | Difficulty | N | no-tool correct | tool necessary |
|---|---|---|---:|---:|---:|

### 表 2: Full-tools 行为混淆矩阵

| Gold \ Pred | NONE | A | B | C |
|---|---:|---:|---:|---:|
| NONE | | | | |
| A | | | | |
| B | | | | |
| C | | | | |

### 表 3: A/B/C 调用错误分析

| Gold class | Recall | UnderCall | WrongCat | CorrectCatWrongAnswer | FinalAcc |
|---|---:|---:|---:|---:|---:|

### 表 4: Probe 解码能力

| Feature | Task | Acc | MacroF1 | AUROC/OVR-AUROC |
|---|---|---:|---:|---:|
| residual hidden all-layer | tool vs no-tool | | | |
| residual hidden all-layer | NONE/A/B/C | | | |
| MLP activation neurons | NONE/A/B/C | | | |

### 表 5: 神经元分布

| Class | rho | total neurons | peak layers | top layers | overlap with others |
|---|---:|---:|---|---|---:|

### 表 6: 因果 mask

| Mask class | Mask type | ActionAcc | MacroF1 | Recall target | FinalAcc target | TotalTC |
|---|---|---:|---:|---:|---:|---:|

### 表 7: 训练后 baseline 对比

| Method | Tool scope | Accuracy | TotalTC | TC reduction | ActionAcc | MacroF1 | Recall_A | Recall_B | Recall_C | OverCall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## 11. 成功判据

> 交接提示：下列条目保留原始方案语义，但统计阶段的正式判据已被当前冻结契约取代。由于 `NONE` 是多数类，不能再以 `ActionAcc > 25%` 单独判定成功；必须同时报告 balanced accuracy、Macro-F1、逐类 recall、majority/prior-matched baseline 与配对置信区间，详见根 README 和阶段报告。

最低成功标准：

1. 统计阶段能显示四分类不是随机：`ActionAcc` 明显高于 25%，且 A/B/C/NONE confusion 有结构性差异。
2. MLP 神经元 mask 有因果性：target mask 对目标类别的 recall/action accuracy 破坏明显大于 random mask。
3. 训练后相比 When2Tool baseline 至少满足一条：
   - 在相近 accuracy 下，总工具调用更少。
   - 在相近 TotalTC 下，accuracy 更高。
   - 在 full-tools 设置下，`MacroF1_action` 或 B/C recall 高于 Probe&Prefill。

不需要现阶段追求：

- 大规模多模型全覆盖。
- multi-hop 主结果。
- 搜索类真实 benchmark 泛化。
- 大量新训练方法。

先把 single-hop + qwen3-4b-instruct 跑通，并证明比 baseline 有提升即可。
