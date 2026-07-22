# Stage 5 前统计结果与旧行为 checkpoint

> 这是 2026-07-22 的历史快照，不代表当前项目进度。它保留 residual probe 结果和一个可恢复但未完成的行为实验。当前 Stage 5–8 交接见 [`docs/STAGE5_PLUS_HANDOFF.md`](../../docs/STAGE5_PLUS_HANDOFF.md)。

## 1. 这一阶段做了什么

项目把原始“是否调用工具”扩展为四个动作：`NONE`、`A`、`B`、`C`。`NONE` 表示模型可以直接答对；A/B/C 分别对应计算规模、知识边界和执行追踪类工具。

主实验使用固定 full menu，避免 scoped menu 直接暴露题目所属环境。full、scoped-adapted、scoped-original-W2T 是三套不同的 prompt/标签协议，不能混在一起比较 gold action。

- 模型：Qwen3-4B-Instruct-2507
- 数据：When2Tool single-hop，train/test = 900/2250
- 上游 commit：`66f100089d1f3f7e7f2acee279c4dbf6e7ae5e2c`
- full menu SHA256：`9fe32b5541d03e6948982b1669fb0d289325f0124ef204d159c239487766f117`

## 2. 标签分布

| 协议 | Split | NONE | A | B | C | 总数 |
|---|---|---:|---:|---:|---:|---:|
| full-menu | train | 498 | 125 | 193 | 84 | 900 |
| full-menu | test | 1185 | 318 | 502 | 245 | 2250 |
| scoped-adapted | train | 431 | 144 | 189 | 136 | 900 |
| scoped-adapted | test | 1064 | 369 | 503 | 314 | 2250 |
| scoped-original-W2T | train | 441 | 146 | 189 | 124 | 900 |
| scoped-original-W2T | test | 1076 | 363 | 506 | 305 | 2250 |

full 与 original scoped 的标签一致率只有 train 85.67%、test 87.16%。这说明标签会随工具菜单和 prompt 改变，不能把一套标签当成跨条件不变的人工真值。

## 3. Residual probe 结果

### 是否需要工具

| 协议 | Train Acc | Test Acc | Test AUROC |
|---|---:|---:|---:|
| scoped-original-W2T | 0.9278 | 0.8853 | 0.9467 |
| scoped-adapted | 0.9078 | 0.8720 | 0.9392 |
| full-menu | 0.9144 | 0.8542 | 0.9239 |

三套协议下都有明显的线性可解码信号，但协议不同，数值高低不能直接解释成同一个机制被增强或削弱。

### Full-menu 四动作

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.8569 |
| Balanced Accuracy | 0.7840 |
| Macro-F1 | 0.8085 |
| OVR AUROC | 0.9633 |
| Majority baseline | 0.5267 |

A/B/C/NONE recall 分别为 0.8616/0.9064/0.4490/0.9190。needed-only A/B/C probe 在同分布测试上达到 1.0，但 environment-held-out 的 Accuracy/Balanced Accuracy/Macro-F1 降到 0.7005/0.6714/0.6117，说明部分信号来自 environment/template。

逐层结果从 layer 1 起已高于随机参考，并在约 layer 20–21 达峰。这只能说明动作信息可解码，不能证明决策在早层形成，更不能证明某个 FFN 神经元具有因果作用。

## 4. 旧行为实验暂停点

旧行为实验按 seed 原子保存；中断时未完成的 seed 没有写入目标文件。已完成 11 个 run，共 24,750 条轨迹。

| scoped-adapted setting | 已完成 seeds | 状态 |
|---|---|---|
| `force_tool_no_reasoning_scoped` | 0/1/2 | 完整 |
| `current_no_reasoning_scoped` | 0/1/2 | 完整 |
| `necessary_tool_no_reasoning_scoped` | 0/1/2 | 完整 |
| `sparse_tool_no_reasoning_scoped` | 0/1 | 缺 seed 2 |
| 其余 prompt/reasoning/P&P settings | 无或未完成 | 待恢复 |

暂停时的最终准确率：

| Setting | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| `current` | 0.8747 | 0.8827 | 0.8831 |
| `force` | 0.9102 | 0.9133 | 0.9098 |
| `necessary` | 0.8511 | 0.8507 | 0.8440 |

这些数值只用于确认 checkpoint 完整，不能拿来提前选择最佳策略。

### Checkpoint 校验信息

- 行为生成 commit：`0d38dcfcfbc89e6df181cf9d4f0110f55ddbb82c`
- runtime provenance SHA256：`30522e4d52ce3882b4a47d9891dad9ed917248de907ae436acda8e426eac1dbe`
- 配置 SHA256：`0cb3ad38b4ef42b6431a5f41fac8d90d8f8655ca1aa4019313599c4a89a37d8c`

| 文件 | SHA256 |
|---|---|
| `force_tool_no_reasoning_scoped.json` | `116924e0fb3ea71787d0cfab268cdccad5ca6539bda83019972e492f2dd27b4d` |
| `current_no_reasoning_scoped.json` | `1657a59fe16654aaf49e4b5c315e1c508f9df01f5d9eec003401778201c4ee74` |
| `necessary_tool_no_reasoning_scoped.json` | `b7c7fb95aef8993a5a04c5e5dcf5bbb8629d44c6cc6d5ab4bdac84e9eee89994` |
| `sparse_tool_no_reasoning_scoped.json` | `2868998a3bfa618ffd03eee741ebe8fb905be163734e14874bb0adaf4a72f839` |

## 5. 如需恢复旧行为面板

只有确实要补齐旧 behavior/P&P 面板时才执行下面的历史恢复流程。行为生成必须使用原 commit，不能修改 provenance 绕过校验。

```bash
test -z "$(git status --porcelain)"
git switch --detach 0d38dcfcfbc89e6df181cf9d4f0110f55ddbb82c
set -o pipefail
git show main:scripts/run_statistics_stage.sh | STAGE_START=behavior bash

git switch main
git pull --ff-only
test -z "$(git status --porcelain)"
STAGE_START=statistics bash scripts/run_statistics_stage.sh
```

恢复程序会重新核验配置、标签、provenance、task 顺序和已有轨迹，只补缺失 seed。旧面板目前没有完整 summary，因此不能报告正式多 setting 结论。

## 6. 结论边界

这一阶段只支持两点：

1. residual stream 中存在可解码的“是否需要工具”和 A/B/C/NONE 动作信号；
2. 同分布结果受 environment/template 影响，environment-held-out 结果更可信。

它不支持“已经找到关键 FFN 神经元”“早层信号具有因果性”“mask 会特异破坏某类动作”或“定向训练优于 Probe&Prefill”。这些问题属于当前 Stage 5–8 路线。
