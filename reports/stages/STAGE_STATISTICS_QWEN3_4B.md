# 阶段性交接：Qwen3-4B When2Tool 精确统计

- 日期：2026-07-22
- 状态：按用户指令主动暂停；GPU 进程已停止，现有行为 checkpoint 可严格恢复。
- 阶段边界：已完成 single-hop 数据、模型相关标签、hidden 与 residual probe；scoped prompt 行为只完成部分 setting/seeds，尚未完成其余行为、Probe&Prefill、三套最终统计、FFN/MLP 神经元筛选、因果 mask、定向训练或 LoRA。

## 1. 本阶段回答的问题

本阶段把原始 When2Tool 的“是否调用工具”扩展成四动作：

- `NONE`：当前模型在 hard-no-tool 下能直接答对；
- `A`：需要计算规模类工具；
- `B`：需要知识边界类工具；
- `C`：需要执行追踪类工具。

主实验让每题看到固定的 15 环境、33 个全局唯一原名工具，避免原 scoped menu 直接暴露题目所属环境。行为统计使用整条轨迹：从未真实路由调用才是 `NONE`；否则以第一次真实路由调用的 A/B/C 类别为动作，未知类别显式记为 `INVALID`，同时保留全部调用序列、调用成本、跨类调用、解析失败、终止状态和安全拒绝。

本阶段能够检验“内部是否线性可解码”和“生成行为是否有结构”，但不能证明某个 FFN 神经元是因果机制。用户定义的后续神经元仍是第 `l` 层 FFN intermediate hidden state 的第 `i` 个分量；本报告中的 37×2560 residual hidden components 只是阶段性诊断特征。

## 2. 冻结协议与必要修正

- 模型：Qwen3-4B-Instruct-2507，单 GPU。
- 数据：When2Tool single-hop train/test = 900/2250。
- 上游：When2Tool commit `66f100089d1f3f7e7f2acee279c4dbf6e7ae5e2c`。
- 全菜单 SHA256：`9fe32b5541d03e6948982b1669fb0d289325f0124ef204d159c239487766f117`。
- full 与 scoped-adapted 标签：seed 0、hard-no-tool；直接答对为 `NONE`，否则取静态 A/B/C 类。scoped-original-W2T 标签来自带完整 receipt/SHA 的原始 pipeline 审计迁移，不冒充当前 adapted 标签。
- 行为：seeds `0/1/2`；prompt temperature 0.7；Probe&Prefill probe temperature 2.0；阈值 0.1/0.3/0.5/0.7/0.9。
- 标签生成固定 12 轮；正式 prompt/P&P 行为固定 10 轮。该拆分来自 pinned 上游：`extract_features.py` 使用 12，而 `run_eval.py` / `run_probe_eval.py` 使用 10。正式行为启动前已修正，因此没有 12 轮行为混入。
- full-menu 的 ListManipulation contract 对所有样本一致展示；旧 gold-env 条件化 smoke 会泄露类别，已作废且不进入任何正式分析。
- full、scoped-adapted、scoped-original-W2T 三种 label protocol 分目录统计，禁止把不同 gold action 混成同一 action 表。

安全执行器保留 benchmark 范围约束与 CodeExecutor 子进程隔离。scoped-original 行为只有在安全拒绝总数为 0 时，才能说明安全改造没有实际改变本次轨迹；否则必须称为 safety-adapted reproduction。

## 3. 环境与可复现性

- 行为生成代码 commit：`0d38dcfcfbc89e6df181cf9d4f0110f55ddbb82c`。
- 当前交接代码 commit：以本报告所在 Git 提交为准；完整统计尚未执行，因此没有 statistics-execution commit。
- 恢复行为生成必须使用 commit：`0d38dcfcfbc89e6df181cf9d4f0110f55ddbb82c`。
- runtime provenance SHA256：`30522e4d52ce3882b4a47d9891dad9ed917248de907ae436acda8e426eac1dbe`。
- 配置 SHA256：`0cb3ad38b4ef42b6431a5f41fac8d90d8f8655ca1aa4019313599c4a89a37d8c`。

验证环境：Python 3.11.15、PyTorch 2.6.0、CUDA 12.4、Transformers 4.55.2、vLLM 0.8.5、NumPy 2.2.6、pandas 3.0.3、scikit-learn 1.9.0，单张 RTX 4090。完整模型、tokenizer、配置、生成数据、依赖和 GPU 信息见数据盘 `manifests/runtime_provenance.json`。

## 4. 数据与标签审计

### 4.1 full-menu 标签

| Split | NONE | A | B | C | 总数 |
|---|---:|---:|---:|---:|---:|
| train | 498 | 125 | 193 | 84 | 900 |
| test | 1185 | 318 | 502 | 245 | 2250 |

full test 按 difficulty：easy 为 A/B/C/NONE = 20/116/52/562；medium = 83/139/64/464；hard = 215/247/129/159。测试标签 2242 条以 boxed answer 结束，8 条达到标签协议的 12 轮上限；25 条出现至少一次工具格式解析失败。hard-no-tool 中真实路由调用严格为 0。

### 4.2 scoped-adapted 标签

| Split | NONE | A | B | C | 总数 |
|---|---:|---:|---:|---:|---:|
| train | 431 | 144 | 189 | 136 | 900 |
| test | 1064 | 369 | 503 | 314 | 2250 |

scoped test 中 2213 条 boxed、37 条达到 12 轮上限；hard-no-tool 中真实路由调用严格为 0。adapted 与导入的 original-W2T gold action 在 train/test 上分别一致 856/900（95.11%）和 2144/2250（95.29%）。

### 4.3 scoped-original-W2T 标签

| Split | NONE | A | B | C | 总数 |
|---|---:|---:|---:|---:|---:|
| train | 441 | 146 | 189 | 124 | 900 |
| test | 1076 | 363 | 506 | 305 | 2250 |

### 4.4 full 与 original scoped 标签不可混用

full 与 original-W2T gold action 在 train/test 上分别一致 771/900（85.67%）和 1961/2250（87.16%）。差异来自工具 schema/menu 改变后，模型的 no-tool 生成也会变化；因此本项目把 label protocol 当成实验条件，而不是把标签视为跨 prompt 永恒不变的人工真值。

### 4.5 hidden states 完整性

full 与 scoped 的 train/test tensors 均为 float32、全部 finite，shape 分别严格为 `(900, 37, 2560)` 和 `(2250, 37, 2560)`；task ID、顺序、gold action、prompt hash 与对应 labels/manifest 逐条一致。

| Scope | Split | Tensor SHA256 |
|---|---|---|
| full | train | `4a1f104ce11146c81d2ce4fc643ba4e05e28d52ccc7d29bb644520e31498dff11` |
| full | test | `066e9aec79686644cd6dc1ad6910069cffd45b8b87d995e68c1b0f7f83e5d6b011` |
| scoped-adapted | train | `78e37cc755b81bab4cdc046f654b8c1c1fdf0d8c79f8610dbb3035b3bfef00f1f` |
| scoped-adapted | test | `9ed6611e4d4a79771e45f3243978ff63c29c725f05927f6b988623388554dd607` |

## 5. residual probe 结果

### 5.1 是否需要工具（二分类 all-layer）

| 协议 | Train Acc | Test Acc | Test AUROC |
|---|---:|---:|---:|
| scoped-original-W2T | 0.9278 | 0.8853 | 0.9467 |
| scoped-adapted | 0.9078 | 0.8720 | 0.9392 |
| full-menu | 0.9144 | 0.8542 | 0.9239 |

这三组数值说明各自 prompt/标签口径下都存在强线性可解码的“是否需要工具”信号；菜单长度和标签协议不同，不能把它们的高低直接解释成同一个机制被增强或削弱。

### 5.2 full-menu 四动作 all-layer probe

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.8569 |
| Balanced Accuracy | 0.7840 |
| Macro-F1 | 0.8085 |
| OVR AUROC | 0.9633 |
| Majority baseline | 0.5267 |
| Prior-matched expected accuracy | 0.3691 |

测试混淆矩阵（行 gold A/B/C/NONE，列预测 A/B/C/NONE）：

```text
A       274    0    0    44
B         0  455    0    47
C         0    0  110   135
NONE     23   23   50  1089
```

A/B/C/NONE recall 分别为 0.8616/0.9064/0.4490/0.9190。needed-only A/B/C probe 在同分布测试上为 1.0，但 environment-held-out 的 mean Accuracy/Balanced Accuracy/Macro-F1 降为 0.7005/0.6714/0.6117，说明一部分信号来自 environment/template，不能只引用同分布满分；该结果只检验 When2Tool 内跨 environment 迁移，不代表跨数据集、跨 schema 或现实工具生态泛化。

### 5.3 逐层诊断

layer 0（embedding 输出）的四分类结果等于 uniform reference（Balanced Accuracy 0.25，AUROC 0.5；不同于 0.5267 的多数类 accuracy baseline）；layer 1（第一个 Transformer block 输出）的四动作可解码性已达到 Balanced Accuracy 0.5228、Macro-F1 0.5678、OVR AUROC 0.8789。单层 Balanced Accuracy 在约 layer 20 达峰，Macro-F1 在约 layer 21 达峰。

这支持“在固定 prompt、模型生成标签与当前 train/test split 下，四动作 aggregate 标签从第一个 block 输出起已存在高于 reference 的线性可解码结构，并在中层继续增强”，但不能单独证明三种工具类型已在 layer 1 稳定分离，更不等于“决策在第一层形成”或“早层 FFN 神经元具有因果性”。关键图为数据盘 `probes/fulltools/layerwise_action_probe.png`。

## 6. 正式行为暂停点

用户要求先整理文件夹后，已向唯一 GPU 进程发送 `SIGTERM` 并确认显存进程清零。行为文件采用 seed 级原子 checkpoint；被中断的未完成 seed 没有写入目标 JSON，所有已写文件均重新解析并核对每 run 2250 个 rows。

| scoped-adapted setting | 已完成 seeds | 每 seed rows | 状态 |
|---|---|---:|---|
| `force_tool_no_reasoning_scoped` | 0/1/2 | 2250 | 完整 |
| `current_no_reasoning_scoped` | 0/1/2 | 2250 | 完整 |
| `necessary_tool_no_reasoning_scoped` | 0/1/2 | 2250 | 完整 |
| `sparse_tool_no_reasoning_scoped` | 0/1 | 2250 | seed 2 未写入，待恢复 |
| `no_tool_no_reasoning_scoped` | 无 | — | 仅协议模板 |
| 五个 reasoning settings | 无 | — | 仅协议模板 |

已完成的 11 个 run 共 24,750 条轨迹。停止前观察到：`current` seeds 0/1/2 的最终准确率为 0.8747/0.8827/0.8831；`force` 为 0.9102/0.9133/0.9098；`necessary` 为 0.8511/0.8507/0.8440。它们只是暂停点完整性诊断，不是最终多 setting 结论；没有对未完成面板提前运行统计或挑选最佳策略。

暂停时四个非空行为文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `force_tool_no_reasoning_scoped.json` | `116924e0fb3ea71787d0cfab268cdccad5ca6539bda83019972e492f2dd27b4d` |
| `current_no_reasoning_scoped.json` | `1657a59fe16654aaf49e4b5c315e1c508f9df01f5d9eec003401778201c4ee74` |
| `necessary_tool_no_reasoning_scoped.json` | `b7c7fb95aef8993a5a04c5e5dcf5bbb8629d44c6cc6d5ab4bdac84e9eee89994` |
| `sparse_tool_no_reasoning_scoped.json` | `2868998a3bfa618ffd03eee741ebe8fb905be163734e14874bb0adaf4a72f839` |

仍待完成的正式面板为：

- full-menu：3 prompt + 5 P&P = 8 settings，24 runs，54,000 task trajectories；
- scoped-adapted：10 prompt + 5 P&P = 15 settings，45 runs，101,250 trajectories；
- scoped-original-W2T：同一 10 prompt 行为严格重标 + 5 个 original probe P&P = 15 settings，45 runs，101,250 trajectories。

因此当前没有三套 `analysis/*/summary.json`，也不会生成会误导接手者的 `formal_stage_audit.json` 或 `stage_handoff.json`。完整后仍需报告三 seed mean±population-SD、paired bootstrap、A/B/C/NONE recall、错误类型、INVALID、mixed valid categories、termination/parse/safety、per-difficulty、Accuracy–TotalTC tradeoff 与描述性 Pareto frontier。

## 7. 结论边界

当前已完成的 probe 结果支持两条有限结论：

1. Qwen3-4B 的 residual stream 中不只存在“是否需要工具”的强线性信号，A/B/C/NONE 四动作也有明显高于多数类与先验随机基线的可解码结构；
2. 四动作 aggregate 结构从 layer 1 已高于 reference，但同分布满分会受 environment/template 影响，environment-held-out 结果是更可辩护的 When2Tool 内跨环境证据。

尚不能宣称：已经找到关键 FFN 神经元、早层信号是因果机制、mask 会特异破坏某类 action，或定向训练优于 Probe&Prefill。这些都属于下一阶段，需要按用户定义抽取 FFN intermediate 第 `(l,i)` 个分量并完成 target-vs-random 因果验证。

## 8. 交接目录与恢复方式

Git 仓库只保留代码、配置、测试、实验计划和本报告；大型产物保存在同级数据目录。详细树和资源来源见根 README 与 `docs/data_manifest.md`。

行为文件按 seed 原子 checkpoint，`--resume` 会重新核验 config/data/labels/provenance、task 顺序与每条轨迹字段后，只从缺失 seed 继续。统计输出使用 sibling staging，全部文件成功后才整体替换目标目录，`summary.json` 最后发布。

当前代码仓库整理完成后可位于新版 `main`，但暂停行为严格绑定旧生成 commit。恢复前必须先执行 `git switch --detach 0d38dcfcfbc89e6df181cf9d4f0110f55ddbb82c` 并确认 worktree clean；完成所有 behavior/P&P 后再切回 `main` 执行统计与 formal-stage audit。不得修改 runtime provenance 来绕过这一绑定。

当前 checkpoint 的恢复命令如下。第一段只补齐行为、Probe&Prefill 和 scoped 重标；第二段回到新版代码后生成三套统计，全部完成后再按根 README 运行 formal-stage audit 与 handoff：

```bash
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = main

git switch --detach 0d38dcfcfbc89e6df181cf9d4f0110f55ddbb82c
set -o pipefail
git show main:scripts/run_statistics_stage.sh | STAGE_START=behavior bash

git switch main
git pull --ff-only
test -z "$(git status --porcelain)"
STAGE_START=statistics bash scripts/run_statistics_stage.sh
```

## 9. 文件夹整理与旧方案清理

- 旧 `experiments_2d` 数据树已在 original-W2T 可复用子集完成 destination-only SHA 校验后删除；迁移 receipt 与自包含审计源保留在当前 `probes/scoped_original_w2t/`。
- 旧 onset 源码已由 Git commit `8ac519b` 删除且仍可从历史追溯；整理时又清掉了只含 38 个 `.pyc` 的本地 `experiments_2d/` 残壳。当前仍保留的 `legacy_scoped.py` 是 formal-stage destination-only 审计依赖，不属于待删旧代码。
- 删除 4 个仅供调试的 `fulltools_smoke` 标签/生成文件；正式 full/scoped 标签、hidden 与 probe 未改动。
- 删除 run root 中两个自动生成的 `.ipynb_checkpoints` 副本，以及受旧 ListManipulation contract 污染、已被 `full_labels_clean.log` 替代的日志。
- 已废弃 onset 流程的旧日志目录被清理；唯一仍有复现价值的环境安装日志移动为当前 `logs/setup/environment_install_initial.log`。其余正式日志按 `labels/hidden/probes/behavior` 分层。
- `CallTool_data/cache/` 是明确隔离的可再生缓存，当前保留；独立 Conda 环境、When2Tool 数据、共享模型和所有 formal/partial 产物均保留。
- 代码仓库只提交代码、配置、测试和小型 Markdown；大型数据不进入 Git。服务器根/data README 由 Git `docs/` 中的模板发布为普通文件。

整理后的 run root 约 3.7 GiB：`data/` 32 MiB、`labels/` 59 MiB、`probes/` 3.4 GiB、partial `outputs/` 243 MiB、`logs/` 2.2 MiB；`analysis/` 目前为空是因为正式统计尚未运行。独立 Conda 环境约 8.4 GiB，可再生 pip cache 约 3.8 GiB，二者与 run root 分层存放。

## 10. 后续建议

1. 先基于 full-menu labels 抽取每层 MLP SwiGLU intermediate 最后输入 token 激活；神经元严格定义为 `(layer l, intermediate component i)`。
2. 按 frozen `rho` 与 signed/positive/absolute 变体做 Precise Shield 风格 saliency、top-k、set difference，并控制 difficulty/environment。
3. 对每类做 target mask 与逐层等数量 random mask（至少 5 seeds）；只有 target degradation 明显超过 random 后才进入 masked LoRA。
4. 后续论文主张继续区分“可解码”“行为相关”“因果必要”“训练可利用”四个证据等级。
