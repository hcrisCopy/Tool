# Stage 5–8 实验与交接说明

## 1. 当前进度

| 阶段 | 要回答的问题 | 状态 |
|---|---|---|
| Stage 5：FFN 探测 | 四动作信息能否从 FFN intermediate 中读出 | **已完成** |
| Stage 6：因果验证 | 候选神经元是否选择性影响对应动作 | 代码已完成，待正式运行 |
| Stage 7：训练 | 定向训练这些神经元是否有效 | 代码已完成，须先检查 Stage 6 |
| Stage 8：训练后评测 | target LoRA 是否优于 dense/random 并改善 Pareto | 代码已完成，待训练后运行 |

前置 residual probe 已确认 full-menu 特征中有较强的四动作信号：binary Accuracy/AUROC 为 `0.8542/0.9239`；`NONE/A/B/C` 的 Accuracy/Balanced Accuracy/Macro-F1/OVR-AUROC 为 `0.8569/0.7840/0.8085/0.9633`。这只能说明“可以读出”，不能说明 FFN 神经元具有因果作用。

研究创新、相关工作和不能越界的结论见[研究定位](RESEARCH_POSITIONING.md)。旧统计阶段的 partial checkpoint 见[前置阶段报告](../reports/stages/STAGE_STATISTICS_QWEN3_4B.md)。

## 2. 代码和输出在哪里

默认实验根目录为 `../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507`。代码只在 Git 仓库，大型产物只在数据目录。

| 阶段 | 一键入口 | 核心实现 | 输出目录 |
|---|---|---|---|
| Stage 5 | `python scripts/run_stage5_probing.py --start fresh` | `when2tool_action/mlp_activations.py`、`when2tool_action/neuron_probing.py` | `stages/05_probing/` |
| Stage 6 | `python scripts/run_stage6_causal.py` | `when2tool_action/hf_agent.py`、`when2tool_action/neuron_ablation.py` | `stages/06_causal/` |
| Stage 7 | `python scripts/run_stage7_training.py --step prepare`，之后用 `--step train --causal-gate-passed` | `when2tool_action/sft.py`、`when2tool_action/masked_lora.py` | `stages/07_training/` |
| Stage 8 | `python scripts/run_stage8_evaluation.py --step all` | `when2tool_action/adapter_evaluation.py` | `stages/08_evaluation/` |

脚本会自动运行该阶段的完整矩阵并保存 provenance。Stage 5 的 `--start activations` 只抽 activation，`--start discovery` 复用完整 activation 做探测；Stage 8 可用 `--step evaluation|summary` 分步恢复。正式 Stage 5 已有结果，不要原地重跑；复现应按根 [README](../README.md) 使用新的输出根。完整文件 SHA 只在[数据清单](data_manifest.md)维护。

## 3. Stage 5：FFN 探测

### 3.1 固定设置

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-4B-Instruct-2507 |
| 数据 | single-hop train 900 / test 2250 |
| 工具菜单 | full menu，15 environments / 33 tools |
| prompt | current + no_reasoning |
| 特征位置 | 最后一个有效输入 token |
| FFN 特征 | `SiLU(gate_proj(x)) * up_proj(x)` |
| 特征 shape | `[N,36,9728]`，float16 保存 |
| 神经元选择 | 只使用 train；test 不参与选择 |
| 对照采样 | one-vs-rest，按 difficulty 匹配，最大可行数量、无放回，seed 42 |
| 主设置 | `rho=0.003, activation_variant=signed` |
| 稳健性设置 | `rho={0.001,0.003,0.005}` × `signed/positive/abs` |

每个类别、每一层先分别计算 target 和匹配 control 的 saliency，再取 `TopK(target) - TopK(control)`。差集有多少就保留多少，不用随机神经元补齐。主设置在看 test 结果前已经固定，不能事后改选为分数最高的一组。

### 3.2 九组结果

四动作多数类基线的 Accuracy/Balanced Accuracy/Macro-F1 是 `0.5267/0.2500/0.1725`。下表保留影响判断的主要指标；完整 metrics 在各目录的 `probe_results.json`。

| rho | variant | unique neurons | binary Acc / AUROC | 四动作 Acc / BalAcc / F1 / AUROC |
|---:|---|---:|---|---|
| .001 | signed | 103 | .6422 / .8451 | .5267 / .2500 / .1725 / .9056 |
| .001 | positive | 125 | .6067 / .8627 | .5267 / .2500 / .1725 / .9132 |
| .001 | abs | 101 | .6404 / .8455 | .5267 / .2500 / .1725 / .9056 |
| **.003** | **signed（primary）** | **306** | **.8018 / .8827** | **.6031 / .3365 / .3101 / .9349** |
| .003 | positive | 312 | .7920 / .8887 | .6378 / .3759 / .3531 / .9370 |
| .003 | abs | 306 | .8013 / .8844 | .6076 / .3415 / .3161 / .9361 |
| .005 | signed | 512 | .8200 / .9016 | .7244 / .4906 / .4901 / .9452 |
| .005 | positive | 482 | .7933 / .8965 | .6960 / .4426 / .4137 / .9442 |
| .005 | abs | 497 | .8213 / .9019 | .7187 / .4836 / .4843 / .9448 |

Primary 的 binary 完整指标（Accuracy/Balanced Accuracy/Macro-F1/AUROC）为 `.8018/.7923/.7917/.8827`。四动作硬分类指标只比多数类基线高 `.0764/.0865/.1376`，而且明显低于 residual probe；AUROC 较高说明排序信号存在，但不能据此说路由分类已经可靠。

### 3.3 Primary 结构和稳定性

- 四类共得到 375 个 assignments：`NONE/A/B/C=58/110/109/98`；去重后为 306 个神经元。
- 306 个神经元中，241 个只属于一类，61 个属于两类，4 个属于三类；类别间 Jaccard 为 `0.0248–0.1123`。集合大体区分，但并不互斥。
- assignments 在 early/middle/late 层的比例为 `10.4%/41.1%/48.5%`，各类 layer entropy 为 `.892–.917`。应表述为“后半层偏重、仍跨层分布”。
- primary 与同 rho 的 `abs` union Jaccard 为 `.9245`，与 `positive` 只有 `.0369`；signed 跨 rho 的 Jaccard 只有 `.0124/.0863`。具体神经元身份对 rho 和激活变体敏感。
- `.001` 的四动作 hard decision 完全退化到 majority；`.005` 更好，但使用了更多特征。它不能证明定位更精准，也不能取代预注册 primary。

**Stage 5 结论：FFN 中存在可探测动作信号这一假设得到有限但明确的支持。** 目前不能说这 306 个神经元是唯一、充分、必要或主要的决策位置。Stage 5 已通过输入、tensor、mask、train-only selection 和运行记录一致性检查。

## 4. Stage 6：因果验证

### 4.1 正式矩阵

使用 primary mask，在同一 HF backend 上运行完整的 2250 条 full-menu test：

| 条件 | 数量 | 设置 |
|---|---:|---|
| no-mask | 1 | 共享基线 |
| target mask | 4 | 分别 mask `N_NONE/N_A/N_B/N_C` |
| random mask | 20 | 每类 5 个 mask seeds：`0,1,2,3,4` |

random mask 必须在每层匹配 target 的真实数量，并从四类 target union 的补集中采样。mask 覆盖 prompt prefill 和整个 autoregressive decode；每个条件都跑全部测试集，不能只跑目标类别。

主矩阵使用 generation seed `0`，共 25 个 condition-seed checkpoints。资源允许时再严格追加 generation seeds `1,2`，扩展为 75 个；mask seed 和 generation seed 必须分开记录。

### 4.2 必看指标和通过门槛

必须同时报告：

- ActionAcc、Macro-F1、`Recall_NONE/A/B/C`；
- FinalAcc、TotalTC、OverCall、UnderCall、WrongCat；
- invalid、schema/parse failure、mixed-category、exact environment/tool；
- target recall drop、off-target mean drop、FinalAcc drop。

每一类的预注册门槛是：

```text
target_recall_drop > mean(random_recall_drop) + std(random_recall_drop)
```

这只是“值得继续训练”的门槛，不等于统计显著。若 target mask 只是让 FinalAcc、格式或所有类别一起变差，就只能说明这些神经元对生成重要，不能称为类别专属的动作神经元。

## 5. Stage 7：SFT 与 masked LoRA

### 5.1 统一训练数据

只使用 900 条 train 派生数据：

- `gold_action=NONE`：采用 full hard-no-tool 下答对的 direct answer，并重建到 full/current prompt；
- `gold_action=A/B/C`：采用 scoped/current/no_reasoning 的成功工具轨迹，再把对话尾部移到重建的 full/current prompt；
- 最终答案错误、首个工具类别错误、INVALID 或未 boxed 结束的轨迹全部丢弃，并逐项记录数量。

所有训练组必须共用同一份 `sft/train_action_trajectories.jsonl` 和对应 manifest，不能为某个方法单独过滤数据。

### 5.2 训练组

| 条件 | 更新范围 | seed |
|---|---|---:|
| `base` | 不训练，只用于评测 | - |
| `dense_mlp_lora` | gate/up 的全部 output rows | 42 |
| `target_neuron_lora` | primary 四类神经元 union | 42 |
| `random_neuron_lora` | 逐层同数量补集采样 | 0、1、2 |

LoRA-A 仍是完整可训练矩阵，mask 只限制 LoRA-B 的输出行。因此 manifest 必须同时记录原始 trainable parameters 和实际选中的 output rows，不能把后者冒充真实参数量。

### 5.3 固定超参

| 参数 | 值 |
|---|---|
| epochs / learning rate | `1 / 1e-5` |
| LoRA rank / alpha / dropout | `8 / 16 / 0` |
| target modules | `gate_proj`, `up_proj` |
| precision | bf16 |
| gradient checkpointing | on |
| max length | 8192；超长直接报错，不截断 |
| global / per-device batch | `8 / 1` |
| scheduler / warmup / weight decay | `constant / 0 / 0` |
| training seed | 42 |
| loss | 只计算 assistant response tokens |
| external logging | none |

只有 Stage 6 的选择性门槛通过后，才能把 target 训练解释为“利用已验证的动作子空间”。即使门槛未通过，也可把训练作为探索性实验，但报告必须明确降级表述。

## 6. Stage 8：训练后评测

正式矩阵是 2 个 scope × 6 个条件 × 3 个 generation seeds：

- scope：`scoped-tools`、`full-tools`；
- 条件：base、dense、target、random-0、random-1、random-2；
- generation seeds：`0,1,2`。

所有条件继续使用同一份冻结的 base-model gold action，禁止为 adapter 重新打标签。

| scope | 主要指标 |
|---|---|
| scoped-tools | Accuracy、Total/Avg TC、TCR、TC reduction、Accuracy loss、Pareto |
| full-tools | Accuracy、ActionAcc、Macro-F1、ToolNeed-F1、四类 recall、OverCall、UnderCall、WrongCat、mixed-category |

自动汇总只在统一 HF backend 的 base/dense/random/target 之间计算 Pareto 和 `target - control` delta。Prompt-only、Reason-then-Act、Probe&Prefill 等 vLLM 结果只能作为上下文对照；在没有同 backend 复现前，不能把差值解释为纯训练因果效应。

seed 只能严格追加：可先跑 seed 0，再扩展到 `0,1,2`；恢复时程序会先重验已有 checkpoint，不能缩减或替换旧 seed。

## 7. 最终判断标准

| 假设 | 当前判断 | 后续什么结果才算支持 |
|---|---|---|
| H1：四动作信息进入 FFN | **有限支持** | Stage 5 已超过 prior，但弱于 residual 且对 feature budget 敏感 |
| H2：存在类别选择性的因果子空间 | **尚未判断** | target recall drop 超过同层同量 random，同时 off-target、格式和 FinalAcc 损伤受控 |
| H3：该子空间可被训练利用 | **尚未判断** | target 在同数据、同步数下优于 dense 和 random，并改善端到端 Pareto |

无论 Stage 6 是阳性还是阴性都应保留：阳性支持局部因果贡献；阴性则说明 residual 中的强信号可能是分布式或冗余表征。单模型、单 benchmark、single-hop 的结果不能外推到普适“工具神经元”。
