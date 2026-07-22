# When2Tool full-menu action experiments

本仓库实现新的统计阶段：在 Qwen3-4B-Instruct-2507 面前固定展示 When2Tool 原始 15 个环境、33 个**原名工具**，把首个真实路由调用归为 `A/B/C`，全程没有调用则为 `NONE`。代码同时保留原始 scoped-tool pipeline，二者分开统计，不能直接把 full-tools 数字冒充论文 scoped 数字。

当前阶段只做到：数据与标签、scoped baseline、full-tools 行为、all-layer residual probes、Probe&Prefill、统计表和图。尚未进入 MLP 神经元、mask 因果验证和 LoRA。

## 一次性准备

以下命令都在仓库根目录 `CallTool_code` 执行。不要在本机安装依赖；远程进入此前创建的独立 conda 环境后执行：

```bash
conda activate /root/autodl-tmp/CallTool/CallTool_data/conda_envs/calltool_qwen3
python --version
python -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)"
git submodule update --init --recursive
python -m pip check
python -m pytest -q
```

要求 Python 3.11、Transformers 4.55.2、vLLM 0.8.5。配置文件只使用相对路径：

```text
when2tool_action/configs/qwen3_4b_instruct_2507.yaml
├── model        ../../Qwen/Qwen3-4B-Instruct-2507
├── dataset      ../CallTool_data/When2Tool
└── output_root  ../CallTool_data/when2tool_precise_shield
```

## 先做 45 题 smoke test

```bash
python -m when2tool_action.scripts.prepare_category_fulltools \
  --config when2tool_action/configs/qwen3_4b_instruct_2507.yaml

python -m when2tool_action.scripts.extract_tool_labels \
  --config when2tool_action/configs/qwen3_4b_instruct_2507.yaml \
  --tool-scope full --smoke
```

脚本默认拒绝覆盖已有产物。只有明确要重跑时才加 `--overwrite`。

## 完整统计阶段

全新目录可直接运行：

```bash
bash scripts/run_statistics_stage.sh
```

这是长任务，依次完成：

1. 严格审计 900/2250 single-hop 数据并写 category/fulltools 两套 JSON；
2. 分别生成 full/scoped 的 seed-0 hard-no-tool 标签；
3. 分别抽取 full/scoped 最后输入 token 的 37×2560 public hidden states；
4. 运行 pinned When2Tool binary probe；full-tools 额外跑四动作、needed-only 三分类、逐层和 environment-held-out probes；
5. scoped 的 5 prompt × 2 reasoning × 3 seeds 及 Probe&Prefill 五阈值；
6. full-tools 的 current/necessary/sparse no-reasoning × 3 seeds及五阈值 Probe&Prefill；
7. 生成逐 run、mean±population-SD、配对 task×seed bootstrap、CSV、JSON 和 PNG 图。

如需分步执行，直接查看 [scripts/run_statistics_stage.sh](scripts/run_statistics_stage.sh)，每条命令都是独立入口。

## 可选：导入原始 scoped When2Tool baseline

旧实验中的原始 `P_env` binary probe 只能作为论文原始 scoped baseline，不能称为当前流水线的 `scoped-adapted` probe。导入命令不会被统计总脚本隐式调用；确认旧资产仍在后，显式执行：

```bash
python -m when2tool_action.scripts.import_legacy_scoped \
  --config when2tool_action/configs/qwen3_4b_instruct_2507.yaml \
  --legacy-root ../CallTool_data/experiments_2d/qwen3-4b-instruct-2507 \
  --output-root ../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507 \
  --transfer-mode hardlink
```

脚本会逐项验证 manifest 与文件 SHA256、hidden shape/dtype/finite、所有 task/prompt hash、`C=1e-4` all-layer probe，并按官方双 `StandardScaler` 路径重算全部保存指标。`hardlink` 不复制数 GB hidden 数据；如果源与目标不在同一文件系统，请显式改成 `--transfer-mode copy`，程序不会静默回退。输出固定隔离在 `probes/scoped_original_w2t/` 和 `*_scoped_original_w2t.json`，审计结论写入 `migration_receipt.json`。

## 关键实验契约

- 数据中始终只保存 gold environment；`--tool-scope full` 只在运行时创建 15 个新环境实例并注入固定 33-tool menu。
- `ListManipulation` 格式说明按“菜单是否暴露该工具”决定：scoped 仅 List 任务加入，full-tools 则所有任务统一加入；绝不按 full-tools 的 gold environment 条件化 system message。
- 全工具按 environment/name 固定排序，菜单保存 SHA256；不重命名工具，不在 prompt 泄露 A/B/C、gold env 或 gold tool。
- `total_tool_calls` 只统计通过 reasoning/no-tool 检查、真正进入路由器的调用。显式断言 `len(routed_tool_events) == tool_calls`。
- 无调用为 `NONE`；否则首个 routed event 的类别为 `A/B/C`。未知工具为 `INVALID`，不能静默塞入四类。
- 三次运行固定 seed `0/1/2`；同一 `(seed, task_id)` 在 setting 间配对。6750 行不当作 IID 样本。
- raw `ActionAcc > 25%` 不是成功判据，因为 `NONE` 是多数类。主证据为 balanced accuracy、Macro-F1、逐类 recall、majority/prior-matched baseline 和配对置信区间。
- 行为混淆矩阵只能证明行为结构；内部信号由 residual probes 单独支持。A/B/C 是本项目从 When2Tool 三组必要性任务派生的 coarse functional actions，并非原论文已验证的 routing taxonomy。

## 工具执行安全

官方 `CodeExecutorEnv` 会在评测进程直接 `exec`，本项目没有调用它。模型只有在参数与当前 benchmark 指令里的 fenced code **完全一致**时，才会在临时目录的受限子进程运行；其余代码返回显式 `[SAFETY_REJECTED]`，但这次调用仍计为 C 类。数学、矩阵、正则等工具有 benchmark 范围上限和 5 秒超时。轨迹会保留安全拒绝，不把它包装成工具成功。

## 产物目录

大文件都在代码目录的同级数据盘目录：

```text
../CallTool_data/when2tool_precise_shield/qwen3-4b-instruct-2507/
├── data/
├── labels/qwen3-4b-instruct-2507/
├── probes/{scoped,fulltools}/
├── outputs/{scoped,fulltools}/
├── analysis/{scoped,fulltools}/
└── reports/
```

代码仓库不保存模型、数据、hidden states 或生成轨迹。最终阶段报告会写入 `reports/`，并记录 commit、menu hash、环境版本、复用产物及任何协议偏离。
