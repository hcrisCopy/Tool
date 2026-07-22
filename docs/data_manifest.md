# 数据、模型与大型产物清单

本文只记录可移植的来源和相对布局。`manifests/runtime_provenance.json` 固定生成代码、配置、原始数据与模型快照；后生成的 labels/hidden/probes 由各自 manifest 或 result receipt 约束，当前 partial 行为 checkpoint 的四个文件 SHA 记录在阶段报告。只有完整阶段的 `formal_stage_audit.json` 与 `stage_handoff.json` 才是全树 inventory；当前 partial 交付并不宣称已有完整机器清单。数据、模型、hidden states、probe 权重、行为轨迹和图表均不进入 Git。

## 期望布局

从 `CallTool_code/` 解析：

```text
../CallTool_data/
├── When2Tool/
├── conda_envs/calltool_qwen3/
└── when2tool_precise_shield/
    └── qwen3-4b-instruct-2507/
        ├── data/
        ├── labels/qwen3-4b-instruct-2507/
        ├── manifests/
        ├── probes/{fulltools,scoped,scoped_original_w2t}/
        ├── outputs/{fulltools,scoped_adapted,scoped_original_w2t}/
        ├── analysis/{fulltools,scoped_adapted,scoped_original_w2t}/
        ├── stages/
        │   ├── 05_probing/{activations,discovery,logs,manifests}/
        │   ├── 06_causal/{conditions,metrics,figures,logs,manifests}/
        │   ├── 07_training/{sft,adapters,logs,manifests}/
        │   └── 08_evaluation/{outputs,metrics,figures,manifests}/
        ├── logs/
        └── reports/

../../Qwen/Qwen3-4B-Instruct-2507/
```

`Qwen/` 是共享模型目录，不复制到项目数据目录。配置文件只能保存上述相对路径；换机器时保持同级关系，或显式修改配置后重新生成 provenance，禁止代码静默换路径。

## 外部资源

| 名称 | 类型 | 来源 | 固定版本 | 目标相对路径 | 用途 |
|---|---|---|---|---|---|
| When2Tool | Hugging Face dataset | `cesun/When2Tool` | revision `4a6d05f2ac8fc366c9fe5760c395ebe0f0320537` | `../CallTool_data/When2Tool` | 900/2250 single-hop 主实验；multi-hop 本阶段不使用 |
| Qwen3-4B-Instruct-2507 | Hugging Face model | `Qwen/Qwen3-4B-Instruct-2507` | 本地文件级 SHA256 见 runtime provenance；下载 revision 未由现有目录单独记录 | `../../Qwen/Qwen3-4B-Instruct-2507` | 唯一主模型 |
| When2Tool code | Git submodule | `Trustworthy-ML-Lab/when2tool` | commit `66f100089d1f3f7e7f2acee279c4dbf6e7ae5e2c` | `third_party/when2tool` | prompt、环境、probe 与 baseline 对齐 |

本次交接直接提供本地数据与共享模型快照，不在代码、配置或文档中保存任何访问 token。When2Tool 可以按表中 revision 重新取得并用下列 LFS SHA 验证；Qwen 模型因原下载 revision 缺失，精确复现必须传递当前共享模型目录并核对 runtime provenance 的逐文件 SHA，不能把远端仓库的“当前最新版”当成同一快照。

When2Tool 当前四个 parquet 的上游 LFS SHA256：

| 文件 | SHA256 |
|---|---|
| `single_hop/train-00000-of-00001.parquet` | `9ef525737e47b02f7f94badcd9b7a64d30e0387b3406fe127091d01d81c7d430` |
| `single_hop/test-00000-of-00001.parquet` | `d8e4997be7bf697a1d8ffa9b273c7f81c30f47a70d92d2fad40b2ad7c20cfc55` |
| `multi_hop/train-00000-of-00001.parquet` | `3ce97ecbff40e1bdcab0863fc67b9aa012b7b3b5310d4935d8c28f575abe8be2` |
| `multi_hop/test-00000-of-00001.parquet` | `2a8bb3450f5afcc834dae260aa56e33674f5edbc9c6ce7096693198d372f634a` |

## 本阶段生成数据

`prepare_category_fulltools` 从 single-hop split 生成四个文件，并写 `data/data_manifest.json`：

- `tasks_v1_train_category.json`：900 条，原始 scoped menu；
- `tasks_v1_test_category.json`：2250 条，原始 scoped menu；
- `tasks_v1_train_fulltools_category.json`：900 条，gold 环境仍只保留在数据中，运行时注入完整 menu；
- `tasks_v1_test_fulltools_category.json`：2250 条，同上。

固定完整菜单包含 15 个环境、33 个全局唯一原名工具，SHA256 为 `9fe32b5541d03e6948982b1669fb0d289325f0124ef204d159c239487766f117`。

## 第 5 阶段以后的大型产物

Qwen3-4B-Instruct-2507 固定为 36 个 decoder blocks、hidden size 2560、SwiGLU intermediate size 9728。第 5 阶段的 neuron 定义为 `SiLU(gate_proj(x)) * up_proj(x)` 的一个 intermediate component，不是现有 `(37, 2560)` residual hidden component。

| 阶段 | 数据侧目录 | 主要产物 | 必需 provenance |
|---|---|---|---|
| 05 probing | `stages/05_probing` | train/test float16 activation、9 组 neuron mask/CSV/PNG、冻结 probe metrics | task/label/model/config SHA、shape/dtype、prompt/menu SHA、selection split、control IDs |
| 06 causal | `stages/06_causal` | 25 个完整条件的逐样本轨迹、condition metrics、recall-drop 图 | mask SHA、HF backend config、target/random indices、generation seed、完成状态 |
| 07 training | `stages/07_training` | SFT JSONL、过滤清单、target/random/dense adapter、trainer state | source trajectory SHA、retained/dropped 分布、mask snapshot、全部训练超参、raw/effective parameter count |
| 08 evaluation | `stages/08_evaluation` | scoped/full 三种子轨迹、统计表和 Pareto 图 | frozen label SHA、adapter/base SHA、scope、generation config、seed |

以 float16 保存完整 train/test MLP activation 约需要 2 GiB 量级，逐样本因果轨迹和 adapter 还会继续增长；这些文件不得复制进 Git。每个 stage 的 manifest 必须能从输入 SHA、mask indices 和冻结参数重建实验，不允许只依赖日志文件名。

## 原始 scoped probe 的迁移边界

旧方案中可复用的原始 When2Tool `P_env` binary probe 已导入 `probes/scoped_original_w2t/`。`migration_receipt.json` 精确绑定：

- 17 个原始源产物；
- 19 个目标产物；
- 10 个自包含 `audit_source` 文件；
- scoped train/test task 文件与数据 manifest。

旧 `experiments_2d` 目录只有在 destination-only SHA256 校验通过后才允许删除。`P_all`、`P_no_schema`、旧 onset、旧 smoke 和诊断缓存不属于当前方案正式输入。

## 真源与验收

- 代码与运行环境：`manifests/runtime_provenance.json`；
- 数据转换：`data/data_manifest.json`；
- 原始 scoped 导入：`probes/scoped_original_w2t/migration_receipt.json`；
- scoped prompt 重标：`outputs/scoped_original_w2t/relabel_receipt.json`；
- 三套统计：各 `analysis/<protocol>/summary.json`；
- 完整阶段语义审计：`manifests/formal_stage_audit.json`；
- 阶段总体交接：`manifests/stage_handoff.json` 与 run root 的普通文件 `reports/STAGE_STATISTICS_QWEN3_4B.md`；其 Git 维护源是 `reports/stages/STAGE_STATISTICS_QWEN3_4B.md`。

当前交接处于 partial checkpoint：只有 scoped-adapted 的 `force/current/necessary` 三设置三种子完整，`sparse` 完成 seeds 0/1，共 11 runs、24,750 trajectories；其余行为/P&P/analysis 尚未生成。精确文件 SHA 与恢复提交见阶段报告。`formal_stage_audit.json` 和 `stage_handoff.json` 只有完整面板通过语义审计后才会出现，当前缺失不表示文件丢失。

服务器项目根 `README.md` 的维护源是 Git `docs/REMOTE_ROOT_README.md`；数据根 `CallTool_data/README.md` 的维护源是 Git `docs/DATA_ROOT_README.md`。发布时复制为普通文件，避免打包或迁移后出现断链。

任何 SHA、task ID、seed panel、tool scope 或 label protocol 不一致都应直接失败，不允许自动下载、跳过或改用缓存中的其他资源。
