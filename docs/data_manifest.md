# 数据、模型与大型产物清单

本文只记录可移植的来源和相对布局。`manifests/runtime_provenance.json` 固定生成代码、配置、原始数据与模型快照；后生成的 labels/hidden/probes 由各自 manifest 或 result receipt 约束。旧统计行为面板仍是 partial checkpoint，其四个文件 SHA 记录在阶段报告；这不代表已独立完成的第 5 阶段也是 partial。第 5 阶段以本阶段 runtime receipt、两份 activation manifest 以及 9 组 mask/probe receipt 为权威清单。只有完整旧行为面板的 `formal_stage_audit.json` 与 `stage_handoff.json` 才是旧方案全树 inventory。数据、模型、hidden states、probe 权重、行为轨迹和图表均不进入 Git。

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
        │   ├── 06_causal/{conditions,logs,manifests}/
        │   ├── 07_training/{sft,adapters,logs,manifests}/
        │   └── 08_evaluation/{outputs,comparison,logs,manifests}/
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

### 第 5 阶段实际 inventory

第 5 阶段已于 2026-07-22 完整运行，生成 commit 为 `3f4bf60c1b9e34665fbfcd22ea830950dac2ce4e`，阶段 runtime receipt SHA256 为 `0a78f15d5327f90038792213da5a7798896d0e79532ee47d6e4b44541d12ed24`。阶段目录共有 70 个文件、`2,214,392,758` bytes（约 2.06 GiB）。

| 相对 `stages/05_probing/` 的文件 | shape / dtype | bytes | 文件 SHA256 | manifest SHA256 |
|---|---|---:|---|---|
| `activations/train_mlp_lasttoken_fulltools.pt` | `[900,36,9728]` / float16 | 630,375,615 | `c35675432b2ad338a95462b80c5d3b6e37edeba601d4a18318a92170b9337db4` | `21f4b9ed592b5b357e0e6de60ce1296a7ccd14e466b598f73240d688e72fbefb` |
| `activations/test_mlp_lasttoken_fulltools.pt` | `[2250,36,9728]` / float16 | 1,575,937,215 | `712a6cc1f45ed68217c6e0c27e15a7fcf55a511b38408cc3ad548496e62af719` | `dfaee8ff42e258feee6fb6c2b3110a4d3ac575f6925d369caa5940b375e09bbb` |
| `activations/down_proj_column_norms.pt` | `[36,9728]` / float32 | 1,402,047 | `27e1fbcc4960c700ef7855ada7130e2640d4de2b732510a313aa7a6c37ed7d46` | 两份 activation manifest 共同引用 |

共同 config SHA256 为 `0cb3ad38b4ef42b6431a5f41fac8d90d8f8655ca1aa4019313599c4a89a37d8c`，model config SHA256 为 `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`。train/test 的 IDs SHA256 分别为 `c579dc3138e1de280a6445d5948015cea45705222811d4588bd894c221d8769b`、`93a7a0fc8326d7c6139d0077266eda8f1bfe7ea91f1db240a37f93c306b9faf6`；labels SHA256 分别为 `e088367bc888d2023135f151b0a0c15bb0dbda6975cfa5df65dc9f9b80b309e0`、`4c0d5d80a2a2f9cca07ffc39e66b2f08af5e66a1aafcdfdf275111bcb7cc0e8f`。

每个 discovery 目录恰有 7 个文件。下表的 `probe SHA` 是 `probe_results.json` 的文件 SHA256，`mask SHA` 是 `tool_action_neurons.json` 的文件 SHA256；它们和目录中的模型、CSV、三张图只存数据侧。

| discovery group | files | unique neurons | probe SHA256 | mask SHA256 |
|---|---:|---:|---|---|
| `rho0.001_signed` | 7 | 103 | `f6c8867a3502bf2b9b03005141f4c375f94e1b4afa0fbf7d254b0b3a69f61b0d` | `a7a1b2b5dadfae5f4967a531f0bef76649f51c77db4dbfaec5253df1ac72c946` |
| `rho0.001_positive` | 7 | 125 | `b0f77483764a049e24903d7edc4c4aaa021a8a3defca9c5c13573abb1dce0733` | `06ca8104e3fbade33d14e87a274a167d6cc1d483585592754e56e7134b1f68a4` |
| `rho0.001_abs` | 7 | 101 | `5cd590e7980ed147f5662744628cd0010400c97cdb635e745505ecac5ce5e9e9` | `0cf8e7b1c28e1badb642ca662310a21f7d993713c3d3ae7355937b9091351b9f` |
| `rho0.003_signed` **primary** | 7 | 306 | `321233ebd0b2b726581519eca3f91f79105cf02830bf6d0fbdc5cde63e9d53da` | `12b94f0a65f244b8ef0bbdad5c76d75d72b4793bfded50312be5c63375b42e1c` |
| `rho0.003_positive` | 7 | 312 | `883b35fb097ca9248100d36941520ad47df9dd3c6459ca97e8a788dd0d531c59` | `945055602eafa2f8004ddb26066a3d60d57e28cc40a09d4cba2e3482b6c8d3e5` |
| `rho0.003_abs` | 7 | 306 | `6320b92daae16d8608d0d12065cf83fd49c9d2f3250c77d66216a81bc8ebc3c4` | `586a7ce2eb1e15c7a3287f599b64006927e5b3784de6c52cdd56f9c24ac7c1f5` |
| `rho0.005_signed` | 7 | 512 | `b889fdf26e29628e5858ee82c25ac7c5fc0b279f61127ea0d602b0fcd3d06669` | `e21a0c4110fc3657a1f4eef019cfa79dea883abaa53d683ed31f8cfa1069f87b` |
| `rho0.005_positive` | 7 | 482 | `afb07269e9736b27389ad890dad965ee395737b80813eb46db184860abfa7d4f` | `a2c224a342e906f4871510f16fa86602031c88e9e4acef1bc6f06c4e874d9640` |
| `rho0.005_abs` | 7 | 497 | `455b26901754f615449696b12eb811d8a02810a35b753a36cecd54fb1d7c4d99` | `4016886940c807f1bf3d476a36125923abe11d0ba88dc33c0ba18ba7861af979` |

Primary union feature SHA256 为 `750bce8df5082d9704e31c159d12978cbb3542259ac004e1cf50735db2b016df`。硬审计重新计算并通过了 9 个 canonical mask SHA、共享 selection/evaluation/control hashes、全部 tensor SHA 与 train-only selection 合同。实际结果解释见[第 5 阶段以后交接](STAGE5_PLUS_HANDOFF.md)。

后续逐样本因果轨迹和 adapter 还会继续增长；这些文件不得复制进 Git。每个 stage 的 manifest 必须能从输入 SHA、mask indices 和冻结参数重建实验，不允许只依赖日志文件名。

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

旧统计行为面板处于 partial checkpoint：只有 scoped-adapted 的 `force/current/necessary` 三设置三种子完整，`sparse` 完成 seeds 0/1，共 11 runs、24,750 trajectories；其余行为/P&P/analysis 尚未生成。这个 partial 状态不影响第 5 阶段的独立完整性。精确文件 SHA 与恢复提交见阶段报告；`formal_stage_audit.json` 和 `stage_handoff.json` 只有旧行为完整面板通过语义审计后才会出现，当前缺失不表示文件丢失。

服务器项目根 `README.md` 的维护源是 Git `docs/REMOTE_ROOT_README.md`；数据根 `CallTool_data/README.md` 的维护源是 Git `docs/DATA_ROOT_README.md`。发布时复制为普通文件，避免打包或迁移后出现断链。

任何 SHA、task ID、seed panel、tool scope 或 label protocol 不一致都应直接失败，不允许自动下载、跳过或改用缓存中的其他资源。
