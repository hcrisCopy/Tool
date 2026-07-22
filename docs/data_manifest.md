# 数据、模型与大型产物清单

本文只回答三件事：资源从哪里来、产物放在哪里、用什么 SHA 校验。实验设置和结果解释见[阶段交接](STAGE5_PLUS_HANDOFF.md)。大型文件不进入 Git。

## 1. 目录约定

所有路径都从 `CallTool_code/` 解析：

```text
../CallTool_data/
├── When2Tool/
├── conda_envs/calltool_qwen3/
└── when2tool_precise_shield/qwen3-4b-instruct-2507/
    ├── data/                       # 处理后的任务
    ├── labels/                     # 冻结标签
    ├── probes/ outputs/ analysis/  # 前置实验
    ├── manifests/                  # 前置实验 provenance
    └── stages/
        ├── 05_probing/
        ├── 06_causal/
        ├── 07_training/
        └── 08_evaluation/

../../Qwen/Qwen3-4B-Instruct-2507/  # 共享模型，不重复复制
```

代码默认使用这些相对位置。迁移到新机器时，应保持同级布局或显式修改配置并重新生成 provenance，不能静默换模型或数据。

## 2. 固定输入

| 资源 | 固定版本 | 相对路径 | 用途 |
|---|---|---|---|
| When2Tool | Hugging Face `cesun/When2Tool`，revision `4a6d05f2ac8fc366c9fe5760c395ebe0f0320537` | `../CallTool_data/When2Tool` | 900/2250 single-hop 主实验 |
| Qwen3-4B-Instruct-2507 | 当前本地快照；逐文件 SHA 见 runtime provenance | `../../Qwen/Qwen3-4B-Instruct-2507` | 唯一主模型 |
| When2Tool 官方代码 | commit `66f100089d1f3f7e7f2acee279c4dbf6e7ae5e2c` | `third_party/when2tool` | prompt、环境、probe 和 baseline 对齐 |

When2Tool parquet 的上游 LFS SHA256：

| 文件 | SHA256 |
|---|---|
| `single_hop/train-00000-of-00001.parquet` | `9ef525737e47b02f7f94badcd9b7a64d30e0387b3406fe127091d01d81c7d430` |
| `single_hop/test-00000-of-00001.parquet` | `d8e4997be7bf697a1d8ffa9b273c7f81c30f47a70d92d2fad40b2ad7c20cfc55` |
| `multi_hop/train-00000-of-00001.parquet` | `3ce97ecbff40e1bdcab0863fc67b9aa012b7b3b5310d4935d8c28f575abe8be2` |
| `multi_hop/test-00000-of-00001.parquet` | `2a8bb3450f5afcc834dae260aa56e33674f5edbc9c6ce7096693198d372f634a` |

当前模型目录没有单独保存下载 revision，因此精确复现应传递现有模型快照，并用 runtime provenance 核对逐文件 SHA；不能把仓库“最新版”当作同一快照。

## 3. 处理后的任务

`data/data_manifest.json` 记录转换过程和输入 SHA。当前有四个文件：

| 文件 | 数量 | 说明 |
|---|---:|---|
| `tasks_v1_train_category.json` | 900 | 原始 scoped menu |
| `tasks_v1_test_category.json` | 2250 | 原始 scoped menu |
| `tasks_v1_train_fulltools_category.json` | 900 | full-menu 实验输入 |
| `tasks_v1_test_fulltools_category.json` | 2250 | full-menu 实验输入 |

完整菜单固定为 15 个环境、33 个全局唯一工具，SHA256 为 `9fe32b5541d03e6948982b1669fb0d289325f0124ef204d159c239487766f117`。

## 4. Stage 5 正式产物

Stage 5 已于 2026-07-22 完成，共 70 个文件、`2,214,392,758` bytes（约 2.06 GiB）。生成 commit 为 `3f4bf60c1b9e34665fbfcd22ea830950dac2ce4e`；`stages/05_probing/manifests/runtime_provenance.json` 的 SHA256 为 `0a78f15d5327f90038792213da5a7798896d0e79532ee47d6e4b44541d12ed24`。

### 4.1 Activation

| 相对 `stages/05_probing/` 的文件 | shape / dtype | 文件 SHA256 | manifest SHA256 |
|---|---|---|---|
| `activations/train_mlp_lasttoken_fulltools.pt` | `[900,36,9728]` / float16 | `c35675432b2ad338a95462b80c5d3b6e37edeba601d4a18318a92170b9337db4` | `21f4b9ed592b5b357e0e6de60ce1296a7ccd14e466b598f73240d688e72fbefb` |
| `activations/test_mlp_lasttoken_fulltools.pt` | `[2250,36,9728]` / float16 | `712a6cc1f45ed68217c6e0c27e15a7fcf55a511b38408cc3ad548496e62af719` | `dfaee8ff42e258feee6fb6c2b3110a4d3ac575f6925d369caa5940b375e09bbb` |
| `activations/down_proj_column_norms.pt` | `[36,9728]` / float32 | `27e1fbcc4960c700ef7855ada7130e2640d4de2b732510a313aa7a6c37ed7d46` | 由两份 activation manifest 共同引用 |

共同 config SHA256 为 `0cb3ad38b4ef42b6431a5f41fac8d90d8f8655ca1aa4019313599c4a89a37d8c`，model config SHA256 为 `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`。train/test 的 ID SHA256 分别为 `c579dc3138e1de280a6445d5948015cea45705222811d4588bd894c221d8769b`、`93a7a0fc8326d7c6139d0077266eda8f1bfe7ea91f1db240a37f93c306b9faf6`；label SHA256 分别为 `e088367bc888d2023135f151b0a0c15bb0dbda6975cfa5df65dc9f9b80b309e0`、`4c0d5d80a2a2f9cca07ffc39e66b2f08af5e66a1aafcdfdf275111bcb7cc0e8f`。

### 4.2 九组 discovery

每个目录有 mask、CSV、probe 结果、probe 模型和三张图。下表分别给出 `probe_results.json` 和 `tool_action_neurons.json` 的文件 SHA256。

| 目录 | unique neurons | probe SHA256 | mask SHA256 |
|---|---:|---|---|
| `discovery/rho0.001_signed` | 103 | `f6c8867a3502bf2b9b03005141f4c375f94e1b4afa0fbf7d254b0b3a69f61b0d` | `a7a1b2b5dadfae5f4967a531f0bef76649f51c77db4dbfaec5253df1ac72c946` |
| `discovery/rho0.001_positive` | 125 | `b0f77483764a049e24903d7edc4c4aaa021a8a3defca9c5c13573abb1dce0733` | `06ca8104e3fbade33d14e87a274a167d6cc1d483585592754e56e7134b1f68a4` |
| `discovery/rho0.001_abs` | 101 | `5cd590e7980ed147f5662744628cd0010400c97cdb635e745505ecac5ce5e9e9` | `0cf8e7b1c28e1badb642ca662310a21f7d993713c3d3ae7355937b9091351b9f` |
| `discovery/rho0.003_signed` **primary** | 306 | `321233ebd0b2b726581519eca3f91f79105cf02830bf6d0fbdc5cde63e9d53da` | `12b94f0a65f244b8ef0bbdad5c76d75d72b4793bfded50312be5c63375b42e1c` |
| `discovery/rho0.003_positive` | 312 | `883b35fb097ca9248100d36941520ad47df9dd3c6459ca97e8a788dd0d531c59` | `945055602eafa2f8004ddb26066a3d60d57e28cc40a09d4cba2e3482b6c8d3e5` |
| `discovery/rho0.003_abs` | 306 | `6320b92daae16d8608d0d12065cf83fd49c9d2f3250c77d66216a81bc8ebc3c4` | `586a7ce2eb1e15c7a3287f599b64006927e5b3784de6c52cdd56f9c24ac7c1f5` |
| `discovery/rho0.005_signed` | 512 | `b889fdf26e29628e5858ee82c25ac7c5fc0b279f61127ea0d602b0fcd3d06669` | `e21a0c4110fc3657a1f4eef019cfa79dea883abaa53d683ed31f8cfa1069f87b` |
| `discovery/rho0.005_positive` | 482 | `afb07269e9736b27389ad890dad965ee395737b80813eb46db184860abfa7d4f` | `a2c224a342e906f4871510f16fa86602031c88e9e4acef1bc6f06c4e874d9640` |
| `discovery/rho0.005_abs` | 497 | `455b26901754f615449696b12eb811d8a02810a35b753a36cecd54fb1d7c4d99` | `4016886940c807f1bf3d476a36125923abe11d0ba88dc33c0ba18ba7861af979` |

Primary union feature SHA256 为 `750bce8df5082d9704e31c159d12978cbb3542259ac004e1cf50735db2b016df`。九组均通过 tensor、mask、selection split 和 control receipt 一致性检查。

## 5. 后续阶段产物

| 阶段 | 目录 | 主要产物 |
|---|---|---|
| Stage 6 | `stages/06_causal/` | 25 个条件的逐样本轨迹、指标和汇总 |
| Stage 7 | `stages/07_training/` | SFT JSONL、过滤清单、5 个 adapter 及训练 manifest |
| Stage 8 | `stages/08_evaluation/` | 两种 scope、三 seeds 的结果、对照表和 Pareto 图 |

这些文件会继续增长，不得复制进 Git。每个 stage 都有自己的 `manifests/runtime_provenance.json`，不能覆盖前置实验或其他 stage 的 receipt。

## 6. 哪个文件算真源

- 代码、环境、原始数据和模型快照：对应阶段的 `manifests/runtime_provenance.json`；
- 数据转换：`data/data_manifest.json`；
- Stage 5 activation：两份 `*_manifest.json`；
- Stage 5 mask/probe：各 discovery 目录的 `tool_action_neurons.json` 与 `probe_results.json`；
- 原始 scoped probe 迁移：`probes/scoped_original_w2t/migration_receipt.json`；
- 旧统计行为面板：`reports/stages/STAGE_STATISTICS_QWEN3_4B.md`，当前仍是 partial checkpoint；
- Stage 5 以后的人类可读结论：[阶段交接](STAGE5_PLUS_HANDOFF.md)。

任何 task ID、seed、scope、label protocol 或 SHA 不一致都应直接失败。服务器项目根和数据根 README 的维护源分别是 `docs/REMOTE_ROOT_README.md` 与 `docs/DATA_ROOT_README.md`。
