# CallTool 2D

本项目验证 Qwen3-4B-Instruct-2507 内部是否形成二维工具决策状态：

1. 是否需要调用工具（binary necessity）；
2. 应调用哪种类型工具（A/B/C tool type）。

本文将第 `l` 层 hidden state 的第 `i` 个标量分量操作性定义为神经元
`n=(l,i)`。层定位使用 residual write `r_l=h_l-h_(l-1)`；神经元选择、
probe 与干预使用 hidden activation `h_l[i]`。

## 目录约定

代码仓库与大文件目录必须同级：

```text
project_root/
├── CallTool_code/                  # 本仓库，只放代码和小配置
├── CallTool_data/
│   ├── When2Tool/                  # 官方 Parquet 数据
│   ├── conda_envs/calltool_qwen3/  # 独立环境
│   └── experiments_2d/             # 标签、hidden states、图表、日志
└── Qwen/Qwen3-4B-Instruct-2507/    # 模型权重
```

所有程序从仓库根目录解析相对路径。路径不正确时会直接报错，不会自动下载、
跳过样本或静默换用其他数据。

## 一次性安装

```bash
git clone --recurse-submodules <repository-url> CallTool_code
cd CallTool_code
conda create -p ../CallTool_data/conda_envs/calltool_qwen3 python=3.11 -y
../CallTool_data/conda_envs/calltool_qwen3/bin/python -m pip install -r requirements.txt
../CallTool_data/conda_envs/calltool_qwen3/bin/python -m pip check
```

上游 When2Tool 以 submodule 固定到论文指定 commit，仅用于公平复现其数据环境和
评测协议；本仓库不复制其源码。`CodeExecutorEnv` 的工具执行不会直接采用上游
`exec` 实现。

三个 prompt variant 已冻结：`P_env` 使用样本原始工具列表；`P_no_schema` 不给工具
schema；`P_all` 给每条样本完全相同、按 environment 命名空间隔离的全候选工具菜单。
因此 `P_all` 既不会泄漏当前 environment，后续也能实际统计 wrong-category calls。

## 当前阶段命令

下面所有命令均在 `CallTool_code` 根目录执行：

```bash
PY=../CallTool_data/conda_envs/calltool_qwen3/bin/python

# 1. 审计并物化官方 Parquet
$PY -m experiments_2d.scripts.prepare_data

# 2. 先取每个 environment × difficulty 一条，共 45 条，生成官方 hard-no-tool 标签
$PY -m experiments_2d.scripts.prepare_labels --smoke

# 3. 主标签：seed=0 与固定 When2Tool/vLLM 默认完全一致
$PY -m experiments_2d.scripts.prepare_labels --split train
$PY -m experiments_2d.scripts.prepare_labels --split test

# 4. 抽取 raw h_0...h_36；batch=1 与官方路径对齐
$PY -m experiments_2d.scripts.extract_hidden --split train
$PY -m experiments_2d.scripts.extract_hidden --split test

# 5. 复现官方 37×2560 all-layer binary probe
$PY -m experiments_2d.scripts.reproduce_w2t_probe

# 6. 主 onset：200 次冻结置换；输出 CSV、JSON 和两张关键曲线图
$PY -m experiments_2d.scripts.onset_scan --mode full --label-seed 0

# 7. 生成设置的 seed=1/2 重复实验（较慢，主结果完成后分 split 运行）
$PY -m experiments_2d.scripts.prepare_labels --seed 1 --split train
$PY -m experiments_2d.scripts.prepare_labels --seed 1 --split test
$PY -m experiments_2d.scripts.prepare_labels --seed 2 --split train
$PY -m experiments_2d.scripts.prepare_labels --seed 2 --split test
```

标签直接复用固定上游的 prompt、parser、state machine 与 scorer，并在 hard-no-tool
分支额外安装“任何工具执行即报错”的安全闸。seed 0 只显式写出 vLLM 原本的默认值；
seed 1/2 仅改变生成随机种子。任何旧产物都必须先归档，脚本不会把旧结果当成新结果。

每个阶段都会写 manifest、配置快照和进度日志。神经元选择、因果干预与
Probe&Prefill 命令将在对应实现完成并通过冒烟后补充。
