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

## 当前阶段命令

下面所有命令均在 `CallTool_code` 根目录执行：

```bash
PY=../CallTool_data/conda_envs/calltool_qwen3/bin/python

# 1. 审计并物化官方 Parquet
$PY -m experiments_2d.scripts.prepare_data

# 2. 先取每个 environment × difficulty 一条，共 45 条，生成 no-tool 标签
$PY -m experiments_2d.scripts.prepare_labels --smoke

# 3. 通过 45 条冒烟后再删除 --smoke 跑完整 train/test
$PY -m experiments_2d.scripts.prepare_labels
```

每个阶段都会写 manifest、配置快照和进度日志。后续 onset、神经元选择、因果干预
与 Probe&Prefill 命令将在对应实现完成并通过冒烟后补充。

