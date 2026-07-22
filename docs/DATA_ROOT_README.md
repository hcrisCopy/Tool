# CallTool_data 说明

本目录只保存不进入 Git 的环境和大型产物；其维护代码位于同级 `../CallTool_code/`。

```text
CallTool_data/
├── README.md
├── When2Tool/                         # 固定上游数据快照
├── cache/pip/                         # 可再生安装缓存，不是正式产物
├── conda_envs/calltool_qwen3/        # 独立运行环境
├── environment.lock.txt              # 本次环境冻结记录
├── hfd.sh                             # 通用资源下载辅助脚本，不含凭据
└── when2tool_precise_shield/
    └── qwen3-4b-instruct-2507/
        ├── data/
        ├── labels/
        ├── manifests/
        ├── probes/
        ├── outputs/
        ├── analysis/
        ├── logs/
        └── reports/
```

主 run root 为 `when2tool_precise_shield/qwen3-4b-instruct-2507/`。其中：

- `manifests/runtime_provenance.json` 固定行为生成时的代码、配置、GPU、依赖、数据和模型文件 SHA；
- `manifests/formal_stage_audit.json` 与 `manifests/stage_handoff.json` 只在完整正式面板和三套统计都结束后生成；当前暂停状态下不存在是有意的；
- `reports/STAGE_STATISTICS_QWEN3_4B.md` 是 Git 阶段报告的普通文件副本，当前记录 partial checkpoint；
- `analysis/{fulltools,scoped_adapted,scoped_original_w2t}/` 保存统计 CSV、JSON 与关键 PNG；
- `logs/{setup,labels,hidden,probes,behavior}/` 只用于诊断，不在 stage handoff 的哈希范围内。

不要把这些文件复制进 Git 仓库，也不要用缓存中的同名模型或数据覆盖现有快照。资源来源、revision 与已知 SHA 见 `../CallTool_code/docs/data_manifest.md`。
