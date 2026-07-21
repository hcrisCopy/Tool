# 阶段 1：Qwen3-4B-Instruct-2507 onset 探测

日期：2026-07-21  
状态：完成正式 seed-0 标签、全量 hidden 提取、When2Tool 全层 probe 基线和 200 次置换 onset 扫描；按预注册里程碑暂停，尚未进入神经元筛选与因果干预。

## 1. 冻结定义与数据

- 神经元定义保持为最终 prompt token 上原始 hidden state 的标量分量 `(layer l, component i)`。
- `h_0` 是 block 1 输入，`h_l` 是 decoder block `l` 的原始输出；Qwen3 共 36 个 block、hidden size 2560。
- residual write 为 `r_l = h_l - h_{l-1}`，逐维按 train 统计量使用 `(r - mean) / (std + 1e-6)` 标准化。
- necessity 主分析使用 `P_no_schema`；type 主分析使用所有样本完全一致、无 A/B/C 文本且顺序固定的 33 工具 `P_all`。
- clean discovery sets 为 `E_c=(c,easy,y=0)` 与 `H_c=(c,hard,y=1)`；medium 不参与 discovery。
- clean set 数量：`E_A=81, E_B=60, E_C=66, H_A=84, H_B=100, H_C=56`。
- 正式 seed-0 标签：train 900 条，`tool_necessary=459`；test 2250 条，`tool_necessary=1174`。

## 2. 计算方法

每层计算逐维 Welch 统计量，并聚合为：

```text
T_li = (mean(P_li) - mean(N_li)) /
       sqrt(var(P_li)/n_P + var(N_li)/n_N + 1e-6)
S_l  = mean_i(T_li^2)
Z_l  = (S_l - mean(S_l^null)) / (std(S_l^null) + 1e-6)
```

- 置换次数 `B=200`，三点居中平滑，边界使用两个可用层。
- onset `l*` 是平滑曲线第一次达到全局峰值 95% 的层，不是“第一次显著”的层。
- necessity 对 A/B/C 分别比较 `H_c` 与 `E_c`，共同曲线为逐层 `min(A,B,C)`。
- type 在 E/H 内分别做每类 one-vs-rest，再取两种状态的逐层最小值；三类总体曲线为类别均值。
- necessity 的 null 在 environment 内置换 y；type 的 null 以 environment 为块置换 category，避免把 environment 身份误当成 A/B/C 类型知识。

最后一项是相对 PDF 字面“逐样本普通 shuffle”的统计增强。它更严格，但会改变 Z、`l*` 和 `R_peak`；后续报告必须同时保留这一偏离说明，不可声称完全未改协议。

## 3. When2Tool 全层 probe 基线

使用固定上游 commit `66f100089d1f3f7e7f2acee279c4dbf6e7ae5e2c` 的 `train_probe.py --reg 10000 --all_layers`，拼接 37×2560=94,720 维，保持上游两次 `StandardScaler` 行为不变。

| 指标 | 本次 seed-0 单卡复现 |
|---|---:|
| train accuracy | 0.9278 |
| test AUROC | 0.9467 |
| test accuracy | 0.8853 |

该结果说明标签、hidden 和固定上游 probe 的对接正确。本次 vLLM 显式设 seed=0、单 GPU 且 train/test 分别重置随机流，因此应称为“固定 seed-0 单卡复现实例”，不是论文默认 `seed=None` 随机流的逐比特复现。

## 4. Necessity onset 结果

| 曲线 | FWER p | 个体峰值层 | 个体 95% 层 |
|---|---:|---:|---:|
| A | 0.004975 | 21 | 20 |
| B | 0.004975 | 23 | 22 |
| C | 0.004975 | 28 | 15 |

共同曲线结果：

| 指标 | 结果 |
|---|---:|
| onset `l_y*` | 23 |
| 全局峰值层 | 28 |
| `Z(l_y*)` | 43.5063 |
| 全局峰值 Z | 44.1159 |
| `R_peak` | 1.6583 |
| 默认后续窗口 | 21--25 |
| `l_y*/36 <= 0.45` | **失败** |
| `R_peak >= 1.5` | 通过 |

结论：三类 necessity 对比都显著，且从很早层就有可检测差异；但按预注册的“第一次达到峰值 95%”定义，写入成熟点在第 23 层，不是早层。共同曲线是宽峰，不能表述成“第 23 层之前不存在、到第 23 层突然出现”。当前结果支持“necessity 信息贯穿多层并在中后段成熟”，不支持更强的“早层局部形成”结论。

## 5. Type onset 结果

总体均值曲线给出 `l_c*=20`、峰值层 20、峰值 Z=2.4139、`R_peak=6.5074`、候选窗口 18--22。然而总体峰值主要由 B 类贡献：

| 类别 | E 状态 FWER p | H 状态 FWER p | `min(E,H)` 峰值层 | 峰值 Z |
|---|---:|---:|---:|---:|
| A | 0.3881 | 0.2289 | 15 | 1.4493 |
| B | 0.0448 | 0.0597 | 20 | 5.4765 |
| C | 0.2438 | 0.5622 | 20 | 0.3730 |

结论：目前不能把总体 `type onset=20` 解释为 A/B/C 三类都存在的统一类型机制。B 有清楚峰值但 H 状态仅临界；A 和 C 未通过 environment-block FWER 检验。直接进入“共享/特异 type 神经元”会有把 B 类或 environment 语义误写成三类机制的风险。

## 6. 产物

远端数据根目录均相对代码仓库为：

```text
../CallTool_data/experiments_2d/qwen3-4b-instruct-2507/
├── labels/full/seed_0/
├── hidden/full/
│   ├── P_env/
│   ├── P_all/
│   └── P_no_schema/
├── baseline/w2t_all/
│   ├── baseline_input_manifest.json
│   ├── probe_no_reasoning.pt
│   └── probe_results_no_reasoning.json
└── onset/full/label_seed_0/shuffles_200/
    ├── onset_summary.json
    ├── onset_curves.csv
    ├── onset_tool_necessity_write_signal.png
    ├── onset_category_write_signal.png
    └── residual_scalers.pt
```

输入文件 SHA256、固定上游 commit、hidden 协议版本和 `P_all` 菜单 SHA256 均已写入 `onset_summary.json`。

## 7. 建议的下一步（等待确认后执行）

在进入 `(l,i)` 神经元筛选前，建议增加一个不改变主分析的确认层：

1. 保留当前 `l_y*=23` 作为预注册主结论；新增“最早 FWER 显著层”作为明确标注的 secondary 指标，不可事后替换主定义。
2. necessity 使用 difficulty/environment 条件内的对照或 medium held-out 检验，确认当前早层可检测信号不是 easy-vs-hard 难度或词面差异。
3. 只需补跑 seed 1/2 的 no-tool train 标签即可复用现有 hidden，检查 `l_y*` 和 clean-set 构成对生成随机性的稳定性。
4. type 先做 environment-held-out / leave-one-environment-out 的类别验证；若仍只有 B 稳定，则把结论收缩为类型非对称，并分别报告 A/B/C，而不是用总体均值宣称统一 type onset。
5. 通过以上确认后，再冻结 necessity 与 type 的候选层窗口，进入 raw hidden 标量筛选、稀疏 probe 及 mask/patch 因果实验。

