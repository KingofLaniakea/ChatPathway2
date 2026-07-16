# 实验运行目录规范

本规范定义 CFFF 上所有 ChatPathway 数据版本与实验运行的强制目录结构。目标是让一次运行的代码、输入、冻结资产、日志、指标和图表能够独立审计与恢复。

## 1. 固定层级

资产根目录为：

```text
/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui
```

每个数据/算法版本使用一个独立版本目录，每次正式运行使用单调递增的 `run_N`：

```text
runs/
└── <version>/
    ├── code/
    ├── VERSION_MANIFEST.json
    ├── RUN_DIRECTORY_STANDARD.md
    └── run_<N>/
        ├── RUN_MANIFEST.json
        ├── reports/
        ├── results/
        └── logs/
```

例如本轮 v3 固定为：

```text
runs/v3/
├── code/
├── VERSION_MANIFEST.json
├── RUN_DIRECTORY_STANDARD.md
└── run_1/
    ├── RUN_MANIFEST.json
    ├── reports/
    ├── results/
    └── logs/
```

不得覆盖既有 `run_N`。重新训练、改变 checkpoint、改变数据切分、改变评测代码或改变关键参数时，新建下一个编号。

## 2. 各目录的唯一职责

### `code/`

保存该版本真实可恢复的代码快照，而不是只写一个分支名。至少包含：

- Git bundle 或完整源代码归档；
- 精确 Git commit；
- 原始仓库地址；
- 快照 SHA-256；
- 恢复命令。

同一版本的多个 `run_N` 复用版本级 `code/`；如果算法代码发生实质变化，应升级版本或在新 run 的 manifest 中记录新的代码快照。

### `run_N/reports/`

保存本轮可直接阅读或再次分析的评估材料：

- 汇总指标与逐样本指标；
- 图表、报告、共同样本清单；
- 评测候选、评测输入的冻结小型副本；
- 数据/评测审计清单与哈希；
- 生成图表和汇总表所需的小型派生数据。

这里不放模型权重，也不把数十 GB 的原始训练数据重复复制进来。

### `run_N/results/`

保存或索引本轮产生结果所必需的冻结资产：

```text
results/
├── checkpoints/
├── models/
├── datasets/
├── inference/
├── latents/
└── ASSET_MANIFEST.json
```

- checkpoint、AE、dynamics 权重等体积合理的资产优先做真实副本；
- 大型基础模型或全量数据集可以使用指向共享只读资产的符号链接，但必须在 `ASSET_MANIFEST.json` 中记录原始绝对路径、解析后的真实路径、大小、文件数及稳定哈希/官方 revision；
- 仅有符号链接而没有 manifest 不算冻结；
- 不得移动原始共享资产来“整理目录”，以免破坏已有命令和历史运行。

### `run_N/logs/`

集中保存训练、推理、评测与系统状态日志：

```text
logs/
├── training/
├── inference/
├── evaluation/
└── system/
```

日志必须包含启动命令、配置、开始/结束时间、退出状态；GPU 状态与错误诊断放在 `system/`。不得只依赖 tmux 滚屏或终端历史。

## 3. `RUN_MANIFEST.json` 必填信息

每次运行至少记录：

- `version`、`run_id`、状态和创建时间；
- Git commit、代码快照路径与 SHA-256；
- 数据集 ID、切分规则、样本数与输入哈希；
- 基础模型 revision、SFT/AE/dynamics/checkpoint 路径与哈希；
- 完整训练、推理和评测命令或其配置文件路径；
- 随机种子、最大 token 长度、batch/gradient accumulation、训练轮数与早停选择；
- GPU 型号、数量和设备映射；
- 结果、日志和报告相对路径；
- 已知限制与不允许做出的科学结论。

`RUN_MANIFEST.json` 是该运行的入口。报告中的模型名或“best checkpoint”文字不能替代它。

## 4. 完成条件

一次运行只有同时满足以下条件才可标记 `complete`：

1. 所有必需报告、结果和日志均位于对应目录；
2. 代码快照可以恢复到记录的 commit；
3. 冻结资产链接全部可解析，哈希/版本校验通过；
4. 逐样本数量与汇总数量一致；
5. 仓库工作树状态和服务器目录清单已记录；
6. 没有正在写入这些冻结文件的训练或评测进程。

## 5. 后续运行约定

- v3 的第一次正式整理固定为 `runs/v3/run_1`；后续 v3 运行依次使用 `run_2`、`run_3`。
- v4 从 `runs/v4/run_1` 开始，不能写入 v3 目录。
- 临时测试可以放在 `runs/<version>/scratch/`，但不得被论文或正式报告引用；一旦用于正式比较，必须转入新的 `run_N` 并冻结 manifest。
- `latest` 符号链接只能用于导航，不能作为论文、脚本或 manifest 中的权威路径。
