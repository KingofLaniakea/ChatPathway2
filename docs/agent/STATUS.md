# Agent 当前状态

> 这是 AI 工作用的可持续状态页，不是正式方法文档。只记录已经核实的事实、正在进行的动作与阻塞；不得写入密码、token、SSH 私钥或 `rclone.conf` 内容。

## 1. 当前结论

- 正式主文档为 `docs/项目SPEC.md`、`docs/实验规划.md`、`docs/实验矩阵.xlsx` 和 `docs/FROZEN_TASK_SPEC_2026-07-13.md`。
- 本地已实现 `pathway_continuation_v3` 构建、严格审计、完整 JSON 训练闸门和三次推理重试；尚未在 CFFF 全量物化，因此不能写出 v3 的最终 row/record/token 数。
- phenotype 暂不进入核心 SFT、AE 或 dynamics。`not_annotated` 只表示无标注，不表示阴性。
- 当前可执行机器矩阵只有共享 SFT/AE、stage-1 SFT、第二次 SFT 对照、joint HNN 和 joint forced/damped HNN 五行；人类版 A/B/C/D 文档还包含后续实验，不应把后续行说成已经实现。

## 2. canonical 数据源

NRC 已核实的 `processed_graph` corpus：

- 1,368,605 个 JSON；
- 10,859 个 organism/target 目录；
- 16 个独立 `tar.gz` 分片，总计 16,200,057,532 bytes，逐片 SHA-256 已在 NRC 与本地暂存验证；
- pipeline snapshot 与归档 checksum 一同保存。

旧 `processed` -> CSV 统计是 32,258,032 train rows 和 36,327 test rows。它按 paragraph 保存 layer，canonical event 边界已经丢失；只能作为 v2 历史数据，不能冒充 v3。

## 3. v3 已实现的数据契约

- 直接读取 `processed_graph` relation/reaction；不从旧 paragraph 猜 event。
- 所有有端点 structural event 参与 SCC/sink/layer 构图，包括 producer 标为 `renderable=false` 的 event；不同 event 不按相同文本去重。
- `graph_id` 同时绑定相对来源路径与内容 hash；`view_id` 绑定 sink signature；`record_id` 绑定 graph/view；`sample_id` 绑定 record/prefix。
- 一个 sink-SCC view 是一个 biological record。layer 只有上游到下游的序数含义；同层 event 是无序并行集合。
- 模型只看到 observed structured prefix、任务说明和目标 JSON shape；prompt/target 不显示 pathway 名称、类别、ID、title、block、organism 或 phenotype。
- target 是闭合 `pathway_continuation_v3` JSON。真实 tokenizer 计算 prompt + 完整 answer + end token，超过 8192 的 row 在物化前排除；trainer 再次 fail closed，不截断 assistant JSON。
- train 每 family 最多 256 records，每 record 最多物化 first/middle/last 三个 prefix；每个 epoch 确定性只取一个 prefix。
- test 同时要求预留 organism 与预留五位 KEGG family；validation 预留另一组 family；train 排除两者。

当前 split 仍有一个明确边界：五位 KEGG family disjoint 不是 graph-homology disjoint。KO set/edge/subgraph similarity cluster split 尚未实现。

## 4. 生成审计

`data_audit.json` 由程序原子写入并设为 `0444`。当前实现检查：

- train/validation/test row、record、source、family、organism 数；
- source/record/sample/family/organism overlap；
- duplicate sample、record identity collision；
- phenotype/parser/substep schema；
- layer/event/entity 与 token 分布、排除的超预算 row；
- CSV 与 record JSONL 精确互相重建；
- record 与原始 `processed_graph` 重新构建的 sink-SCC view 精确一致；
- graph artifact 存在、来源路径+内容 hash、manifest/CSV/JSONL SHA-256。

CFFF scheduler 在分配 GPU 前重新核对 audit 状态、只读权限与所有上述 hash。

## 5. 当前模型与矩阵

第一批机器可执行路径：

```text
Qwen3-8B
  -> 4 x A100 shared SFT
  -> shared 4096 -> 128 -> 4096 reconstruction AE
  -> stage-2 SFT-only control
     OR joint HNN: dz/dt = J grad H
     OR joint FDHNN: dz/dt = (J-rI) grad H + F(t)
  -> direct greedy v3 JSON inference, invalid output at most three attempts
```

- `J=Q^T J0 Q` 是全 128 维严格反对称 Poisson structure，不把前/后 64 维任意命名为 q/p。
- `r=softplus(raw_r)>=0`；第一批使用 isotropic damping。
- `F(t)` 只依赖统一 surrogate layer time，不伪装成 sample-specific intervention `u`。
- upstream pathway 通过初态 `z0` 影响 rollout。
- dynamics 一次推进对应一个完整 graph layer；同层 `A relation B` event spans 先集合池化，不人为编造同层时间顺序。
- AE 与 stage-1 SFT checkpoint 在同一 seed 的 HNN/FDHNN/SFT-control 间共享；validation-selected checkpoint、早停、seed、输入 hash、训练/验证时间和 token throughput 都写入日志。

四卡调度：shared SFT 用 4 卡；主 FDHNN 用 4 卡；HNN 与 stage-2 SFT control 各用 2 卡并行；AE 用 1 卡并与 dependency-ready inference shards 穿插；direct inference 用 4 个互斥 shard。

## 6. 正在进行的 CFFF 落地

- Drive -> 本机 `/private/tmp` 暂存已完成，约 15 GiB，49 files。
- CFFF 第 1/16 个 `processed_graph` 分片已上传并通过 SHA-256；第 2/16 个分片正在独立续传会话中。
- 临时 NRC rclone 配置只位于 `/private/tmp`、权限 `0600`；在 CFFF 完整上传、远端 SHA、解压 inventory 与 v3 构建审计全部完成前不得删除，也不得提交 Git。

传输完成后的顺序：

1. CFFF 对 16 片执行完整 `sha256sum -c`；
2. 解压到临时 restore 目录；
3. 核对 1,368,605 JSON 与 10,859 目录；
4. 目标不存在时再原子落到 `KEGG_all_new/processed_graph`；
5. 运行 v3 cap256 builder 和 `data_audit.json`；
6. 在 CFFF `.venv` 执行完整单测、matrix audit、CPU/GPU smoke；
7. 全部通过后删除临时配置和本机 staging。

## 7. 仍未完成

- CFFF `processed_graph` 全部传输、校验与恢复；
- v3 全量物化后的实际审计数字和真实一天训练时间；
- graph-similarity cluster split；
- dynamics-only、dynamics pretrain -> joint、Neural ODE、AE geometry/predictive、rerank 与 latent fusion 等 A/B/C/D 后续行；
- Task 3/4 专用人工数据、Task 5 single-cell 迁移、Task 6 BioMaze 数据。
