# Agent 当前状态

> 这是 AI 工作用的可持续状态页，不是正式方法文档。只记录已经核实的事实、正在进行的动作与阻塞；不得写入密码、token、SSH 私钥或 `rclone.conf` 内容。

## 1. 当前结论

- 正式主文档为 `docs/项目SPEC.md`、`docs/实验规划.md`、`docs/实验矩阵.xlsx` 和 `docs/FROZEN_TASK_SPEC_2026-07-13.md`。
- 本地已实现 `pathway_continuation_v4` 全量 canonical index、70/20/10 family-aware split、五分区发布、严格审计、完整 JSON 训练闸门和三次推理重试；CFFF 正在全量索引，因此不能写出 v4 最终 row/record/token 数。
- phenotype 暂不进入核心 SFT、AE 或 dynamics。`not_annotated` 只表示无标注，不表示阴性。
- 当前机器矩阵有 9 行：共享 SFT/AE、stage-1 SFT、第二次 SFT 对照、HNN/FDHNN 的 D2 独立预训练、D3 稳定后联合，以及 D4 直接联合消融。B2/B3 损失已实现但尚未注册独立 row；Neural ODE、C2/C3、PHNN 仍未进入当前矩阵。
- v4 的 checkpoint/run 路径同时绑定 immutable `dataset_build_id` 与 training seed；旧 release 的成功标记不能跳过新 release。第一轮只启动一个 seed 的四卡 SFT，单轮有 48 小时硬上限。

## 2. canonical 数据源

NRC 已核实的 `processed_graph` corpus：

- 1,368,605 个 JSON；
- 10,859 个 organism/target 目录；
- 16 个独立 `tar.gz` 分片，总计 16,200,057,532 bytes，逐片 SHA-256 已在 NRC 与本地暂存验证；
- pipeline snapshot 与归档 checksum 一同保存。

旧 `processed` -> CSV 统计是 32,258,032 train rows 和 36,327 test rows。它按 paragraph 保存 layer，canonical event 边界已经丢失；只能作为历史数据，不能冒充 v4。

## 3. v4 已实现的数据契约

- 直接读取 `processed_graph` relation/reaction；不从旧 paragraph 猜 event。
- 所有有端点 structural event 参与 SCC/sink/layer 构图，包括 producer 标为 `renderable=false` 的 event；不同 event 不按相同文本去重。
- `graph_id` 同时绑定相对来源路径与内容 hash；`view_id` 绑定 sink signature；`record_id` 绑定 graph/view；`sample_id` 绑定 record/prefix。
- 一个 sink-SCC view 是一个 biological record。layer 只有上游到下游的序数含义；同层 canonical traversal 只是确定性遍历，不是实测时间。
- P0 主 prompt 显示实际已知的 KEGG 来源代码并保留 source-native IDs；P1 只去掉显式来源名，P2 仅使用天然物种中立 ID。所有条件都不显示 pathway 名称、类别、ID、title、block 或 phenotype。
- target 是闭合 `pathway_continuation_v4` JSON。真实 tokenizer 计算 prompt + 完整 answer + end token，超过 8192 的候选在物化前排除；trainer 再次 fail closed，不截断 assistant JSON。
- 全量 index 不抽样、不设 family cap；515M token 是候选 release 上限，不再等同于必然启动的训练集。正式 SFT 选择保守估计不超过 48 小时的最大独立审计 release，每个 record 只选一个全局平衡的 prefix。
- seen 来源内部按完整五位 family 优化 train/validation/test 约 70/20/10；另有 held-out source 的 `test_organism` 与 source+family 双留出的 `test_strict`。

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
  -> shared 4096 -> 128 -> 4096 pure-MSE reconstruction AE
  -> stage-2 SFT-only control
     OR fixed-latent HNN (D2) -> stability gate -> low-LR joint D3
     OR fixed-latent FDHNN (D2) -> stability gate -> low-LR joint D3
     OR random HNN/FDHNN direct-joint D4 ablations
  -> direct greedy v4 JSON inference, invalid output at most three attempts
```

- `J=Q^T J0 Q` 是全 128 维严格反对称 Poisson structure，不把前/后 64 维任意命名为 q/p。
- `r=softplus(raw_r)>=0`；第一批使用 isotropic damping。
- `F(t)` 只依赖统一 surrogate layer time，不伪装成 sample-specific intervention `u`。
- upstream pathway 通过初态 `z0` 影响 rollout。
- dynamics 每步编码完整 event object；同层 event 用 `dt=1/512` 快步，进入新 layer 用 `dt=1/128` 慢步，layer boundary 不丢失，但 traversal 不得解释成实测生物时间。
- D2 固定 LoRA/AE 训练 1--3 轮并要求稳定性闸门；D3 LoRA LR `1e-5`、dynamics LR `2e-4`、前 10% dynamics-to-LoRA 梯度渐增、stage-1 KL `0.02`、每 100 optimizer steps 记录梯度夹角。
- D3 保存 best SFT、best dynamics、best composite 与 last；正式推理只用 best composite。AE 与 stage-1 SFT checkpoint 在同 seed 的所有对照间共享。

四卡调度：shared SFT 用 4 卡 DDP、PyTorch SDPA、每 rank 4 个 tokenizer workers、按长度相邻的 global batch，并把 validation 无重复分到四卡后汇总 token-weighted loss；AE、D2、D3、D4 与 SFT-control 每个 run 用 1 卡，调度器跨方法/seed 任务并行占满节点；direct inference 用 4 个互斥 shard。

## 6. 正在进行的 CFFF 落地

2026-07-16 04:10 CST 快照：

- `dataprocess/run_cfff_dataset_v4.sh` 与 canonical index worker 仍存活；
- 已索引 225,000 / 1,368,605 graphs（16.4%），生成 528,686 canonical records；约 31 graphs/s，若保持该速度，index 尚需约 10.2 小时，之后仍有 split、token 物化和全量 audit；
- 四张 A100-80GB 当前均为 0 MiB、0% utilization，不在数据阶段提前占用；
- CFFF 与本地均为 `f4e9ccf`；method 51、experiments 32、dataprocess 79 项测试，以及 9-row/18-wrapper/365-record 审计全部通过；
- 每小时监督已升级为：数据 audit 通过后选择不超过 48 小时的最大独立审计 release，只用 `20260711` 启动四卡 SFT；不会自动继续 AE/HNN。
- GitHub `origin/main` 仍停在 `3f9457e`，本地 `gh` 当前凭据无效；CFFF 已同步，不影响正在运行的数据任务。

## 7. 仍未完成

- v4 canonical index、五分区物化后的实际审计数字和真实一轮训练时间；
- graph-similarity cluster split；
- 第一 seed 的四卡 SFT 实际吞吐、显存、validation 和 checkpoint；其余 seed 等第一轮完成后再决定；
- Neural ODE、B2/B3 正式 row、B4、rerank 与 latent fusion 等后续行；
- Task 3/4 专用人工数据、Task 5 single-cell 迁移、Task 6 BioMaze 数据。
