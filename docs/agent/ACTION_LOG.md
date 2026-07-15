# Agent 行动记录

> 本文档是可持续更新的行动史与待办队列，不是面向用户的方法主文档。只记录已做的动作、
> 证据、未竟工作和明确决策；数据、服务器与结果的当前真值在 [`STATUS.md`](STATUS.md) 更新。

## 1. 更新规则

每次记录包含：日期/时间、作用域、动作、证据或 artifact、结果、后续。
不保存密码、token、SSH 私钥、OAuth refresh token 或 `rclone.conf` 内容。

## 2. 已完成里程碑

### 源码迁移与目录整理

- `a419f53`：导入经审计的服务器源码快照，初始 37 个 Python 文件在删除旧根目录副本前完成 byte-check。
- `2044df2`：将方法、脚本、baseline、downstream 和 experiment wrapper 收拢到维护目录。
- `ff2a71a`：将通路推理入口改为 CLI 可配置，增加输出目录创建、覆盖保护与 run record，
  未改变旧 checkpoint 的推理语义。

### 训练工程修正

- SFT 从硬编码 DDP-only 脚本改为 argparse 入口，同时支持普通单进程与 `torchrun`。
- SFT、AE 和 stage-2 加入固定 seed、family-grouped validation、early stopping、validation-selected
  `checkpoint_best`、`run_config`/history/metrics log 和输入/checkpoint provenance。
- AE 修复了 reconstruction history 未初始化、MSE/cosine 计数不完整与 tail gradient accumulation。
- Framework A 修复了 HNN trajectory detach、tensor time column、prompt boundary 由 padding 判断错误、
  tail accumulation，并使目标 layer stop-gradient而预测路径保持可导。
- 统一了 attention mask 与真实 sequence length，避免 PAD/EOS 共享 token ID 时判断错误。
- 加入 `run_steps`、wrapper dry-run、matrix/runtime-manifest consistency audit、asset preflight、
  smoke-input 准备与 CFFF 四 GPU scheduler。

### Hamiltonian 算法整理

- 移除将前 64/后 64 latent 轴直接命名 q/p 的任意分割，实现 `J=Q^T J0 Q` 的学习正交
  Poisson frame。
- 固定第一轮两个向量场：`J∇H` 与 `(J-rI)∇H+F(t)`，`r>=0`、`F(t)` time-only。
- 移除将 prompt embedding 假装成 PHNN control `u` 的错误原型；PHNN 延后到数据有独立 port 契约。
- dynamics target 从 token-by-token 纠正为 graph-layer-by-graph-layer，同层 event span pooling，
  `dt=1/128`、最多 128 layer advances。
- 固化 stage-1、compute-matched stage-2 SFT-only、HNN、FDHNN 四条归因臂与三 seed 共享 SFT/AE 规则。

### 2026-07-16 — v4 event/layer dynamics 与 D3 主线

- 将 v4 dynamics span 从 event 的 `text` 扩展为完整 compact event object，包含参与者、action、mediators、target 与 text；仍按 graph layer 分组。
- 实现两层 surrogate-time：层内 canonical event 使用 `dt=1/512`，进入新 layer 使用 `dt=1/128`；明确不将层内遍历声称为实测生物时间。
- B1 AE 主基准改为纯 MSE；B2 下一层预测和 B3 latent 均值/方差/协方差损失以默认权重 0 注册。
- 新增 `hamiltonian_pretrain.py`，固定 SFT/AE 后独立训练 HNN/FDHNN 1--3 轮，并以有限性、coverage、改善和回退阈值决定 `stability_passed`。
- 新增 E010→E011 与 E020→E021：稳定 checkpoint 才能进入低 LR D3；D4 的 exp001/exp002 保留为直接联合消融。
- D3 加入 stage-1 KL、dynamics-to-LoRA 前 10% 梯度渐增、每 100 optimizer steps 梯度夹角和 best-SFT/best-dynamics/best-composite/last 四类 checkpoint。
- 调度器不再用“文件存在”判断预训练成功；`run_complete.json` 必须明确 `status=completed`，否则依赖任务阻断。
- 本地矩阵/路径/单测和 Excel 渲染审计通过；CFFF PyTorch 单测待代码同步后运行，不加载基础模型。

### 推理与 artifact 审计

- 确认历史 Framework A inference 只加载 LoRA，不运行 HNN/AE rollout。
- 推理增加 finish reason、generated token count、JSON/schema validity、source identity 与逐样本 progress JSONL。
- 实测 batch size 8 改变部分贪心生成轨迹后，将受控 inference 固定为每进程 batch size 1，
  四卡通过互斥 shard 并行。
- 历史 v2 strict core 曾按 gold 长度使用 1024-token generation cap；v3 已改为首次 4096、最多三次、末次 8192，并保留每次生成历史。

### 数据修正与审计

- 确认 `/Users/tpmam/Projects/ChatPathway/dataprocess` 是更早的六阶段知识对齐流程，
  不是当前 CSV builder 的旧版。
- 补齐 `record_id`、`sample_id`、source/organism/pathway/block/family 字段，并要求推理输出保留。
- 将 phenotype 状态从模糊 `missing` 改为 `not_annotated`；不当阴性，不进核心损失。
- 将 strict core 改为 organism + 五位 KEGG pathway-family disjoint，把允许 family overlap 的七物种表单独称为
  organism-transfer evaluation。
- 建立 record-balanced 0.1% 派生池，每 record 最多三 prefix，trainer 每 epoch 每 record 确定性选一 prefix。
- 修正 pandas 自动将 `pathway_family_id="00051"` 转成整数导致的 validation split 错误，共享 CSV reader 改为字符串读取。
- 实现 family coverage selector，生成 cap32/cap64 嵌套数据、固定 validation CSV、selection manifest 与 token-budget report。
- 完成 NRC corpus/Step 12/processed vs processed_graph 审计，发现每 layer 写盘为一个 paragraph，
  canonical `unit_id`/endpoint/event-to-view provenance 没有进入 `processed`。
- 确认 `sentence_parser_v1` 只是句子边界，不能称 canonical biological event；提出 structured view v3。
- 发现 `method/training/sequence.py` 可在 assistant JSON token 中间截断；v3 要求 materializer
  按完整 layer/horizon 适配预算。

### 数据归档与传输

- `processed` 已分 16 片归档到 Google Drive，并有逐片 SHA-256、总 checksum、manifest 和恢复说明。
- `processed_graph` 已在 NRC 分 16 片打包，实际 archive 合计约 15.09 GiB；加 pipeline snapshot
  的 17 项 SHA-256 在 NRC 复验通过。
- Drive 续传已完成初步 `rclone check`，并后续增加 cap64 数据与审计 artifact。
- 用户已批准仅为本次 Drive->CFFF 传输临时使用 NRC `rclone.conf`；用完必须删除本地临时副本。

### GPU 观测

- 完成四卡 A100-80GB、8192-token 的 SFT/AE/HNN/FDHNN 工程贯通测试。
- 完成 cap32 SFT 四卡完整单轮吞吐/显存实测；证据与限制见 `STATUS.md`。
- 确认“节点可占满四卡”与“每个单独 trainer 都是四卡 DDP”不是同一声明；
  当前 SFT 为四卡 DDP，AE/stage-2 可通过多任务并行占满节点。

### 下游任务

- 在 `downstream/new_tasks/` 固化 Task 0--6 的 schema、metric、manifest 与 claim gates。
- Task 0/2 共用 `method.analysis.semantic_latent_export`，保证 representation 与 Framework A layer construction 一致。
- Task 1 默认评估 layer-set event；只有独立 causal ordering provenance 才评平面 substep 顺序。
- Task 3 拒绝 raw energy-delta 因果 proxy，要求 expert-validated candidate 与 validation calibration。
- Task 4 要求真实 intervention evidence 和校准 phenotype probability，拒绝 `F(t)` 充当 knockout `u`。
- Task 5 要求 cell/gene/perturbation/normalization manifest 和 controlled ablation；Task 6 要求 BioMaze version/license/split/contamination audit。

### 文档整理

- 2026-07-13：将 `docs/` 根目录收敛为项目 SPEC、实验规划、实验矩阵、冻结任务定义与 `agent/` 工作目录。
- 方法、数据 contract、prompt、训练与 downstream 合并至 `项目SPEC.md`。
- 受控矩阵、执行闸门和后续轴合并至 `实验规划.md`。
- 运行性、服务器、data/result/checkpoint/provenance 状态合并至 `agent/STATUS.md`，
  行动史与待办合并至本文档。
- 用户提供的 PHNN PDF 只移入 `agent/references/` 保留，没有将全文复制到主文档。

## 3. 当前待办（按依赖顺序）

### 2026-07-16 04:10 — v4 artifact 隔离与四卡 SFT 启动门

- 作用域：本地 / CFFF / GitHub。
- 动作：checkpoint/run 改为 dataset-build + seed 双重命名空间；增加只运行 shared SFT 的调度入口、48 小时硬门、derived release 透传、SDPA、长度分组 DDP、并行 tokenization 和四卡 validation。
- 输入：`pathway_v4_full` immutable manifest/audit；commit `145735e`。
- 输出：commit `f5a9aef`、dry-run 修复 `f4e9ccf`；CFFF 已 fast-forward。
- 验证：method 51/51、experiments 32/32、dataprocess 79（5 historical skip）、9 matrix rows、18 wrappers、365 consistency records。
- 结果：算法和启动链完成；数据索引进行中，225,000 / 1,368,605 graphs。
- 后续：audit 通过后选择 48 小时内最大 release，只启动 seed 20260711 的四卡 SFT；GitHub 凭据需重新登录后推送。

### P0：数据落地与安全清理

- [ ] 完成 Drive->CFFF 的 `processed_graph`/pipeline/cap64 资产传输。
- [ ] 在 CFFF 运行完整 SHA-256，逐 archive 解压，用 inventory 核对文件数与内容。
- [ ] 传输完成后删除 `/private/tmp` 中临时 `rclone.conf`，并不在日志中回显内容。
- [ ] 把 job ID、路径、hash、成功/失败数量更新到 `STATUS.md`。

### P0：structured view v3

- [x] 本地实现由 `processed_graph` 直接输出稳定 `graph_id/view_id/record_id/event_id`。
- [x] 先使用全部有端点 structural event 计算 SCC/sink/layer，再保留 producer renderability。
- [x] 不按文本去重，并在 release audit 中由源 graph 重建每个被选 view 做逐 record 精确比对。
- [ ] CFFF 全量物化后冻结 inventory/hash/schema/build manifest；producer commit/config 仍需从 pipeline snapshot 固化。

### P0：合法 JSON 训练物化

- [x] canonical record JSONL 与训练 CSV prefix materialization 分层；trainer 每 epoch 每 record 选一个 prefix。
- [x] 用真实 tokenizer 只保留完整闭合 JSON，训练器遇到超预算 row 直接失败。
- [x] 核心 answer 移除 `predicted_phenotype:null`；phenotype 只留 `not_annotated` metadata。
- [x] 当前正式 prompt 固定 prefix-only，不显示 pathway/organism metadata。
- [x] 加入 target schema、CSV/JSONL/source graph 重建、token 与 inference 三次重试测试。
- [ ] metadata-rich prompt 仅作为未来消融实现，不进入当前首轮数据 release。

### P1：冻结正式 dataset revision

- [x] 用户选择从 canonical `processed_graph` 构建 v3 cap256，不以 v2 cap64 启动主矩阵。
- [ ] 在 CFFF 生成并冻结 v3 dataset/split/selection/token-budget manifest，与旧 checkpoint 分离。
- [ ] 在 graph feature index 到位后增加 graph-similarity cluster split 和 coverage selector。

### P1：正式训练与时间实测

- [ ] 先用冻结 revision 做 disposable 小规模端到端贯通。
- [ ] 分别实测四卡 SFT、单卡 AE，以及任务并行的单卡 HNN/FDHNN/SFT-control throughput、显存、验证与保存开销。
- [ ] 对 `20260711/12/13` 运行 shared base000 与 exp000/003/010/011/020/021/001/002，不混用 SFT/AE digest。
- [ ] 只对 validation-selected checkpoint 运行 strict core 与 organism-transfer direct inference。
- [ ] 汇总三 seed 结果、失败样本、coverage、config、hash 和完整 logs。

### P1：下游数据

- [ ] Task 3：人工/专家验证 true path 与 direction reversal、跨层 shuffle、matched unrelated negatives，
  保存 changed event IDs 与 annotation provenance。
- [ ] Task 4：构建真实 WT/KO/rescue intervention 数据与独立 validation calibration，不复用当前 missing phenotype。
- [ ] Task 5：将 AutoDL single-cell 资产迁往 CFFF，固定 gene/cell/perturbation/normalization/control manifest。
- [ ] Task 6：下载 BioMaze，记录 version/source/license/split/contamination audit。

### P2：第一轮后的方法轴

- [ ] graph-layer JSON boundary controller。
- [x] v4 完整 event-object dynamics：层内快步、跨 graph-layer 慢步，并保留 layer identity。
- [x] 实现 D3：固定 SFT+AE 预训练 HNN/FDHNN 1--3 轮并通过 validation 稳定判据，再以
  LoRA `1e-5`、dynamics `2e-4` 联合训练；加入 dynamics-to-LoRA 梯度 warmup、
  stage-1 SFT 输出 KL、LoRA 梯度夹角日志和三类 validation-best checkpoint。
- [x] 实现 D2/D3 的成功终止标记与调度依赖；稳定性失败不能被文件存在性误判为完成。
- [x] 将 B1 AE 固定为纯 MSE 主基准；实现但不默认启用 B2 predictive 与 B3 geometry losses。
- [ ] 独立 token-resolution dynamics。
- [ ] 前述 controller 通过后的 multiscale hybrid。
- [ ] AE 表示、frozen teacher target、latent dimension/geometry 消融。
- [ ] Neural ODE/GENERIC/Koopman/SINDy 等共享 SFT/AE 对照。
- [ ] 只在有独立 observed port/control 后实现 PHNN。

## 4. 新记录模板

```markdown
### YYYY-MM-DD HH:MM — 标题

- 作用域：本地 / NRC / CFFF / Drive / GitHub
- 动作：
- 输入：路径、revision、hash
- 输出：路径、revision、hash
- 验证：命令、job ID、计数、测试、日志
- 结果：已完成 / 进行中 / 失败 / 待决定
- 后续：
```
