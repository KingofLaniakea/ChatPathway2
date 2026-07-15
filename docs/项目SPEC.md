# ChatPathway 项目精神，Pipeline和Mehtod

Disclaimer： 本文件为项目核心约束，AI coding agent不可以擅自操作本文件，除非得到我的明确允许。

Disclaimer：为避免把计划写成既成事实，本文使用三种标记：

- **当前事实**：数据或代码现在确实如此；
- **目标（v4）**：下一代数据契约，实现完并全量校验后才能使用该声明；
- **尚未完成**：仍缺数据、程序或科学契约，不得当作实验结果。

## 项目精神
- 首先，目前llm很擅长reasoning，我们打算把它应用在pathway reasoning上面。我们的任务设计是输入上游pathway补全下游pathway。但是llm擅长文本内容reasoning不一定适合pathway这个domain（这个只是一个belief，现在通用大模型，像gpt5.6sol/claude mythos有多强感觉已经能覆盖这个domain了）。所以我们觉得pathway可以用某种微分方程表示，我们用hnn/phnn等微分方程神经网络/能量神经网络学习latent dynamcis，从而监督约束pathway step by step generation。
- 其次，我们可以整合hypothesis generation与hypothesis testing，testing这块以前的黑箱模型是做不到的。这也引出了需要设计的下游任务。
- 第三，我们的项目也因此从kegg 图表数据构造了一个能用于大模型训练/微调的文本数据集。
- 最后，我们用ae压缩到latent space学习，认为latent space比较能学到重要的生物信息。

## 1. 总体路线

```text
KEGG KGML 与 support files
  -> canonical processed_graph
  -> sink-SCC rooted structured view
  -> 一条 view 一个 biological record
  -> 固定 split manifest + 训练时动态 prefix/horizon
  -> messages JSONL/ CSV
  -> Qwen3-8B 第一阶段 SFT
  -> 共享重建 AE（4096 -> 128 -> 4096）
  -> 第二阶段 SFT 对照 / HNN / forced-damped HNN
  -> 直接贪心生成（json格式）
  -> Task 0--6 下游任务
```

注意！源图、文本视图、模型样本和展开的 prefix 是四种不同对象。不应继续把它们统称为 “一个 JSON 转 CSV 数据集”。

## 2. KEGG 数据到规范图事件

### 2.1 不可变输入与全量索引

CFFF 上的 `KEGG_all_new/processed_graph` 是结构事实源：共 1,368,605 个 JSON、10,859 个来源目录。每个文件保存 node、relation、reaction 和 pathway metadata。`processed` 是由这些事件渲染并按 layer 拼接得到的历史文本视图，不能反向当作 event truth；v4 只用它逐事件验证重建的 `legacy_text` 能在对应旧段落中精确找到，失败的 graph 整体隔离。

v4 首先对全部 `processed_graph` 做一次可恢复并行扫描，写入压缩 SQLite canonical index。这个阶段不做 train/test 划分、不抽样，也不设 family cap。索引为每个源文件保存内容 SHA-256、状态、排除原因与统计；为每个可用 sink view 保存完整 record。生成代码或模板 hash 改变时，旧索引拒绝续跑，必须使用新目录，因此不会把两种定义混在一起。

### 2.2 实体、action 与文本

一个非 group KGML entry 是一个模型实体。其第一个 resolved ID 是 `canonical_id`，其余 resolved ID 是 `aliases`，不能把同一 entry 错拆成多个参与者。group entry 才递归展开为成员实体。任何端点缺失、未解析或字段矛盾都会隔离整张 graph，而不是删掉坏边后重算拓扑。

每个模型事件保留：

- `event_type`：`relation` 或 `reaction`；
- `source / mediators / target`：规范实体及 aliases；
- `action`：relation class、全部 subtype，或 conversion 的可逆性；
- `producer_event_ids` 与 node provenance；
- `legacy_text`：按归档 Step12 模板精确复现的旧句子；
- `text`：使用同一结构事实和固定模板生成、修正已知方向/措辞错误后的训练句子。

因此 action 不再被压成一个模糊动词。例如 phosphorylation 与 activation 可以同时存在；reversible reaction 仍是一个带 `reversibility=reversible` 的 reaction，而不是两条互相矛盾的独立反应。

历史 Step12 的顺序是“event 先生成句子，再组成 SCC/layer，最后按文本去重并拼段落”，所以不能从 `processed/*.json` 的 paragraph 无损恢复 event。v4 直接从 `processed_graph` 重建 event；只有在 layer 已确定之后，才合并同层语义完全相同的重复事件，并保留所有 producer event ID。

### 2.3 topology、SCC 与 layer

只有有明确方向证据的 relation subtype 和 reaction substrate→product 方向进入 backbone。association、dissociation、state change、修饰、compound mediator 等信息作为 context 保留，但不凭空制造先后边。可逆 reaction 在拓扑上可双向连通，目标 JSON 中仍保留为一个 reaction event。

管线先压缩 SCC，再在 condensation DAG 中按到 sink 的最长拓扑距离建立 sink-rooted view。layer 表示上游到下游的序数位置，不表示秒、分钟或真实等间隔时间；同层事件也没有被观测到的内部生物时间顺序。

canonical record 同时保留 `layer_index`、每个独立 event 和 producer provenance，因此后续可以训练两层动力学：event/substep 是层内快事件，graph layer 是层间慢推进。数据本身不会强迫当前 HNN 立即采用某一种粒度。

## 3. 从 canonical index 到正式数据发布

### 3.1 身份与存储

身份层次固定为：

| 层 | 身份 | 含义 |
| --- | --- | --- |
| graph | `graph_id` | source path 与内容 hash 绑定的一个结构图 |
| view | `view_id` | 一个 sink-rooted trajectory |
| record | `record_id` | 一个等权的完整 biological trajectory |
| base sample | `record_id:prefix=<n>` | 一个 prefix→continuation 问题 |
| profile sample | base sample + profile | P0/P1/P2 的一种 prompt 条件 |

完整 canonical record 压缩保存在 `data/pathway_v4_canonical_index/canonical_index_v4.sqlite3`；正式训练兼容文件保存在 `data/pathway_v4_full`。JSONL 一行一个完整 record，CSV 一行一个已选 prefix 问题。推理结果必须原样保留 source、graph、view、record 和 sample identity，不能再次退回局部 `entry_id=0/1/...`。

### 3.2 数据内生的 70/20/10 与跨来源测试

划分只依赖这次数据快照，不依赖会变化的外部 organism 清单。

1. 先按每个来源代码的 record、graph、family、layer 和 event 覆盖规模做十个分位箱；每箱固定 seed 留出约 10% 来源代码。`hsa` 固定留在 seen，`ko/ec` 作为物种中立参考来源也不进入 unseen-source 测试。
2. 在剩余 seen 来源上，以完整五位 KEGG family 为不可拆单位，用全局及各来源 record 权重优化 train/validation/test = 70%/20%/10%。比例是优化目标；当一个大 family 使精确比例不可能时，审计记录实际最优偏差，绝不拆 family 伪造精确比例。
3. 发布五个 split：
   - `train`：seen 来源、train family；
   - `validation`：seen 来源、validation family；
   - `test`：seen 来源、test family；
   - `test_organism`：held-out 来源、train family；
   - `test_strict`：held-out 来源、任意非 train family；只在 held-out 来源出现的 family 也保留在这里。

任意两个 split 的 source graph、graph、view、record 和 sample identity 必须零重叠。三个主 split 的 family 必须互斥。`test_organism` 有意与 train 共享 family，`test_strict` 同时留出来源和 family。这个设计准确测量“数据覆盖分层的未见来源代码”，不声称系统发育分类均衡，也不声称不同 family 之间不存在同源子图；后者需要另建图相似度 cluster 对照。

### 3.3 最大数据量与 token 预算

全量 index 永久保留所有可用 record；正式训练物化才受计算预算约束。当前一次 SFT 目标是四张 A100 上约 2–3 天，因此默认上限为一轮 515,000,000 个完整输入 token。候选顺序先保证每个训练来源、再优先全部 human record、再保证每张 graph、最后按固定 hash 补满，不设 family cap，也不再设 18,000-record 人工上限。

每个 record 先计算 long/middle/short 三种可用 prefix；完整 chat prompt + 闭合 answer + 结束标记超过 8192 token 的候选在写盘前排除，绝不截断 JSON。全局流算法在实际可用 horizon 约束下证明最优平衡；正式 CSV 每个 record 只出现一次。validation/test 默认各最多 20,000 record。第一轮推荐 1 epoch，探索上限 5 epoch；首轮实测吞吐必须写回实验记录。

### 3.4 phenotype

核心 v4 SFT/AE/HNN 不使用 phenotype。`phenotype_status=not_annotated` 只表示未标注，绝不表示阴性。旧的 graph-level phenotype 聚合可能把一个支路的标签复制到同图其他 sink view；在建立 phenotype event→sink/view 映射前，不允许把它重新放入核心目标。

### 3.5 不可手改审计

构建结束必须生成只读 0444 的 `data_audit.json`。至少固定：

- 全量 graph inventory、状态、隔离原因和 canonical index hash；
- 每个 split 的 row、record、source、family、organism/source-code 数；
- 所有 identity/family/organism overlap；
- duplicate record/sample/graph ID；
- phenotype policy、parser source、substep/alias/producer coverage；
- layer-length、horizon、token 长度与超长排除量；
- processed counterpart 和 source graph hash coverage；
- CSV、JSONL、manifest、split assignment、模板与 canonical DB 的 SHA-256。

任何必需检查失败时 audit 为 `failed`，训练调度器拒绝启动。

## 4. Prompt 与目标 JSON

### 4.1 模型看到什么

主条件 P0 在实际应用已知的情况下显示 `Organism/source context (KEGG code)`，并显示 observed upstream structured layers。prompt 直接给出完整可解析的目标 JSON 格式，减少 SFT 仅为学习括号和 key 所消耗的容量。它不显示 pathway 名称、类别、ID、title、block 或 phenotype。

同时发布两个严格对照：

- P1：不显示显式来源代码，但保留原生 ID；若 ID 是 `hsa:...`，它仍泄露来源，所以只能叫“无显式名称对照”；
- P2：只接受天然为 KO、compound、glycan、reaction、EC 等物种中立 ID 的完整样本，不允许靠删除前缀伪造中立映射；实体 `name` 和事件 `text` 同时改写为只含中立 ID 的确定性表述，才称为真正无物种条件。

### 4.2 闭合 v4 answer

模型只生成 downstream continuation：

```json
{
  "schema_version": "pathway_continuation_v4",
  "remaining_layers": [
    {
      "layer_index": 2,
      "events": [
        {
          "event_type": "relation",
          "source": [
            {
              "canonical_id": "hsa:207",
              "aliases": ["ncbigene:207"],
              "name": "AKT1"
            }
          ],
          "action": {
            "kind": "relation",
            "relation_class": "PPrel",
            "subtypes": ["activation", "phosphorylation"],
            "reversibility": null
          },
          "mediators": [],
          "target": [
            {
              "canonical_id": "hsa:572",
              "aliases": [],
              "name": "BAD"
            }
          ],
          "text": "AKT1 phosphorylates and activates BAD."
        }
      ]
    }
  ]
}
```

answer 不生成内部 event ID、pathway metadata 或 phenotype；这些只留在 record/CSV provenance。训练样本 JSON 不闭合就不物化。推理最多三次：第一次正常生成，失败后按同一 schema 修复重生；第三次仍不闭合或不通过严格 schema 时记录错误并退出，不能把坏 JSON 当结果。

### 4.3 如何理解 JSON 化

JSON 不会自动提高生物推理，也不会天然损害 LLM 的推理能力；它把边界、层、参与者和 action 变成可审计监督，避免 paragraph 中的分支与重复事件静默丢失。代价是模型要学习严格语法且答案 token 稍多，因此 prompt 直接展示 schema、训练只保留闭合目标，并同时评估事件正确性与 JSON 有效率。真正的生物学上限仍由 KEGG graph 的方向、覆盖与缺失决定，而不是由 JSON 外壳决定。

## 5. 训练管线

### 5.1 第一阶段 SFT

基础模型是 Qwen3-8B，通过 LoRA 学习“observed graph-layer prefix -> remaining structured continuation”。当前主设置为 prompt+answer 最大 8192 token、每 GPU batch size 1，四卡通过 DDP 并行。`8192` 是文本 token 预算，不是 HNN 步数。

短句本身不会使 HNN 无定义：只要至少有一个完整 event span，它就是有效目标；短目标只是估计方差可能更大。没有完整 semantic layer 的 row 只参与 SFT CE，不参与动力学损失。

### 5.2 共享重建 AE

AE 将 4096 维 Qwen contextual hidden state 压缩到 128 维，再重建回 4096 维。它在 answer states 以及 final prompt anchor 上训练，然后在第二阶段冻结。同一 seed 下的 stage-2 SFT 对照、HNN 和 FDHNN 必须共用完全相同的 SFT/AE checkpoint 与内容 hash。

### 5.3 不再任意将 128 维切成 q/p

任意把前 64 维叫 `q`、后 64 维叫 `p`，相当于把一箱未标签的工具直接分两堆，再宣称一堆是“位置”、一堆是“动量”。AE 没有提供这种语义，所以这个命名没有科学依据。

当前方法从 canonical Poisson matrix `J0` 出发，用 Householder reflections 学习正交基变换 `Q`：

```text
J = Q^T J0 Q
```

这相当于先让模型在 128 维空间里“旋转坐标系”，再寻找适合的成对共轭方向，而不是把 AE 输出的原始轴强行分半。构造严格满足 `J^T=-J`、`J^2=-I`，从而保留 Hamiltonian 结构；但这仍不证明某一个 latent 轴有可直译的生化学含义。

### 5.4 HNN 与 forced-damped HNN

当前可执行的两个向量场是：

```text
HNN:       dz/dt = J ∇H(z)
FDHNN:     dz/dt = (J - rI) ∇H(z) + F(t),  r = softplus(raw_r) >= 0
```

因此 `(J-R)∇H+F(t)` 在当前受控实现中具体为 `R=rI`：一个非负、各向同性的阻尼。这样不会再通过每轴不同阻尼偷偷引入未证明的 latent 坐标语义。`F(t)`只依赖序数时间，零初始化并单独正则化；它不是 knockout/control input `u`。

保守项满足：

```text
∇H^T J ∇H = 0
```

强迫/阻尼项下：

```text
dH/dt = -r ||∇H||^2 + ∇H^T F(t)
```

所以有 forcing 时总能量不保证单调下降。学得的 `H` 是 latent structural potential，不是已验证的生化自由能，也不能单独当作因果方向分数。

注意！这不是宣称精确复现某一篇物理系统论文，而是一个受控的方法组合：用 HNN 的反对称保守骨架、耗散 Hamiltonian 的正半定阻尼约束以及显式时间强迫项，在同一数据和对照中测试。每个结构都有数学来源，但它们在 pathway-language latent 中是新的实验假设，必须靠消融而不是靠名称证明。

### 5.5 semantic-layer trajectory 与损失

当前 Framework A 一次只沿一个完整 graph layer 前进：

```text
z0        = frozen_AE_encoder(final_prompt_hidden)
target[k] = mean(contextual hidden states of all complete event spans in layer k)
z[k+1]    = RK4(z[k], dt=1/128)
pred[k]   = frozen_AE_decoder(z[k+1])
```

同层 event 先做 pooling，不各自消耗一个伪时间步。最多使用 128 个 semantic-layer advance，超过部分只从 dynamics loss 中排除，并单独记数。`dt=1/128` 是统一的 surrogate coordinate，不是测量的生物时间。

因 decoder 非线性，velocity 对齐使用：

```text
D(z[k+1]) - D(z[k])   versus   h[k+1] - h[k]
```

不能直接解码 `z[k+1]-z[k]`。辅助目标包含 decoded velocity cosine、decoded state cosine 和 latent SmoothL1；结构、force 和 damping 另行正则化。AE 冻结但在计算图中可导，目标 layer state 停止梯度，防止预测与目标在同一步一起移动来虚假降低 loss。

## 6. 推理边界

**当前事实**：核心矩阵只使用 validation-selected LoRA 做 direct greedy generation，不在生成时加载 AE/HNN。因此准确名称是“经过动力学辅助目标训练的 LoRA 直接生成”，不是“HNN 在推理时 rollout”。

当前动力学按 graph layer 训练，因而不能每生成一个 token 就调用同一个 vector field。之后仍保留三个独立实验轴：

1. 完成一个合法 JSON layer 后前进一步的 graph-layer controller；
2. 单独训练 token-resolution dynamics 后的 token-by-token controller；
3. 前两者分别验证后的 multiscale hybrid，并与 direct greedy 做同等 decoding budget 消融实验。

直接推理的每个完成样本写入 progress JSONL，最终合并必须无重复、无缺失且保留 source identity。不同 batch size 可能改变浮点计算并进而改变贪心 token 轨迹，所以受控评测固定每进程 batch size 1，四 GPU 只做互斥数据分片并行。

## 7. 下游 Task 0--6

见 [FROZEN_TASK_SPEC_2026-07-13.md](FROZEN_TASK_SPEC_2026-07-13.md)，当前维护的严格任务位于 `downstream/new_tasks/`。每个结果都必须携带 dataset split、parser/representation version 和不变 checkpoint identity。

| Task | 问题 | 当前可报告输出 | 关键闸门 |
| --- | --- | --- | --- |
| 0 AE/HNN self-consistency | AE 是否保真，ODE 是否跟随 held-out latent trajectory | reconstruction MSE/cosine、固定 horizon rollout error | 必须用与 Framework A 一致的 layer representation/dt |
| 1 substep CSP | 下一层 event 集与剩余层序列是否正确 | layer-set event precision/recall/F1 等 | v4 直接使用完整 action 的 structured event；v2/v3 parser fallback 报告覆盖；有独立因果顺序 provenance 才评 ordered substep |
| 2 PCTE | prediction/gold 在同一固定表示中的轨迹差异 | DTW PCTE | 不能把 PCTE 写成 HNN self-consistency |
| 3 causal reranking | 模型能否将真路径排在反向/洗牌/无关候选前 | LLM Top-1、MRR、rejection | candidate 需专家验证与 provenance；HNN 只能经 validation 校准后组合 |
| 4 knockout/rescue | phenotype 预测与真实 KO/rescue 是否一致 | Brier/accuracy、KO direction、rescue Hit@1/MRR | 需真实干预数据与校准 scorer；缺失 label 不计为阴性 |
| 5 cell transfer | cell-adapted checkpoint 能否预测 held-out 扰动响应 | expression/delta correlation 及 controlled-ablation difference | 需对齐 gene/cell/perturbation、normalization 和 control manifest |
| 6 BioMaze | 冻结 checkpoint 在独立 mechanistic QA 上表现如何 | option accuracy 与 validity | 需官方版本、license、split 和 contamination audit |

## 8. 哪些结论现在可以说

当前 v2/v3 文本层数据可用于复现旧基线。v4 全量索引器、训练 fail-closed 检查、三次 JSON 推理闭环和 scheduler audit gate 已实现；只有 CFFF 全量 `data_audit.json` 通过后，v4 才能成为新的正式训练输入。当前不得宣称：

- parser 拆出的每句话就是完整、可追溯的 canonical biological event；
- 同层句子顺序是生物时间顺序；
- `dt=1/128` 具有实验时间单位；
- phenotype 与 sink/block 一一对应；
- 五位 KEGG family 不重叠就等于没有同源图泄漏；
- latent Hamiltonian 就是生化自由能；
- 直接 LoRA 生成就是 inference-time HNN rollout。

要支持 event-level 生物学动力学主张，必须先让 v4 全量质量闸门通过，再实现并验证两层 event/layer 动力学；若要声称同源结构严格隔离，还必须另建 graph-similarity cluster 对照。
