# ChatPathway 项目精神，Pipeline和Mehtod

Disclaimer： 本文件为项目核心约束，AI coding agent不可以擅自操作本文件，除非得到我的明确允许。

Disclaimer：为避免把计划写成既成事实，本文使用三种标记：

- **当前事实**：数据或代码现在确实如此；
- **目标 （v3）**：下一代数据契约，实现完并全量校验后才能使用该声明；
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

## 2. KEGG 数据到图与文本视图

### 2.1 不可变输入与 provenance

**当前事实**：服务器上已有大规模 KGML 解析产物，其中
`processed_graph/<organism>/<pathway>.json` 是结构事实层，
`processed/<organism>/<pathway>.json` 是面向文本模型的派生视图。生产数据必须同时固定：

- 源文件 inventory 和内容 hash；
- 下载日期、数据来源与使用范围；
- producer Git commit、配置 hash 和 schema version；
- 每次构建唯一的 `dataset_build_id`；
- 输出分片 hash 及自身带 hash 的 manifest。

原始快照应只读，新一次构建不覆盖旧版本。

### 2.2 canonical graph

**当前事实**：`processed_graph` 保存 pathway metadata、node、relation、reaction，以及源 KGML 真实存在时的 phenotype edge。node 具有 entry/node ID、类型、canonical ID、display name、alias 和解析状态；event 具有端点、方向、subtype 和可渲染状态。

**目标 v3** 要保证：

- `graph_id` 与源 KGML 内容绑定；
- relation、reaction、phenotype 分别有不变 `event_id`，文本不充当身份；
- raw token、canonical name、aliases 和 unresolved reason 全部保留；
- group/component 不在展开后丢失；
- 每个 event 能追回源 XML element 或稳定 source reference。

### 2.3 sink view、SCC 与 layer 的真正含义

`processed` 中的一个 `pathway N` 是一个 sink-rooted view。管线先对有环图做 SCC 压缩，再按到 sink 的拓扑距离形成 layer。因此：

- 不同 layer 提供上游到下游的**序数坐标**；
- layer 不是秒、分钟或小时，层间也没有可观测的不等时间差；
- 同一 layer 内的 event 是并行集合，序列化顺序仅用于重现；
- 同一 canonical event 可被多个 sink view 引用，它们不是多个独立实验观测。

**当前缺陷**：Step 12 在内存中已为 event 构建过带 `unit_id`、event type、source/target node IDs 和 text 的 `text_unit`，但写盘时按文本去重并拼成了每层一个 paragraph。现有 `processed` 的 layer 虽是 list，实际通常只有一个拼接字符串。因此 `sentence_parser_v1` 只能被称为句子边界恢复，不能被宣称为 canonical 生物 event。

还有两个结构风险：

1. 两条不同 canonical edge 渲染成同一句话时，按文本去重会静默丢 event；
2. 如果只用 `renderable=true` event 构建 SCC，“无法渲染英文”会被错当成“这条边不存在”，从而制造伪 sink、伪断连和错误 layer。

**目标 v3** 应先用所有 canonical structural event 构图，再把可渲染性作为文本层状态，
并写出如下的结构化引用：

```json
{
  "graph_id": "...",
  "view_id": "<graph_id>:sink=<stable_sink_signature>",
  "sink_node_ids": [31, 42],
  "layers": [
    {
      "layer_index": 0,
      "distance_to_sink": 3,
      "events": [
        {
          "event_id": "relation:17",
          "event_type": "relation",
          "source_node_ids": [2],
          "target_node_ids": [8],
          "text": "A activates B.",
          "renderable": true
        }
      ]
    }
  ]
}
```

`view_id` 应由排序后的 canonical sink node IDs 计算，不再依赖可能随 producer 变化的`pathway N`。

## 3. 从 graph/view 到模型数据集

### 3.1 四层身份

目标数据契约明确区分：

| 层 | 单位 | 身份与用途 |
| --- | --- | --- |
| graph | 一个 KGML pathway graph | `graph_id`，保存 canonical node/event |
| view | 一个 sink-rooted trajectory | 稳定 `view_id`，保存 ordered layers 和 same-layer event sets |
| record | 一个可等权采样的 biological record | `record_id = hash(graph_id, view_id)` |
| sample | record 在某一 prefix/horizon 下的监督问题 | `sample_id`，由 record、prefix、horizon 和 schema version 唯一确定 |

**当前 v2** 已用 organism、`source_json`、pathway ID 和 pathway block 的 hash 补上稳定`record_id`，并使用 `record_id:prefix=<n>` 作为 `sample_id`。旧 `entry_id=0/1/...`只是文件内 block 编号，绝不是全局身份。推理产物必须原样保留所有 identity/source字段，包括以字符串保留 `pathway_family_id` 的前导零。

### 3.2 record-centric 存储和动态 prefix

**当前事实**：旧全量 CSV 将一条长为 `k` 的 record 展开成多个 prefix row，因而让长轨迹按 prefix 数被反复加权，同时生成大量重复 prompt/answer 文本。当前派生训练池已缩减为每 record 最多 first/middle/last 三个候选 prefix，训练 Dataset 又每个 epoch 每 record确定性选一个 prefix。

**目标 v3** 的主数据应是 sharded Parquet/Arrow，并保留 JSONL 作为可读与流式交换形式。一行存一个完整 record，训练时再按固定 seed、epoch 和 sampler manifest 动态生成prefix/horizon 与 messages JSONL。CSV 只做小规模审计和兼容导出。这样才能同时做到：

- 先对 biological record 等权，再由实验明确控制 horizon；
- 长、中、短 continuation 在 epoch 间系统轮换；
- SFT、AE、HNN/FDHNN 和下游任务共用同一 canonical record；
- 不再用数千万个展开 row 混淆生物覆盖度与训练次数。

### 3.3 split 是实验声明的一部分

至少固化三种不同的 evaluation profile：

1. **strict pathway-family split**：organism 与 KEGG 五位 map family 均不跨 train/test；
2. **graph-cluster split（目标 v3）**：基于 KO set、边类型和子图相似度聚类，整个近同源component 只进入一个 split；
3. **organism transfer**：只 hold out species，允许 family overlap，并显式报告 overlap，不得冒充 unseen-pathway 分数。

五位 family disjoint 是必要而非充分条件：不同 map ID 仍可共享大量 KO 或近同构子图。每个 split manifest 必须保存阈值、seed、cluster membership、输入 inventory hash 和 overlap audit。validation 从训练候选 family/cluster 中整组留出，checkpoint 选择和早停只看 validation，test 不参与选择。

**v3 当前实现**采用第 1 种 profile：test 同时限定为预留物种和预留 family，train 排除这些物种与 family；validation 再预留另一组完整 family。graph-cluster split 仍未实现，因此 v3 的严格声明是“物种与五位 KEGG family 均不重叠”，不是“无所有同源子图泄漏”。

### 3.4 phenotype 政策

**当前事实**：当前准备表中 phenotype 实质上不可用，核心 SFT/AE/HNN 不使用它。
`not_annotated` 表示没有标注，不表示阴性 phenotype。

旧逻辑的风险是从整个 graph 汇总 phenotype，再复制给该 JSON 内所有 sink block，从而将一条支路的信号污染到另一条支路。下一版只能接受明确的 phenotype event -> sink/view映射；缺乏映射时只能明确称为 graph-level 弱监督。

**目标 v3 prompt**：核心 pathway-continuation SFT 不再要求每个样本生成
`"predicted_phenotype": null`。`phenotype_status` 仅保留为数据 metadata。未来 Task 4 用独立的 intervention/phenotype task schema，并要求真实干预、validation 校准 scorer 和 view-grounded label。

### 3.5 必须 fail closed 的数据闸门

- 每个输入 KGML 的 graph/view 引用数可对账；
- graph event 与 XML 计数相符，每个 canonical event 都有 text unit 或明确的 unrenderable 状态；
- 所有 view event reference 都能解析回 graph，端点、sink reachability 和 layer 单调性相容；
- 相同文本背后的不同 event ID 不能静默丢失；
- JSON Schema 对 100% 文件验证，不是只验几个例子；
- train/validation/test 的 source、record、sample、family/cluster overlap 全部审计；
- archive 解压后用 per-file inventory 端到端复核；
- 每个训练物化版本记录完整 layer/event 保留率、排除原因和 token 预算。

## 4. Prompt 与目标 JSON

### 4.1 早期六阶段对齐流程值得保留什么

早期 `dataprocess` 是“名称抽取 -> phenotype 匹配 -> pathway 匹配 -> trajectory 匹配 -> sentence 匹配 -> prompt 构造”的六阶段知识对齐流程，不是现有 CSV builder 的旧版。它值得借鉴的是：

- 保留实体 alias、canonical name 与 unresolved token；
- 让每个自然语言 event 能追回 source/target endpoint；
- pathway 名称、类别、物种可保留为 provenance，但核心续写 prompt 不再显式提供，避免检索捷径；
- 保存 graph traversal/evidence trace，而不是只留最终段落。

早期 `006_step3_design_prompt.py` 的 prompt 只包含领域角色、pathway 名称/类别/物种和“初始反应物”，answer 是将 layer 句子逐行拼接的平面文本；它没有 JSON schema、稳定 ID、边界 provenance 或 split contract。更重要的是，其中“cells are exposed to initial reactants”并没有 KGML实验干预依据，不应再使用这种因果措辞。正确任务是：给定 observed upstream graph events，续写 downstream graph continuation。

早期 sentence matching 的“端点 + evidence span”思路有价值，但 substring 匹配不是 edge 证据，“每层只取第一条 outgoing edge”还会丢失分支与并行 biology。目标 v3 必须由canonical event 直接产生文本与引用，不再靠 substring 和日志行号重建 truth。

### 4.2 当前 v2 prompt 和 answer

**当前事实**：prompt 已从“根据起始物预测所有反应”改成了 prefix-to-remainder 任务，包含 organism、KEGG pathway ID、block、title 和已观测 Steps，并明确说明层间有序、同层事件无序。
现行 answer 为：

```json
{
  "remaining_steps": [
    {
      "step": 2,
      "layer": "layer 2",
      "substeps": [
        {"substep": 0, "text": "AKT1 phosphorylates BAD."}
      ]
    }
  ],
  "predicted_phenotype": null
}
```

把下游 JSON 契约直接写入 prompt 确实能降低模型仅为学格式而消耗的 SFT 容量，但“请输出 JSON”不会自动解决 event provenance、长答案截断或训练/推理 schema 不一致。

### 4.3 v3 prompt、答案与物化闸门

**已实现，待 CFFF 全量物化审计**：v3 将 canonical record 与训练 CSV 分层。当前契约固定：

- dataset build/split 由 manifest 与文件固定；CSV 保存 `question_type`、record/sample identity，answer 内保存 `schema_version`；
- observed layer 和 same-layer event-set 边界；
- prompt 中的紧凑目标 JSON 骨架；当前尚未实现 constrained decoding；
- canonical record 中的完整 `event_id`、source/target node、relation/reaction ID 和 boundary provenance；
- CSV 中的 prefix/target layer 数；选择政策与逐 split token 分布由 manifest 和生成审计固定。

核心 prompt 只包含任务说明、紧凑 JSON 骨架与 observed structured events，不显式加入 pathway 名称、类别、ID、title、block、物种或 phenotype。核心 answer 只输出 downstream continuation，不包含恒定 `predicted_phenotype:null`。同一 layer 内 events 是 permutation-invariant set；若将它们排序，只能说是确定性 serialization，不能说是生物时间。

推荐的 model-visible 骨架是：

```json
{
  "schema_version": "pathway_continuation_v3",
  "remaining_layers": [
    {
      "layer_index": 2,
      "events": [
        {
          "source": [{"canonical_id": "hsa:207", "name": "AKT1"}],
          "relation": "phosphorylates",
          "target": [{"canonical_id": "hsa:572", "name": "BAD"}],
          "text": "AKT1 phosphorylates BAD."
        }
      ]
    }
  ]
}
```

canonical record JSONL 另行保存这些目标对应的 `event_id`、node ID、view ID 和 source reference，manifest 固定 record 文件与 CSV 的 hash。若 `event_id` 只是 producer 生成的局部不透明编号，就不应强迫模型凭空生成它，也不应用编号不匹配否定语义正确的事件。产品化前需固定：是让模型输出有语义的 canonical entity ID/triple，还是由评估器将预测 triple 对齐到 gold event ID；
不能混合两种契约。

SFT 中 system/user token 全部 mask，assistant answer 默认全监督。只有在训练和推理都始终使用 schema-constrained decoding 时，才可将恒定 JSON key 作为一个明确对照轴降权；step/layer/event 的值不能 mask。

物化器用真实 Qwen tokenizer 计算完整 chat prompt、完整 answer 与结束 token。总长超过 8192 的 row 在写入训练集前整条排除；训练编码器再次检查，任何漏网 row 直接报错，绝不在 JSON token 中间截断。推理第一次输出不闭合或不符合 v3 schema 时，加入明确 repair turn 并扩大生成预算；第三次仍失败就记录完整 attempt history 并报错。

默认发布目录为 `data/pathway_v3_cap256/`：family cap 为 256，每个 train record 最多物化三个 prefix，低于 12,000 个可训练 record 就拒绝发布。该门槛是为了避免再次得到只够很短训练的表；最终训练时长只能由第一轮 v3 实测 token throughput 确认。

发布时程序生成只读 `data_audit.json`，至少包含 train/validation/test 的 row、record、source JSON 与 family 数；source/record/sample/family strict overlap；organism overlap；重复 ID；phenotype/parser 状态；structured event 覆盖；layer/token 长度分布；超长排除率；graph artifact coverage。CFFF 调度器在分配 GPU 前复核 audit pass 状态、只读权限、manifest hash、三个 CSV 和三个 record JSONL hash。

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
| 1 substep CSP | 下一层 event 集与剩余层序列是否正确 | layer-set event precision/recall/F1 等 | v3 直接使用 structured event；v2 parser fallback 报告覆盖；有独立因果顺序 provenance 才评 ordered substep |
| 2 PCTE | prediction/gold 在同一固定表示中的轨迹差异 | DTW PCTE | 不能把 PCTE 写成 HNN self-consistency |
| 3 causal reranking | 模型能否将真路径排在反向/洗牌/无关候选前 | LLM Top-1、MRR、rejection | candidate 需专家验证与 provenance；HNN 只能经 validation 校准后组合 |
| 4 knockout/rescue | phenotype 预测与真实 KO/rescue 是否一致 | Brier/accuracy、KO direction、rescue Hit@1/MRR | 需真实干预数据与校准 scorer；缺失 label 不计为阴性 |
| 5 cell transfer | cell-adapted checkpoint 能否预测 held-out 扰动响应 | expression/delta correlation 及 controlled-ablation difference | 需对齐 gene/cell/perturbation、normalization 和 control manifest |
| 6 BioMaze | 冻结 checkpoint 在独立 mechanistic QA 上表现如何 | option accuracy 与 validity | 需官方版本、license、split 和 contamination audit |

## 8. 哪些结论现在可以说

当前 v2 文本层数据可用于复现旧基线。v3 builder、训练 fail-closed 检查、三次 JSON 推理闭环和 scheduler audit gate 已实现；只有 CFFF 全量 `data_audit.json` 通过后，v3 才能成为新的正式训练输入。当前不得宣称：

- parser 拆出的每句话就是完整、可追溯的 canonical biological event；
- 同层句子顺序是生物时间顺序；
- `dt=1/128` 具有实验时间单位；
- phenotype 与 sink/block 一一对应；
- 五位 KEGG family 不重叠就等于没有同源图泄漏；
- latent Hamiltonian 就是生化自由能；
- 直接 LoRA 生成就是 inference-time HNN rollout。

要支持 event-level 生物学动力学主张，必须先完成 structured view v3、完整 JSON 目标物化、graph-cluster split 与对应的全量质量闸门。
