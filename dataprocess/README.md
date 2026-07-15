# Pathway-continuation v4 数据管线

正式构建分成两个阶段：

1. `index_structured_graphs_v4.py` 并行扫描全部 `processed_graph`，生成可恢复的 canonical SQLite index；
2. `materialize_dataset_v4.py` 只用 index 内部统计完成划分、token 筛选和正式发布。

`processed_graph/<source>/<pathway>.json` 是 relation、reaction、action、实体和图层结构的事实源。`processed/<source>/<pathway>.json` 只是历史文本对应物；索引要求每个 producer event 的 `legacy_text` 都能在对应旧段落中逐字找到，但不从拼接 paragraph 反推 event。

## Canonical biological record

- 非 group entry 是一个参与者；多个 resolved ID 分为一个 `canonical_id` 与 aliases，不拆成多个虚假参与者。同一规范实体由多个 occurrence node 重复出现时，模型侧按原事件顺序稳定去重并合并 aliases，record 侧仍保留每个 node provenance。
- group entry 递归展开成员。
- relation 保存 relation class 和全部 subtype；reaction 保存 substrate/product 与可逆性。
- 只有有明确方向证据的关系和 reaction 方向参与 SCC/topology。无方向信息保留为 context，不制造时间边。
- 任意坏端点、未解析实体或互相矛盾的结构字段会隔离整张 graph。
- 先完成 SCC 和 layer 归属，再合并同层语义完全相同的 event；所有 producer event ID 和端点 provenance 仍保留。
- `legacy_text` 复现固定的历史 Step12 模板；若合并后的不同 producer 使用了不同旧显示名，record 以 `legacy_text_overrides` 保留逐 producer 差异。`text` 使用去重后的结构事实和已审计模板修正已知方向或语法问题。不会调用 LLM 生成 gold text。

身份顺序为 `graph_id -> view_id -> record_id -> base_sample_id -> profile sample_id`。JSONL 保存完整 record 与 provenance；CSV 一行是一个被选中的 prefix→continuation 问题。

## Prompt 与闭合目标

主条件 P0 显示已知的 KEGG source/organism code 和 observed upstream layers，并直接展示目标 JSON key 结构。它不显示 pathway 名称、类别、ID、title、block 或 phenotype。另发布：

- P1：去掉显式 source code，但保留 native ID，因此不是严格无物种条件；
- P2：只保留天然物种中立 ID 全覆盖的样本，不删除物种前缀伪造映射；模型可见 `name` 与 `text` 也用中立 ID 确定性重写，避免名称侧漏。

目标是完整 `pathway_continuation_v4`：

```json
{
  "schema_version": "pathway_continuation_v4",
  "remaining_layers": [
    {
      "layer_index": 2,
      "events": [
        {
          "event_type": "relation",
          "source": [{"canonical_id": "ko:K00001", "aliases": [], "name": "A"}],
          "action": {
            "kind": "relation",
            "relation_class": "PPrel",
            "subtypes": ["activation"],
            "reversibility": null
          },
          "mediators": [],
          "target": [{"canonical_id": "ko:K00002", "aliases": [], "name": "B"}],
          "text": "A activates B."
        }
      ]
    }
  ]
}
```

完整 chat prompt、answer 和结束标记超过 8192 token 的候选在写盘前排除；JSON 永不截断。推理最多三次，第三次仍不闭合或 schema 不合法就记录错误并失败。

## 数据内生划分

划分不调用外部 taxonomy。它只使用 canonical index 中每个 source code 的 record、graph、family、layer 和 event 覆盖：

1. 按覆盖规模分成最多十个分位层，每层固定 seed 留出约 10% source；`hsa`、`ko`、`ec` 固定保留；
2. 在 seen sources 上把完整五位 KEGG family 作为不可拆单位，优化 train/validation/test 的 record 比例到 70/20/10；
3. 发布 `train`、`validation`、`test`、`test_organism`（未见 source + train family）和 `test_strict`（未见 source + 任意非 train family，含只在留出来源出现的 family）。

这能严格测试“数据覆盖分层的未见 source code”，但不声称系统发育平衡，也不声称不同 family 没有相似子图。后者需要另建 graph-similarity cluster 对照。

canonical index 不抽样、不做 family cap。正式 train 发布按固定候选顺序尽量装满 515,000,000 个完整 token，默认 validation/test 各最多 20,000 record。每个 record 只发布一个由全局匹配器平衡的 long/middle/short horizon；默认第一轮 SFT 为 1 epoch。

## CFFF 构建

首选入口是带锁和断点策略的脚本：

```bash
bash dataprocess/run_cfff_dataset_v4.sh
```

默认使用 64 个 graph parser worker 和 32 个 tokenizer worker。canonical index 可以从已完成的 source 继续；若正式物化被打断，脚本保留 index，只从头重建 release，并在开始时删除旧 audit，防止调度器误用半成品。

输出目录 `data/pathway_v4_full/` 包含五组 P0 CSV/record JSONL、P1/P2 controls、`source_graph_hashes.tsv`、`split_assignments.json`、`dataset_manifest.json` 和只读 0444 `data_audit.json`。audit 固定全量 inventory、每个 split 的计数与 overlap、duplicate identities、parser/substep/layer/token 统计、processed counterpart 的逐事件文本精确匹配，以及所有正式文件/source graph hash。训练调度器会再次验证这些契约。

## 历史入口

`build_pathway_csv.py`、`prepare_experiment_data.py`、`select_training_coverage.py` 和 `build_structured_dataset.py` 只用于复现 v2/v3。它们不是 v4 正式构建器。
