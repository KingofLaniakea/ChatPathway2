#!/usr/bin/env bash
set -euo pipefail

# Formal CFFF pathway-v4 data build.  The expensive canonical index is
# resumable and never sampled; release materialization is a separate step.
BASE="${CHATPATHWAY_ASSET_ROOT:-/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui}"
REPO="${CHATPATHWAY_REPO:-$BASE/ChatPathway2}"
GRAPH_ROOT="${CHATPATHWAY_GRAPH_ROOT:-$BASE/KEGG_all_new/processed_graph}"
PROCESSED_ROOT="${CHATPATHWAY_PROCESSED_ROOT:-$BASE/KEGG_all_new/processed}"
TOKENIZER="${CHATPATHWAY_TOKENIZER:-$BASE/models/qwen3_8B}"
INDEX_DIR="${CHATPATHWAY_V4_INDEX_DIR:-$BASE/data/pathway_v4_canonical_index}"
RELEASE_DIR="${CHATPATHWAY_V4_RELEASE_DIR:-$BASE/data/pathway_v4_full}"
RUN_DIR="${CHATPATHWAY_V4_RUN_DIR:-$BASE/runs/data/pathway_v4}"
PYTHON="${CHATPATHWAY_V4_PYTHON:-$REPO/.venv/bin/python}"
INDEX_WORKERS="${CHATPATHWAY_V4_INDEX_WORKERS:-64}"
INDEX_BATCH_SIZE="${CHATPATHWAY_V4_INDEX_BATCH_SIZE:-8}"
TOKEN_WORKERS="${CHATPATHWAY_V4_TOKEN_WORKERS:-32}"
TOKEN_BATCH_SIZE="${CHATPATHWAY_V4_TOKEN_BATCH_SIZE:-8}"
SEED="${CHATPATHWAY_V4_SEED:-20260715}"
TRAIN_TOKEN_BUDGET="${CHATPATHWAY_V4_TRAIN_TOKEN_BUDGET:-515000000}"
EVALUATION_RECORDS="${CHATPATHWAY_V4_EVALUATION_RECORDS:-20000}"

mkdir -p "$INDEX_DIR" "$RELEASE_DIR" "$RUN_DIR"
exec 9>"$RUN_DIR/build.lock"
if ! flock -n 9; then
  echo "A pathway-v4 build already holds $RUN_DIR/build.lock" >&2
  exit 75
fi

for required in "$REPO" "$GRAPH_ROOT" "$PROCESSED_ROOT" "$TOKENIZER"; do
  if [[ ! -e "$required" ]]; then
    echo "Required CFFF asset is missing: $required" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  echo "Configured Python is not executable: $PYTHON" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=4
ulimit -n 65536

cd "$REPO"
GRAPH_FILES=$(find "$GRAPH_ROOT" -type f -name '*.json' | wc -l)
SOURCE_DIRS=$(find "$GRAPH_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [[ "$GRAPH_FILES" -ne 1368605 || "$SOURCE_DIRS" -ne 10859 ]]; then
  echo "Pinned processed_graph inventory mismatch: files=$GRAPH_FILES dirs=$SOURCE_DIRS" >&2
  exit 2
fi

if [[ -f "$RELEASE_DIR/data_audit.json" ]]; then
  AUDIT_STATUS=$(
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "missing"))' \
      "$RELEASE_DIR/data_audit.json"
  )
  if [[ "$AUDIT_STATUS" == "passed" ]]; then
    echo "Formal pathway-v4 release already passed: $RELEASE_DIR/data_audit.json"
    exit 0
  fi
fi

echo "stage=index graph_files=$GRAPH_FILES source_dirs=$SOURCE_DIRS workers=$INDEX_WORKERS batch=$INDEX_BATCH_SIZE"
"$PYTHON" -m dataprocess.index_structured_graphs_v4 \
  --processed-graph-root "$GRAPH_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --output-dir "$INDEX_DIR" \
  --workers "$INDEX_WORKERS" \
  --batch-size "$INDEX_BATCH_SIZE" \
  --seed "$SEED" \
  --progress-every 1000

MATERIALIZE_ARGS=(
  --index-dir "$INDEX_DIR"
  --output-dir "$RELEASE_DIR"
  --tokenizer "$TOKENIZER"
  --processed-root "$PROCESSED_ROOT"
  --source-holdout-fraction 0.10
  --protected-sources hsa,ko,ec
  --train-token-budget "$TRAIN_TOKEN_BUDGET"
  --maximum-evaluation-records "$EVALUATION_RECORDS"
  --minimum-train-records 12000
  --max-length 8192
  --priority-organism hsa
  --token-workers "$TOKEN_WORKERS"
  --token-worker-batch-size "$TOKEN_BATCH_SIZE"
  --seed "$SEED"
  --progress-every 1000
)
if [[ "${CHATPATHWAY_V4_OVERWRITE_RELEASE:-0}" == "1" ]] || \
   [[ -n "$(find "$RELEASE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "stage=materialize_restart reason=partial_or_explicit_overwrite canonical_index_preserved"
  MATERIALIZE_ARGS+=(--overwrite)
fi

echo "stage=materialize token_workers=$TOKEN_WORKERS batch=$TOKEN_BATCH_SIZE train_token_budget=$TRAIN_TOKEN_BUDGET"
"$PYTHON" -m dataprocess.materialize_dataset_v4 "${MATERIALIZE_ARGS[@]}"

AUDIT_MODE=$(stat -c '%a' "$RELEASE_DIR/data_audit.json")
AUDIT_STATUS=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
    "$RELEASE_DIR/data_audit.json"
)
if [[ "$AUDIT_STATUS" != "passed" || "$AUDIT_MODE" != "444" ]]; then
  echo "Formal v4 audit gate failed: status=$AUDIT_STATUS mode=$AUDIT_MODE" >&2
  exit 1
fi
echo "stage=complete release=$RELEASE_DIR audit_status=$AUDIT_STATUS audit_mode=$AUDIT_MODE"
