#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pack_kegg_directory.sh SOURCE_ROOT DATASET_DIR OUTPUT_DIR [PARTS] [JOBS] [PIGZ_THREADS]

Example:
  pack_kegg_directory.sh \
    /home/shl003/work/ChatPathway_dev/datasets/KEGG_all_new \
    processed_graph \
    /home/shl003/work/KEGG_all_new_processed_graph_archives_20260713 \
    16 8 2

The source must be SOURCE_ROOT/DATASET_DIR. Each independent archive stores
paths as DATASET_DIR/<target>/..., so no binary concatenation is needed when
restoring the shards.
EOF
}

if [[ $# -lt 3 || $# -gt 6 ]]; then
  usage >&2
  exit 2
fi

source_root=$1
dataset_dir=$2
output_dir=$3
parts=${4:-16}
jobs=${5:-8}
pigz_threads=${6:-2}
source_dir="$source_root/$dataset_dir"

if [[ ! -d "$source_dir" ]]; then
  echo "Source directory does not exist: $source_dir" >&2
  exit 2
fi
if ! [[ "$parts" =~ ^[1-9][0-9]*$ && "$jobs" =~ ^[1-9][0-9]*$ && "$pigz_threads" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARTS, JOBS, and PIGZ_THREADS must be positive integers." >&2
  exit 2
fi
if ! command -v pigz >/dev/null 2>&1; then
  echo "pigz is required." >&2
  exit 2
fi

mkdir -p "$output_dir/lists" "$output_dir/logs" "$output_dir/tmp"
run_log="$output_dir/pack.run.log"

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source_root=$source_root"
  echo "dataset_dir=$dataset_dir"
  echo "output_dir=$output_dir"
  echo "parts=$parts"
  echo "jobs=$jobs"
  echo "pigz_threads=$pigz_threads"
  df -h "$source_dir" "$output_dir" 2>/dev/null || true
} >"$run_log"

find "$source_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | LC_ALL=C sort >"$output_dir/lists/all_targets.txt"

target_count=$(wc -l <"$output_dir/lists/all_targets.txt" | tr -d ' ')
if [[ "$target_count" -eq 0 ]]; then
  echo "No target directories found under $source_dir" >&2
  exit 2
fi
echo "target_count=$target_count" >>"$run_log"

find "$output_dir/lists" -maxdepth 1 -type f -name 'part_*.txt' -delete
awk -v out="$output_dir" -v parts="$parts" -v dataset="$dataset_dir" '
  {
    part_index = (NR - 1) % parts + 1
    printf "%s/%s\n", dataset, $0 >> sprintf("%s/lists/part_%02d.txt", out, part_index)
  }
' "$output_dir/lists/all_targets.txt"

pack_one() {
  local part=$1
  local part_padded
  part_padded=$(printf '%02d' "$part")
  local list="$output_dir/lists/part_${part_padded}.txt"
  local archive="$output_dir/KEGG_all_new_${dataset_dir}.part_${part_padded}_of_${parts}.tar.gz"
  local temporary="$output_dir/tmp/$(basename "$archive").tmp"
  local log="$output_dir/logs/part_${part_padded}.log"

  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "list=$list"
    echo "archive=$archive"
    echo "target_entries=$(wc -l <"$list" | tr -d ' ')"
  } >"$log"

  if [[ -s "$archive" ]] && gzip -t "$archive" >>"$log" 2>&1; then
    echo "status=skipped_existing_valid" >>"$log"
    return 0
  fi

  rm -f "$temporary"
  if tar -C "$source_root" -I "pigz -p $pigz_threads -1" -cf "$temporary" -T "$list" >>"$log" 2>&1; then
    mv "$temporary" "$archive"
    sha256sum "$archive" >"$archive.sha256"
    gzip -t "$archive"
    echo "status=complete" >>"$log"
  else
    local status=$?
    echo "status=failed exit_code=$status" >>"$log"
    return "$status"
  fi
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$log"
}

fail=0
running=0
for part in $(seq 1 "$parts"); do
  pack_one "$part" &
  running=$((running + 1))
  if [[ "$running" -ge "$jobs" ]]; then
    wait -n || fail=1
    running=$((running - 1))
  fi
done
while [[ "$running" -gt 0 ]]; do
  wait -n || fail=1
  running=$((running - 1))
done

if [[ "$fail" -ne 0 ]]; then
  echo "status=failed" >>"$run_log"
  exit 1
fi

(
  cd "$output_dir"
  sha256sum KEGG_all_new_"${dataset_dir}".part_*_of_"${parts}".tar.gz >SHA256SUMS
  sha256sum -c SHA256SUMS
)

{
  echo "source=$source_dir"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "target_count=$target_count"
  echo "parts=$parts"
  echo "archive_format=independent_tar_gzip_shards"
  echo "restore_parent=/path/to/KEGG_all_new"
  echo "restore_command=for archive in KEGG_all_new_${dataset_dir}.part_*_of_${parts}.tar.gz; do tar -xzf \"\$archive\"; done"
  echo "archives:"
  ls -lh "$output_dir"/KEGG_all_new_"${dataset_dir}".part_*_of_"${parts}".tar.gz
} >"$output_dir/MANIFEST.txt"

echo "status=complete" >>"$run_log"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_log"
