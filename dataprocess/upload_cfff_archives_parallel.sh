#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  upload_cfff_archives_parallel.sh \
    ARCHIVE_DIR SSH_TARGET REMOTE_DIR [JOBS] [PORT] [CONTROL_PATH]

Example:
  upload_cfff_archives_parallel.sh \
    /private/tmp/chatpathway_drive_to_cfff_stage \
    lihaorui@10.193.2.99 \
    /cpfs01/projects-HDD/.../KEGG_all_new_processed_graph_archives_20260713 \
    4 30456 /private/tmp/chatpathway_cfff_20260713.sock

The script uploads every artifact listed in SHA256SUMS. Each worker:

  1. skips an already complete remote artifact whose SHA-256 matches;
  2. resumes .incoming.<name> with SFTP reput;
  3. verifies the completed temporary artifact remotely;
  4. atomically renames it to the final name only after verification.

All workers may share an existing SSH ControlMaster. No password, token, or
private-key material is stored by this script.
EOF
}

if [[ $# -lt 3 || $# -gt 6 ]]; then
  usage >&2
  exit 2
fi

archive_dir=$1
ssh_target=$2
remote_dir=${3%/}
jobs=${4:-4}
port=${5:-22}
control_path=${6:--}

if [[ ! -d "$archive_dir" || ! -s "$archive_dir/SHA256SUMS" ]]; then
  echo "Archive directory or SHA256SUMS is missing: $archive_dir" >&2
  exit 2
fi
if ! [[ "$jobs" =~ ^[1-9][0-9]*$ && "$port" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS and PORT must be positive integers." >&2
  exit 2
fi
for command_name in ssh sftp sha256sum sed; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 2
  fi
done
if [[ "$control_path" != "-" && ! -S "$control_path" ]]; then
  echo "SSH ControlPath is not a socket: $control_path" >&2
  exit 2
fi

log_dir="$archive_dir/cfff_parallel_upload_logs"
mkdir -p "$log_dir"
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_log="$log_dir/run_${run_id}.log"

ssh_options=(-p "$port" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
sftp_options=(-P "$port" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
if [[ "$control_path" != "-" ]]; then
  ssh_options+=(-S "$control_path")
  sftp_options+=(-o "ControlPath=$control_path")
fi

remote_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

sftp_quote() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '"%s"' "$value"
}

remote_hash() {
  local path=$1
  local quoted_path
  quoted_path=$(remote_quote "$path")
  ssh "${ssh_options[@]}" "$ssh_target" \
    "if test -f $quoted_path; then sha256sum $quoted_path | cut -d ' ' -f 1; fi"
}

remote_size() {
  local path=$1
  local quoted_path
  quoted_path=$(remote_quote "$path")
  ssh "${ssh_options[@]}" "$ssh_target" \
    "if test -f $quoted_path; then stat -c %s $quoted_path; else printf '%s\\n' missing; fi"
}

upload_one() {
  local name=$1
  local expected=$2
  local source_path="$archive_dir/$name"
  local final_path="$remote_dir/$name"
  local incoming_path="$remote_dir/.incoming.$name"
  local item_log="$log_dir/${run_id}.${name}.log"
  local actual
  local batch_file
  local status
  local local_bytes
  local incoming_bytes
  local transfer_command

  local_bytes=$(wc -c <"$source_path" | tr -d ' ')

  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "name=$name"
    echo "bytes=$local_bytes"
    echo "expected_sha256=$expected"
  } >"$item_log"

  actual=$(remote_hash "$final_path" | tr -d '\r\n')
  if [[ "$actual" == "$expected" ]]; then
    echo "status=skipped_remote_verified" >>"$item_log"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SKIP $name sha256=$expected" | tee -a "$run_log"
    return 0
  fi

  incoming_bytes=$(remote_size "$incoming_path" | tr -d '\r\n')
  if [[ "$incoming_bytes" == "missing" ]]; then
    transfer_command=put
  elif ! [[ "$incoming_bytes" =~ ^[0-9]+$ ]]; then
    echo "status=remote_size_invalid value=$incoming_bytes" >>"$item_log"
    return 1
  elif [[ "$incoming_bytes" -gt "$local_bytes" ]]; then
    echo "status=remote_temporary_larger_than_source remote_bytes=$incoming_bytes" >>"$item_log"
    return 1
  else
    transfer_command=reput
  fi
  echo "transfer_command=$transfer_command incoming_bytes=$incoming_bytes" >>"$item_log"

  batch_file=$(mktemp "${TMPDIR:-/tmp}/chatpathway-sftp.XXXXXX")
  printf '%s %s %s\n' \
    "$transfer_command" \
    "$(sftp_quote "$source_path")" \
    "$(sftp_quote "$incoming_path")" >"$batch_file"
  set +e
  sftp "${sftp_options[@]}" -b "$batch_file" "$ssh_target" >>"$item_log" 2>&1
  status=$?
  set -e
  rm -f "$batch_file"
  if [[ "$status" -ne 0 ]]; then
    echo "status=upload_failed exit_code=$status" >>"$item_log"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAIL $name upload_exit=$status" | tee -a "$run_log" >&2
    return "$status"
  fi

  actual=$(remote_hash "$incoming_path" | tr -d '\r\n')
  if [[ "$actual" != "$expected" ]]; then
    echo "status=remote_checksum_failed actual_sha256=$actual" >>"$item_log"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAIL $name checksum=$actual" | tee -a "$run_log" >&2
    return 1
  fi

  ssh "${ssh_options[@]}" "$ssh_target" \
    "mv -- $(remote_quote "$incoming_path") $(remote_quote "$final_path")"
  echo "status=complete" >>"$item_log"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$item_log"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) OK $name sha256=$expected" | tee -a "$run_log"
}

ssh "${ssh_options[@]}" "$ssh_target" "mkdir -p -- $(remote_quote "$remote_dir")"

names=()
hashes=()
while IFS= read -r checksum_line || [[ -n "$checksum_line" ]]; do
  [[ -z "$checksum_line" ]] && continue
  expected=${checksum_line%% *}
  name=${checksum_line#"$expected"}
  name=${name# }
  name=${name# }
  name=${name#\*}
  if [[ ! "$expected" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "Invalid SHA256SUMS entry: $checksum_line" >&2
    exit 2
  fi
  case "$name" in
    ""|*/*|*".."*|*$'\n'*|*$'\r'*)
      echo "Unsafe or nested SHA256SUMS name: $name" >&2
      exit 2
      ;;
  esac
  if [[ ! -s "$archive_dir/$name" ]]; then
    echo "Listed source artifact is missing or empty: $archive_dir/$name" >&2
    exit 2
  fi
  names+=("$name")
  hashes+=("$(printf '%s' "$expected" | tr 'A-F' 'a-f')")
done <"$archive_dir/SHA256SUMS"

if [[ ${#names[@]} -eq 0 ]]; then
  echo "SHA256SUMS contains no artifacts." >&2
  exit 2
fi

{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "archive_dir=$archive_dir"
  echo "remote_dir=$remote_dir"
  echo "jobs=$jobs"
  echo "artifacts=${#names[@]}"
} >"$run_log"

fail=0
pids=()
pid_names=()
reap_one() {
  local index
  while true; do
    for index in "${!pids[@]}"; do
      if ! kill -0 "${pids[$index]}" 2>/dev/null; then
        if ! wait "${pids[$index]}"; then
          echo "worker_failed=${pid_names[$index]}" >>"$run_log"
          fail=1
        fi
        unset 'pids[index]'
        unset 'pid_names[index]'
        pids=("${pids[@]}")
        pid_names=("${pid_names[@]}")
        return
      fi
    done
    sleep 1
  done
}

for ((index=0; index<${#names[@]}; index++)); do
  upload_one "${names[$index]}" "${hashes[$index]}" &
  pids+=("$!")
  pid_names+=("${names[$index]}")
  if [[ ${#pids[@]} -ge "$jobs" ]]; then
    reap_one
  fi
done
while [[ ${#pids[@]} -gt 0 ]]; do
  reap_one
done

if [[ "$fail" -ne 0 ]]; then
  echo "status=failed" >>"$run_log"
  exit 1
fi

manifest_hash=$(sha256sum "$archive_dir/SHA256SUMS" | cut -d ' ' -f 1)
upload_one "SHA256SUMS" "$manifest_hash"
ssh "${ssh_options[@]}" "$ssh_target" \
  "cd $(remote_quote "$remote_dir") && sha256sum -c SHA256SUMS"

echo "status=complete" >>"$run_log"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_log"
