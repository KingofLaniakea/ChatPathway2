#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: upload_kegg_archives.sh ARCHIVE_DIR RCLONE_DESTINATION [RCLONE_BINARY]" >&2
  exit 2
fi

archive_dir=$1
destination=$2
rclone_binary=${3:-rclone}

if [[ ! -d "$archive_dir" || ! -s "$archive_dir/SHA256SUMS" ]]; then
  echo "Archive directory or SHA256SUMS is missing: $archive_dir" >&2
  exit 2
fi
if [[ ! -x "$rclone_binary" ]] && ! command -v "$rclone_binary" >/dev/null 2>&1; then
  echo "Working rclone binary not found: $rclone_binary" >&2
  exit 2
fi
if find "$archive_dir/tmp" -maxdepth 1 -type f -name '*.tmp' -print -quit 2>/dev/null | grep -q .; then
  echo "Temporary archive files remain; packing has not completed cleanly." >&2
  exit 2
fi

(
  cd "$archive_dir"
  sha256sum -c SHA256SUMS
)

"$rclone_binary" copy "$archive_dir" "$destination" \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 128M \
  --exclude 'lists/**' \
  --exclude 'logs/**' \
  --exclude 'tmp/**' \
  --exclude 'rclone_upload.log' \
  --exclude 'rclone_check.log' \
  --log-file "$archive_dir/rclone_upload.log" \
  --log-level INFO \
  --stats 30s \
  --stats-one-line

"$rclone_binary" check "$archive_dir" "$destination" \
  --one-way \
  --exclude 'lists/**' \
  --exclude 'logs/**' \
  --exclude 'tmp/**' \
  --exclude 'rclone_upload.log' \
  --exclude 'rclone_check.log' \
  --log-file "$archive_dir/rclone_check.log" \
  --log-level INFO
