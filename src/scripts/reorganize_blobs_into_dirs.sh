#!/usr/bin/env bash
set -euo pipefail

src_path="../data/cryoem_blobs"

# ensure destination root exists
mkdir -p "$src_path/grouped_blobs"

# enable nullglob so an empty glob expands to nothing
shopt -s nullglob

# collect files and count
files=("$src_path"/*.npz)
total=${#files[@]}
i=0

for src in "${files[@]}"; do
  file=$(basename "$src")
  # use the last underscore-separated token and strip the .npz extension for the group name
  group="${file##*_}"
  group="${group%.npz}"
  group_dir="$src_path/grouped_blobs/$group"
  mkdir -p "$group_dir"
  mv -- "$src" "$group_dir/$file"
  i=$((i+1))
  printf '\rMoved %d/%d files' "$i" "$total"
done

printf '\n'
