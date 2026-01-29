#!/bin/bash
DIR_PATH="$(dirname "$(realpath "$0")")"
echo "$DIR_PATH"

target_files=(
  "$DIR_PATH/skrl_train/pyproject.toml"
)

for target_file in "${target_files[@]}"; do
  if [ -f "$target_file" ]; then
    echo "Updating $target_file"
    sed -i "s|abs_path/to/rl-training|$DIR_PATH|g" "$target_file"
  else
    echo "File $target_file does not exist. Skipping."
  fi
done