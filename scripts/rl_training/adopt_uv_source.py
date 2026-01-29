#!/usr/bin/env python3
import os
from pathlib import Path

# Get the directory where the script is located
DIR_PATH = Path(__file__).parent.resolve()
print(DIR_PATH)

# Define target files
target_files = [
    DIR_PATH / "skrl_train" / "pyproject.toml",
]

# Process each target file
for target_file in target_files:
    if target_file.exists():
        print(f"Updating {target_file}")
        # Read file content
        content = target_file.read_text(encoding="utf-8")
        # Replace the path (use forward slashes for consistency)
        new_content = content.replace("/abs_path/to/rl-training", DIR_PATH.as_posix())
        # Write back to file
        target_file.write_text(new_content, encoding="utf-8")
    else:
        print(f"File {target_file} does not exist. Skipping.")

print("Done.")
