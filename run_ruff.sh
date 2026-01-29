#!/bin/bash

# Run ruff on all Python files in scripts and src directories
echo "Running ruff on Python files in scripts and src directories..."

# Find all Python files and run ruff (excluding .venv directories)
find scripts src -name "*.py" -type f ! -path "*/.venv/*" | while read -r file; do
    echo "Processing: $file"
    source .venv/bin/activate && ruff format "$file"
    source .venv/bin/activate && ruff check "$file" --fix
done

echo "Ruff processing completed!"