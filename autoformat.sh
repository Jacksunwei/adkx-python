#!/bin/bash
# Copyright 2025 Wei Sun (Jack)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Autoformat ADKX codebase.
# Usage: ./autoformat.sh [path1 path2 ...]
#   If no paths are provided, formats src/, tests/, and examples/
#   Paths can be files or directories and can be repeated

if ! command -v isort &> /dev/null
then
    echo "isort not found, please install dev dependencies first: uv pip install -e '.[dev]'"
    exit
fi

if ! command -v pyink &> /dev/null
then
    echo "pyink not found, please install dev dependencies first: uv pip install -e '.[dev]'"
    exit
fi

# If no arguments provided, use default paths
if [ $# -eq 0 ]; then
    paths=("src/" "tests/" "examples/")
else
    paths=("$@")
fi

# Process each path
for path in "${paths[@]}"; do
    echo "Formatting $path..."
    echo "  → Organizing imports (isort)..."
    isort "$path"

    echo "  → Auto-formatting code (pyink)..."
    if [ -d "$path" ]; then
        # If it's a directory, find all .py files
        find -L "$path" -not -path "*/.*" -type f -name "*.py" -exec pyink --config pyproject.toml {} +
    elif [ -f "$path" ]; then
        # If it's a file, format it directly
        pyink --config pyproject.toml "$path"
    else
        echo "  ✗ Warning: $path does not exist, skipping..."
        continue
    fi

    echo "  ✓ Done"
    echo
done

echo "All done! ✨ 🍰 ✨"
