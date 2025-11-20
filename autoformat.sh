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

# Autoformat ADKX codebase using pre-commit hooks.
# Usage: ./autoformat.sh [path1 path2 ...]
#   If no paths are provided, formats all files
#   Otherwise formats specified paths (files or directories)

set -e  # Exit on error

# If no arguments provided, run on all files
if [ $# -eq 0 ]; then
    echo "Formatting all files..."
    uv run pre-commit run --all-files
else
    # Collect all Python files from specified paths
    files=()
    for path in "$@"; do
        if [ -d "$path" ]; then
            # If it's a directory, find all .py files
            while IFS= read -r -d '' file; do
                files+=("$file")
            done < <(find -L "$path" -not -path "*/.*" -type f -name "*.py" -print0)
        elif [ -f "$path" ]; then
            # If it's a file, add it directly
            files+=("$path")
        else
            echo "Warning: $path does not exist, skipping..."
        fi
    done

    # Run pre-commit on collected files
    if [ ${#files[@]} -gt 0 ]; then
        echo "Formatting ${#files[@]} file(s)..."
        uv run pre-commit run --files "${files[@]}"
    else
        echo "No files found to format."
        exit 1
    fi
fi

echo
echo "All done! ✨ 🍰 ✨"
