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

echo '---------------------------------------'
echo '|  Organizing imports for src/...'
echo '---------------------------------------'

isort src/
echo 'All done! ✨ 🍰 ✨'

echo '---------------------------------------'
echo '|  Organizing imports for tests/...'
echo '---------------------------------------'

isort tests/
echo 'All done! ✨ 🍰 ✨'

echo '---------------------------------------'
echo '|  Auto-formatting src/...'
echo '---------------------------------------'

find -L src/ -not -path "*/.*" -type f -name "*.py" -exec pyink --config pyproject.toml {} +

echo '---------------------------------------'
echo '|  Auto-formatting tests/...'
echo '---------------------------------------'

find -L tests/ -not -path "*/.*" -type f -name "*.py" -exec pyink --config pyproject.toml {} +
