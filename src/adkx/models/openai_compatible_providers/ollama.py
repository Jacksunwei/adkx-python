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

"""Ollama provider implementation."""

from __future__ import annotations

from openai import AsyncOpenAI

from ..openai_compatible import OpenAICompatibleLlm


class Ollama(OpenAICompatibleLlm):
  """Ollama LLM provider using OpenAI-compatible API.

  Connects to local Ollama instances using the OpenAI-compatible
  endpoint at /v1/chat/completions.

  Attributes:
    model: The name of the Ollama model (e.g., "qwen3-coder:30b", "deepseek-r1:8b").
    base_url: The base URL of the Ollama instance (default: "http://localhost:11434/v1").
  """

  model: str = "qwen3-coder:30b"
  base_url: str = "http://localhost:11434/v1"

  def _create_client(self) -> AsyncOpenAI:
    """Create AsyncOpenAI client configured for Ollama.

    Returns:
      AsyncOpenAI: Client configured to connect to Ollama.
    """
    return AsyncOpenAI(
        base_url=self.base_url,
        api_key="unused",  # Ollama doesn't require an API key
    )
