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

from typing import ClassVar

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import Field
from typing_extensions import override

from ..openai_compatible import _ToolCallBuffer
from ..openai_compatible import AssistantTurnAccumulator
from ..openai_compatible import OpenAICompatibleLlm


class OllamaAssistantTurnAccumulator(AssistantTurnAccumulator):
  """Ollama-specific accumulator handling ID-based tool call tracking.

  Ollama Streaming Pattern:
  - Uses same index (often 0) for all tool calls
  - Each tool call has unique non-null ID
  - Arguments complete in single chunk (no incremental streaming)

  Solution: Assign synthetic sequential indices as new IDs arrive,
  reusing parent's _tool_call_buffers_by_index dict.
  """

  def __init__(self):
    """Initialize with ID tracking for synthetic index generation."""
    super().__init__()
    self._current_id: str | None = None
    self._next_synthetic_index: int = 0

  @override
  def _process_tool_calls(self, tool_calls: list[ChoiceDeltaToolCall]) -> None:
    """Process tool calls, tracking by ID and assigning synthetic indices.

    Args:
      tool_calls: Tool call deltas from chunk.
    """
    for tool_call_delta in tool_calls:
      # New tool call detected by ID change (Ollama always provides non-null ID)
      if tool_call_delta.id and tool_call_delta.id != self._current_id:
        # Start new buffer with next synthetic index
        if tool_call_delta.function and tool_call_delta.function.name:
          new_buffer = _ToolCallBuffer(
              id=tool_call_delta.id,
              name=tool_call_delta.function.name,
          )
          # Add arguments if present (Ollama sends complete args in single chunk)
          if tool_call_delta.function.arguments:
            new_buffer.args_fragments.append(tool_call_delta.function.arguments)

          self._tool_call_buffers_by_index[self._next_synthetic_index] = (
              new_buffer
          )
          self._current_id = tool_call_delta.id
          self._next_synthetic_index += 1


class Ollama(OpenAICompatibleLlm):
  """Ollama LLM provider using OpenAI-compatible API.

  Connects to local Ollama instances using the OpenAI-compatible
  endpoint at /v1/chat/completions.

  Attributes:
    model: The name of the Ollama model (e.g., "qwen3-coder:30b", "deepseek-r1:8b").
    base_url: The base URL of the Ollama instance (default: "http://localhost:11434/v1").
  """

  model: str = Field(default="qwen3:8b")
  base_url: str = Field(default="http://localhost:11434/v1")

  # Override accumulator class for Ollama-specific streaming behavior
  accumulator_class: ClassVar[type[AssistantTurnAccumulator]] = (
      OllamaAssistantTurnAccumulator
  )

  @override
  def _create_client(self) -> AsyncOpenAI:
    """Create AsyncOpenAI client configured for Ollama.

    Returns:
      AsyncOpenAI: Client configured to connect to Ollama.
    """
    return AsyncOpenAI(
        base_url=self.base_url,
        api_key="unused",  # Ollama doesn't require an API key
    )
