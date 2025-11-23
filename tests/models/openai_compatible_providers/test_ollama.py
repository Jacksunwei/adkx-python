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

"""Tests for Ollama provider."""

from __future__ import annotations

from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction
import pytest

from adkx.models import Ollama
from adkx.models.openai_compatible_providers.ollama import OllamaAssistantTurnAccumulator


@pytest.fixture
def ollama() -> Ollama:
  """Create an Ollama instance with default configuration."""
  return Ollama()


class TestOllamaInitialization:
  """Tests for Ollama initialization."""

  def test_model_defaults_to_qwen3_8b(self, ollama):
    """Test that model defaults to qwen3:8b."""
    assert ollama.model == "qwen3:8b"

  def test_base_url_defaults_to_localhost_11434(self, ollama):
    """Test that base_url defaults to http://localhost:11434/v1."""
    assert ollama.base_url == "http://localhost:11434/v1"

  def test_model_can_be_customized(self):
    """Test that model can be set to custom value."""
    custom_ollama = Ollama(model="llama2:7b")
    assert custom_ollama.model == "llama2:7b"

  def test_base_url_can_be_customized(self):
    """Test that base_url can be set to custom value."""
    custom_ollama = Ollama(base_url="http://custom:8080/v1")
    assert custom_ollama.base_url == "http://custom:8080/v1"

  def test_both_model_and_base_url_can_be_customized(self):
    """Test that both model and base_url can be customized simultaneously."""
    custom_ollama = Ollama(model="llama2:7b", base_url="http://custom:8080/v1")
    assert custom_ollama.model == "llama2:7b"
    assert custom_ollama.base_url == "http://custom:8080/v1"


class TestOllamaClient:
  """Tests for Ollama client property."""

  def test_client_property_uses_configured_base_url(self, ollama):
    """Test that client property creates client with configured base_url."""
    client = ollama.client
    # OpenAI client normalizes URLs with trailing slash
    assert str(client.base_url) == "http://localhost:11434/v1/"

  def test_client_property_uses_unused_api_key(self, ollama):
    """Test that client property sets api_key to 'unused' for Ollama."""
    client = ollama.client
    assert client.api_key == "unused"

  def test_client_property_uses_custom_base_url(self):
    """Test that client property respects custom base_url."""
    custom_ollama = Ollama(base_url="http://custom:8080/v1")
    client = custom_ollama.client
    assert str(client.base_url) == "http://custom:8080/v1/"


class TestOllamaAssistantTurnAccumulator:
  """Tests for OllamaAssistantTurnAccumulator ID-based tool call handling."""

  def _create_tool_call_delta(
      self,
      tool_call_id: str | None = None,
      name: str | None = None,
      arguments: str | None = None,
      index: int = 0,
  ) -> ChoiceDeltaToolCall:
    """Create a tool call delta mimicking Ollama's streaming pattern."""
    return ChoiceDeltaToolCall(
        index=index,
        id=tool_call_id,
        function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
        type="function" if tool_call_id else None,
    )

  def test_process_tool_calls_assigns_synthetic_index_for_first_tool_call(self):
    """Test that _process_tool_calls assigns synthetic index 0 for first tool call."""
    accumulator = OllamaAssistantTurnAccumulator()

    # Ollama pattern: index=0, id="call_1", name="get_weather", args complete
    tool_calls = [
        self._create_tool_call_delta(
            tool_call_id="call_1",
            name="get_weather",
            arguments='{"location": "SF"}',
            index=0,
        )
    ]

    accumulator._process_tool_calls(tool_calls)

    # Should be stored at synthetic index 0
    assert 0 in accumulator._tool_call_buffers_by_index
    buffer = accumulator._tool_call_buffers_by_index[0]
    assert buffer.id == "call_1"
    assert buffer.name == "get_weather"
    assert buffer.args_fragments == ['{"location": "SF"}']

  def test_process_tool_calls_assigns_synthetic_index_for_multiple_tool_calls(
      self,
  ):
    """Test that _process_tool_calls assigns sequential synthetic indices for multiple tool calls."""
    accumulator = OllamaAssistantTurnAccumulator()

    # First tool call: Ollama uses index=0, id="call_1"
    tool_calls_1 = [
        self._create_tool_call_delta(
            tool_call_id="call_1",
            name="get_weather",
            arguments='{"location": "SF"}',
            index=0,
        )
    ]
    accumulator._process_tool_calls(tool_calls_1)

    # Second tool call: Ollama also uses index=0, but different id="call_2"
    tool_calls_2 = [
        self._create_tool_call_delta(
            tool_call_id="call_2",
            name="get_time",
            arguments='{"timezone": "PST"}',
            index=0,
        )
    ]
    accumulator._process_tool_calls(tool_calls_2)

    # Should be stored at synthetic indices 0 and 1
    assert 0 in accumulator._tool_call_buffers_by_index
    assert 1 in accumulator._tool_call_buffers_by_index

    buffer_0 = accumulator._tool_call_buffers_by_index[0]
    assert buffer_0.id == "call_1"
    assert buffer_0.name == "get_weather"

    buffer_1 = accumulator._tool_call_buffers_by_index[1]
    assert buffer_1.id == "call_2"
    assert buffer_1.name == "get_time"

  def test_process_tool_calls_ignores_same_id_repeated(self):
    """Test that _process_tool_calls ignores repeated chunks with same ID."""
    accumulator = OllamaAssistantTurnAccumulator()

    # First chunk with call_1
    tool_calls_1 = [
        self._create_tool_call_delta(
            tool_call_id="call_1",
            name="get_weather",
            arguments='{"location": "SF"}',
            index=0,
        )
    ]
    accumulator._process_tool_calls(tool_calls_1)

    # Repeated chunk with same call_1 (should be ignored)
    tool_calls_2 = [
        self._create_tool_call_delta(
            tool_call_id="call_1",
            name="get_weather",
            arguments='{"location": "NYC"}',
            index=0,
        )
    ]
    accumulator._process_tool_calls(tool_calls_2)

    # Should only have one buffer at index 0
    assert len(accumulator._tool_call_buffers_by_index) == 1
    buffer = accumulator._tool_call_buffers_by_index[0]
    # Original arguments should remain unchanged
    assert buffer.args_fragments == ['{"location": "SF"}']

  def test_process_tool_calls_tracks_current_id(self):
    """Test that _process_tool_calls tracks _current_id correctly."""
    accumulator = OllamaAssistantTurnAccumulator()

    assert accumulator._current_id is None

    tool_calls = [
        self._create_tool_call_delta(
            tool_call_id="call_1", name="test", arguments="{}", index=0
        )
    ]
    accumulator._process_tool_calls(tool_calls)

    assert accumulator._current_id == "call_1"

  def test_process_tool_calls_increments_synthetic_index(self):
    """Test that _process_tool_calls increments _next_synthetic_index for each new tool call."""
    accumulator = OllamaAssistantTurnAccumulator()

    assert accumulator._next_synthetic_index == 0

    # Add first tool call
    accumulator._process_tool_calls([
        self._create_tool_call_delta(
            tool_call_id="call_1", name="test1", index=0
        )
    ])
    assert accumulator._next_synthetic_index == 1

    # Add second tool call
    accumulator._process_tool_calls([
        self._create_tool_call_delta(
            tool_call_id="call_2", name="test2", index=0
        )
    ])
    assert accumulator._next_synthetic_index == 2

  def test_process_tool_calls_requires_both_id_and_name_for_new_buffer(self):
    """Test that _process_tool_calls requires both ID and name to create new buffer."""
    accumulator = OllamaAssistantTurnAccumulator()

    # Only ID, no name - should not create buffer
    tool_calls_only_id = [
        self._create_tool_call_delta(tool_call_id="call_1", name=None, index=0)
    ]
    accumulator._process_tool_calls(tool_calls_only_id)
    assert len(accumulator._tool_call_buffers_by_index) == 0

    # Only name, no ID - should not create buffer (ID check fails first)
    tool_calls_only_name = [
        self._create_tool_call_delta(tool_call_id=None, name="test", index=0)
    ]
    accumulator._process_tool_calls(tool_calls_only_name)
    assert len(accumulator._tool_call_buffers_by_index) == 0

  def test_process_tool_calls_handles_complete_arguments_in_single_chunk(self):
    """Test that _process_tool_calls handles Ollama's pattern of complete args in single chunk."""
    accumulator = OllamaAssistantTurnAccumulator()

    # Ollama sends complete arguments in the first chunk
    complete_args = '{"location": "San Francisco", "units": "celsius"}'
    tool_calls = [
        self._create_tool_call_delta(
            tool_call_id="call_1",
            name="get_weather",
            arguments=complete_args,
            index=0,
        )
    ]

    accumulator._process_tool_calls(tool_calls)

    buffer = accumulator._tool_call_buffers_by_index[0]
    assert buffer.args_fragments == [complete_args]
    assert len(buffer.args_fragments) == 1
