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

"""Tests for OpenAI-compatible LLM base class."""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator
from typing import Literal
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction
from openai.types.completion_usage import CompletionUsage
import pytest

from adkx.models import OpenAICompatibleLlm
from adkx.models.openai_compatible import AssistantTurnAccumulator


class MockProvider(OpenAICompatibleLlm):
  """Concrete test implementation of OpenAICompatibleLlm."""

  model: str = "test-model"
  base_url: str = "http://test:8000/v1"

  def _create_client(self) -> AsyncOpenAI:
    """Create test client."""
    return AsyncOpenAI(base_url=self.base_url, api_key="test-key")


@pytest.fixture
def test_provider() -> MockProvider:
  """Create a test provider instance."""
  return MockProvider()


@pytest.fixture
def llm_request() -> LlmRequest:
  """Create a basic LlmRequest for testing."""
  return LlmRequest(
      model="test-model",
      contents=[types.UserContent(parts=[types.Part(text="Hello")])],
  )


def create_completion(
    text: str,
    finish_reason: Literal[
        "stop", "length", "tool_calls", "content_filter", "function_call"
    ] = "stop",
) -> ChatCompletion:
  """Create a mock ChatCompletion."""
  return ChatCompletion(
      id="test-id",
      choices=[
          CompletionChoice(
              finish_reason=finish_reason,
              index=0,
              message=ChatCompletionMessage(role="assistant", content=text),
          )
      ],
      created=1234567890,
      model="test-model",
      object="chat.completion",
  )


def create_chunk(
    text: str | None = None,
    finish_reason: (
        Literal[
            "stop", "length", "tool_calls", "content_filter", "function_call"
        ]
        | None
    ) = None,
) -> ChatCompletionChunk:
  """Create a mock ChatCompletionChunk."""
  return ChatCompletionChunk(
      id="test-id",
      choices=[
          ChunkChoice(
              delta=ChoiceDelta(content=text),
              finish_reason=finish_reason,
              index=0,
          )
      ],
      created=1234567890,
      model="test-model",
      object="chat.completion.chunk",
  )


async def mock_stream_generator(
    chunks: list[ChatCompletionChunk],
) -> AsyncGenerator[ChatCompletionChunk, None]:
  """Create an async generator from a list of chunks."""
  for chunk in chunks:
    yield chunk


def setup_streaming_mock(
    mock_client: MagicMock, chunks: list[ChatCompletionChunk]
) -> None:
  """Setup mock client for streaming responses."""

  async def mock_stream(*args, **kwargs):
    return mock_stream_generator(chunks)

  mock_client.chat.completions.create = AsyncMock(side_effect=mock_stream)


async def collect_responses(
    provider: OpenAICompatibleLlm, llm_request: LlmRequest, stream: bool = False
) -> list[LlmResponse]:
  """Helper to collect all responses from generate_content_async."""
  responses = []
  async for response in provider.generate_content_async(
      llm_request, stream=stream
  ):
    responses.append(response)
  return responses


class TestOpenAICompatibleLlmAbstraction:
  """Tests for OpenAICompatibleLlm abstract base class."""

  def test_cannot_instantiate_abstract_class(self):
    """Test that OpenAICompatibleLlm cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
      OpenAICompatibleLlm(model="test")  # type: ignore[abstract]

  def test_concrete_implementation_requires_create_client(self):
    """Test that concrete implementations must implement _create_client."""

    class IncompleteProvider(OpenAICompatibleLlm):
      model: str = "test"

    with pytest.raises(TypeError, match="abstract"):
      IncompleteProvider()  # type: ignore[abstract]

  def test_custom_provider_implementation(self):
    """Test that custom providers can be created by implementing _create_client."""

    class CustomProvider(OpenAICompatibleLlm):
      model: str = "custom-model"

      def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url="http://custom:9000/v1", api_key="custom-key"
        )

    provider = CustomProvider()
    assert provider.model == "custom-model"
    # OpenAI client normalizes URLs with trailing slash
    assert str(provider.client.base_url) == "http://custom:9000/v1/"


class TestOpenAICompatibleNonStreaming:
  """Tests for non-streaming mode."""

  @pytest.mark.asyncio
  async def test_generate_content_async_yields_single_response_when_not_streaming(
      self, test_provider, llm_request
  ):
    """Test that generate_content_async yields exactly one response in non-streaming mode."""
    mock_completion = create_completion("Hi there")

    with patch.object(test_provider, "client", create=True) as mock_client:
      mock_client.chat.completions.create = AsyncMock(
          return_value=mock_completion
      )

      responses = await collect_responses(
          test_provider, llm_request, stream=False
      )

      assert len(responses) == 1
      assert responses[0].content.parts[0].text == "Hi there"
      assert responses[0].turn_complete is True

  @pytest.mark.asyncio
  async def test_generate_content_async_includes_usage_metadata_when_not_streaming(
      self, test_provider, llm_request
  ):
    """Test that generate_content_async includes usage metadata in non-streaming mode."""
    completion = create_completion("Response")
    completion.usage = CompletionUsage(
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )

    with patch.object(test_provider, "client", create=True) as mock_client:
      mock_client.chat.completions.create = AsyncMock(return_value=completion)

      responses = await collect_responses(
          test_provider, llm_request, stream=False
      )

      assert len(responses) == 1
      assert responses[0].usage_metadata.total_token_count == 30
      assert responses[0].usage_metadata.prompt_token_count == 10
      assert responses[0].usage_metadata.candidates_token_count == 20


class TestOpenAICompatibleStreaming:
  """Tests for streaming mode."""

  @pytest.mark.asyncio
  async def test_generate_content_async_yields_partial_and_final_when_streaming(
      self, test_provider, llm_request
  ):
    """Test that generate_content_async yields partial responses + final complete response when streaming."""
    chunks = [
        create_chunk("Hi"),
        create_chunk(" there", finish_reason="stop"),
    ]

    with patch.object(test_provider, "client", create=True) as mock_client:
      setup_streaming_mock(mock_client, chunks)
      responses = await collect_responses(
          test_provider, llm_request, stream=True
      )

      # Should yield 2 partial + 1 final
      assert len(responses) == 3
      assert responses[0].partial is True
      assert responses[1].partial is True
      assert responses[2].partial is False
      assert responses[2].turn_complete is True
      assert responses[2].content.parts[0].text == "Hi there"

  @pytest.mark.asyncio
  async def test_generate_content_async_merges_consecutive_text_parts_when_streaming(
      self, test_provider, llm_request
  ):
    """Test that generate_content_async merges consecutive text parts when streaming."""
    chunks = [
        create_chunk("Hello"),
        create_chunk(" world"),
        create_chunk("!", finish_reason="stop"),
    ]

    with patch.object(test_provider, "client", create=True) as mock_client:
      setup_streaming_mock(mock_client, chunks)
      responses = await collect_responses(
          test_provider, llm_request, stream=True
      )

      final_response = responses[-1]
      assert len(final_response.content.parts) == 1
      assert final_response.content.parts[0].text == "Hello world!"


class TestOpenAICompatibleFieldHandling:
  """Tests for field handling in responses."""

  @pytest.mark.asyncio
  async def test_generate_content_async_final_response_uses_last_chunk_fields(
      self, test_provider, llm_request
  ):
    """Test that generate_content_async final response uses last chunk's fields."""
    chunks = [
        create_chunk("A"),
        create_chunk("B", finish_reason="stop"),
    ]

    with patch.object(test_provider, "client", create=True) as mock_client:
      setup_streaming_mock(mock_client, chunks)
      responses = await collect_responses(
          test_provider, llm_request, stream=True
      )

      final_response = responses[-1]
      assert final_response.finish_reason == types.FinishReason.STOP
      assert final_response.usage_metadata is None  # No usage in stream mode


class TestAssistantTurnAccumulator:
  """Tests for AssistantTurnAccumulator streaming behavior."""

  def _create_chunk_with_tool_call(
      self,
      index: int,
      tool_call_id: str | None = None,
      name: str | None = None,
      arguments: str | None = None,
      finish_reason: (
          Literal[
              "stop", "length", "tool_calls", "content_filter", "function_call"
          ]
          | None
      ) = None,
  ) -> ChatCompletionChunk:
    """Create chunk with tool call delta."""
    tool_call = ChoiceDeltaToolCall(
        index=index,
        id=tool_call_id,
        function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
        type="function" if tool_call_id else None,
    )
    return ChatCompletionChunk(
        id="test-id",
        choices=[
            ChunkChoice(
                delta=ChoiceDelta(tool_calls=[tool_call]),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )

  def _process_and_collect(
      self, accumulator: AssistantTurnAccumulator, chunk: ChatCompletionChunk
  ) -> list[LlmResponse]:
    """Process chunk and collect all yielded responses."""
    return list(accumulator.process_chunk(chunk))

  def test_has_content_returns_false_when_empty(self):
    """Test that has_content returns False when accumulator is empty."""
    accumulator = AssistantTurnAccumulator()

    assert accumulator.has_content() is False

  def test_has_content_returns_true_after_text(self):
    """Test that has_content returns True after processing text."""
    accumulator = AssistantTurnAccumulator()

    self._process_and_collect(accumulator, create_chunk("Hello"))

    assert accumulator.has_content() is True

  def test_has_content_returns_true_after_tool_calls(self):
    """Test that has_content returns True after processing tool calls."""
    accumulator = AssistantTurnAccumulator()

    chunk = self._create_chunk_with_tool_call(
        index=0, tool_call_id="call_1", name="test", arguments=""
    )
    self._process_and_collect(accumulator, chunk)

    assert accumulator.has_content() is True

  def test_process_chunk_yields_partial_for_text(self):
    """Test that process_chunk yields partial response for text chunks."""
    accumulator = AssistantTurnAccumulator()

    responses = self._process_and_collect(accumulator, create_chunk("Hello"))

    assert len(responses) == 1
    response = responses[0]
    assert response.partial is True
    assert response.content.parts[0].text == "Hello"

  def test_process_chunk_buffers_tool_calls_silently(self):
    """Test that process_chunk buffers tool calls without yielding."""
    accumulator = AssistantTurnAccumulator()

    # Tool call start (id, name, type) - no yield
    start_chunk = self._create_chunk_with_tool_call(
        index=0, tool_call_id="call_1", name="get_weather", arguments=""
    )
    assert len(self._process_and_collect(accumulator, start_chunk)) == 0

    # Arguments chunk - still no yield
    args_chunk = self._create_chunk_with_tool_call(
        index=0, arguments='{"location": "SF"}'
    )
    assert len(self._process_and_collect(accumulator, args_chunk)) == 0

  def test_create_final_response_includes_tool_calls(self):
    """Test that create_final_response includes accumulated tool calls."""
    accumulator = AssistantTurnAccumulator()

    # Accumulate tool call across chunks
    start_chunk = self._create_chunk_with_tool_call(
        index=0, tool_call_id="call_1", name="get_weather", arguments=""
    )
    args_chunk = self._create_chunk_with_tool_call(
        index=0, arguments='{"location": "SF"}'
    )
    self._process_and_collect(accumulator, start_chunk)
    self._process_and_collect(accumulator, args_chunk)

    final = accumulator.create_final_response()
    tool_call = final.content.parts[0].function_call

    assert len(final.content.parts) == 1
    assert tool_call.name == "get_weather"
    assert tool_call.args == {"location": "SF"}
    assert tool_call.id == "call_1"

  def test_create_final_response_includes_multiple_tool_calls(self):
    """Test that create_final_response includes multiple tool calls by index."""
    accumulator = AssistantTurnAccumulator()

    # First tool call (index 0): get_weather
    self._process_and_collect(
        accumulator,
        self._create_chunk_with_tool_call(
            index=0, tool_call_id="call_1", name="get_weather", arguments=""
        ),
    )
    self._process_and_collect(
        accumulator,
        self._create_chunk_with_tool_call(
            index=0, arguments='{"location": "SF"}'
        ),
    )

    # Second tool call (index 1): get_time
    self._process_and_collect(
        accumulator,
        self._create_chunk_with_tool_call(
            index=1, tool_call_id="call_2", name="get_time", arguments=""
        ),
    )
    self._process_and_collect(
        accumulator,
        self._create_chunk_with_tool_call(
            index=1, arguments='{"timezone": "PST"}'
        ),
    )

    final = accumulator.create_final_response()
    parts = final.content.parts

    assert len(parts) == 2
    assert parts[0].function_call.name == "get_weather"
    assert parts[0].function_call.args == {"location": "SF"}
    assert parts[1].function_call.name == "get_time"
    assert parts[1].function_call.args == {"timezone": "PST"}

  def test_create_final_response_includes_both_text_and_tool_calls(self):
    """Test that create_final_response includes both text and tool calls."""
    accumulator = AssistantTurnAccumulator()

    # Accumulate text
    self._process_and_collect(
        accumulator, create_chunk("Let me check the weather.")
    )

    # Accumulate tool call
    start_chunk = self._create_chunk_with_tool_call(
        index=0, tool_call_id="call_1", name="get_weather", arguments=""
    )
    args_chunk = self._create_chunk_with_tool_call(
        index=0, arguments='{"location": "SF"}'
    )
    self._process_and_collect(accumulator, start_chunk)
    self._process_and_collect(accumulator, args_chunk)

    final = accumulator.create_final_response()
    parts = final.content.parts

    assert len(parts) == 2
    assert parts[0].text == "Let me check the weather."
    assert parts[1].function_call.name == "get_weather"

  def test_create_final_response_merges_consecutive_text(self):
    """Test that create_final_response merges consecutive text parts."""
    accumulator = AssistantTurnAccumulator()

    # Process multiple text chunks
    for text in ["Hello", " world", "!"]:
      self._process_and_collect(accumulator, create_chunk(text))

    final = accumulator.create_final_response()

    assert len(final.content.parts) == 1
    assert final.content.parts[0].text == "Hello world!"

  def test_create_final_response_concatenates_incremental_arguments(self):
    """Test that create_final_response concatenates incremental arguments."""
    accumulator = AssistantTurnAccumulator()

    # Arguments split across three chunks
    self._process_and_collect(
        accumulator,
        self._create_chunk_with_tool_call(
            index=0, tool_call_id="call_1", name="test", arguments='{"key"'
        ),
    )
    self._process_and_collect(
        accumulator,
        self._create_chunk_with_tool_call(index=0, arguments=': "val'),
    )
    self._process_and_collect(
        accumulator,
        self._create_chunk_with_tool_call(index=0, arguments='ue"}'),
    )

    final = accumulator.create_final_response()

    assert final.content.parts[0].function_call.args == {"key": "value"}

  def test_create_final_response_includes_finish_reason(self):
    """Test that create_final_response includes finish reason from chunks."""
    accumulator = AssistantTurnAccumulator()

    self._process_and_collect(accumulator, create_chunk("Hello"))
    self._process_and_collect(
        accumulator, create_chunk(" world", finish_reason="stop")
    )

    final = accumulator.create_final_response()

    assert final.finish_reason == types.FinishReason.STOP

  def test_create_final_response_maps_tool_calls_finish_reason(self):
    """Test that create_final_response maps tool_calls finish reason to STOP."""
    accumulator = AssistantTurnAccumulator()

    chunk = self._create_chunk_with_tool_call(
        index=0,
        tool_call_id="call_1",
        name="get_weather",
        arguments='{"location": "SF"}',
        finish_reason="tool_calls",
    )
    self._process_and_collect(accumulator, chunk)

    final = accumulator.create_final_response()

    assert final.finish_reason == types.FinishReason.STOP
