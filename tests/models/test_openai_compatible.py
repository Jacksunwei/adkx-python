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
from openai.types.completion_usage import CompletionUsage
import pytest

from adkx.models import OpenAICompatibleLlm

# ============================================================================
# Test Concrete Implementation
# ============================================================================


class MockProvider(OpenAICompatibleLlm):
  """Concrete test implementation of OpenAICompatibleLlm."""

  model: str = "test-model"
  base_url: str = "http://test:8000/v1"

  def _create_client(self) -> AsyncOpenAI:
    """Create test client."""
    return AsyncOpenAI(base_url=self.base_url, api_key="test-key")


# ============================================================================
# Test Fixtures
# ============================================================================


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


# ============================================================================
# Test Helpers
# ============================================================================


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


# ============================================================================
# Base Class Tests
# ============================================================================


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


# ============================================================================
# Non-Streaming Tests
# ============================================================================


class TestOpenAICompatibleNonStreaming:
  """Tests for non-streaming mode."""

  @pytest.mark.asyncio
  async def test_yields_single_response(self, test_provider, llm_request):
    """Test that non-streaming mode yields exactly one response."""
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
  async def test_includes_usage_metadata(self, test_provider, llm_request):
    """Test that usage metadata is included."""
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


# ============================================================================
# Streaming Tests
# ============================================================================


class TestOpenAICompatibleStreaming:
  """Tests for streaming mode."""

  @pytest.mark.asyncio
  async def test_yields_partial_and_final(self, test_provider, llm_request):
    """Test that streaming yields partial responses + final complete response."""
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
  async def test_merges_consecutive_text_parts(
      self, test_provider, llm_request
  ):
    """Test that consecutive text parts are merged."""
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


# ============================================================================
# Field Handling Tests
# ============================================================================


class TestOpenAICompatibleFieldHandling:
  """Tests for field handling in responses."""

  @pytest.mark.asyncio
  async def test_final_response_uses_last_chunk_fields(
      self, test_provider, llm_request
  ):
    """Test that final response uses last chunk's fields."""
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
