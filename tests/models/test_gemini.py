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

"""Tests for Gemini LLM implementation."""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from adkx.models import Gemini

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def gemini() -> Gemini:
  """Create a Gemini instance for testing."""
  return Gemini(model="gemini-2.5-flash")


@pytest.fixture
def llm_request() -> LlmRequest:
  """Create a basic LlmRequest for testing."""
  return LlmRequest(
      model="gemini-2.5-flash",
      contents=[types.UserContent(parts=[types.Part(text="Hello")])],
  )


# ============================================================================
# Test Helpers
# ============================================================================


def create_mock_response(
    text: str, finish: bool = False
) -> types.GenerateContentResponse:
  """Create a mock GenerateContentResponse.

  Args:
    text: The text content of the response.
    finish: Whether this response has a finish reason.

  Returns:
    A mock GenerateContentResponse.
  """
  candidate = types.Candidate(
      content=types.ModelContent(parts=[types.Part(text=text)]),
  )
  if finish:
    candidate.finish_reason = types.FinishReason.STOP
  return types.GenerateContentResponse(candidates=[candidate])


def create_thought_response(
    text: str, finish: bool = False
) -> types.GenerateContentResponse:
  """Create a mock response with thought content."""
  candidate = types.Candidate(
      content=types.ModelContent(parts=[types.Part(thought=True, text=text)]),
  )
  if finish:
    candidate.finish_reason = types.FinishReason.STOP
  return types.GenerateContentResponse(candidates=[candidate])


async def mock_stream_generator(
    chunks: list[types.GenerateContentResponse],
) -> AsyncGenerator[types.GenerateContentResponse, None]:
  """Create an async generator from a list of chunks."""
  for chunk in chunks:
    yield chunk


def setup_streaming_mock(
    mock_client: MagicMock, chunks: list[types.GenerateContentResponse]
) -> None:
  """Setup mock client for streaming responses."""
  mock_client.aio.models.generate_content_stream = MagicMock(
      return_value=mock_stream_generator(chunks)
  )


async def collect_responses(
    gemini: Gemini, llm_request: LlmRequest, stream: bool = False
) -> list[LlmResponse]:
  """Helper to collect all responses from generate_content_async."""
  responses = []
  async for response in gemini.generate_content_async(
      llm_request, stream=stream
  ):
    responses.append(response)
  return responses


# ============================================================================
# Non-Streaming Tests
# ============================================================================


class TestGeminiNonStreaming:
  """Tests for non-streaming mode."""

  @pytest.mark.asyncio
  async def test_yields_single_response(self, gemini, llm_request):
    """Test that non-streaming mode yields exactly one response."""
    mock_response = create_mock_response("Hi there", finish=True)

    with patch.object(gemini, "api_client", create=True) as mock_client:
      mock_client.aio.models.generate_content = AsyncMock(
          return_value=mock_response
      )

      responses = await collect_responses(gemini, llm_request, stream=False)

      assert len(responses) == 1
      assert responses[0].content.parts[0].text == "Hi there"
      assert responses[0].partial is None
      assert responses[0].turn_complete is None


# ============================================================================
# Streaming Tests
# ============================================================================


class TestGeminiStreaming:
  """Tests for streaming mode."""

  @pytest.mark.asyncio
  async def test_yields_partial_and_final(self, gemini, llm_request):
    """Test that streaming yields partial responses + final complete response."""
    chunks = [
        create_mock_response("Hi"),
        create_mock_response(" there", finish=True),
    ]

    with patch.object(gemini, "api_client", create=True) as mock_client:
      setup_streaming_mock(mock_client, chunks)
      responses = await collect_responses(gemini, llm_request, stream=True)

      # Should yield 2 partial + 1 final
      assert len(responses) == 3
      assert responses[0].partial is True
      assert responses[1].partial is True
      assert responses[2].partial is False
      assert responses[2].turn_complete is True
      assert responses[2].content.parts[0].text == "Hi there"

  @pytest.mark.asyncio
  async def test_merges_consecutive_text_parts(self, gemini, llm_request):
    """Test that consecutive text parts are merged."""
    chunks = [
        create_mock_response("Hello"),
        create_mock_response(" world"),
        create_mock_response("!", finish=True),
    ]

    with patch.object(gemini, "api_client", create=True) as mock_client:
      setup_streaming_mock(mock_client, chunks)
      responses = await collect_responses(gemini, llm_request, stream=True)

      final_response = responses[-1]
      assert len(final_response.content.parts) == 1
      assert final_response.content.parts[0].text == "Hello world!"

  @pytest.mark.asyncio
  async def test_merges_thought_parts(self, gemini):
    """Test that consecutive thought parts are merged."""
    llm_request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[types.UserContent(parts=[types.Part(text="Think")])],
    )

    chunks = [
        create_thought_response("Let me"),
        create_thought_response(" think", finish=True),
    ]

    with patch.object(gemini, "api_client", create=True) as mock_client:
      setup_streaming_mock(mock_client, chunks)
      responses = await collect_responses(gemini, llm_request, stream=True)

      final_response = responses[-1]
      assert len(final_response.content.parts) == 1
      assert final_response.content.parts[0].thought is True
      assert final_response.content.parts[0].text == "Let me think"

  @pytest.mark.asyncio
  async def test_keeps_text_and_thought_separate(self, gemini, llm_request):
    """Test that text and thought parts are not merged together."""
    chunks = [
        create_thought_response("Thinking"),
        create_mock_response("Response", finish=True),
    ]

    with patch.object(gemini, "api_client", create=True) as mock_client:
      setup_streaming_mock(mock_client, chunks)
      responses = await collect_responses(gemini, llm_request, stream=True)

      final_response = responses[-1]
      assert len(final_response.content.parts) == 2
      assert final_response.content.parts[0].thought is True
      assert final_response.content.parts[0].text == "Thinking"
      assert final_response.content.parts[1].thought is None
      assert final_response.content.parts[1].text == "Response"


# ============================================================================
# Metadata Merging Tests
# ============================================================================


class TestGeminiMetadataMerging:
  """Tests for metadata merging in streaming mode."""

  @pytest.mark.asyncio
  async def test_custom_metadata_later_overrides_earlier(
      self, gemini, llm_request
  ):
    """Test that later custom_metadata values override earlier ones."""
    chunks = [
        create_mock_response("A"),
        create_mock_response("B", finish=True),
    ]

    async def mock_stream(*args, **kwargs):
      for chunk in chunks:
        yield chunk

    with patch.object(gemini, "api_client", create=True) as mock_client:
      mock_client.aio.models.generate_content_stream = mock_stream

      with patch.object(LlmResponse, "create") as mock_create:
        mock_create.side_effect = [
            LlmResponse(
                content=types.ModelContent(parts=[types.Part(text="A")]),
                custom_metadata={"key1": "value1"},
            ),
            LlmResponse(
                content=types.ModelContent(parts=[types.Part(text="B")]),
                custom_metadata={"key1": "value1_updated", "key2": "value2"},
            ),
        ]

        responses = await collect_responses(gemini, llm_request, stream=True)

        final_response = responses[-1]
        assert final_response.custom_metadata == {
            "key1": "value1_updated",
            "key2": "value2",
        }

  @pytest.mark.asyncio
  async def test_citation_metadata_accumulates(self, gemini, llm_request):
    """Test that citations from all chunks are accumulated."""
    chunks = [
        create_mock_response("A"),
        create_mock_response("B", finish=True),
    ]

    async def mock_stream(*args, **kwargs):
      for chunk in chunks:
        yield chunk

    with patch.object(gemini, "api_client", create=True) as mock_client:
      mock_client.aio.models.generate_content_stream = mock_stream

      with patch.object(LlmResponse, "create") as mock_create:
        citation1 = types.Citation(start_index=0, end_index=5, uri="http://a")
        citation2 = types.Citation(start_index=6, end_index=10, uri="http://b")

        mock_create.side_effect = [
            LlmResponse(
                content=types.ModelContent(parts=[types.Part(text="A")]),
                citation_metadata=types.CitationMetadata(citations=[citation1]),
            ),
            LlmResponse(
                content=types.ModelContent(parts=[types.Part(text="B")]),
                citation_metadata=types.CitationMetadata(citations=[citation2]),
            ),
        ]

        responses = await collect_responses(gemini, llm_request, stream=True)

        final_response = responses[-1]
        assert len(final_response.citation_metadata.citations) == 2
        assert final_response.citation_metadata.citations[0].uri == "http://a"
        assert final_response.citation_metadata.citations[1].uri == "http://b"


# ============================================================================
# Field Handling Tests
# ============================================================================


class TestGeminiFieldHandling:
  """Tests for field handling in final response."""

  @pytest.mark.asyncio
  async def test_final_response_uses_last_chunk_fields(
      self, gemini, llm_request
  ):
    """Test that final response uses last chunk's fields."""
    chunks = [
        create_mock_response("A"),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.ModelContent(parts=[types.Part(text="B")]),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=20,
                total_token_count=30,
            ),
        ),
    ]

    with patch.object(gemini, "api_client", create=True) as mock_client:
      setup_streaming_mock(mock_client, chunks)
      responses = await collect_responses(gemini, llm_request, stream=True)

      final_response = responses[-1]
      assert final_response.finish_reason == types.FinishReason.STOP
      assert final_response.usage_metadata.total_token_count == 30
