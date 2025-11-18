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

"""Gemini LLM implementation strictly following BaseLlm contract."""

from __future__ import annotations

from functools import cached_property
from itertools import groupby
import logging
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import Client
from google.genai import types
from typing_extensions import override

logger = logging.getLogger(__name__)


class Gemini(BaseLlm):
  """Gemini LLM implementation.

  A clean implementation of Gemini that strictly follows the BaseLlm contract.
  Supports both streaming and non-streaming content generation.

  Attributes:
    model: The name of the Gemini model (e.g., "gemini-2.5-flash").
  """

  model: str = "gemini-2.5-flash"

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content from the Gemini model.

    This method strictly follows the BaseLlm contract:
    - For non-streaming: yields exactly one LlmResponse
    - For streaming: yields multiple LlmResponse objects that should be merged

    Args:
      llm_request: The request to send to the LLM.
      stream: Whether to use streaming mode.

    Yields:
      LlmResponse: The model response(s).
    """
    # Apply the helper method from BaseLlm to ensure proper user content
    self._maybe_append_user_content(llm_request)

    logger.info(
        "Sending request to model: %s, stream: %s", llm_request.model, stream
    )

    if stream:
      # Streaming mode: yield multiple responses
      async for llm_response in self._generate_content_streaming(llm_request):
        yield llm_response
    else:
      # Non-streaming mode: yield exactly one response
      response = await self.api_client.aio.models.generate_content(
          model=llm_request.model or self.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      logger.info("Response received from model")
      yield LlmResponse.create(response)

  async def _generate_content_streaming(
      self, llm_request: LlmRequest
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generate content in streaming mode.

    Per BaseLlm contract: all streaming responses are treated as one turn.
    Yields each chunk as a partial response, then yields a final complete
    response with all accumulated content.

    Args:
      llm_request: The request to send to the LLM.

    Yields:
      LlmResponse: Partial responses for each chunk, then final complete response.
    """
    accumulated_responses: list[LlmResponse] = []

    stream = await self.api_client.aio.models.generate_content_stream(
        model=llm_request.model or self.model,
        contents=llm_request.contents,
        config=llm_request.config,
    )
    async for response in stream:
      partial_response = self._create_partial_response(response)
      accumulated_responses.append(partial_response)
      yield partial_response

    if accumulated_responses:
      final_response = self._create_complete_response(accumulated_responses)
      yield final_response

  def _create_partial_response(
      self, response: types.GenerateContentResponse
  ) -> LlmResponse:
    """Create a partial response (marked with partial=True)."""
    llm_response = LlmResponse.create(response)
    llm_response.partial = True
    return llm_response

  def _create_complete_response(
      self, accumulated_responses: list[LlmResponse]
  ) -> LlmResponse:
    """Create the final complete response from all streaming chunks.

    Merging strategy for generate_content API:
    - content: Merge all parts from all chunks
    - citation_metadata: Merge all citations from all chunks
    - custom_metadata: Merge dicts from early to late (later overrides)
    - grounding_metadata: Use last (most complete)
    - finish_reason: Use last (only final chunk has this)
    - usage_metadata: Use last (cumulative usage)
    - error_code/error_message: Use last (errors terminate generation)
    - Other fields: Use last chunk's values

    Args:
      accumulated_responses: All partial responses from streaming.

    Returns:
      LlmResponse: Complete response with all accumulated content.
    """
    all_parts: list[types.Part] = self._merge_response_parts(
        accumulated_responses
    )
    last_response: LlmResponse = accumulated_responses[-1]
    citation_metadata: types.CitationMetadata | None = (
        self._merge_citation_metadata(accumulated_responses)
    )
    custom_metadata: dict | None = self._merge_custom_metadata(
        accumulated_responses
    )

    return LlmResponse(
        model_version=last_response.model_version,
        content=types.ModelContent(parts=all_parts) if all_parts else None,
        grounding_metadata=last_response.grounding_metadata,
        partial=False,
        turn_complete=True,
        finish_reason=last_response.finish_reason,
        error_code=last_response.error_code,
        error_message=last_response.error_message,
        custom_metadata=custom_metadata,
        usage_metadata=last_response.usage_metadata,
        avg_logprobs=last_response.avg_logprobs,
        logprobs_result=last_response.logprobs_result,
        cache_metadata=last_response.cache_metadata,
        citation_metadata=citation_metadata,
    )

  def _merge_custom_metadata(self, responses: list[LlmResponse]) -> dict | None:
    """Merge custom_metadata dicts (later values override earlier)."""
    merged: dict = {}
    for resp in responses:
      if resp.custom_metadata:
        merged.update(resp.custom_metadata)
    return merged or None

  def _merge_citation_metadata(
      self, responses: list[LlmResponse]
  ) -> types.CitationMetadata | None:
    """Merge citations from all responses."""
    all_citations = []
    for resp in responses:
      if resp.citation_metadata and resp.citation_metadata.citations:
        all_citations.extend(resp.citation_metadata.citations)
    return (
        types.CitationMetadata(citations=all_citations)
        if all_citations
        else None
    )

  def _merge_response_parts(
      self, responses: list[LlmResponse]
  ) -> list[types.Part]:
    """Merge all content parts, combining consecutive parts of same type."""
    all_parts = self._collect_all_parts(responses)
    if not all_parts:
      return []

    return self._group_and_merge_parts(all_parts)

  def _collect_all_parts(
      self, responses: list[LlmResponse]
  ) -> list[types.Part]:
    """Collect all parts from all responses."""
    all_parts = []
    for resp in responses:
      if resp.content and resp.content.parts:
        all_parts.extend(resp.content.parts)
    return all_parts

  def _group_and_merge_parts(self, parts: list[types.Part]) -> list[types.Part]:
    """Group consecutive parts by type and merge them."""

    def get_part_type(part: types.Part) -> str:
      if part.thought:
        return "thought"
      if part.text:
        return "text"
      return "other"

    merged = []
    for part_type, group in groupby(parts, key=get_part_type):
      group_parts = list(group)

      if part_type == "text":
        merged.append(self._merge_text_parts(group_parts))
      elif part_type == "thought":
        merged.append(self._merge_thought_parts(group_parts))
      else:
        merged.extend(group_parts)

    return merged

  def _merge_text_parts(self, parts: list[types.Part]) -> types.Part:
    """Merge consecutive text parts into one."""
    merged_text = "".join(p.text for p in parts if p.text)
    return types.Part(text=merged_text)

  def _merge_thought_parts(self, parts: list[types.Part]) -> types.Part:
    """Merge consecutive thought parts into one."""
    merged_text = "".join(p.text for p in parts if p.text)
    return types.Part(thought=True, text=merged_text)

  @cached_property
  def api_client(self) -> Client:
    """Creates and returns the Google GenAI client.

    Returns:
      Client: The configured GenAI client.
    """
    return Client()
