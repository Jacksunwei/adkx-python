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

"""Ollama LLM implementation using OpenAI-compatible API."""

from __future__ import annotations

from functools import cached_property
import json
import logging
import subprocess
from typing import AsyncGenerator
from typing import cast
from typing import Optional

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call_param import ChatCompletionMessageFunctionToolCallParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from typing_extensions import override

from . import _streaming_utils

logger = logging.getLogger(__name__)

# Mapping from OpenAI finish reasons to ADK FinishReason types
_FINISH_REASON_MAP: dict[str, types.FinishReason] = {
    "stop": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "tool_calls": types.FinishReason.STOP,
    "content_filter": types.FinishReason.OTHER,
    "function_call": types.FinishReason.STOP,
}


class Ollama(BaseLlm):
  """Ollama LLM implementation using OpenAI-compatible API.

  Connects to Ollama instances (local or Cloud Run) using the OpenAI-compatible
  endpoint at /v1/chat/completions.

  Note on thought/reasoning content:
    - When receiving responses: thought text (from reasoning field) is captured as
      Part(thought=True, text=...) for models like DeepSeek R1
    - When sending requests: thought parts are EXCLUDED from conversation history
      as the OpenAI Chat Completions API does not have a standard field for
      reasoning content in request messages

  Attributes:
    model: The name of the Ollama model (e.g., "deepseek-r1:8b").
    base_url: The base URL of the Ollama instance (e.g., "http://localhost:11434/v1").
    api_key: API key (not used by Ollama, defaults to "unused").
  """

  model: str = "qwen3-coder:30b"
  base_url: str = "http://localhost:11434/v1"
  api_key: str = "unused"
  use_gcp_auth: bool = False

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content from the Ollama model.

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
        "Sending request to Ollama model: %s, stream: %s",
        llm_request.model or self.model,
        stream,
    )

    # Convert ADK request to OpenAI format
    messages = self._convert_contents_to_messages(llm_request.contents)
    tools = self._convert_tools(llm_request)
    temperature = llm_request.config.temperature
    max_tokens = llm_request.config.max_output_tokens

    if stream:
      # Streaming mode: yield multiple responses
      async for llm_response in self._generate_content_streaming(
          llm_request, messages, tools, temperature, max_tokens
      ):
        yield llm_response
    else:
      # Non-streaming mode: yield exactly one response
      response: ChatCompletion = await self.client.chat.completions.create(  # type: ignore[call-overload]
          model=llm_request.model or self.model,
          messages=messages,
          tools=tools or [],
          temperature=temperature,
          max_tokens=max_tokens,
          stream=False,
      )
      logger.info("Response received from Ollama model")
      yield self._convert_completion(response)

  async def _generate_content_streaming(
      self,
      llm_request: LlmRequest,
      messages: list[ChatCompletionMessageParam],
      tools: Optional[list[ChatCompletionToolParam]],
      temperature: Optional[float],
      max_tokens: Optional[int],
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generate content in streaming mode.

    Per BaseLlm contract: all streaming responses are treated as one turn.
    Yields each chunk as a partial response, then yields a final complete
    response with all accumulated content.

    Args:
      llm_request: The request to send to the LLM.
      messages: OpenAI-format messages.
      tools: OpenAI-format tools (if any).
      temperature: Temperature parameter.
      max_tokens: Maximum output tokens.

    Yields:
      LlmResponse: Partial responses for each chunk, then final complete response.
    """
    accumulated_responses: list[LlmResponse] = []

    # TODO: Add support for stream_options={"include_usage": True} to get usage
    # metadata in streaming mode. This requires handling the final chunk which has
    # empty choices array but contains usage data.
    stream = await self.client.chat.completions.create(  # type: ignore[call-overload]
        model=llm_request.model or self.model,
        messages=messages,
        tools=tools or [],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    async for chunk in stream:
      partial_response = self._convert_streaming_chunk(chunk)
      accumulated_responses.append(partial_response)
      yield partial_response

    if accumulated_responses:
      final_response = self._create_complete_response(accumulated_responses)
      yield final_response

  # Request conversion methods (ADK -> OpenAI format)

  def _convert_contents_to_messages(
      self, contents: list[types.Content]
  ) -> list[ChatCompletionMessageParam]:
    """Convert ADK Content objects to OpenAI messages format."""
    messages: list[ChatCompletionMessageParam] = []
    for content in contents:
      messages.extend(self._convert_content_to_message(content))
    return messages

  def _convert_content_to_message(
      self, content: types.Content
  ) -> list[ChatCompletionMessageParam]:
    """Convert ADK Content to OpenAI messages.

    Returns a list that can contain multiple messages when user content has
    multiple function responses (one tool message per function response).
    """
    if content.role == "model":
      message = self._convert_model_content_to_message(content)
      return [message] if message else []
    else:
      return self._convert_user_content_to_message(content)

  def _convert_user_content_to_message(
      self, content: types.Content
  ) -> list[ChatCompletionMessageParam]:
    """Convert user Content to OpenAI user/tool messages."""
    if not content.parts:
      return []

    if any(part.function_response for part in content.parts):
      return self._convert_function_responses_to_tool_messages(content)
    else:
      message = self._convert_user_text_to_message(content)
      return [message] if message else []

  def _convert_function_responses_to_tool_messages(
      self, content: types.Content
  ) -> list[ChatCompletionMessageParam]:
    """Convert function response parts to tool messages.

    Each function response part becomes a separate tool message with matching tool_call_id.
    """
    if not content.parts:
      return []

    tool_messages: list[ChatCompletionMessageParam] = []
    for part in content.parts:
      if part.function_response:
        tool_message: ChatCompletionToolMessageParam = {
            "role": "tool",
            "tool_call_id": part.function_response.id or "call_unknown",
            "content": json.dumps(part.function_response.response or {}),
        }
        tool_messages.append(tool_message)
    return tool_messages

  def _convert_user_text_to_message(
      self, content: types.Content
  ) -> Optional[ChatCompletionMessageParam]:
    """Convert user text parts to a single user message.

    User text content always maps 1:1 to a single user message.
    """
    if not content.parts:
      return None

    text_parts = [part.text for part in content.parts if part.text]

    if not text_parts:
      return None

    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": "".join(text_parts),
    }
    return user_message

  def _convert_model_content_to_message(
      self, content: types.Content
  ) -> Optional[ChatCompletionMessageParam]:
    """Convert model Content to OpenAI assistant message.

    Model content can contain text, thought text, and function_calls.
    Thought parts are excluded (not supported in OpenAI Chat Completions API).
    Model content always maps 1:1 to a single assistant message.
    """
    if not content.parts:
      return None

    # Collect text parts (excluding thought parts)
    text_parts = [
        part.text for part in content.parts if part.text and not part.thought
    ]

    # Collect function calls and convert to tool calls
    tool_calls = [
        self._convert_function_call_to_tool_call(part.function_call)
        for part in content.parts
        if part.function_call
    ]

    # Return None if no content and no tool_calls
    if not text_parts and not tool_calls:
      return None

    # Create assistant message
    assistant_message: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
    }

    if text_parts:
      assistant_message["content"] = "".join(text_parts)

    if tool_calls:
      assistant_message["tool_calls"] = tool_calls

    return assistant_message

  def _convert_function_call_to_tool_call(
      self, function_call: types.FunctionCall
  ) -> ChatCompletionMessageFunctionToolCallParam:
    """Convert ADK FunctionCall to OpenAI tool call format.

    Args:
      function_call: The ADK FunctionCall to convert.

    Returns:
      OpenAI-format tool call dict.
    """
    return cast(
        ChatCompletionMessageFunctionToolCallParam,
        {
            "id": function_call.id or "call_unknown",
            "type": "function",
            "function": {
                "name": function_call.name,
                "arguments": json.dumps(function_call.args or {}),
            },
        },
    )

  def _convert_tools(
      self, llm_request: LlmRequest
  ) -> Optional[list[ChatCompletionToolParam]]:
    """Convert ADK tools to OpenAI tools format.

    Args:
      llm_request: The LLM request containing tools.

    Returns:
      List of OpenAI-format tool dicts, or None if no tools.
    """
    if not llm_request.config.tools:
      return None

    # Convert ADK Tool objects to OpenAI tool format
    # Each Tool can contain multiple function_declarations
    openai_tools: list[ChatCompletionToolParam] = []
    for tool in llm_request.config.tools:
      adk_tool = cast(types.Tool, tool)
      if adk_tool.function_declarations:
        for func_decl in adk_tool.function_declarations:
          if not func_decl.name:
            continue
          openai_tool: ChatCompletionToolParam = {
              "type": "function",
              "function": {
                  "name": func_decl.name,
                  "description": func_decl.description or "",
                  "parameters": func_decl.parameters_json_schema or {},
              },
          }
          openai_tools.append(openai_tool)

    return openai_tools if openai_tools else None

  # Response conversion methods (OpenAI -> ADK format)

  def _extract_parts_from_delta(self, delta: ChoiceDelta) -> list[types.Part]:
    """Extract ADK Parts from OpenAI streaming delta.

    Handles reasoning (thought), text content, and tool calls.

    Args:
      delta: OpenAI ChoiceDelta from streaming chunk.

    Returns:
      List of ADK Part objects.
    """
    parts: list[types.Part] = []

    # Handle DeepSeek R1 reasoning (non-standard field)
    reasoning = getattr(delta, "reasoning", None)
    if reasoning:
      parts.append(types.Part(thought=True, text=reasoning))

    # Handle text content
    if delta.content:
      parts.append(types.Part(text=delta.content))

    # Handle tool calls
    for tool_call in delta.tool_calls or []:
      if tool_call.function:
        args_str = tool_call.function.arguments or "{}"
        try:
          args_dict = json.loads(args_str)
        except json.JSONDecodeError:
          args_dict = {}

        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=args_dict,
                )
            )
        )

    return parts

  def _convert_streaming_chunk(self, chunk: ChatCompletionChunk) -> LlmResponse:
    """Convert streaming chunk to LlmResponse.

    Args:
      chunk: OpenAI ChatCompletionChunk.

    Returns:
      Partial LlmResponse object.
    """

    if not chunk.choices:
      return LlmResponse(
          error_code="NO_CHOICES",
          error_message="No choices returned from model",
      )

    choice = chunk.choices[0]
    parts = self._extract_parts_from_delta(choice.delta)
    content = types.ModelContent(parts=parts) if parts else None

    finish_reason = None
    if choice.finish_reason:
      finish_reason = _FINISH_REASON_MAP.get(
          choice.finish_reason, types.FinishReason.OTHER
      )

    return LlmResponse(
        model_version=chunk.model,
        content=content,
        partial=True,
        turn_complete=None,
        finish_reason=finish_reason,
        usage_metadata=None,
    )

  def _extract_parts_from_message(
      self, message: ChatCompletionMessage
  ) -> list[types.Part]:
    """Extract ADK Parts from OpenAI completion message.

    Handles reasoning (thought), text content, and tool calls.

    Args:
      message: OpenAI ChatCompletionMessage from non-streaming completion.

    Returns:
      List of ADK Part objects.
    """
    parts: list[types.Part] = []

    # Handle DeepSeek R1 reasoning (non-standard field)
    if reasoning := getattr(message, "reasoning", None):
      parts.append(types.Part(thought=True, text=reasoning))

    # Handle text content
    if message.content:
      parts.append(types.Part(text=message.content))

    # Handle tool calls
    for tool_call in message.tool_calls or []:
      if isinstance(tool_call, ChatCompletionMessageToolCall):
        args_str = tool_call.function.arguments
        try:
          args_dict = json.loads(args_str)
        except json.JSONDecodeError:
          args_dict = {}

        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=args_dict,
                )
            )
        )

    return parts

  def _convert_completion(self, completion: ChatCompletion) -> LlmResponse:
    """Convert non-streaming completion to LlmResponse.

    Args:
      completion: OpenAI ChatCompletion.

    Returns:
      Complete LlmResponse object.
    """

    if not completion.choices:
      return LlmResponse(
          error_code="NO_CHOICES",
          error_message="No choices returned from model",
      )

    choice = completion.choices[0]
    parts = self._extract_parts_from_message(choice.message)
    content = types.ModelContent(parts=parts) if parts else None

    finish_reason = None
    if choice.finish_reason:
      finish_reason = _FINISH_REASON_MAP.get(
          choice.finish_reason, types.FinishReason.OTHER
      )

    usage_metadata = None
    if completion.usage:
      usage = completion.usage

      # Extract reasoning tokens if available (for reasoning models like o1/o3/DeepSeek R1)
      thoughts_tokens = None
      if usage.completion_tokens_details:
        thoughts_tokens = usage.completion_tokens_details.reasoning_tokens

      # Extract cached tokens if available
      cached_tokens = None
      if usage.prompt_tokens_details:
        cached_tokens = usage.prompt_tokens_details.cached_tokens

      usage_metadata = types.GenerateContentResponseUsageMetadata(
          prompt_token_count=usage.prompt_tokens,
          candidates_token_count=usage.completion_tokens,
          total_token_count=usage.total_tokens,
          thoughts_token_count=thoughts_tokens,
          cached_content_token_count=cached_tokens,
      )

    return LlmResponse(
        model_version=completion.model,
        content=content,
        turn_complete=True,
        finish_reason=finish_reason,
        usage_metadata=usage_metadata,
    )

  def _create_complete_response(
      self, accumulated_responses: list[LlmResponse]
  ) -> LlmResponse:
    """Create the final complete response from all streaming chunks.

    Merging strategy:
    - content: Merge all parts from all chunks
    - finish_reason: Use last (only final chunk has this)
    - usage_metadata: Use last (cumulative usage)
    - Other fields: Use last chunk's values

    Args:
      accumulated_responses: All partial responses from streaming.

    Returns:
      LlmResponse: Complete response with all accumulated content.
    """
    all_parts: list[types.Part] = _streaming_utils.merge_response_parts(
        accumulated_responses
    )
    last_response: LlmResponse = accumulated_responses[-1]

    return LlmResponse(
        model_version=last_response.model_version,
        content=types.ModelContent(parts=all_parts) if all_parts else None,
        partial=False,
        turn_complete=True,
        finish_reason=last_response.finish_reason,
        usage_metadata=last_response.usage_metadata,
    )

  # Utility methods

  def _get_gcp_auth_token(self) -> str:
    """Get GCP authentication token for Cloud Run.

    Returns:
      str: The authentication token.

    Raises:
      RuntimeError: If unable to fetch authentication token.
    """
    # TODO: This method is for local development with user credentials (gcloud auth login).
    # For production deployments, use service account keys or GCP metadata server instead.
    # Note: Cannot use google-auth library to generate ID tokens with user credentials.
    # For Cloud Run, we need an ID token not an access token
    # Use gcloud to get the ID token (works with user credentials)
    try:
      result = subprocess.run(
          ["gcloud", "auth", "print-identity-token"],
          capture_output=True,
          text=True,
          check=True,
      )
      token = result.stdout.strip()
      if not token:
        raise RuntimeError("Empty token returned from gcloud")
      return token
    except subprocess.CalledProcessError as e:
      raise RuntimeError(
          f"Failed to fetch GCP authentication token: {e.stderr}"
      ) from e
    except FileNotFoundError as e:
      raise RuntimeError(
          "gcloud command not found. Please install Google Cloud SDK."
      ) from e

  # Properties

  @cached_property
  def client(self) -> AsyncOpenAI:
    """Creates and returns the AsyncOpenAI client.

    Returns:
      AsyncOpenAI: The configured OpenAI client pointing to Ollama.
    """
    api_key = self.api_key
    default_headers = None

    if self.use_gcp_auth:
      api_key = self._get_gcp_auth_token()
      # For Cloud Run, we need to explicitly set the Authorization header
      default_headers = {"Authorization": f"Bearer {api_key}"}

    return AsyncOpenAI(
        base_url=self.base_url,
        api_key=api_key,
        default_headers=default_headers,
    )
