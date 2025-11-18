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

"""Base tool class for adkx extensions."""

from __future__ import annotations

from typing import Any
from typing import TypedDict

from google.adk.tools import BaseTool as ADKBaseTool
from google.genai.types import Blob
from google.genai.types import FileData
from google.genai.types import FunctionResponse
from google.genai.types import Part
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import NotRequired
from typing_extensions import override


class _FunctionResponseData(TypedDict):
  """Structure for FunctionResponse.response field (internal).

  Attributes:
    status: Execution status (always present, defaults to "success").
    text_result: Concatenated string content from tool execution (optional).
    structured_result: Merged dict content from tool execution (optional).
  """

  status: str
  text_result: NotRequired[str]
  structured_result: NotRequired[dict[str, Any]]


class ToolResult(BaseModel):
  """Result returned by a tool execution.

  ToolResult encapsulates the output of a tool execution, including multi-modal
  content for the LLM, execution status, and optional runtime directives.
  Use `to_parts()` to convert to LLM-consumable Parts.

  Attributes:
    details: Multi-modal content for the LLM (optional). Supports:
      - Text strings: Concatenated in FunctionResponse as "text_result"
      - Dicts: Merged in FunctionResponse as "structured_result"
      - Blob: Images/binary data as standalone Parts for visual understanding
      - FileData: File references as standalone Parts
      If empty, only status is returned in FunctionResponse.
    status: Execution status. Defaults to "success". Always included in
      FunctionResponse. Common values: "success", "error", "pending".
    adk_context: Optional dictionary for ADK runtime directives. Not passed to
      the LLM. Each ADK component defines its own schema for keys it recognizes.

  Returns (via to_parts):
    List of Parts with FunctionResponse first, then media Parts. FunctionResponse
    contains:
      - status: Execution status (always present)
      - text_result: Concatenated string content (if any)
      - structured_result: Merged dict content (if any)

  Example:
    ```python
    # Text-only result
    result = ToolResult(details=["Temperature is 72°F"])
    # → FunctionResponse: {"status": "success", "text_result": "Temperature is 72°F"}

    # Structured data result
    result = ToolResult(details=[{"temperature": 72, "unit": "F"}])
    # → FunctionResponse: {"status": "success", "structured_result": {...}}

    # Multi-modal result with image
    result = ToolResult(details=[
        "Here's the weather map:",
        Blob(data=image_bytes, mime_type="image/png")
    ])
    # → FunctionResponse: {"status": "success", "text_result": "Here's..."}
    #   + Standalone Part with image (for LLM visual understanding)

    # Mixed content (text + structured data)
    result = ToolResult(details=[
        "Temperature is 72°F",
        {"temperature": 72, "unit": "F", "condition": "sunny"}
    ])
    # → FunctionResponse: {"status": "success", "text_result": "...", "structured_result": {...}}

    # Custom status (e.g., error)
    result = ToolResult(details=["Network timeout"], status="error")
    # → FunctionResponse: {"status": "error", "text_result": "Network timeout"}

    # With ADK runtime context (not sent to LLM)
    result = ToolResult(
        details=["Task delegated"],
        adk_context={"transfer_to_agent": "specialist_agent"}
    )
    ```
  """

  status: str = "success"
  details: list[str | dict[str, Any] | Blob | FileData] = Field(
      default_factory=list
  )
  adk_context: dict[str, Any] | None = None

  def to_parts(self, *, name: str, id: str) -> list[Part]:
    """Convert ToolResult to Content parts for LLM consumption.

    Returns a list of Parts to be added to Content.parts in the LLM request.
    The first Part contains the FunctionResponse, followed by standalone Parts
    for any Blob/FileData items.

    Args:
      name: The function/tool name to include in the response.
      id: The function call ID to include in the response.

    Returns:
      List of Parts containing FunctionResponse and standalone media.

    FunctionResponse structure:
      - id: Function call ID (always present)
      - status: Execution status (always present)
      - text_result: Concatenated string content (optional)
      - structured_result: Merged dict content (optional)

    Example:
      ```python
      # Text only
      result = ToolResult(details=["Temperature is 72°F"])
      parts = result.to_parts(name="get_weather", id="call-123")
      # → [Part(function_response=FunctionResponse(
      #     name="get_weather",
      #     id="call-123",
      #     response={"status": "success", "text_result": "Temperature is 72°F"}
      #   ))]

      # With image (recommended for visual understanding)
      result = ToolResult(details=[
          "Here's a dog",
          Blob(data=image_bytes, mime_type="image/jpeg")
      ])
      parts = result.to_parts(name="generate_image", id="call-456")
      # → [
      #     Part(function_response=FunctionResponse(...)),
      #     Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
      #   ]
      ```
    """
    texts: list[str] = []
    details_dict: dict[str, Any] = {}
    media_parts: list[Part] = []

    for item in self.details:
      if isinstance(item, str):
        texts.append(item)
      elif isinstance(item, dict):
        # Dict content merges into details
        details_dict.update(item)
      elif isinstance(item, Blob):
        # Blob becomes standalone Part for visual understanding
        if item.data is not None and item.mime_type is not None:
          media_parts.append(
              Part.from_bytes(data=item.data, mime_type=item.mime_type)
          )
      elif isinstance(item, FileData):
        # FileData becomes standalone Part
        if item.file_uri is not None:
          media_parts.append(
              Part.from_uri(file_uri=item.file_uri, mime_type=item.mime_type)
          )

    # Build response data
    response_data: _FunctionResponseData = {"status": self.status}
    if texts:
      response_data["text_result"] = "\n".join(texts)
    if details_dict:
      response_data["structured_result"] = details_dict

    # Build Parts list: FunctionResponse first, then media
    result_parts: list[Part] = [
        Part(
            function_response=FunctionResponse(
                id=id,
                name=name,
                response=dict(response_data) if response_data else {},
            )
        )
    ]
    result_parts.extend(media_parts)

    return result_parts


class BaseTool(ADKBaseTool):
  """Extended base class for adkx tools.

  Simple extension of google.adk.tools.BaseTool with redesigned result interface.
  Tools return ToolResult containing multi-modal content and optional ADK
  runtime directives.

  Inherited from ADKBaseTool:
  - name: str - The tool's name
  - description: str - The tool's description

  Example:
    ```python
    class WeatherTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="get_weather",
                description="Get current weather"
            )

        async def run_async(
            self, *, args: dict[str, Any], tool_context
        ) -> ToolResult:
            # Fetch weather data
            temp = get_temperature(args["location"])

            return ToolResult(
                content=[f"Temperature in {args['location']} is {temp}°F"]
            )
    ```
  """

  @override
  def __init__(
      self,
      *,
      name: str,
      description: str,
  ) -> None:
    """Initialize the tool.

    Args:
      name: The tool's name.
      description: The tool's description.
    """
    super().__init__(name=name, description=description)

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context
  ) -> ToolResult:
    """Run the tool with the given arguments.

    This is a redesigned interface for tool execution. Override this method
    in your tool implementation.

    Args:
      args: The LLM-filled arguments.
      tool_context: The context of the tool.

    Returns:
      A ToolResult containing multi-modal content and optional ADK runtime
      directives.
    """
    del args, tool_context  # Unused in base class
    raise NotImplementedError(f"{type(self)} has not implemented run_async")
