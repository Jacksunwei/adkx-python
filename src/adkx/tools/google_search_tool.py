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

"""Google Search tool for adkx.

Migrated from google.adk.tools.GoogleSearchTool to use adkx.tools.BaseTool.
"""

from __future__ import annotations

from google.adk.models.llm_request import LlmRequest
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool


def _is_gemini_model(model_name: str | None) -> bool:
  """Check if model is a Gemini model."""
  if not model_name:
    return False
  model_lower = model_name.lower()
  return "gemini" in model_lower


class _GoogleSearchTool(BaseTool):
  """A built-in tool that enables Google Search within Gemini models.

  This tool operates internally within the model and does not require local
  code execution. It configures the LLM request to enable Google Search
  capabilities in Gemini 2.0+ models.

  Note: This class is private. Use the `google_search` singleton instance instead.

  Example:
    ```python
    from adkx.agents import Agent
    from adkx.tools import google_search

    agent = Agent(
        name="research_assistant",
        model="gemini-2.5-flash",
        instruction="Use Google Search to answer user questions.",
        tools=[google_search],
    )
    ```
  """

  def __init__(self) -> None:
    """Initialize the Google Search tool."""
    # Name and description are not used because this is a model built-in tool
    super().__init__(name="google_search", description="Google Search")

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    """Configure the LLM request to enable Google Search.

    This method modifies the GenerateContentConfig to add the Google Search
    tool configuration for Gemini 2.0+ models.

    Args:
      tool_context: Tool execution context (unused).
      llm_request: The LLM request to configure.

    Raises:
      ValueError: If the model doesn't support Google Search.
    """
    del tool_context  # Not used for built-in tools

    # Ensure config exists
    llm_request.config = llm_request.config or types.GenerateContentConfig()
    llm_request.config.tools = llm_request.config.tools or []

    model_name = llm_request.model

    if _is_gemini_model(model_name):
      # Gemini 2.0+: Use GoogleSearch (can combine with other tools)
      llm_request.config.tools.append(
          types.Tool(google_search=types.GoogleSearch())
      )
    else:
      raise ValueError(
          f"Google Search tool is not supported for model '{model_name}'. "
          "Only Gemini models support this feature."
      )


# Singleton instance for convenience
google_search = _GoogleSearchTool()
