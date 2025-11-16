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

"""Tests for BaseTool."""

from __future__ import annotations

from typing import Any

from google.genai.types import Blob
import pytest

from adkx.tools import BaseTool
from adkx.tools import ToolResult


class TestBaseTool:
  """Tests for BaseTool class."""

  @pytest.mark.asyncio
  async def test_creation_and_execution(self):
    """Test BaseTool subclass creation and execution."""

    class TestTool(BaseTool):

      async def run_async(
          self, *, args: dict[str, Any], tool_context
      ) -> ToolResult:
        del args, tool_context  # Unused in test
        return ToolResult(details=["test result"])

    # Test instantiation
    tool = TestTool(name="test_tool", description="A test tool")
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"

    # Test execution
    result = await tool.run_async(args={}, tool_context=None)
    assert isinstance(result, ToolResult)
    assert result.details == ["test result"]
    assert result.status == "success"


class TestToolResult:
  """Tests for ToolResult class."""

  def test_field_assignment(self):
    """Test ToolResult field assignment and options."""
    # Custom status
    result = ToolResult(details=["Processing..."], status="pending")
    assert result.status == "pending"

    # Error status with adk_context
    result = ToolResult(
        details=["Network timeout"], status="error", adk_context={"retry": True}
    )
    assert result.status == "error"
    assert result.adk_context == {"retry": True}

    # Structured content (mixed types)
    result = ToolResult(
        details=[
            {"temperature": 72, "unit": "F", "condition": "sunny"},
            "Additional context: perfect weather",
        ]
    )
    assert len(result.details) == 2
    assert result.details[0] == {
        "temperature": 72,
        "unit": "F",
        "condition": "sunny",
    }
    assert result.details[1] == "Additional context: perfect weather"

  def test_to_parts_text_only(self):
    """Test converting text-only ToolResult to Parts."""
    result = ToolResult(details=["Temperature is 72°F"])
    parts = result.to_parts(name="get_weather")

    assert len(parts) == 1
    assert parts[0].function_response.name == "get_weather"
    assert parts[0].function_response.response == {
        "status": "success",
        "text_result": "Temperature is 72°F",
    }

  def test_to_parts_multiple_text(self):
    """Test converting multiple text items to Parts."""
    result = ToolResult(details=["Line 1", "Line 2"])
    parts = result.to_parts(name="test_tool")

    assert len(parts) == 1
    assert parts[0].function_response.response == {
        "status": "success",
        "text_result": "Line 1\nLine 2",
    }

  def test_to_parts_with_dict(self):
    """Test converting dict content to Parts."""
    result = ToolResult(details=[{"temperature": 72, "unit": "F"}])
    parts = result.to_parts(name="get_weather")

    assert len(parts) == 1
    assert parts[0].function_response.response == {
        "status": "success",
        "structured_result": {"temperature": 72, "unit": "F"},
    }

  def test_to_parts_with_mixed_content(self):
    """Test converting mixed text and dict content to Parts."""
    result = ToolResult(
        details=[
            "Temperature is 72°F",
            {"temperature": 72, "unit": "F", "condition": "sunny"},
        ]
    )
    parts = result.to_parts(name="get_weather")

    assert len(parts) == 1
    assert parts[0].function_response.response == {
        "status": "success",
        "text_result": "Temperature is 72°F",
        "structured_result": {
            "temperature": 72,
            "unit": "F",
            "condition": "sunny",
        },
    }

  def test_to_parts_with_blob(self):
    """Test converting Blob content to Parts with standalone media."""
    result = ToolResult(
        details=[
            "Here's an image:",
            Blob(data=b"fake_image_data", mime_type="image/png"),
        ]
    )
    parts = result.to_parts(name="get_image")

    assert len(parts) == 2
    assert parts[0].function_response.response == {
        "status": "success",
        "text_result": "Here's an image:",
    }
    assert parts[1].inline_data.data == b"fake_image_data"
    assert parts[1].inline_data.mime_type == "image/png"

  def test_to_parts_with_status(self):
    """Test converting ToolResult with error status to Parts."""
    result = ToolResult(details=["Error occurred"], status="error")
    parts = result.to_parts(name="failing_tool")

    assert len(parts) == 1
    assert parts[0].function_response.response == {
        "status": "error",
        "text_result": "Error occurred",
    }

  def test_to_parts_empty_details(self):
    """Test converting ToolResult with no details (action-only tool)."""
    result = ToolResult()  # Empty details, default status
    parts = result.to_parts(name="action_tool")

    assert len(parts) == 1
    assert parts[0].function_response.response == {"status": "success"}
