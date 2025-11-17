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

"""Tests for FunctionTool."""

from __future__ import annotations

from typing import Any
from typing import Optional
from unittest.mock import Mock

from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel
import pytest

from adkx.tools import FunctionTool
from adkx.tools import ToolResult


class UserInput(BaseModel):
  """User input data for testing Pydantic model parameters."""

  name: str
  age: int
  email: str | None = None


class WeatherData(BaseModel):
  """Weather data for testing Pydantic model return types."""

  temperature: float
  conditions: str
  humidity: int | None = None


@pytest.fixture
def mock_tool_context():
  """Create a mock ToolContext for testing."""
  return Mock(spec=ToolContext)


class TestFunctionTool:
  """Tests for FunctionTool class."""

  @pytest.mark.asyncio
  async def test_basic_function(self, mock_tool_context):
    """Test FunctionTool with a basic async function."""

    async def greet(name: str, *, tool_context) -> str:
      """Greet a person by name."""
      del tool_context  # Unused
      return f"Hello, {name}!"

    tool = FunctionTool(
        func=greet,
        description="Greet a person",
    )

    assert tool.name == "greet"
    assert tool.description == "Greet a person"

    # Check declaration
    declaration = tool._get_declaration()
    assert declaration.name == "greet"
    assert declaration.description == "Greet a person"
    assert declaration.parameters_json_schema == {
        "properties": {"name": {"title": "Name", "type": "string"}},
        "required": ["name"],
        "title": "Parameters",
        "type": "object",
    }
    assert declaration.response_json_schema == {"type": "string"}

    # Test execution - result is auto-wrapped in ToolResult
    result = await tool.run_async(
        args={"name": "Alice"}, tool_context=mock_tool_context
    )
    assert isinstance(result, ToolResult)
    assert result.details == ["Hello, Alice!"]

  @pytest.mark.asyncio
  async def test_function_with_defaults(self, mock_tool_context):
    """Test FunctionTool with default parameter values."""

    async def get_weather(
        location: str, unit: str = "F", *, tool_context
    ) -> str:
      """Get weather for a location."""
      del tool_context  # Unused
      return f"Weather in {location}: 72°{unit}"

    tool = FunctionTool(func=get_weather)

    # Uses docstring as description
    assert tool.description == "Get weather for a location."

    # Check declaration
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema == {
        "properties": {
            "location": {"title": "Location", "type": "string"},
            "unit": {"default": "F", "title": "Unit", "type": "string"},
        },
        "required": ["location"],
        "title": "Parameters",
        "type": "object",
    }
    assert declaration.response_json_schema == {"type": "string"}

    # Test execution with default
    result = await tool.run_async(
        args={"location": "Seattle"}, tool_context=mock_tool_context
    )
    assert result.details == ["Weather in Seattle: 72°F"]

    # Test execution with explicit unit
    result = await tool.run_async(
        args={"location": "Seattle", "unit": "C"},
        tool_context=mock_tool_context,
    )
    assert result.details == ["Weather in Seattle: 72°C"]

  @pytest.mark.asyncio
  async def test_function_with_multiple_types(self, mock_tool_context):
    """Test FunctionTool with multiple parameter types."""

    async def calculate(
        a: int, b: float, operation: str = "add", *, tool_context
    ) -> dict[str, Any]:
      """Perform calculation."""
      del tool_context  # Unused
      if operation == "add":
        result = a + b
      elif operation == "multiply":
        result = a * b
      else:
        result = 0
      return {"result": result, "operation": operation}

    tool = FunctionTool(
        calculate,
        description="Perform mathematical operations",
    )

    # Check declaration
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "number"},
            "operation": {
                "default": "add",
                "title": "Operation",
                "type": "string",
            },
        },
        "required": ["a", "b"],
        "title": "Parameters",
        "type": "object",
    }
    assert declaration.response_json_schema == {
        "additionalProperties": True,
        "title": "Value",
        "type": "object",
    }

    # Test execution
    result = await tool.run_async(
        args={"a": 5, "b": 3.5}, tool_context=mock_tool_context
    )
    assert result.details == [{"result": 8.5, "operation": "add"}]

    result = await tool.run_async(
        args={"a": 5, "b": 3.5, "operation": "multiply"},
        tool_context=mock_tool_context,
    )
    assert result.details == [{"result": 17.5, "operation": "multiply"}]

  @pytest.mark.asyncio
  async def test_tool_context_excluded_from_schema(self, mock_tool_context):
    """Test that tool_context is excluded from parameters schema."""

    async def use_context(param: str, *, tool_context) -> str:
      """Function that uses tool_context."""
      # Return both param and tool_context to verify forwarding
      return f"{param}:{tool_context}"

    tool = FunctionTool(use_context)

    # tool_context should not be in parameters schema
    declaration = tool._get_declaration()
    assert (
        "tool_context" not in declaration.parameters_json_schema["properties"]
    )

    # Test that tool_context is correctly forwarded to the function
    result = await tool.run_async(
        args={"param": "test"}, tool_context=mock_tool_context
    )
    assert result.details == [f"test:{mock_tool_context}"]

  @pytest.mark.asyncio
  async def test_sync_function_support(self, mock_tool_context):
    """Test FunctionTool with synchronous functions."""

    def sync_greet(name: str, *, tool_context) -> str:
      """Sync greeting function."""
      del tool_context  # Unused
      return f"Sync hello, {name}!"

    tool = FunctionTool(
        sync_greet,
        description="Synchronous greeting",
    )

    # Should work with sync functions too
    result = await tool.run_async(
        args={"name": "Bob"}, tool_context=mock_tool_context
    )
    assert result.details == ["Sync hello, Bob!"]

  @pytest.mark.asyncio
  async def test_no_parameters_function(self, mock_tool_context):
    """Test FunctionTool with a function that has no parameters at all."""

    async def get_timestamp() -> str:
      """Get current timestamp."""
      return "2025-11-16T12:00:00Z"

    tool = FunctionTool(
        get_timestamp,
        description="Get timestamp",
    )

    # Parameters schema should be empty
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema == {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # Test execution - tool_context should not be passed
    result = await tool.run_async(args={}, tool_context=mock_tool_context)
    assert result.details == ["2025-11-16T12:00:00Z"]

  @pytest.mark.asyncio
  async def test_function_without_tool_context(self, mock_tool_context):
    """Test FunctionTool with a function that doesn't use tool_context."""

    async def add_numbers(a: int, b: int) -> int:
      """Add two numbers."""
      return a + b

    tool = FunctionTool(add_numbers, description="Add two numbers")

    # Check declaration
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "Parameters",
        "type": "object",
    }
    assert declaration.response_json_schema == {"type": "integer"}

    # Test execution - tool_context should not be passed to function
    # Note: int return value is converted to string for ToolResult compatibility
    result = await tool.run_async(
        args={"a": 5, "b": 3}, tool_context=mock_tool_context
    )
    assert result.details == ["8"]

  @pytest.mark.asyncio
  async def test_default_name_and_description(self, mock_tool_context):
    """Test that name and description default to function name and docstring."""

    async def my_custom_tool(arg: str, *, tool_context) -> str:
      """This is my custom tool docstring."""
      del tool_context  # Unused
      return f"Result: {arg}"

    tool = FunctionTool(my_custom_tool)

    # Name should default to function name
    assert tool.name == "my_custom_tool"
    # Description should default to docstring
    assert tool.description == "This is my custom tool docstring."

    # Test execution
    result = await tool.run_async(
        args={"arg": "test"}, tool_context=mock_tool_context
    )
    assert result.details == ["Result: test"]

  @pytest.mark.asyncio
  async def test_dict_parameter(self, mock_tool_context):
    """Test FunctionTool with dict parameter type."""

    async def process_config(config: dict[str, Any], *, tool_context) -> str:
      """Process configuration dictionary."""
      del tool_context  # Unused
      return f"Processed {len(config)} items"

    tool = FunctionTool(process_config, description="Process config")

    # Check declaration
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema == {
        "properties": {
            "config": {
                "additionalProperties": True,
                "title": "Config",
                "type": "object",
            },
        },
        "required": ["config"],
        "title": "Parameters",
        "type": "object",
    }
    assert declaration.response_json_schema == {"type": "string"}

    # Test execution
    result = await tool.run_async(
        args={"config": {"key1": "value1", "key2": "value2"}},
        tool_context=mock_tool_context,
    )
    assert result.details == ["Processed 2 items"]

  @pytest.mark.asyncio
  async def test_pydantic_model_parameter(self, mock_tool_context):
    """Test FunctionTool with Pydantic model parameter."""

    async def process_user(user: UserInput, *, tool_context) -> str:
      """Process user input."""
      del tool_context  # Unused
      # FunctionTool automatically converts dict to UserInput
      email_part = f", {user.email}" if user.email else ""
      return f"{user.name} ({user.age}){email_part}"

    tool = FunctionTool(process_user, description="Process user data")

    # Check declaration - Pydantic model should be represented in schema
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema == {
        "$defs": {
            "UserInput": {
                "description": (
                    "User input data for testing Pydantic model parameters."
                ),
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "age": {"title": "Age", "type": "integer"},
                    "email": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "title": "Email",
                    },
                },
                "required": ["name", "age"],
                "title": "UserInput",
                "type": "object",
            }
        },
        "properties": {
            "user": {"$ref": "#/$defs/UserInput"},
        },
        "required": ["user"],
        "title": "Parameters",
        "type": "object",
    }
    assert declaration.response_json_schema == {"type": "string"}

    # Test execution with Pydantic model instance
    user_data = UserInput(name="Alice", age=30, email="alice@example.com")
    result = await tool.run_async(
        args={"user": user_data.model_dump()}, tool_context=mock_tool_context
    )
    assert result.details == ["Alice (30), alice@example.com"]

    # Test execution without optional field
    user_data_no_email = UserInput(name="Bob", age=25)
    result = await tool.run_async(
        args={"user": user_data_no_email.model_dump()},
        tool_context=mock_tool_context,
    )
    assert result.details == ["Bob (25)"]

  @pytest.mark.asyncio
  async def test_optional_pydantic_model_parameter(self, mock_tool_context):
    """Test FunctionTool with Optional Pydantic model parameter."""

    async def process_optional_user(
        user: Optional[UserInput], *, tool_context
    ) -> str:
      """Process optional user input."""
      del tool_context  # Unused
      if user is None:
        return "No user provided"
      email_part = f", {user.email}" if user.email else ""
      return f"{user.name} ({user.age}){email_part}"

    tool = FunctionTool(
        process_optional_user, description="Process optional user"
    )

    # Check declaration - Optional should still produce the same schema
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema is not None
    assert "$defs" in declaration.parameters_json_schema
    assert "UserInput" in declaration.parameters_json_schema["$defs"]

    # Test execution with user data
    user_data = UserInput(name="Charlie", age=35, email="charlie@example.com")
    result = await tool.run_async(
        args={"user": user_data.model_dump()}, tool_context=mock_tool_context
    )
    assert result.details == ["Charlie (35), charlie@example.com"]

    # Test execution with None
    result = await tool.run_async(
        args={"user": None}, tool_context=mock_tool_context
    )
    assert result.details == ["No user provided"]

  @pytest.mark.asyncio
  async def test_pydantic_model_return_type(self, mock_tool_context):
    """Test FunctionTool with Pydantic model return type."""

    async def get_weather(location: str, *, tool_context) -> WeatherData:
      """Get weather data for a location."""
      del location, tool_context  # Unused
      return WeatherData(temperature=72.5, conditions="sunny", humidity=65)

    tool = FunctionTool(get_weather, description="Get weather data")

    # Check declaration - Pydantic model return type should generate schema
    declaration = tool._get_declaration()
    assert declaration.parameters_json_schema == {
        "properties": {
            "location": {"title": "Location", "type": "string"},
        },
        "required": ["location"],
        "title": "Parameters",
        "type": "object",
    }
    assert declaration.response_json_schema == {
        "description": "Weather data for testing Pydantic model return types.",
        "properties": {
            "temperature": {"title": "Temperature", "type": "number"},
            "conditions": {"title": "Conditions", "type": "string"},
            "humidity": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "Humidity",
            },
        },
        "required": ["temperature", "conditions"],
        "title": "WeatherData",
        "type": "object",
    }

    # Test execution - Pydantic model should be converted to dict
    result = await tool.run_async(
        args={"location": "San Francisco"}, tool_context=mock_tool_context
    )
    assert len(result.details) == 1
    assert isinstance(result.details[0], dict)
    assert result.details[0] == {
        "temperature": 72.5,
        "conditions": "sunny",
        "humidity": 65,
    }
