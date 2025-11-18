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

"""Tests for Agent."""

from __future__ import annotations

from unittest.mock import Mock

from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from adkx.agents import Agent
from adkx.tools import FunctionTool


def test_agent_creation():
  """Test that Agent can be instantiated with minimal fields."""
  agent = Agent(
      name="test_agent",
      model="gemini-2.5-flash",
      description="A test agent",
  )
  assert agent.name == "test_agent"
  assert agent.model == "gemini-2.5-flash"
  assert agent.description == "A test agent"
  assert agent.instruction == ""
  assert agent.tools == []


def test_agent_with_instruction():
  """Test that Agent can be created with an instruction."""
  agent = Agent(
      name="test_agent",
      model="gemini-2.5-flash",
      instruction="You are a helpful assistant",
  )
  assert agent.instruction == "You are a helpful assistant"


def test_agent_with_tools():
  """Test that Agent can be created with tools."""

  async def get_weather(location: str, *, tool_context) -> str:
    """Get weather for a location."""
    del tool_context  # Unused
    return f"Weather in {location}: 72°F"

  tool = FunctionTool(get_weather, description="Get weather")

  agent = Agent(
      name="test_agent",
      model="gemini-2.5-flash",
      tools=[tool],
  )
  assert len(agent.tools) == 1
  assert agent.tools[0].name == "get_weather"


# ============================================================================
# Extension Point Tests
# ============================================================================


class TestExtensionPoints:
  """Tests for Agent extension points that can be overridden by subclasses."""

  def test_build_identity_instruction_with_name_only(self):
    """Test identity instruction with just name."""
    agent = Agent(name="test_agent", model="gemini-2.5-flash")

    identity = agent._build_identity_instruction()

    assert identity == 'You are known as "test_agent".'

  def test_build_identity_instruction_with_name_and_description(self):
    """Test identity instruction with name and description."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash",
        description="A helpful assistant",
    )

    identity = agent._build_identity_instruction()

    assert (
        identity
        == 'You are known as "test_agent". The description about you is "A'
        ' helpful assistant".'
    )

  def test_build_system_instruction_adds_identity(self):
    """Test that system instruction includes identity."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash",
        description="Test description",
    )
    llm_request = LlmRequest(model="gemini-2.5-flash")
    ctx = Mock()

    agent._build_system_instruction(llm_request, ctx)

    # Verify instruction was added (system_instruction is a string after append_instructions)
    assert llm_request.config.system_instruction is not None
    assert isinstance(llm_request.config.system_instruction, str)
    assert (
        llm_request.config.system_instruction
        == 'You are known as "test_agent". The description about you is "Test'
        ' description".'
    )

  def test_build_system_instruction_adds_custom_instruction(self):
    """Test that system instruction includes custom instruction."""
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant.",
    )
    llm_request = LlmRequest(model="gemini-2.5-flash")
    ctx = Mock()

    agent._build_system_instruction(llm_request, ctx)

    # Verify both identity and custom instruction were added (separated by blank line)
    assert isinstance(llm_request.config.system_instruction, str)
    expected = """You are known as "test_agent".

You are a helpful assistant."""
    assert llm_request.config.system_instruction == expected

  def test_build_system_instruction_empty_when_no_instruction(self):
    """Test that system instruction is not set when agent has no instruction."""
    # Note: name is required, so we test with name but no custom instruction
    agent = Agent(name="test_agent", model="gemini-2.5-flash")
    llm_request = LlmRequest(model="gemini-2.5-flash")
    ctx = Mock()

    agent._build_system_instruction(llm_request, ctx)

    # System instruction should still be set with identity
    assert llm_request.config.system_instruction is not None
    assert isinstance(llm_request.config.system_instruction, str)
    assert (
        llm_request.config.system_instruction
        == 'You are known as "test_agent".'
    )

  def test_build_tools_adds_tools_to_request(self):
    """Test that tools are added to LlmRequest."""

    async def test_tool(*, tool_context) -> str:
      """Test tool."""
      del tool_context
      return "test"

    tool = FunctionTool(test_tool)
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash",
        tools=[tool],
    )
    llm_request = LlmRequest(model="gemini-2.5-flash")
    ctx = Mock()

    agent._build_tools(llm_request, ctx)

    # Verify tool was added (tools are in config.tools)
    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    # Verify tool is in the tools_dict
    assert "test_tool" in llm_request.tools_dict
    assert llm_request.tools_dict["test_tool"] == tool

  def test_build_conversation_adds_session_events(self):
    """Test that conversation history is built from session events."""
    agent = Agent(name="test_agent", model="gemini-2.5-flash")
    llm_request = LlmRequest(model="gemini-2.5-flash")

    # Create mock session with events
    event1 = Event(
        author="user",
        content=types.UserContent(parts=[types.Part(text="Hello")]),
    )
    event2 = Event(
        author="test_agent",
        content=types.ModelContent(
            parts=[types.Part(text="Hi, how can I help?")]
        ),
    )

    ctx = Mock()
    ctx.session.events = [event1, event2]

    agent._build_conversation(llm_request, ctx)

    # Verify events were added to contents
    expected_contents = [event1.content, event2.content]
    assert llm_request.contents == expected_contents

  def test_build_conversation_skips_events_without_content(self):
    """Test that events without content are skipped."""
    agent = Agent(name="test_agent", model="gemini-2.5-flash")
    llm_request = LlmRequest(model="gemini-2.5-flash")

    # Create events - one with content, one without
    event_with_content = Event(
        author="user",
        content=types.UserContent(parts=[types.Part(text="Hello")]),
    )
    event_without_content = Event(author="test_agent")  # No content

    ctx = Mock()
    ctx.session.events = [event_with_content, event_without_content]

    agent._build_conversation(llm_request, ctx)

    # Only the event with content should be added
    expected_contents = [event_with_content.content]
    assert llm_request.contents == expected_contents


# ============================================================================
# Function Call ID Tests
# ============================================================================


class TestFunctionCallIds:
  """Tests for function call ID generation and handling."""

  def test_ensure_function_call_ids(self):
    """Test ID generation for missing IDs and preservation of existing IDs."""
    agent = Agent(name="test_agent", model="gemini-2.5-flash")

    # Create content with two function calls: one without ID, one with ID
    existing_id = "model-provided-id-123"
    content = types.ModelContent(
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    name="get_weather", args={"location": "SF"}
                )
            ),
            types.Part(
                function_call=types.FunctionCall(
                    id=existing_id, name="get_time", args={}
                )
            ),
        ]
    )

    agent._ensure_function_call_ids(content)

    # Verify both function calls exist
    assert content.parts[0].function_call is not None
    assert content.parts[1].function_call is not None

    # First call should get a generated ID
    assert content.parts[0].function_call.id is not None
    assert content.parts[0].function_call.id.startswith("adk-")

    # Second call should keep its existing ID
    assert content.parts[1].function_call.id == existing_id

    # IDs should be unique
    assert (
        content.parts[0].function_call.id != content.parts[1].function_call.id
    )


# ============================================================================
# TODO: Integration Tests
# ============================================================================
# TODO: Add integration tests for Agent._run_async_impl with fake BaseLlm client
# - Test complete LLM loop execution
# - Test tool execution triggered by function calls
# - Test event generation (model events and function response events)
# - Test loop continuation until no more function calls
# - Test EventActions merging from concurrent tool executions
