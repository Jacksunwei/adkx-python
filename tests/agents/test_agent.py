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

from adkx.agents import Agent


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


def test_agent_with_instruction():
  """Test that Agent can be created with an instruction."""
  agent = Agent(
      name="test_agent",
      model="gemini-2.5-flash",
      instruction="You are a helpful assistant",
  )
  assert agent.instruction == "You are a helpful assistant"
