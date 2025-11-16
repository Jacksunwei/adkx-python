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

"""Tests for BaseAgent."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest

from adkx.agents import BaseAgent


class ConcreteAgent(BaseAgent):
  """Concrete agent implementation for testing."""

  async def _run_async_impl(self, ctx) -> AsyncGenerator:
    """Minimal implementation for testing."""
    # Empty implementation - just needed to make class instantiable
    yield  # pragma: no cover
    return  # pragma: no cover


def test_base_agent_creation():
  """Test that BaseAgent subclass can be instantiated."""
  agent = ConcreteAgent(name="test_agent", description="A test agent")
  assert agent.name == "test_agent"
  assert agent.description == "A test agent"


def test_frozen_model():
  """Test that agents are immutable after creation."""
  from pydantic import ValidationError

  agent = ConcreteAgent(name="test", description="Test agent")

  # Attempting to modify any field should raise ValidationError
  with pytest.raises(ValidationError, match="Instance is frozen"):
    agent.name = "new_name"  # type: ignore

  with pytest.raises(ValidationError, match="Instance is frozen"):
    agent.description = "new description"  # type: ignore


def test_no_hierarchy_fields():
  """Test that parent_agent and sub_agents are disabled."""
  agent = ConcreteAgent(name="test", description="Test agent")

  # Fields should exist but be empty/None
  assert agent.parent_agent is None
  assert agent.sub_agents == []
