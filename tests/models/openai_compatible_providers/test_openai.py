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

"""Tests for OpenAI provider."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from adkx.models import OpenAI


class TestOpenAI:
  """Tests for OpenAI provider."""

  def test_default_model(self):
    """Test OpenAI default model."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
      openai = OpenAI()
      assert openai.model == "gpt-4o"
      assert openai.api_key == "sk-test"

  def test_explicit_api_key(self):
    """Test OpenAI with explicit API key."""
    openai = OpenAI(api_key="sk-explicit")
    assert openai.api_key == "sk-explicit"

  def test_missing_api_key_raises_error(self):
    """Test that missing API key raises ValueError."""
    with patch.dict("os.environ", {}, clear=True):
      openai = OpenAI(api_key="")
      with pytest.raises(ValueError, match="OpenAI API key required"):
        _ = openai.client

  def test_organization_parameter(self):
    """Test OpenAI with organization parameter."""
    openai = OpenAI(api_key="sk-test", organization="org-test")
    assert openai.organization == "org-test"

  def test_client_creation(self):
    """Test that client is created with correct base URL."""
    openai = OpenAI(api_key="sk-test")
    client = openai.client
    assert str(client.base_url) == "https://api.openai.com/v1/"
