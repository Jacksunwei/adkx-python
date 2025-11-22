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

"""Tests for Groq provider."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from adkx.models import Groq


class TestGroq:
  """Tests for Groq provider."""

  def test_default_model(self):
    """Test Groq default model."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
      groq = Groq()
      assert groq.model == "llama-3.1-70b-versatile"
      assert groq.api_key == "gsk-test"

  def test_explicit_api_key(self):
    """Test Groq with explicit API key."""
    groq = Groq(api_key="gsk-explicit")
    assert groq.api_key == "gsk-explicit"

  def test_missing_api_key_raises_error(self):
    """Test that missing API key raises ValueError."""
    with patch.dict("os.environ", {}, clear=True):
      groq = Groq(api_key="")
      with pytest.raises(ValueError, match="Groq API key required"):
        _ = groq.client

  def test_client_creation(self):
    """Test that client is created with correct base URL."""
    groq = Groq(api_key="gsk-test")
    client = groq.client
    assert str(client.base_url) == "https://api.groq.com/openai/v1/"
