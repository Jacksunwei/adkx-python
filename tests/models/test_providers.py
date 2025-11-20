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

"""Tests for specific OpenAI-compatible provider implementations."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from adkx.models import Groq
from adkx.models import Ollama
from adkx.models import OllamaCloudRun
from adkx.models import OpenAI


class TestOllamaProvider:
  """Tests for Ollama provider."""

  def test_default_configuration(self):
    """Test Ollama default configuration."""
    ollama = Ollama()
    assert ollama.model == "qwen3-coder:30b"
    assert ollama.base_url == "http://localhost:11434/v1"

  def test_custom_configuration(self):
    """Test Ollama custom configuration."""
    ollama = Ollama(model="llama2:7b", base_url="http://custom:8080/v1")
    assert ollama.model == "llama2:7b"
    assert ollama.base_url == "http://custom:8080/v1"

  def test_client_creation(self):
    """Test that client is created correctly."""
    ollama = Ollama()
    client = ollama.client
    assert str(client.base_url) == "http://localhost:11434/v1/"
    # Ollama uses "unused" as API key
    assert client.api_key == "unused"


class TestOpenAIProvider:
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


class TestGroqProvider:
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


class TestOllamaCloudRunProvider:
  """Tests for OllamaCloudRun provider."""

  def test_requires_base_url(self):
    """Test that base_url is required."""
    # This should work - base_url is provided
    cloud_run = OllamaCloudRun(base_url="https://ollama.run.app/v1")
    assert cloud_run.base_url == "https://ollama.run.app/v1"

  def test_default_model(self):
    """Test OllamaCloudRun default model."""
    cloud_run = OllamaCloudRun(base_url="https://ollama.run.app/v1")
    assert cloud_run.model == "qwen3-coder:30b"

  def test_custom_model(self):
    """Test OllamaCloudRun with custom model."""
    cloud_run = OllamaCloudRun(
        base_url="https://ollama.run.app/v1", model="llama2:13b"
    )
    assert cloud_run.model == "llama2:13b"

  @patch("subprocess.run")
  def test_gcp_auth_token_retrieval(self, mock_subprocess):
    """Test that GCP auth token is retrieved correctly."""
    # Mock successful gcloud command
    mock_subprocess.return_value.stdout = "test-token-123\n"
    mock_subprocess.return_value.returncode = 0

    cloud_run = OllamaCloudRun(base_url="https://ollama.run.app/v1")
    token = cloud_run._get_gcp_auth_token()

    assert token == "test-token-123"
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert args == ["gcloud", "auth", "print-identity-token"]
