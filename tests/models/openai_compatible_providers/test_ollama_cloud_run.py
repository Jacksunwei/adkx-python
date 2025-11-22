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

"""Tests for OllamaCloudRun provider."""

from __future__ import annotations

from unittest.mock import patch

from adkx.models import OllamaCloudRun


class TestOllamaCloudRun:
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
