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

"""Ollama Cloud Run provider implementation with GCP authentication."""

from __future__ import annotations

import subprocess

from openai import AsyncOpenAI

from ..openai_compatible import OpenAICompatibleLlm


class OllamaCloudRun(OpenAICompatibleLlm):
  """Ollama provider for GCP Cloud Run with identity token authentication.

  Connects to Ollama instances running on Google Cloud Run using GCP
  identity tokens for authentication.

  Attributes:
    model: The name of the Ollama model (e.g., "qwen3-coder:30b").
    base_url: The base URL of the Cloud Run Ollama instance (required).
  """

  model: str = "qwen3-coder:30b"
  base_url: str  # Required - no default for Cloud Run

  def _create_client(self) -> AsyncOpenAI:
    """Create AsyncOpenAI client configured for Ollama on Cloud Run.

    Returns:
      AsyncOpenAI: Client configured with GCP identity token authentication.
    """
    token = self._get_gcp_auth_token()
    return AsyncOpenAI(
        base_url=self.base_url,
        api_key=token,
        default_headers={"Authorization": f"Bearer {token}"},
    )

  def _get_gcp_auth_token(self) -> str:
    """Get GCP authentication token for Cloud Run.

    Returns:
      str: The authentication token.

    Raises:
      RuntimeError: If unable to fetch authentication token.
    """
    # TODO: This method is for local development with user credentials (gcloud auth login).
    # For production deployments, use service account keys or GCP metadata server instead.
    # Note: Cannot use google-auth library to generate ID tokens with user credentials.
    # For Cloud Run, we need an ID token not an access token
    # Use gcloud to get the ID token (works with user credentials)
    try:
      result = subprocess.run(
          ["gcloud", "auth", "print-identity-token"],
          capture_output=True,
          text=True,
          check=True,
      )
      token = result.stdout.strip()
      if not token:
        raise RuntimeError("Empty token returned from gcloud")
      return token
    except subprocess.CalledProcessError as e:
      raise RuntimeError(
          f"Failed to fetch GCP authentication token: {e.stderr}"
      ) from e
    except FileNotFoundError as e:
      raise RuntimeError(
          "gcloud command not found. Please install Google Cloud SDK."
      ) from e
