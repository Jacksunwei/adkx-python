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

"""OpenAI provider implementation."""

from __future__ import annotations

import os

from openai import AsyncOpenAI
from pydantic import Field

from ..openai_compatible import OpenAICompatibleLlm


class OpenAI(OpenAICompatibleLlm):
  """OpenAI LLM provider.

  Connects to OpenAI's official API at https://api.openai.com/v1.

  Attributes:
    model: The name of the OpenAI model (e.g., "gpt-4o", "gpt-4", "gpt-3.5-turbo").
    api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable).
    organization: Optional organization ID for OpenAI API requests.
  """

  model: str = "gpt-4o"
  api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
  organization: str | None = None

  def _create_client(self) -> AsyncOpenAI:
    """Create AsyncOpenAI client configured for OpenAI.

    Returns:
      AsyncOpenAI: Client configured to connect to OpenAI API.

    Raises:
      ValueError: If API key is not provided.
    """
    if not self.api_key:
      raise ValueError(
          "OpenAI API key required. Set OPENAI_API_KEY environment variable "
          "or pass api_key parameter."
      )

    return AsyncOpenAI(
        base_url="https://api.openai.com/v1",
        api_key=self.api_key,
        organization=self.organization,
    )
