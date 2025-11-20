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

"""Groq provider implementation."""

from __future__ import annotations

import os

from openai import AsyncOpenAI
from pydantic import Field

from ..openai_compatible import OpenAICompatibleLlm


class Groq(OpenAICompatibleLlm):
  """Groq LLM provider with ultra-fast inference.

  Connects to Groq's API at https://api.groq.com/openai/v1.
  Groq provides extremely fast inference using their LPU (Language Processing Unit) architecture.

  Attributes:
    model: The name of the Groq model (e.g., "llama-3.1-70b-versatile", "mixtral-8x7b-32768").
    api_key: Groq API key (defaults to GROQ_API_KEY environment variable).
  """

  model: str = "llama-3.1-70b-versatile"
  api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

  def _create_client(self) -> AsyncOpenAI:
    """Create AsyncOpenAI client configured for Groq.

    Returns:
      AsyncOpenAI: Client configured to connect to Groq API.

    Raises:
      ValueError: If API key is not provided.
    """
    if not self.api_key:
      raise ValueError(
          "Groq API key required. Set GROQ_API_KEY environment variable "
          "or pass api_key parameter."
      )

    return AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=self.api_key,
    )
