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

"""Example of a weather Agent using local Ollama.

This example demonstrates:
1. Using the Ollama LLM with a local Ollama instance
2. Creating a weather assistant with the same tool as weather_agent
3. Swapping LLM providers without changing agent logic

The agent uses the same get_weather tool as the base weather_agent example,
showing how easy it is to switch between Gemini and Ollama.

Tested models:
- qwen3:8b (recommended)
- gpt-oss:20b
- mistral-small:24b
- qwen3-coder:30b

For running agents, see the ADK documentation on sessions and
invocation contexts.
"""

from __future__ import annotations

from pydantic import BaseModel

from adkx.agents import Agent
from adkx.models import Ollama
from adkx.tools import FunctionTool


# 1. Define the output schema using Pydantic
class WeatherData(BaseModel):
  """Weather information."""

  temperature: float
  conditions: str
  humidity: int


# 2. Define a tool function that returns structured data
async def get_weather(location: str, *, tool_context) -> WeatherData:
  """Get the current weather for a location.

  Args:
    location: The city or location to get weather for.
    tool_context: Tool execution context (provided by framework).

  Returns:
    WeatherData with current conditions.
  """
  del tool_context  # Not used in this simple example

  # In a real application, this would call a weather API
  # For this example, we'll generate deterministic weather based on location hash
  location_hash = hash(location.lower())

  temperature = 50.0 + (location_hash % 50)  # Range: 50-100°F
  conditions_list = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
  conditions = conditions_list[abs(location_hash) % len(conditions_list)]
  humidity = 30 + (abs(location_hash) % 60)  # Range: 30-90%

  return WeatherData(
      temperature=float(temperature),
      conditions=conditions,
      humidity=humidity,
  )


# 3. Create the Ollama LLM instance
ollama_llm = Ollama(
    model="qwen3:8b",  # You could try other models as listed above.
)

# 4. Create an agent with the Ollama LLM and tool
root_agent = Agent(
    name="weather_assistant",
    model=ollama_llm,  # Pass the Ollama instance directly
    instruction="""
You are a helpful weather assistant.

Use the get_weather tool to answer questions about the weather.
""".strip(),
    tools=[FunctionTool(func=get_weather)],
)
