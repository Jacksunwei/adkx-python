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

"""Minimal example of an Agent with a simple tool.

This example demonstrates:
1. Creating a simple FunctionTool that returns structured data
2. Creating an Agent with that tool
3. How the agent can be instantiated and configured

This is a reference implementation showing the minimal setup needed.
For running agents, see the ADK documentation on sessions and
invocation contexts.
"""

from __future__ import annotations

from pydantic import BaseModel

from adkx.agents import Agent
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


# 3. Create an agent with the tool
root_agent = Agent(
    name="weather_assistant",
    model="gemini-2.5-flash",
    instruction="""
You are a helpful weather assistant.

Use the get_weather tool to answer questions about the weather.
""".strip(),
    tools=[FunctionTool(func=get_weather)],
)
