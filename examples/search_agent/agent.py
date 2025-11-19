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

"""Example of an Agent with Google Search capability.

This example demonstrates:
1. Using the built-in google_search tool with an Agent
2. How Gemini 2.0+ models can perform web searches internally
3. Creating a research assistant that can search the web

The google_search tool is a model built-in tool that operates within
Gemini 2.0+ models. It doesn't execute locally but enables the model
to search Google and incorporate results into its responses.

For running agents, see the ADK documentation on sessions and
invocation contexts.
"""

from __future__ import annotations

from adkx.agents import Agent
from adkx.tools import google_search

# Create an agent with Google Search capability
root_agent = Agent(
    name="search_assistant",
    model="gemini-2.5-flash",
    instruction="""
You are a helpful research assistant with access to Google Search.

When users ask questions that require current information or web research,
use the google_search tool to find relevant information.

Always cite your sources and provide accurate, well-researched answers.
""".strip(),
    tools=[google_search],
)
