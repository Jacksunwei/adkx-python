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

"""Example of an Agent using Ollama LLM.

This example demonstrates:
1. Using the Ollama LLM with a local or remote Ollama instance
2. Creating a code assistant agent with tool calling capabilities
3. Using a custom LLM backend instead of Gemini

The agent can be configured to use any Ollama model (e.g., qwen3-coder,
deepseek-r1, llama3, etc.) and can connect to:
- Local Ollama instance (default: http://localhost:11434)
- Remote Ollama instance on Cloud Run or other endpoints

For running agents, see the ADK documentation on sessions and
invocation contexts.
"""

from __future__ import annotations

from pydantic import BaseModel

from adkx.agents import Agent
from adkx.models.ollama import Ollama
from adkx.tools import FunctionTool


# 1. Define a simple tool for code analysis
class CodeAnalysisResult(BaseModel):
  """Result of code analysis."""

  language: str
  line_count: int
  has_functions: bool
  has_classes: bool


async def analyze_code(code: str, *, tool_context) -> CodeAnalysisResult:
  """Analyze a code snippet and return basic metrics.

  Args:
    code: The code snippet to analyze.
    tool_context: Tool execution context (provided by framework).

  Returns:
    CodeAnalysisResult with basic code metrics.
  """
  del tool_context  # Not used in this simple example

  lines = code.strip().split("\n")
  line_count = len(lines)

  # Simple heuristic-based language detection
  language = "unknown"
  if "def " in code or "import " in code:
    language = "python"
  elif "function " in code or "const " in code or "let " in code:
    language = "javascript"
  elif "public class" in code or "private class" in code:
    language = "java"
  elif "fn " in code or "let mut" in code:
    language = "rust"

  # Check for functions and classes
  has_functions = any(
      keyword in code
      for keyword in ["def ", "function ", "fn ", "func ", "public static"]
  )
  has_classes = any(
      keyword in code for keyword in ["class ", "struct ", "interface "]
  )

  return CodeAnalysisResult(
      language=language,
      line_count=line_count,
      has_functions=has_functions,
      has_classes=has_classes,
  )


# 2. Create the Ollama LLM instance
# You can customize the model and base_url to match your setup
ollama_llm = Ollama(
    model="qwen3-coder:30b",  # Change this to your preferred model
    base_url="http://localhost:11434/v1",  # Local Ollama instance
    # For remote Cloud Run instances, use:
    # base_url="https://your-service.run.app/v1",
    # use_gcp_auth=True,
)

# 3. Create an agent with the Ollama LLM and tool
root_agent = Agent(
    name="ollama_code_assistant",
    model=ollama_llm,  # Pass the Ollama instance directly
    instruction="""
You are a helpful coding assistant powered by Ollama.

You can analyze code, answer programming questions, and help with software
development tasks.

When asked to analyze code, use the analyze_code tool to get basic metrics
about the code snippet.
""".strip(),
    tools=[FunctionTool(func=analyze_code)],
)
