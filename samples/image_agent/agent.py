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

"""Example of an Agent that generates images using multi-modal tool responses.

This example demonstrates:
1. Creating a tool that returns multi-modal content (text + image)
2. Using ToolResult with Blob for image data
3. How standalone Parts enable visual understanding across backends
4. Creating an agent that can both generate and discuss images

The create_image tool calls Gemini 2.5 Flash Image (Nano Banana) to generate
images and returns both a text description and the actual image as a Blob.
This allows the LLM to visually understand the generated image and provide
meaningful feedback.

For running agents, see the ADK documentation on sessions and
invocation contexts.
"""

from __future__ import annotations

from pathlib import Path

from google.genai import types
from google.genai.client import Client

from adkx.agents import Agent
from adkx.tools import FunctionTool
from adkx.tools import ToolResult


async def create_image(prompt: str) -> ToolResult:
  """Generate an image from a text description using Gemini 2.5 Flash Image.

  Args:
    prompt: Description of the image to generate.

  Returns:
    ToolResult containing both a text description and the generated image.
  """
  client = Client()

  try:
    # Generate image using Gemini 2.5 Flash Image (Nano Banana)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    # Extract generated image from response parts
    image_data = None
    if (
        response.candidates
        and response.candidates[0].content
        and response.candidates[0].content.parts
    ):
      for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
          image_data = part.inline_data.data
          break

    if not image_data:
      return ToolResult(
          details=["Failed to generate image. Please try a different prompt."],
          status="error",
      )

    # Save image to local file in the same folder as this module
    output_path = Path(__file__).parent / "image.png"
    output_path.write_bytes(image_data)

    # Return multi-modal result: text + image Blob
    # The framework converts this to standalone Parts for LLM visual understanding
    return ToolResult(
        details=[
            f"Generated image for: {prompt}",
            types.Blob(data=image_data, mime_type="image/jpeg"),
        ]
    )

  except Exception as e:
    # Return error status with details
    return ToolResult(
        details=[f"Image generation failed: {e!r}"],
        status="error",
    )


# Create an agent with image generation capability
root_agent = Agent(
    name="image_creator",
    model="gemini-2.5-flash",
    instruction="""
You are a creative image generation assistant.

When users request images, use the create_image tool to generate them.
After generating an image, you can see it and provide feedback about
the visual content, composition, colors, and how well it matches the request.

Be creative with prompts and provide helpful suggestions for improvement.
""".strip(),
    tools=[FunctionTool(func=create_image)],
)
