# Image Generation Agent Example

An agent that generates images using Gemini 2.5 Flash Image (Nano Banana) and demonstrates multi-modal tool responses using the standalone Parts pattern ([ADR-0002](../../docs/decisions/0002-multi-modal-function-response.md)).

## Key Implementation

The `create_image` tool returns both text and image data:

```python
async def create_image(prompt: str) -> ToolResult:
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    image_data = response.candidates[0].content.parts[0].inline_data.data

    return ToolResult(
        details=[
            f"Generated image for: {prompt}",
            types.Blob(data=image_data, mime_type="image/jpeg"),
        ]
    )
```

The framework converts this to standalone Parts, allowing the LLM to visually understand and discuss the generated image.

## Sample Queries

```
Create an image of a husky playing in the sand, with a happy face and flying ears
```

## Viewing Generated Images

Generated images are automatically saved to `image.png` in the `samples/image_agent/` folder. The ADK web UI cannot display images yet, but you can open the saved file directly.
