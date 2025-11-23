# ADR-0002: Multi-Modal Function Responses with Standalone Parts

**Date**: 2025-11-16
**Status**: Accepted

## Context

When tools return images or other media to Gemini models, we need a way to ensure the LLM can visually understand the content. Testing revealed three approaches with different outcomes:

**Test Results (gemini-2.5-flash on both Google AI API and Vertex AI)**:

| Approach                      | Google AI API  | Vertex AI      | Result            |
| ----------------------------- | -------------- | -------------- | ----------------- |
| **1. FunctionResponse.parts** | ❌ 400 error   | ❌ 400 error   | Computer Use only |
| **2. Base64 in response**     | ✅ Full visual | ⚠️ No visual   | Backend-specific  |
| **3. Standalone Part**        | ✅ Works\*     | ✅ Full visual | Cross-backend     |

\*Requires omitting `display_name` parameter

**Key findings**:

1. **FunctionResponse.parts fails for normal Gemini models** with error: `'function_response.parts is not supported for this model.'` This field only works for Computer Use models.

1. **Base64 encoding in response dict is backend-dependent**:

   - Google AI API: Understands base64 images, correctly identified Border Collie
   - Vertex AI: Cannot parse base64 in response, hallucinated breed from text description

1. **Standalone Parts work universally** when placed alongside FunctionResponse in Content.parts:

   - Both backends correctly identified Border Collie with detailed visual analysis
   - Requires omitting `display_name` parameter for Google AI API compatibility

**Constraints**:

- Must support both Google AI API and Vertex AI backends
- Must work with normal Gemini models (not just Computer Use)
- Must enable actual visual understanding, not text-based guessing
- Should be simple for tool developers to use

## Decision

We will use **standalone Parts** for multi-modal function responses. The `ToolResult.to_parts()` method returns a list of Parts where:

1. **First Part**: Contains the FunctionResponse with:

   - `status`: Execution status (always present, defaults to "success")
   - `text_result`: Concatenated string content (optional)
   - `structured_result`: Merged dict content (optional)

1. **Subsequent Parts**: Media content as standalone Parts:

   - `Blob` items → `Part.from_bytes(data=..., mime_type=...)`
   - `FileData` items → `Part.from_uri(file_uri=..., mime_type=...)`

**Implementation**:

```python
def to_parts(self, *, name: str, id: str) -> list[Part]:
    """Convert ToolResult to Content parts for LLM consumption."""
    # Build FunctionResponse
    response_data = {"status": self.status}
    if texts:
        response_data["text_result"] = "\n".join(texts)
    if details_dict:
        response_data["structured_result"] = details_dict

    # FunctionResponse first, then media Parts
    parts = [Part(function_response=FunctionResponse(...))]

    for blob in blobs:
        parts.append(Part.from_bytes(data=blob.data, mime_type=blob.mime_type))
    for file_data in file_datas:
        parts.append(
            Part.from_uri(file_uri=file_data.file_uri, mime_type=file_data.mime_type)
        )

    return parts
```

**Usage for tool authors**:

See [examples/image_agent](../../examples/image_agent/) for a complete working example.

```python
async def create_image(prompt: str) -> ToolResult:
    """Generate an image from a text description using Gemini 2.5 Flash Image."""
    # Generate image using Gemini 2.5 Flash Image (Nano Banana)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    # Extract image from response
    image_data = response.candidates[0].content.parts[0].inline_data.data

    # Return multi-modal result: text + image
    return ToolResult(
        details=[
            f"Generated image for: {prompt}",
            types.Blob(data=image_data, mime_type="image/jpeg"),
        ]
    )


# Framework automatically converts ToolResult to Parts:
# [
#   Part(function_response=FunctionResponse(
#     name="create_image",
#     response={"status": "success", "text_result": "Generated image for: ..."}
#   )),
#   Part(inline_data=Blob(data=image_data, mime_type="image/jpeg"))
# ]
```

**Why this over alternatives**:

- **vs. FunctionResponse.parts**: Works with all Gemini models, not just Computer Use
- **vs. Base64 in response dict**: Backend-agnostic, guaranteed visual understanding on both Google AI API and Vertex AI
- **vs. Multiple approaches**: Single consistent pattern, simpler for tool developers

## Consequences

### Positive

- **Universal compatibility**: Works with normal Gemini models (gemini-2.5-flash), Computer Use models, Google AI API, and Vertex AI
- **Guaranteed visual understanding**: Both backends correctly analyze images with detailed reasoning (verified with Border Collie test image)
- **Simple developer experience**: Tool authors return `ToolResult(details=[text, Blob(...)])`, conversion handled automatically
- **Type-safe**: Leverages existing Pydantic models (`Blob`, `FileData`) with proper validation

### Negative / Trade-offs

- **Multiple Parts per function response**: Slightly more complex message structure compared to single FunctionResponse
- **Google AI API quirk**: Must omit `display_name` parameter (handled in implementation, but limits metadata)
- **No partial image streaming**: Images included only in final response (acceptable - visual understanding requires complete image)

### Neutral

- **Backward compatible**: Existing text-only tools continue working unchanged
- **Extensible**: Can support future media types (audio, video) using same pattern
- **Testable**: Image understanding verified experimentally with real model responses, not mocked

## References

- **Design Analysis**: [Multi-modal Function Response Test Results](../design/multi-modal-function-response.md)
- **Implementation**: [ToolResult.to_parts()](../../src/adkx/tools/base_tool.py#L116-L201)
- **Test Image**: Border Collie JPEG (500x342, ~57KB) - correctly identified by both backends
