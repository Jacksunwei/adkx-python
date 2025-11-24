---
type: Technical Analysis
date: 2025-11-16
status: Implemented
category: Tools
tags: [multi-modal, gemini, function-response, visual-understanding]
---

# FunctionResponse Multi-modal Content Test Results

## Summary

Tested 3 approaches for returning images from function calls with normal Gemini models (gemini-2.5-flash).

**Key Findings:**

1. **Standalone Parts (Test 3)** ⭐ — Works on both backends with visual understanding
1. **FunctionResponse.parts** — Only for Computer Use models, NOT normal Gemini
1. **Backend differences exist:**
   - Google AI API: Understands base64 in response dicts; rejects `display_name`
   - Vertex AI: Cannot see base64 images; supports `display_name` parameter

## Test Results

| Test                          | Approach               | Google AI API  | Vertex AI      | Recommendation    |
| ----------------------------- | ---------------------- | -------------- | -------------- | ----------------- |
| **1. FunctionResponse.parts** | Image in `parts` field | ❌ 400 error   | ❌ 400 error   | Computer Use only |
| **2. Base64 in response**     | Encoded image in dict  | ✅ Full visual | ⚠️ No visual   | Backend-specific  |
| **3. Standalone Part**        | Image as separate Part | ✅ Works\*     | ✅ Full visual | **RECOMMENDED**   |

\*Google AI API requires omitting `display_name` parameter

### Test 1: FunctionResponse.parts (FAILED)

```python
types.FunctionResponse(
    name="generate_image",
    response={"description": "Generated image of a dog"},
    parts=[
        types.FunctionResponsePart.from_bytes(data=image_data, mime_type="image/jpeg")
    ],
)
```

**Error:** `400 INVALID_ARGUMENT. 'function_response.parts is not supported for this model.'`

### Test 2: Base64 in Response Dict (BACKEND-DEPENDENT)

```python
types.FunctionResponse(
    name="generate_image",
    response={
        "description": "...",
        "image": base64_encoded_image,
        "mime_type": "image/jpeg",
    },
)
```

- **Google AI API:** ✅ Full visual understanding

  > "The dog in the image appears to be a **Golden Retriever**. Here's why: Golden Coat, Friendly Expression, Build and Features..."

- **Vertex AI:** ⚠️ No visual analysis, appears to guess from text

  > "This dog appears to be a **Goldendoodle**. Its medium size, shaggy/fluffy light brown coat..." (incorrect - actual image is black/white Border Collie)

### Test 3: Standalone Part (RECOMMENDED) ⭐

```python
types.Content(
    role="function",
    parts=[
        types.Part(function_response=types.FunctionResponse(...)),
        types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
        # Note: Omit display_name for cross-backend compatibility
    ],
)
```

- **Google AI API:** ✅ Correct breed identification

  > "I'm going to guess this adorable puppy is a **Border Collie**. The black and white coloring with the distinctive white blaze on the face and a fluffy, medium-length coat are all strong indicators..."

- **Vertex AI:** ✅ Detailed visual analysis with reasoning

  > "Based on the image, the dog appears to be a **Border Collie** (likely a puppy). Key characteristics: Black and white coat, distinct white blaze on face, medium-length fluffy fur, intelligent expression..."

**Critical:** Google AI API rejects `display_name` parameter with ValueError

## Implementation Recommendation

Current `ToolResult.to_function_response()` puts Blob/FileData in `FunctionResponse.parts`, which fails for normal Gemini models.

**Recommended approach:** Add `to_content_parts()` method that returns standalone Parts.

```python
def to_content_parts(self, *, function_name: str) -> list[Part]:
    """Convert ToolResult to Content parts for LLM consumption."""
    parts = [Part(function_response=self.to_function_response(name=function_name))]

    for item in self.content:
        if isinstance(item, Blob):
            parts.append(Part.from_bytes(data=item.data, mime_type=item.mime_type))
        elif isinstance(item, FileData):
            parts.append(
                Part.from_uri(file_uri=item.file_uri, mime_type=item.mime_type)
            )

    return parts
```

**Options:**

1. Return `tuple[FunctionResponse, list[Part]]` from existing method
1. Add new `to_content_parts()` method (recommended for backward compatibility)

## Environment

- **Model:** gemini-2.5-flash
- **Backends:** Google AI API & Vertex AI
- **Test Image:** Border Collie JPEG (500x342, ~57KB)
- **Date:** 2025-11-16
