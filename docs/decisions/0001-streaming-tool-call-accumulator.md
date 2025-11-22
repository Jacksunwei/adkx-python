# ADR-0001: Tool Call Streaming Incompatibility Across OpenAI-Compatible Providers

**Date**: 2025-01-21
**Status**: Accepted

## Context

**OpenAI-compatible providers stream tool calls inconsistently**. For parallel tool calls (`get_weather` and `get_time`):

| **OpenAI**                                          | **Ollama**                                                        |
| --------------------------------------------------- | ----------------------------------------------------------------- |
| Chunk 1: `index=0, id="call_1", name="get_weather"` | Chunk 1: `index=0, id="call_1", name="get_weather", args="{...}"` |
| Chunk 2: `index=0, id=null, args="{...}"`           | Chunk 2: `index=0, id="call_2", name="get_time", args="{...}"`    |
| Chunk 3: `index=1, id="call_2", name="get_time"`    |                                                                   |
| Chunk 4: `index=1, id=null, args="{...}"`           |                                                                   |

OpenAI uses `index` to track calls (0, 1, 2...) with `id=null` in continuations. Ollama reuses `index=0` for multiple calls but uses different `id` values.

**Requirements**: Support both streaming patterns, maintain real-time UX for text/thought, preserve compatibility with existing providers (OpenAI, Groq, OllamaCloudRun), allow future provider variants

## Decision

Introduce `AssistantTurnAccumulator` to handle provider-specific streaming differences. The base implementation follows OpenAI's official spec (index-based tracking with `id=null` continuations), while providers like Ollama override `_process_tool_calls()` to implement their ID-based tracking quirk.

**Streaming behavior**:

- **Text/thought**: Yields partial responses immediately for real-time UX
- **Tool calls**: Buffers silently until complete, included only in final response (they complete before `finish_reason`, so streaming adds no user benefit)

**Provider integration**:

```python
# Base: OpenAI spec-compliant accumulator
class OpenAICompatibleLlm(BaseLlm):
    accumulator_class: ClassVar = AssistantTurnAccumulator  # Subclass-overridable


# Ollama: Override for ID-based tracking quirk
class Ollama(OpenAICompatibleLlm):
    accumulator_class: ClassVar = OllamaAssistantTurnAccumulator
```

**Design choices**:

- **Self-contained**: Accumulator manages all state/logic internally, simplifying `_generate_content_streaming()` from 80 to 20 lines
- **Generator pattern**: `process_chunk()` yields text immediately, buffers tool calls internally
- **ClassVar for accumulator**: Not per-instance configurable (not a pydantic field), but subclass-overridable for provider-specific behavior

## Consequences

**Benefits**:

- OpenAI spec-compliant base with single-method override (`_process_tool_calls()`) for provider quirks
- Streaming logic isolated in accumulator, provider quirks isolated in subclasses
- Independently testable accumulator (24/24 tests passing)

**Trade-offs**:

- Custom providers must understand accumulator pattern to override `_process_tool_calls()` for provider-specific streaming quirks
- No partial tool call progress (acceptable - tool calls arrive in full before stream ends with `finish_reason`)

## References

- **OpenAI Official Docs**: [Function Calling - Streaming](https://platform.openai.com/docs/guides/function-calling#streaming)
- **Experimental verification**: [test_openai_streaming.py](/experiments/test_openai_streaming.py), [test_ollama_streaming.py](/experiments/test_ollama_streaming.py)
- **Implementation**: [AssistantTurnAccumulator](/src/adkx/models/openai_compatible.py#L79-L282)
- **Ollama override**: [OllamaAssistantTurnAccumulator](/src/adkx/models/openai_compatible_providers/ollama.py#L32-L109)
