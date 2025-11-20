# OpenAI-Compatible Provider Analysis

**Date**: 2025-11-19
**Author**: Wei Sun
**Status**: Active

## Summary

Analysis of the current Ollama implementation in ADKX and its potential for generalization to support all OpenAI-compatible LLM providers. Covers technical gaps, popular providers, and recommended architecture for multi-provider support.

## Motivation

The current `Ollama` class ([ollama.py](../../src/adkx/models/ollama.py)) uses the OpenAI Python SDK and implements the OpenAI Chat Completions API. This design is already 95% compatible with other OpenAI-compatible providers (Groq, Together AI, Azure OpenAI, etc.).

**Why investigate now?**

- Reduce code duplication across provider integrations
- Enable users to switch providers without code changes
- Leverage the mature OpenAI SDK ecosystem
- Support emerging high-performance providers (Groq, Cerebras)

## Current Implementation Analysis

### Architecture Overview

The `Ollama` class extends `BaseLlm` and uses `AsyncOpenAI` client:

```python
class Ollama(BaseLlm):
    model: str = "qwen3-coder:30b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "unused"
    use_gcp_auth: bool = False

    @cached_property
    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
```

**Key capabilities:**

- ✅ Streaming and non-streaming generation
- ✅ Tool/function calling
- ✅ Thought/reasoning content (DeepSeek R1)
- ✅ Usage metadata extraction
- ✅ ADK ↔ OpenAI format conversion
- ✅ GCP Cloud Run authentication

### Conversion Layer

**Request conversion** (ADK → OpenAI):

- `Content` → `ChatCompletionMessageParam`
- User/model/tool messages
- Function calls → tool calls
- Tool declarations → OpenAI tools format

**Response conversion** (OpenAI → ADK):

- `ChatCompletion` → `LlmResponse`
- Streaming chunks → partial responses with merging
- Finish reasons mapping
- Usage metadata (including reasoning tokens)

## Findings

### 1. Gaps for Supporting OpenAI Directly

| Component           | Current                       | OpenAI Requirement          | Gap                     |
| ------------------- | ----------------------------- | --------------------------- | ----------------------- |
| **Base URL**        | `http://localhost:11434/v1`   | `https://api.openai.com/v1` | Configuration only      |
| **API Key**         | `"unused"` (hardcoded)        | Actual API key              | Make configurable       |
| **Authentication**  | Bearer token                  | Bearer token                | ✅ Compatible           |
| **Model Names**     | `qwen3-coder:30b`             | `gpt-4`, `gpt-4o`           | Configuration only      |
| **GCP Auth**        | Ollama Cloud Run specific     | Not needed                  | Conditional logic       |
| **Reasoning Field** | `getattr(delta, "reasoning")` | Not exposed (o1/o3)         | Graceful degradation ✅ |

**Verdict**: Only **configuration changes** needed for basic OpenAI support. No architectural changes required.

### 2. Popular OpenAI-Compatible Providers

#### Cloud Inference Providers

1. **OpenAI** - Original API, premium pricing
1. **Azure OpenAI** - Enterprise with Microsoft infrastructure
1. **Google Vertex AI** - Multi-model including OpenAI-compatible endpoints
1. **AWS Bedrock** - Multi-model marketplace

#### High-Speed/Low-Cost Inference

5. **Groq** - Fastest inference (LPU architecture), native OpenAI compatibility
   - Base URL: `https://api.groq.com/openai/v1`
   - Auth: `Authorization: Bearer <api_key>`
1. **Together AI** - Wide model selection, cost-effective
1. **Fireworks AI** - Fast inference, competitive pricing
1. **Anyscale Endpoints** - Serverless LLM hosting
1. **Cerebras** - Ultra-fast inference with custom hardware

#### Aggregators/Routers

10. **OpenRouter** - 300+ models unified API
    - Requires `HTTP-Referer` header for attribution
01. **Perplexity** - Search-augmented models
01. **LiteLLM** - Python library for 100+ providers (not a service)

#### Local/Self-Hosted

13. **Ollama** - Current implementation ✅
01. **LM Studio** - Desktop GUI with OpenAI API
01. **vLLM** - Open-source inference server
01. **LocalAI** - Self-hosted OpenAI alternative

### 3. Technical Gaps for Multi-Provider Support

#### A. Authentication Variations

| Provider          | Auth Pattern                       | Current Support | Implementation Needed      |
| ----------------- | ---------------------------------- | --------------- | -------------------------- |
| OpenAI            | `Authorization: Bearer <api_key>`  | ✅              | None                       |
| Groq              | `Authorization: Bearer <api_key>`  | ✅              | None                       |
| Azure OpenAI      | API key + deployment in URL        | ❌              | Azure-specific URL builder |
| OpenRouter        | Bearer + `HTTP-Referer` header     | ⚠️              | Custom headers support     |
| Anthropic (proxy) | `x-api-key` header                 | ❌              | Header customization       |
| GCP Cloud Run     | `Authorization: Bearer <id_token>` | ✅              | Already implemented        |

**Gap**: Need flexible header customization beyond current `default_headers`.

**Azure OpenAI URL pattern:**

```
https://{resource}.openai.azure.com/openai/deployments/{deployment-id}/chat/completions?api-version={version}
```

#### B. Response Format Differences

**Usage Metadata in Streaming**

- **Current**: TODO comment notes missing `stream_options` support ([ollama.py:160](../../src/adkx/models/ollama.py#L160))
- **OpenAI**: Requires `stream_options={"include_usage": True}` to get usage in final chunk
- **Impact**: Can't get token counts in streaming mode without this
- **Fix**: Add `stream_options` parameter support

**Provider-Specific Fields**

- **DeepSeek R1**: `reasoning` field in responses (currently handled)
- **Perplexity**: `citations` field for search results
- **Impact**: Low - extra fields can be ignored or captured in metadata

**Error Response Formats**

- **Current**: Relies on OpenAI SDK exception handling
- **Gap**: Different providers return different error structures
- **Example**: Azure has different error codes/messages than OpenAI
- **Fix**: Provider-specific error normalization layer

#### C. Feature Support Variations

| Feature         | OpenAI | Ollama             | Groq | Azure | Gap                           |
| --------------- | ------ | ------------------ | ---- | ----- | ----------------------------- |
| Tool calling    | ✅     | ✅                 | ✅   | ✅    | None                          |
| Streaming       | ✅     | ✅                 | ✅   | ✅    | None                          |
| Vision (images) | ✅     | ⚠️ Model-dependent | ⚠️   | ✅    | Multimodal content handling   |
| JSON mode       | ✅     | ⚠️ Model-dependent | ⚠️   | ✅    | Schema validation             |
| Response format | ✅     | ⚠️                 | ❌   | ✅    | Provider capability detection |
| System prompts  | ✅     | ✅                 | ✅   | ✅    | None                          |

**Gap**: No capability detection mechanism. Current implementation assumes all features work.

#### D. Configuration & Client Management

**Current implementation:**

```python
@cached_property
def client(self) -> AsyncOpenAI:
    api_key = self.api_key
    default_headers = None

    if self.use_gcp_auth:
        api_key = self._get_gcp_auth_token()
        default_headers = {"Authorization": f"Bearer {api_key}"}

    return AsyncOpenAI(
        base_url=self.base_url,
        api_key=api_key,
        default_headers=default_headers,
    )
```

**Missing:**

- `organization` parameter (OpenAI teams/orgs)
- `timeout` configuration
- Custom retry policies
- Proxy support
- Token auto-refresh (GCP tokens expire after 1 hour)

#### E. Model Selection & Routing

**Current**: Single model per instance, no fallback logic

**Use cases not supported:**

```python
# 1. Automatic provider fallback
router = OpenAICompatible(
    providers=[
        {"name": "groq", "base_url": "...", "priority": 1},
        {"name": "openai", "base_url": "...", "priority": 2},
    ]
)

# 2. Cost-based routing
llm.generate(routing_strategy="cheapest")  # Use Together AI

# 3. Latency-based routing
llm.generate(routing_strategy="fastest")  # Use Groq
```

**Gap**: No provider abstraction or routing layer.

#### F. Streaming Robustness

**Current**: Basic streaming with accumulation

**Missing:**

- Reconnection on dropped connections
- Partial chunk recovery
- Stream timeout handling
- Provider-specific SSE format handling

## Recommendations

### Minimal Viable Generalization (Quick Win)

**Goal**: Support OpenAI, Groq, Together AI with minimal changes

**Changes needed:**

1. Rename `Ollama` → `OpenAICompatible`
1. Make `base_url`, `api_key`, `model` configurable (remove defaults)
1. Add factory methods:
   ```python
   OpenAICompatible.for_openai(api_key="...")
   OpenAICompatible.for_groq(api_key="...")
   OpenAICompatible.for_ollama(base_url="...")
   ```
1. Move GCP auth to separate strategy/mixin
1. Add `organization`, `timeout`, `max_retries` parameters
1. Support `stream_options` for usage metadata in streaming

**Estimated effort**: 2-4 hours

### Medium-Term Enhancements

**Provider profiles:**

```python
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "auth_type": "bearer",
        "supports_usage_in_streaming": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "auth_type": "bearer",
        "supports_usage_in_streaming": False,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "auth_type": "none",
        "supports_usage_in_streaming": False,
    },
}
```

**Benefits:**

- Automatic capability detection
- Provider-specific optimizations
- Better error messages

**Estimated effort**: 1-2 days

### Long-Term Vision (Full Multi-Provider)

**Architecture:**

1. **Provider abstraction layer** - `ProviderConfig` with auth strategies
1. **Capability detection** - Query `/models` endpoint, feature flags
1. **Error normalization** - Standard error types across providers
1. **Retry/fallback** - Automatic provider switching on failure
1. **Multimodal support** - Images, audio (extends beyond current text-only)
1. **Observability** - Provider-specific metrics, cost tracking
1. **Model routing** - Cost/latency-based provider selection

**Estimated effort**: 1-2 weeks

## Next Steps

1. **Decision needed**:

   - Quick win (rename + factory methods)?
   - Medium-term (provider profiles)?
   - Full architecture redesign?

1. **RFC drafting**: Once direction chosen, draft formal RFC with:

   - Problem statement (user pain with current single-provider approach)
   - Proposed API (code examples)
   - Migration path for existing Ollama users
   - Implementation plan

1. **Prototype**: Build quick proof-of-concept with OpenAI + Groq support

## References

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Groq OpenAI Compatibility](https://console.groq.com/docs/openai)
- [OpenRouter API Reference](https://openrouter.ai/docs/api-reference)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- Current implementation: [src/adkx/models/ollama.py](../../src/adkx/models/ollama.py)
