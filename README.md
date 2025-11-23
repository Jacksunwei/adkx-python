# adkx - Extensions for Google ADK

[![PyPI status](https://img.shields.io/pypi/status/adkx.svg)](https://pypi.org/project/adkx/)
[![PyPI version](https://badge.fury.io/py/adkx.svg)](https://badge.fury.io/py/adkx)
[![Python Versions](https://img.shields.io/pypi/pyversions/adkx.svg)](https://pypi.org/project/adkx/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![Tests](https://github.com/Jacksunwei/adkx-python/actions/workflows/test.yml/badge.svg)](https://github.com/Jacksunwei/adkx-python/actions/workflows/test.yml)
[![Lint](https://github.com/Jacksunwei/adkx-python/actions/workflows/lint.yml/badge.svg)](https://github.com/Jacksunwei/adkx-python/actions/workflows/lint.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**Production-ready extensions and experimental features for Google's Agent Development Kit**

This package provides production-ready extensions and utilities built on top of [Google ADK](https://github.com/google/adk-python). It also serves as a testing ground for experimental features that may introduce breaking changes to ADK 1.x. Successful features may be upstreamed to the official ADK.

## Status

🚧 **Early Development** - APIs are subject to change

## Features

### Multi-Model Support

- **Gemini Models** - Enhanced Gemini integration with streaming support
- **OpenAI-Compatible Providers** - Unified interface for:
  - OpenAI (GPT-4o, GPT-4 Turbo)
  - Local Ollama (Qwen, Llama, etc.)
  - Ollama on Google Cloud Run

### Advanced Tool System

- **FunctionTool** - Convert Python functions to ADK tools with automatic schema generation
- **Multi-Modal Responses** - Tools can return images, files, and structured data ([ADR-0002](docs/decisions/0002-multi-modal-function-response.md))

### Developer Experience

- **Type Safety** - Full type hints and Pydantic models throughout
- **Async First** - Built for async/await workflows
- **Streaming** - Efficient streaming for all supported models

## Installation

```bash
pip install adkx
```

## Quick Start

### Simple Weather Agent

```python
from adkx.agents.agent import Agent
from adkx.tools.function_tool import FunctionTool, ToolResult


async def get_weather(location: str) -> ToolResult:
    """Get current weather for a location."""
    # Your weather API logic here
    return ToolResult(details=[f"Weather in {location}: Sunny, 72°F"])


agent = Agent(
    model="gemini-2.0-flash-exp",
    name="WeatherBot",
    description="A helpful weather assistant",
    tools=[FunctionTool(get_weather)],
)
```

### Multi-Modal Image Generation

```python
from google.genai import types
from adkx.tools.function_tool import FunctionTool, ToolResult


async def create_image(prompt: str) -> ToolResult:
    """Generate an image from a text description."""
    # Generate image using Gemini
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

### Using OpenAI-Compatible Models

```python
from adkx.agents.agent import Agent
from adkx.models.openai_compatible_providers import OpenAI, Ollama

# OpenAI
agent = Agent(model=OpenAI(model="gpt-4o"), tools=[...])

# Local Ollama
agent = Agent(model=Ollama(model="qwen3:8b"), tools=[...])
```

## Samples

Complete working examples are available in the [`samples/`](samples/) directory:

- **[weather_agent](samples/weather_agent/)** - Basic tool usage with function calling
- **[search_agent](samples/search_agent/)** - Google Search integration
- **[image_agent](samples/image_agent/)** - Multi-modal responses with image generation
- **[weather_agent_openai](samples/weather_agent_openai/)** - Using OpenAI models
- **[weather_agent_ollama](samples/weather_agent_ollama/)** - Using local Ollama

## Documentation

- **[Architecture Decisions](docs/decisions/)** - ADRs documenting key design choices
- **[Research](docs/research/)** - Technical analysis and comparisons
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup and guidelines
- **[AGENTS.md](AGENTS.md)** - Python best practices and patterns for AI-assisted coding

## Model Compatibility

| Provider       | Models                           | Streaming | Function Calling |
| -------------- | -------------------------------- | --------- | ---------------- |
| Gemini         | 2.0 Flash, 2.5 Flash, Pro        | ✅        | ✅               |
| OpenAI         | GPT-4o, GPT-4 Turbo, GPT-4o mini | ✅        | ✅               |
| Ollama (Local) | Qwen, Llama, Mistral, DeepSeek   | ✅        | ✅               |
| Ollama (Cloud) | Same as local, hosted on GCP     | ✅        | ✅               |

## Development

```bash
# Clone repository
git clone https://github.com/Jacksunwei/adkx-python.git
cd adkx-python

# Setup environment
uv venv --python python3.11 .venv
source .venv/bin/activate

# Install dependencies
uv sync --all-extras

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
./autoformat.sh
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Code style requirements
- Testing guidelines
- PR submission process

## Feedback

- 💬 [GitHub Discussions](https://github.com/Jacksunwei/adkx-python/discussions) - Questions and ideas
- 🐛 [GitHub Issues](https://github.com/Jacksunwei/adkx-python/issues) - Bug reports
- 📧 Direct feedback: jacksunwei@gmail.com

## License

Apache 2.0 - See [LICENSE](LICENSE)
