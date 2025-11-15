# Contributing to adkx

Thank you for your interest in contributing to adkx!

## Development Setup

### Prerequisites

- Python 3.11+
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))

### Setup

```bash
git clone https://github.com/Jacksunwei/adkx.git
cd adkx

# Create virtual environment
uv venv --python python3.11 .venv
source .venv/bin/activate

# Install dependencies
uv sync --all-extras
```

## Code Style

This project follows the same style guide as main ADK:

**Auto-format before committing:**
```bash
# Format imports
isort src/ tests/ examples/

# Format code
pyink --config pyproject.toml src/ tests/ examples/
```

**Style requirements:**
- Line length: 80 characters
- Indentation: 2 spaces
- Imports: sorted with isort (Google profile)
- Type hints: use `from __future__ import annotations`

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=adkx --cov-report=html
```

## Submitting Changes

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-idea`)
3. **Make your changes** and add tests
4. **Format your code** (isort + pyink)
5. **Commit** with descriptive message
6. **Push** to your fork
7. **Open a Pull Request** with a clear description of your changes

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
test: Add tests
```

## Questions?

- 💬 [GitHub Discussions](https://github.com/Jacksunwei/adkx-python/discussions)
- 🐛 [GitHub Issues](https://github.com/Jacksunwei/adkx-python/issues)
- 📧 Email: jacksunwei@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
