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

This project follows the same style guide as main ADK.

### Automated Formatting with Pre-commit (Recommended)

Pre-commit is included in dev dependencies. After running `uv sync --all-extras`, install the hooks:

```bash
pre-commit install
```

This will automatically run checks before each commit:

- **pyink**: Code formatting (Google style)
- **isort**: Import sorting
- **trailing-whitespace**: Remove trailing spaces
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML syntax
- **check-added-large-files**: Prevent large files (>1MB)
- **check-merge-conflict**: Detect merge conflict markers
- **detect-private-key**: Prevent committing secrets

To run manually on all files:

```bash
pre-commit run --all-files
```

### Manual Formatting

Use the autoformat script to format code:

```bash
# Format all default directories (src/, tests/, samples/)
./autoformat.sh

# Format specific paths
./autoformat.sh src/
./autoformat.sh samples/weather_agent.py
./autoformat.sh src/ tests/

# Or run formatters manually
isort src/ tests/ samples/
pyink src/ tests/ samples/
```

### Style Requirements

- Line length: 80 characters
- Indentation: 2 spaces
- Imports: sorted with isort (Google profile)
- Type hints: use `from __future__ import annotations`

### AI-Assisted Development

[AGENTS.md](AGENTS.md) contains comprehensive Python best practices and coding patterns that can be used to bootstrap AI-assisted coding (e.g., with Claude Code, Gemini CLI, GitHub Copilot, or Cursor). This file serves as:

- **Context for AI assistants** - Coding standards, patterns, and anti-patterns
- **Reference guide** - Python best practices, testing strategies, and debugging tips
- **Project conventions** - Naming, structure, and design decisions specific to adkx

When working with AI coding assistants, reference AGENTS.md to ensure generated code follows project conventions.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=adkx --cov-report=html
```

## Submitting Changes

1. **Fork the repository**
1. **Create a feature branch** (`git checkout -b feature/amazing-idea`)
1. **Make your changes** and add tests
1. **Format your code** (run `./autoformat.sh` or `pre-commit run --all-files`)
1. **Run tests** (`pytest tests/`)
1. **Commit** with descriptive message
1. **Push** to your fork
1. **Open a Pull Request** with a clear description of your changes

### CI Checks

All pull requests must pass automated checks:

- **Lint**: Pre-commit hooks (formatting, imports, file checks)
- **Tests**: pytest with coverage

CI runs the same pre-commit checks as your local environment, so running `pre-commit run --all-files` before pushing ensures your PR will pass.

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
