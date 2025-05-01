# Contributing to Seq2Seq

Thank you for considering contributing to this project! Here's how you can help.

## Development Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd seq2seq
   ```

2. Install development dependencies:
   ```bash
   uv add -d pytest ruff
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Code Style

This project uses [ruff](https://github.com/charliermarsh/ruff) for code formatting and linting. Please ensure your code follows these guidelines:

- Use 4 spaces for indentation
- Maximum line length is 100 characters
- Follow PEP 8 conventions
- Use type hints for function parameters and return values
- Document classes and functions with docstrings

To format your code:

```bash
ruff format .
```

To lint your code:

```bash
ruff check .
```

## Testing

This project uses pytest for testing. Please write tests for new features and ensure existing tests pass:

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_model.py
```

## Pull Request Process

1. Fork the repository and create a branch for your feature or fix
2. Add tests that cover your changes
3. Ensure all tests pass
4. Format your code using ruff
5. Submit a pull request with a clear description of the changes

## Adding New Features

When adding new features, please:

1. Start by writing tests that define the expected behavior
2. Implement the feature
3. Document the feature in the docstrings and update the API.md file if necessary
4. Add examples if appropriate

## Reporting Issues

When reporting issues, please include:

1. A clear description of the issue
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (Python version, OS, etc.)
6. If possible, a minimal code example that reproduces the issue