# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run: `uv run main.py`
- Format: `ruff format .`
- Add packages: `uv add <package_name>`

## Code Style Guidelines
- Use Python 3.12+ features
- **Imports**: Group standard library, third-party, and local imports with a blank line between groups
- **Formatting**: Follow PEP 8 conventions; use ruff for formatting
- **Types**: Use type hints for all function parameters and return values
- **Naming**: Use snake_case for variables/functions, CamelCase for classes, UPPER_CASE for constants
- **Error handling**: Use try/except with specific exception types
- **Documentation**: Document classes and functions with docstrings

## Project Structure
This is a sequence-to-sequence translation model with attention for machine translation between languages (English-French). Follow steps in start.md to implement data loading, encoder-decoder architecture, attention mechanism, and training/evaluation procedures.