# GitHub Copilot Instructions for EasyReflectometryLib

## Project Overview

EasyReflectometryLib is a reflectometry Python package built on the
EasyScience framework. It provides tools for reflectometry analysis and
modeling.

## Development Environment

- **Python Versions**: 3.11, 3.12
- **Supported Platforms**: Linux (ubuntu-latest), macOS (macos-latest),
  Windows (windows-latest)
- **Package Manager**: pip
- **Build System**: hatchling with setuptools-git-versioning

## Code Style and Formatting

### Ruff Configuration

- Use **Ruff** for linting and formatting (configured in
  `pyproject.toml`)
- Maximum line length: 127 characters
- Quote style: single quotes for strings
- Import style: force single-line imports
- To fix issues automatically: `python -m ruff . --fix`

### Code Quality Standards

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write clear, self-documenting code with meaningful variable names
- Maintain consistency with existing code patterns in the repository

### Linting Rules

The project uses Ruff with the following rule sets:

- `E9`, `F63`, `F7`, `F82`: Critical flake8 rules
- `E`: pycodestyle errors
- `F`: Pyflakes
- `I`: isort (import sorting)
- `S`: flake8-bandit (security checks)

Special notes:

- Asserts are allowed in test files (`*test_*.py`)
- Init module imports are ignored
- Exclude `docs` directory from linting

## Testing

### Test Framework

- Use **pytest** for all tests
- Test coverage should be tracked with **pytest-cov**
- Aim for comprehensive test coverage
- Tests are located in the `tests/` directory

### Running Tests

```bash
# Install dev dependencies
pip install -e '.[dev]'

# Run tests with coverage
pytest --cov --cov-report=xml

# Run tests using tox (for multiple Python versions)
pip install tox tox-gh-actions
tox
```

### Test Guidelines

- Write unit tests for all new functionality
- Include tests when fixing bugs to prevent regression
- Test files should match the pattern `test_*.py`
- Use descriptive test function names that explain what is being tested
- Follow the existing test structure and patterns in the repository

## Security

- Follow flake8-bandit security guidelines (enabled via Ruff `S` rules)
- Be cautious with user input and file operations
- Do not commit secrets or sensitive information
- Review security implications of all changes

## Documentation

### Docstring Style

- Include docstrings for all public modules, classes, and functions
- Use **NumPy style** docstrings (`Parameters` / `Returns` / `Raises`
  sections), the style `mkdocstrings` is configured to parse
- Use clear, concise descriptions
- Document parameters, return values, and exceptions
- Example format:

  ```python
  """Brief description of the function.

  Parameters
  ----------
  param_name : type
      Description of the parameter.

  Returns
  -------
  type
      Description of the return value.
  """
  ```

### Documentation Build

- Documentation is built with MkDocs (Material theme), configured in
  `docs/mkdocs.yml`
- Source files are Markdown and Jupyter notebooks under `docs/docs/`;
  every page must be listed in the `nav` section of `docs/mkdocs.yml`
- API reference pages are one-liners rendered by `mkdocstrings`
  (`::: easyreflectometry.<module>`)
- Build locally with `pixi run docs-build` or preview with
  `pixi run docs-serve`
- Include code examples in documentation where appropriate

## Dependencies

### Core Dependencies

- easyscience (EasyScience framework)
- scipp (Scientific computing)
- refnx, refl1d (Reflectometry calculations)
- orsopy (Data format support)
- bumps (Optimization)

### Adding New Dependencies

- Only add dependencies when absolutely necessary
- Add to appropriate section in `pyproject.toml`:
  - `dependencies` for core runtime dependencies
  - `dev` for development tools
  - `docs` for documentation building
- Document why the dependency is needed

## Git and Version Control

### Commit Messages

- Write clear, descriptive commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issue numbers when applicable

### Branch Workflow

- Create feature branches from the main branch
- Use descriptive branch names (e.g., `feature/add-new-calculator`,
  `bugfix/fix-reflection-calculation`)
- Keep changes focused and atomic

## Pull Request Guidelines

1. Include tests for new functionality
2. Update documentation if adding or changing features
3. Ensure all CI checks pass:
   - Code consistency (Ruff)
   - Code testing (pytest on all supported platforms/versions)
   - Package building
4. Code should work on Python 3.11, 3.12 and all supported platforms
5. Write a clear PR description explaining the changes

## Project Structure

```
src/easyreflectometry/     # Main package source code
├── calculators/           # Calculator implementations (refnx, refl1d)
│   └── bornagain/         # BornAgain calculator (not yet functional)
├── model/                 # Reflectometry models
├── sample/                # Sample structures and materials
├── special/               # Special calculations and parsing
├── summary/               # Summary generation
└── project.py             # Main project interface

tests/                     # Test suite
docs/                      # Documentation source
```

## Best Practices

1. **Minimal Changes**: Make the smallest possible changes to accomplish
   the task
2. **Don't Break Existing Code**: Maintain backward compatibility unless
   explicitly required
3. **Test Before Committing**: Always run tests and linting before
   pushing
4. **Follow Existing Patterns**: Look at similar code in the repository
   for guidance
5. **Ask When Uncertain**: If unsure about an approach, ask for
   clarification

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

- **Code Consistency**: Runs Ruff linting on all pushes and PRs
- **Code Testing**: Runs pytest across multiple Python versions and
  platforms
- **Package Testing**: Validates package building and installation
- **Coverage**: Uploads test coverage to Codecov

All CI checks must pass before merging PRs.

## Special Notes

- The project is part of the EasyScience ecosystem
- Built on top of established reflectometry libraries (refnx, refl1d)
- Focuses on providing a user-friendly interface for reflectometry
  analysis
- Maintains compatibility with multiple calculator backends
