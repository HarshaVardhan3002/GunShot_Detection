# Contributing to Gunshot Localization System

Thank you for your interest in contributing to the Gunshot Localization System! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Code Style](#code-style)
8. [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of signal processing and audio systems
- Familiarity with NumPy, SciPy, and audio processing concepts

### Areas for Contribution

- **Algorithm Improvements**: Enhanced detection algorithms, noise filtering
- **Performance Optimization**: Speed improvements, memory optimization
- **Hardware Support**: Additional audio interface support, microphone types
- **Documentation**: User guides, API documentation, tutorials
- **Testing**: Unit tests, integration tests, performance benchmarks
- **Bug Fixes**: Issue resolution, stability improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/GunShot_Detection.git
cd GunShot_Detection
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files (optional)
pre-commit run --all-files
```

### 4. Verify Setup

```bash
# Run tests to ensure everything works
python -m pytest tests/ -v

# Run system validation
python calibrate_system.py --config config/default_config.json
```

## Making Changes

### 1. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 2. Development Guidelines

#### Code Organization

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Follow the existing project structure

#### Performance Considerations

- Profile code changes for performance impact
- Consider real-time processing requirements
- Optimize for low latency where possible
- Test memory usage under load

#### Error Handling

- Use appropriate exception types
- Provide meaningful error messages
- Implement graceful degradation where possible
- Log errors appropriately

### 3. Commit Guidelines

```bash
# Make atomic commits with clear messages
git add specific_files
git commit -m "Add feature: brief description

Detailed explanation of what was changed and why.
Include any breaking changes or migration notes."
```

#### Commit Message Format

```
<type>: <brief description>

<detailed description>

<footer>
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_gunshot_detector.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run performance tests
python tests/run_performance_tests.py
```

### Writing Tests

#### Unit Tests

```python
import unittest
from gunshot_detector import AmplitudeBasedDetector

class TestGunshotDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AmplitudeBasedDetector(
            sample_rate=48000,
            channels=8,
            threshold_db=-25.0
        )

    def test_detection_with_valid_input(self):
        # Test implementation
        pass
```

#### Integration Tests

- Test complete workflows
- Verify component interactions
- Test with realistic data
- Measure performance metrics

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Tests should be deterministic and fast
- Use mocks for external dependencies

## Submitting Changes

### 1. Pre-submission Checklist

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Performance impact is acceptable
- [ ] No breaking changes (or properly documented)

### 2. Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request on GitHub
# Include:
# - Clear description of changes
# - Link to related issues
# - Screenshots/examples if applicable
# - Performance impact notes
```

### 3. Pull Request Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance tests pass

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use Black for code formatting
- Use isort for import sorting
- Maximum line length: 88 characters

### Formatting Tools

```bash
# Format code
black .

# Sort imports
isort .

# Check style
flake8 .

# Type checking
mypy .
```

### Naming Conventions

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Documentation Style

```python
def calculate_tdoa(self, audio_channels: np.ndarray) -> np.ndarray:
    """
    Calculate Time Difference of Arrival between microphone pairs.

    Args:
        audio_channels: Multi-channel audio data (samples, channels)

    Returns:
        TDoA matrix for all microphone pairs

    Raises:
        ValueError: If audio data shape is invalid

    Example:
        >>> localizer = TDoALocalizer(mic_positions)
        >>> tdoa_matrix = localizer.calculate_tdoa(audio_data)
    """
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings, inline comments
2. **User Documentation**: README, installation guides
3. **API Documentation**: Function/class references
4. **Developer Documentation**: Architecture, contributing guides

### Documentation Standards

- Use clear, concise language
- Include examples where helpful
- Keep documentation up-to-date with code changes
- Use proper Markdown formatting

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html
```

## Issue Reporting

### Bug Reports

Include:

- System information (OS, Python version)
- Hardware configuration
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs
- Configuration files (if relevant)

### Feature Requests

Include:

- Use case description
- Proposed solution
- Alternative solutions considered
- Implementation complexity estimate

## Review Process

### Code Review Criteria

- Functionality correctness
- Performance impact
- Code quality and style
- Test coverage
- Documentation completeness
- Security considerations

### Review Timeline

- Initial review: Within 3-5 days
- Follow-up reviews: Within 2 days
- Merge: After approval from maintainers

## Getting Help

### Communication Channels

- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: General questions, ideas
- Code Comments: Implementation-specific questions

### Resources

- [Project Documentation](docs/)
- [Hardware Setup Guide](docs/HARDWARE_SETUP.md)
- [Installation Guide](docs/INSTALLATION_README.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

## Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes for significant contributions
- GitHub contributor statistics

Thank you for contributing to the Gunshot Localization System!
