# Contributing to SmallCoder

Thank you for your interest in contributing to SmallCoder! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/smallLLM.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests (if applicable)
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests (if available)
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small
- Add comments for complex logic

## Areas for Contribution

We welcome contributions in the following areas:

### 1. Model Optimizations
- Implement additional quantization schemes (INT4, mixed precision)
- Add support for Flash Attention 2
- Optimize memory usage during training
- Implement LoRA/QLoRA for parameter-efficient fine-tuning

### 2. Training Improvements
- Add support for distributed training
- Implement curriculum learning
- Add more data preprocessing options
- Create better dataset utilities

### 3. Inference Enhancements
- Improve generation speed
- Add beam search decoding
- Implement speculative decoding
- Add support for ONNX export

### 4. Evaluation & Benchmarking
- Add more coding benchmarks (HumanEval, MBPP)
- Implement automated testing
- Add quality metrics (BLEU, CodeBLEU)
- Create performance profiling tools

### 5. Documentation
- Improve README with more examples
- Add tutorials and guides
- Create video demonstrations
- Translate documentation to other languages

### 6. Hardware Support
- Add Apple Silicon (MPS) support
- Optimize for AMD GPUs (ROCm)
- Add Intel GPU support
- Improve CPU inference

## Pull Request Guidelines

1. **Keep PRs focused**: One feature or fix per PR
2. **Write clear descriptions**: Explain what and why, not just how
3. **Update documentation**: If you change APIs, update docs
4. **Add tests**: If adding features, add corresponding tests
5. **Follow code style**: Match the existing code style

## Testing

Before submitting a PR, please:

1. Test your changes locally
2. Ensure the model still trains/infers correctly
3. Check for memory leaks or performance regressions
4. Verify compatibility with the target hardware

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: How to reproduce the problem
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: OS, Python version, PyTorch version, GPU model
6. **Error messages**: Full error messages and stack traces

## Feature Requests

We welcome feature requests! Please:

1. Check if the feature already exists
2. Explain the use case
3. Describe the proposed solution
4. Consider alternatives

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions, please:

1. Check the documentation
2. Search existing issues
3. Open a new issue with your question

Thank you for contributing to SmallCoder! 🚀
