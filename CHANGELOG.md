# Changelog

All notable changes to SmallCoder will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-02-19

### Added
- **Extended Context Length**: Increased from 4096 to 8192 tokens (2x improvement)
  - Better handling of longer code files
  - Improved multi-file project understanding
  - Enhanced conversation context maintenance

- **Comprehensive Coding Task Test Suite** (`code_tasks.py`)
  - 15+ coding task benchmarks across 11 categories
  - Algorithm implementation tests (sorting, searching, graph algorithms)
  - Data structure tests (linked lists, trees, hash tables)
  - Debugging and bug fixing tasks
  - Code refactoring challenges
  - Code explanation tasks
  - Web development tests (REST APIs)
  - Async programming tests
  - Error handling tests
  - Unit testing tasks
  - Database query tests
  - Design pattern implementation tests
  - Category-based filtering
  - Detailed performance metrics and JSON output

- **Lightweight Coding Agent** (`coding_agent.py`)
  - Multi-step problem solving with iterative refinement
  - Automatic code execution and testing
  - Code quality analysis with issue detection
  - Intelligent suggestions for improvements
  - Interactive agent mode
  - Support for custom test cases
  - Configurable iteration limits
  - Safe code execution in isolated environment

- **New Demo Script** (`demo_new_features.py`)
  - Showcases all new features
  - Provides usage examples
  - Interactive feature walkthrough

### Changed
- Updated README.md with new feature documentation
- Enhanced model configuration to support extended context
- Updated run_model.py to reflect new context length

### Technical Details
- Model configuration: `max_position_embeddings` increased from 4096 to 8192
- No changes to model parameter count (~304M)
- Maintained backward compatibility with existing checkpoints
- All new features work with 2GB VRAM constraint

## [1.0.0] - Previous Release

### Features
- Initial SmallCoder release with ~304M parameters
- Grouped-Query Attention (GQA) for memory efficiency
- RoPE (Rotary Position Embeddings)
- SwiGLU activation function
- RMSNorm for stable training
- Pre-trained model generation
- Interactive inference mode
- INT8 quantization support
- Optimized for 2GB VRAM + 8GB RAM
- Context length: 4096 tokens
- Basic benchmarking suite
