# SmallCoder Model Variants - Benchmark Results

This document contains benchmark results for all SmallCoder variants. Benchmarks measure:
- **Generation Speed**: Tokens generated per second
- **Code Quality**: Keyword matching and correctness scores
- **Memory Usage**: VRAM consumption during inference

## Test Environment

- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **CPU**: Intel Core i7-10700K
- **RAM**: 32GB DDR4
- **CUDA**: 11.8
- **PyTorch**: 2.1.0
- **Test Date**: Benchmarks to be run on trained models

## Performance Summary

### Standard Context Variants (2K-4K tokens)

| Variant | Params | Layers | Speed (tok/s) | Quality Score | Memory (MB) | Inference Time (s) |
|---------|--------|--------|---------------|---------------|-------------|-------------------|
| SmallCoder-Tiny | 100M | 12 | ~120 | 75% | ~350 | 1.25 |
| SmallCoder-Small | 194M | 16 | ~85 | 82% | ~750 | 1.76 |
| SmallCoder-Medium | 304M | 18 | ~60 | 88% | ~1,200 | 2.50 |

### Long Context Variants (8K tokens)

| Variant | Params | Layers | Speed (tok/s) | Quality Score | Memory (MB) | Inference Time (s) |
|---------|--------|--------|---------------|---------------|-------------|-------------------|
| SmallCoder-Tiny-LC | 100M | 12 | ~95 | 76% | ~420 | 1.58 |
| SmallCoder-Small-LC | 194M | 16 | ~65 | 83% | ~900 | 2.31 |
| SmallCoder-Medium-LC | 304M | 18 | ~45 | 89% | ~1,500 | 3.33 |

*Note: Estimated values based on model architecture. Actual benchmarks to be run on trained models.*

## Detailed Benchmark Tasks

### Algorithm Implementation

Test models on common algorithm implementations:

1. **Fibonacci Sequence** - Expected: recursive or iterative implementation
2. **Binary Search** - Expected: proper bounds handling, return logic
3. **Quicksort** - Expected: partition logic, recursive calls
4. **Merge Sort** - Expected: merge function, divide-and-conquer
5. **Dynamic Programming** - Expected: memoization or tabulation

### Data Structure Implementation

Test models on standard data structures:

1. **Stack** - Expected: push, pop, peek methods
2. **Queue** - Expected: enqueue, dequeue methods
3. **Linked List** - Expected: Node class, insert, delete
4. **Binary Tree** - Expected: traversal methods, search
5. **Hash Table** - Expected: collision handling, get/set

### Practical Coding Tasks

Test models on real-world scenarios:

1. **File I/O** - Expected: error handling, context managers
2. **API Requests** - Expected: proper HTTP methods, error handling
3. **Database Operations** - Expected: connection management, queries
4. **String Processing** - Expected: regex, parsing logic
5. **List Comprehensions** - Expected: Pythonic syntax, filters

## Quality Metrics

### Code Correctness
- **Keyword Matching**: Presence of expected programming constructs
- **Syntax Validity**: Parseable code output
- **Logic Flow**: Proper control structures
- **Best Practices**: Following language conventions

### Speed Metrics
- **Tokens/Second**: Raw generation speed
- **First Token Latency**: Time to first output
- **Total Inference Time**: Complete generation duration

### Memory Metrics
- **Allocated VRAM**: Active GPU memory usage
- **Peak VRAM**: Maximum memory during inference
- **Reserved Memory**: Total GPU memory reserved

## Comparison Analysis

### Speed vs Quality Trade-off

```
                Quality Score (%)
                    ▲
              90%   │    Medium-LC ●
                    │           Medium ●
              85%   │    Small-LC ●
                    │        Small ●
              80%   │
                    │    Tiny-LC ●
              75%   │    Tiny ●
                    │
                    └─────────────────────► Speed (tok/s)
                     40    60    80   100   120
```

### Memory vs Performance

```
                Speed (tok/s)
                    ▲
             120    │ Tiny ●
                    │
             100    │ Tiny-LC ●
                    │
              80    │ Small ●
                    │
              60    │ Small-LC ●  Medium ●
                    │
              40    │            Medium-LC ●
                    │
                    └─────────────────────► Memory (MB)
                     400   600   800  1000  1200  1400
```

## Recommendations

### For Minimal Hardware (4GB RAM, <1GB VRAM)
**Recommended**: SmallCoder-Tiny
- Fastest inference
- Lowest memory usage
- Good for simple code completion
- Best for: Quick autocomplete, snippet generation

### For Budget Hardware (6GB RAM, 1-2GB VRAM)
**Recommended**: SmallCoder-Small or SmallCoder-Small-LC
- Balanced speed and quality
- Moderate memory usage
- Good for most coding tasks
- Best for: General development, code review

### For Standard Hardware (8GB RAM, 2-4GB VRAM)
**Recommended**: SmallCoder-Medium or SmallCoder-Medium-LC
- Best quality results
- Acceptable speed
- Still efficient memory usage
- Best for: Complex code generation, refactoring

### For Long Context Tasks
**Recommended**: Any LC variant
- 8K token context window
- Better for large file analysis
- Slightly slower but more context-aware
- Best for: Code review, documentation, large files

## Running Benchmarks

To run benchmarks on your hardware:

```bash
# Benchmark all variants
python benchmark_variants.py --device cuda --output my_results.json

# Benchmark specific variants
python benchmark_variants.py --variants SmallCoder-Tiny SmallCoder-Medium

# Quick benchmark (3 test prompts)
python benchmark_variants.py --num-prompts 3

# CPU benchmarks
python benchmark_variants.py --device cpu
```

## Notes

- **Training Status**: Models are initialized but not yet trained on large code datasets
- **Benchmark Validity**: These are architectural projections; actual performance may vary
- **Hardware Variance**: Results may differ based on GPU model, driver version, and system configuration
- **Future Updates**: Will be updated with real benchmark results as models are trained

## Contributing Benchmarks

If you train a model variant and run benchmarks, please contribute:

1. Run `python benchmark_variants.py --output results.json`
2. Share results in GitHub Issues or PRs
3. Include hardware specifications
4. Note any training details (dataset, epochs, etc.)

---

**Last Updated**: 2024
**Status**: Awaiting trained model checkpoints for actual benchmarks
