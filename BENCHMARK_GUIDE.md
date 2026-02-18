# SmallCoder Benchmarking Guide

This guide explains how to benchmark SmallCoder and compare it with other coding models.

## Overview

The benchmarking system provides:
- **Code completion benchmarks**: Test model's ability to complete various coding tasks
- **Model comparison**: Compare SmallCoder against other models (CodeLlama, CodeGen, StarCoder)
- **Performance metrics**: Accuracy (keyword matching), speed (tokens/sec), memory usage
- **Detailed reports**: JSON and Markdown reports with comparison tables

## Quick Start

### 1. Basic Benchmarking (SmallCoder only)

```bash
python benchmark.py \
    --checkpoint ./checkpoints/best_model.pt \
    --tokenizer codellama/CodeLlama-7b-hf \
    --device cuda
```

### 2. Compare with Other Models

```bash
python benchmark.py \
    --checkpoint ./checkpoints/best_model.pt \
    --tokenizer codellama/CodeLlama-7b-hf \
    --compare \
    --comparison-models codegen-350m starcoder-1b \
    --device cuda
```

### 3. Include Additional Coding Challenges

```bash
python benchmark.py \
    --checkpoint ./checkpoints/best_model.pt \
    --compare \
    --include-challenges \
    --device cuda
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint` | Path to SmallCoder checkpoint | None |
| `--tokenizer` | Tokenizer for SmallCoder | `codellama/CodeLlama-7b-hf` |
| `--device` | Device to use (cuda/cpu) | `cuda` |
| `--output-dir` | Output directory for results | `./benchmark_results` |
| `--compare` | Enable comparison mode | False |
| `--comparison-models` | Models to compare (space-separated) | All available |
| `--include-challenges` | Include extra coding challenges | False |
| `--skip-memory-test` | Skip memory measurement | False |

## Available Comparison Models

| Model Key | Model Name | Size | Description |
|-----------|------------|------|-------------|
| `codellama-7b` | CodeLlama 7B | 7B params | Meta's CodeLlama base model |
| `codegen-350m` | CodeGen 350M | 350M params | Salesforce CodeGen |
| `starcoder-1b` | StarCoder 1B | 1B params | BigCode StarCoder base |

## Benchmark Categories

### Core Code Benchmarks (8 tests)

1. **fibonacci** - Recursive function implementation
2. **binary_search** - Search algorithm implementation  
3. **quicksort** - Sorting algorithm with recursion
4. **class_definition** - Object-oriented programming
5. **list_comprehension** - Pythonic list comprehension
6. **error_handling** - Exception handling
7. **async_function** - Asynchronous programming
8. **decorator** - Python decorators

### Additional Coding Challenges (3 tests)

Use `--include-challenges` to enable:

1. **palindrome_check** - String manipulation
2. **merge_sort** - Divide and conquer algorithm
3. **linked_list** - Data structure implementation

## Output Files

The benchmarking system generates two files:

### 1. JSON Results (`benchmark_results_YYYYMMDD_HHMMSS.json`)

Complete benchmark data including:
- Per-benchmark results for each model
- Generated code samples
- Performance metrics
- Memory usage stats
- Model configurations

### 2. Markdown Report (`benchmark_comparison_YYYYMMDD_HHMMSS.md`)

Human-readable report with:
- Performance summary table
- Memory usage comparison
- Detailed per-benchmark results
- Key findings and insights

## Example Output

### Performance Summary Table

| Model | Success Rate | Avg Keyword Score | Avg Speed (tok/s) | Avg Time (s) |
|-------|-------------|-------------------|-------------------|-------------|
| SmallCoder | 100.0% | 75.5% | 85.3 | 1.76 |
| CodeGen 350M | 100.0% | 68.2% | 92.1 | 1.63 |
| StarCoder 1B | 100.0% | 82.3% | 71.4 | 2.10 |

### Memory Usage Comparison

| Model | Allocated (GB) | Reserved (GB) | Peak (GB) |
|-------|----------------|---------------|-----------|
| SmallCoder | 1.15 | 1.22 | 1.28 |
| CodeGen 350M | 0.98 | 1.05 | 1.10 |
| StarCoder 1B | 2.31 | 2.45 | 2.58 |

## Evaluation Metrics

### Keyword Score

Measures how many expected keywords appear in generated code:
- **Score = keywords_found / total_expected_keywords**
- Range: 0% to 100%
- Higher is better

### Generation Speed

Tokens generated per second:
- **Speed = tokens_generated / generation_time**
- Unit: tokens/second
- Higher is better

### Success Rate

Percentage of benchmarks completed without errors:
- **Rate = successful_runs / total_benchmarks × 100%**
- Range: 0% to 100%
- Higher is better

## CPU vs GPU Benchmarking

### GPU (CUDA)
```bash
python benchmark.py --checkpoint model.pt --device cuda --compare
```

- Faster inference
- Memory tracking available
- Can compare larger models

### CPU Only
```bash
python benchmark.py --checkpoint model.pt --device cpu --compare
```

- Slower inference (5-10x)
- No memory tracking
- May need to skip larger comparison models

## Tips for Accurate Benchmarking

1. **Close other applications** to reduce GPU memory interference
2. **Run multiple times** and average results for consistency
3. **Use same device** for all models in comparison
4. **Check GPU memory** before running: `nvidia-smi`
5. **Use `--skip-memory-test`** if memory measurement causes issues

## Troubleshooting

### Out of Memory Error

```bash
# Use CPU instead
python benchmark.py --checkpoint model.pt --device cpu

# Or skip memory test
python benchmark.py --checkpoint model.pt --skip-memory-test
```

### Model Loading Failed

```bash
# Check available models
python benchmark.py --help

# Specify only models you have access to
python benchmark.py --checkpoint model.pt --compare --comparison-models codegen-350m
```

### Slow Performance

```bash
# Skip additional challenges
python benchmark.py --checkpoint model.pt  # default: core benchmarks only

# Use fewer comparison models
python benchmark.py --checkpoint model.pt --compare --comparison-models codegen-350m
```

## Advanced Usage

### Benchmark Only SmallCoder

```bash
python benchmark.py \
    --checkpoint ./checkpoints/best_model.pt \
    --output-dir ./my_results
```

### Full Comparison Suite

```bash
python benchmark.py \
    --checkpoint ./checkpoints/best_model.pt \
    --compare \
    --comparison-models codellama-7b codegen-350m starcoder-1b \
    --include-challenges \
    --output-dir ./full_benchmark \
    --device cuda
```

### Compare Models Without SmallCoder

```bash
# Just compare other models (useful for baseline)
python benchmark.py \
    --compare \
    --comparison-models codegen-350m starcoder-1b \
    --device cuda
```

## Interpreting Results

### What Makes a Good Score?

**Keyword Score:**
- 70%+ : Excellent - Model understands the task well
- 50-70% : Good - Model captures main concepts
- 30-50% : Fair - Model has basic understanding
- <30% : Poor - Model struggles with the task

**Speed:**
- 100+ tok/s : Very Fast - Real-time interaction
- 50-100 tok/s : Fast - Smooth experience
- 20-50 tok/s : Moderate - Acceptable for most uses
- <20 tok/s : Slow - May feel laggy

**Memory Usage:**
- <2GB : Excellent - Runs on low-end GPUs
- 2-4GB : Good - Runs on mid-range GPUs
- 4-8GB : Moderate - Requires decent GPU
- >8GB : High - Requires high-end GPU

## Contributing

To add new benchmarks:

1. Edit `benchmark.py`
2. Add entries to `CODE_BENCHMARKS` or `CODING_CHALLENGES`
3. Follow the format:
```python
{
    "name": "benchmark_name",
    "prompt": "def example():",
    "expected_keywords": ["keyword1", "keyword2"],
    "description": "Short description",
}
```

## See Also

- [README.md](README.md) - Main documentation
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [PRETRAINED_MODEL_GUIDE.md](PRETRAINED_MODEL_GUIDE.md) - Using pre-trained models
