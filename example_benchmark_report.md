# Example Benchmark Report

This is an example of what the benchmark comparison report looks like.

Generated: 2026-02-18 21:00:00

## Model Performance Summary

| Model | Success Rate | Avg Keyword Score | Avg Speed (tok/s) | Avg Time (s) |
|-------|-------------|-------------------|-------------------|-------------|
| SmallCoder | 100.0% | 75.5% | 85.3 | 1.76 |
| CodeGen 350M | 100.0% | 68.2% | 92.1 | 1.63 |
| StarCoder 1B | 100.0% | 82.3% | 71.4 | 2.10 |
| CodeLlama 7B | 100.0% | 88.1% | 45.2 | 3.32 |

## Memory Usage Comparison

| Model | Allocated (GB) | Reserved (GB) | Peak (GB) |
|-------|----------------|---------------|----------|
| SmallCoder | 1.15 | 1.22 | 1.28 |
| CodeGen 350M | 0.98 | 1.05 | 1.10 |
| StarCoder 1B | 2.31 | 2.45 | 2.58 |
| CodeLlama 7B | 6.85 | 7.12 | 7.45 |

## Detailed Benchmark Results

### fibonacci

| Model | Keyword Score | Time (s) | Speed (tok/s) |
|-------|---------------|----------|---------------|
| SmallCoder | 100.0% | 1.52 | 98.7 |
| CodeGen 350M | 75.0% | 1.35 | 111.1 |
| StarCoder 1B | 100.0% | 1.89 | 79.4 |
| CodeLlama 7B | 100.0% | 3.12 | 48.1 |

### binary_search

| Model | Keyword Score | Time (s) | Speed (tok/s) |
|-------|---------------|----------|---------------|
| SmallCoder | 80.0% | 1.68 | 89.3 |
| CodeGen 350M | 60.0% | 1.55 | 96.8 |
| StarCoder 1B | 80.0% | 2.02 | 74.3 |
| CodeLlama 7B | 100.0% | 3.28 | 45.7 |

### quicksort

| Model | Keyword Score | Time (s) | Speed (tok/s) |
|-------|---------------|----------|---------------|
| SmallCoder | 80.0% | 1.71 | 87.7 |
| CodeGen 350M | 60.0% | 1.58 | 94.9 |
| StarCoder 1B | 100.0% | 2.15 | 69.8 |
| CodeLlama 7B | 100.0% | 3.35 | 44.8 |

### class_definition

| Model | Keyword Score | Time (s) | Speed (tok/s) |
|-------|---------------|----------|---------------|
| SmallCoder | 80.0% | 1.85 | 81.1 |
| CodeGen 350M | 80.0% | 1.72 | 87.2 |
| StarCoder 1B | 80.0% | 2.28 | 65.8 |
| CodeLlama 7B | 80.0% | 3.48 | 43.1 |

### list_comprehension

| Model | Keyword Score | Time (s) | Speed (tok/s) |
|-------|---------------|----------|---------------|
| SmallCoder | 100.0% | 1.42 | 105.6 |
| CodeGen 350M | 100.0% | 1.38 | 108.7 |
| StarCoder 1B | 100.0% | 1.78 | 84.3 |
| CodeLlama 7B | 100.0% | 2.95 | 50.8 |

### error_handling

| Model | Keyword Score | Time (s) | Speed (tok/s) |
|-------|---------------|----------|---------------|
| SmallCoder | 66.7% | 1.88 | 79.8 |
| CodeGen 350M | 66.7% | 1.75 | 85.7 |
| StarCoder 1B | 66.7% | 2.32 | 64.7 |
| CodeLlama 7B | 100.0% | 3.52 | 42.6 |

### async_function

| Model | Keyword Score | Time (s) | Speed (tok/s) |
|-------|---------------|----------|---------------|
| SmallCoder | 66.7% | 1.92 | 78.1 |
| CodeGen 350M | 33.3% | 1.68 | 89.3 |
| StarCoder 1B | 66.7% | 2.38 | 63.0 |
| CodeLlama 7B | 100.0% | 3.58 | 41.9 |

### decorator

| Model | Keyword Score | Time (s) | Speed (tok/s) |
|-------|---------------|----------|---------------|
| SmallCoder | 75.0% | 1.95 | 76.9 |
| CodeGen 350M | 75.0% | 1.82 | 82.4 |
| StarCoder 1B | 75.0% | 2.45 | 61.2 |
| CodeLlama 7B | 100.0% | 3.62 | 41.4 |

## Key Findings

- **Best Accuracy**: CodeLlama 7B (88.1%)
- **Best Speed**: CodeGen 350M (92.1 tokens/s)
- **Best Memory Efficiency**: CodeGen 350M (1.10 GB peak)

### SmallCoder Performance

- Success Rate: 8/8
- Average Keyword Score: 75.5%
- Average Speed: 85.3 tokens/s
- Peak Memory Usage: 1.28 GB

**Analysis:**

SmallCoder demonstrates strong performance across all benchmarks with an excellent balance of:
- **Quality**: 75.5% keyword score shows good code understanding
- **Speed**: 85.3 tok/s is faster than larger models like CodeLlama (45.2 tok/s)
- **Efficiency**: Only 1.28 GB memory, enabling deployment on budget GPUs
- **Reliability**: 100% success rate across all tests

SmallCoder achieves 85.6% of CodeLlama-7B's accuracy while being:
- **1.89x faster** in generation
- **5.82x smaller** in memory footprint
- Suitable for consumer hardware (2GB VRAM GPUs)

This makes SmallCoder ideal for developers who need:
- Real-time code completion on limited hardware
- Fast iteration during development
- Local deployment without cloud costs
- Privacy-preserving code generation
