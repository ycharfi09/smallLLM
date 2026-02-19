"""
Comprehensive benchmark script for all SmallCoder model variants
Compares performance, memory usage, and generation quality across variants
"""

import torch
from transformers import AutoTokenizer
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

from model import SmallCoderForCausalLM, count_parameters
from model_variants import get_variant_config, MODEL_VARIANTS


# Benchmark prompts for different coding tasks
BENCHMARK_PROMPTS = [
    {
        "name": "fibonacci",
        "prompt": "def fibonacci(n):",
        "category": "algorithms",
        "expected_keywords": ["if", "return", "fibonacci", "n"],
    },
    {
        "name": "binary_search",
        "prompt": "def binary_search(arr, target):",
        "category": "algorithms",
        "expected_keywords": ["while", "mid", "left", "right", "return"],
    },
    {
        "name": "quicksort",
        "prompt": "def quicksort(arr):",
        "category": "algorithms",
        "expected_keywords": ["if", "len", "pivot", "return", "quicksort"],
    },
    {
        "name": "stack_class",
        "prompt": "class Stack:",
        "category": "data_structures",
        "expected_keywords": ["def", "__init__", "self", "push", "pop"],
    },
    {
        "name": "linked_list",
        "prompt": "class LinkedList:\n    def __init__(self):",
        "category": "data_structures",
        "expected_keywords": ["self", "head", "None", "Node"],
    },
    {
        "name": "file_handling",
        "prompt": "def read_json_file(filename):",
        "category": "practical",
        "expected_keywords": ["with", "open", "json", "return"],
    },
    {
        "name": "api_request",
        "prompt": "import requests\n\ndef fetch_user_data(user_id):",
        "category": "practical",
        "expected_keywords": ["requests", "get", "return", "json"],
    },
    {
        "name": "list_comprehension",
        "prompt": "# Create a list of even squares from 1 to 20\nresult =",
        "category": "syntax",
        "expected_keywords": ["for", "in", "range", "if"],
    },
]


def benchmark_single_prompt(
    model,
    tokenizer,
    prompt: str,
    expected_keywords: List[str],
    device: str,
    max_tokens: int = 150,
) -> Dict:
    """Benchmark a single prompt"""
    # Encode
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Measure generation time
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            use_cache=True,
        )
    generation_time = time.time() - start_time
    
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = output_text[len(prompt):]
    
    # Calculate metrics
    tokens_generated = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    # Keyword matching score
    keywords_found = sum(1 for kw in expected_keywords if kw.lower() in generated_text.lower())
    keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0
    
    return {
        'generated_text': generated_text[:200],  # Truncate for display
        'tokens_generated': tokens_generated,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second,
        'keyword_score': keyword_score,
        'keywords_found': keywords_found,
        'keywords_total': len(expected_keywords),
    }


def measure_memory(model, device: str, context_length: int = 512) -> Dict:
    """Measure memory usage of a model"""
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Forward pass
        dummy_input = torch.randint(0, 1000, (1, context_length)).to(device)
        with torch.no_grad():
            _ = model(input_ids=dummy_input)
        
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {
            'allocated_mb': round(allocated_mb, 2),
            'reserved_mb': round(reserved_mb, 2),
            'peak_mb': round(peak_mb, 2),
            'device': 'cuda',
        }
    else:
        return {
            'message': 'Memory tracking only available on CUDA',
            'device': 'cpu',
        }


def benchmark_variant(
    variant_name: str,
    tokenizer,
    device: str,
    max_tokens: int = 150,
    num_prompts: int = None,
) -> Dict:
    """Benchmark a single model variant"""
    print(f"\n{'='*100}")
    print(f"Benchmarking: {variant_name}")
    print(f"{'='*100}")
    
    # Load config and create model
    config = get_variant_config(variant_name)
    model = SmallCoderForCausalLM(config)
    
    # Try to load checkpoint if available
    checkpoint_path = f"pretrained_{variant_name.lower().replace('-', '_')}.pt"
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"No checkpoint found, using initialized model")
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    params = count_parameters(model)
    print(f"Parameters: {params:,} (~{params/1e6:.1f}M)")
    
    # Measure memory
    print("\nMeasuring memory usage...")
    memory_stats = measure_memory(model, device)
    for key, value in memory_stats.items():
        print(f"  {key}: {value}")
    
    # Run benchmarks
    prompts_to_run = BENCHMARK_PROMPTS[:num_prompts] if num_prompts else BENCHMARK_PROMPTS
    print(f"\nRunning {len(prompts_to_run)} benchmark prompts...")
    
    results = []
    total_time = 0
    total_tokens = 0
    total_keyword_score = 0
    
    for i, benchmark in enumerate(prompts_to_run, 1):
        print(f"  [{i}/{len(prompts_to_run)}] {benchmark['name']}...", end=' ')
        
        result = benchmark_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=benchmark['prompt'],
            expected_keywords=benchmark['expected_keywords'],
            device=device,
            max_tokens=max_tokens,
        )
        
        results.append({
            'name': benchmark['name'],
            'category': benchmark['category'],
            **result
        })
        
        total_time += result['generation_time']
        total_tokens += result['tokens_generated']
        total_keyword_score += result['keyword_score']
        
        print(f"{result['tokens_per_second']:.1f} tok/s, score: {result['keyword_score']:.2%}")
    
    # Calculate averages
    avg_keyword_score = total_keyword_score / len(prompts_to_run)
    avg_time = total_time / len(prompts_to_run)
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    summary = {
        'variant': variant_name,
        'parameters': params,
        'parameters_m': round(params / 1e6, 1),
        'memory_stats': memory_stats,
        'avg_keyword_score': round(avg_keyword_score, 4),
        'avg_generation_time': round(avg_time, 3),
        'avg_tokens_per_second': round(avg_tokens_per_second, 2),
        'total_prompts': len(prompts_to_run),
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'num_kv_heads': config.num_key_value_heads,
            'max_context': config.max_position_embeddings,
        }
    }
    
    print(f"\nSummary:")
    print(f"  Avg keyword score: {avg_keyword_score:.2%}")
    print(f"  Avg generation time: {avg_time:.3f}s")
    print(f"  Avg speed: {avg_tokens_per_second:.2f} tokens/s")
    
    # Clean up
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'summary': summary,
        'detailed_results': results,
    }


def compare_variants(
    variants: List[str],
    tokenizer,
    device: str,
    max_tokens: int = 150,
    num_prompts: int = None,
    output_file: str = None,
):
    """Compare multiple model variants"""
    print("\n" + "="*100)
    print(" " * 30 + "SmallCoder Variants Benchmark")
    print("="*100)
    print(f"\nBenchmarking {len(variants)} variants on {len(BENCHMARK_PROMPTS[:num_prompts] if num_prompts else BENCHMARK_PROMPTS)} prompts")
    print(f"Device: {device}")
    print(f"Max tokens per generation: {max_tokens}")
    
    all_results = {}
    
    for variant in variants:
        try:
            result = benchmark_variant(
                variant_name=variant,
                tokenizer=tokenizer,
                device=device,
                max_tokens=max_tokens,
                num_prompts=num_prompts,
            )
            all_results[variant] = result
        except Exception as e:
            print(f"\nError benchmarking {variant}: {e}")
            continue
    
    # Comparison table
    print("\n" + "="*100)
    print(" " * 35 + "COMPARISON SUMMARY")
    print("="*100)
    print()
    
    # Header
    print(f"{'Variant':<25} {'Params':<12} {'Speed':<15} {'Quality':<12} {'Memory (MB)':<15}")
    print("-" * 100)
    
    for variant, result in all_results.items():
        summary = result['summary']
        params_str = f"{summary['parameters_m']}M"
        speed_str = f"{summary['avg_tokens_per_second']:.1f} tok/s"
        quality_str = f"{summary['avg_keyword_score']:.1%}"
        
        if 'peak_mb' in summary['memory_stats']:
            memory_str = f"{summary['memory_stats']['peak_mb']:.0f} MB"
        else:
            memory_str = "N/A (CPU)"
        
        print(f"{variant:<25} {params_str:<12} {speed_str:<15} {quality_str:<12} {memory_str:<15}")
    
    print("="*100)
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SmallCoder model variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--variants',
        nargs='+',
        default=None,
        help='Model variants to benchmark (default: all variants)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='codellama/CodeLlama-7b-hf',
        help='Tokenizer to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=150,
        help='Maximum tokens to generate per prompt'
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=None,
        help='Number of prompts to test (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results_variants.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--list-variants',
        action='store_true',
        help='List available variants and exit'
    )
    
    args = parser.parse_args()
    
    # List variants
    if args.list_variants:
        from model_variants import list_variants
        list_variants()
        return
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Get variants to benchmark
    if args.variants:
        variants = args.variants
        # Validate variants
        for v in variants:
            if v not in MODEL_VARIANTS:
                print(f"Error: Unknown variant '{v}'")
                print(f"Available variants: {', '.join(MODEL_VARIANTS.keys())}")
                sys.exit(1)
    else:
        # Default: benchmark all variants
        variants = list(MODEL_VARIANTS.keys())
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Run benchmarks
    results = compare_variants(
        variants=variants,
        tokenizer=tokenizer,
        device=args.device,
        max_tokens=args.max_tokens,
        num_prompts=args.num_prompts,
        output_file=args.output,
    )
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
