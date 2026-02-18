"""
Evaluation benchmarks for SmallCoder
"""

import torch
from transformers import AutoTokenizer
import time
import argparse
from pathlib import Path
import json

from model import SmallCoderConfig, SmallCoderForCausalLM
from inference import load_model


# Code completion benchmarks
CODE_BENCHMARKS = [
    {
        "name": "fibonacci",
        "prompt": "def fibonacci(n):",
        "expected_keywords": ["if", "return", "fibonacci", "n"],
    },
    {
        "name": "binary_search",
        "prompt": "def binary_search(arr, target):",
        "expected_keywords": ["while", "mid", "left", "right", "return"],
    },
    {
        "name": "quicksort",
        "prompt": "def quicksort(arr):",
        "expected_keywords": ["if", "len", "pivot", "return", "quicksort"],
    },
    {
        "name": "class_definition",
        "prompt": "class Stack:",
        "expected_keywords": ["def", "__init__", "self", "push", "pop"],
    },
    {
        "name": "list_comprehension",
        "prompt": "# Create a list of squares from 1 to 10\nsquares =",
        "expected_keywords": ["for", "in", "range"],
    },
    {
        "name": "error_handling",
        "prompt": "def read_file(filename):\n    try:",
        "expected_keywords": ["open", "except", "return"],
    },
    {
        "name": "async_function",
        "prompt": "async def fetch_data(url):",
        "expected_keywords": ["await", "async", "return"],
    },
    {
        "name": "decorator",
        "prompt": "def timer(func):",
        "expected_keywords": ["def", "wrapper", "return", "time"],
    },
]


def evaluate_completion(model, tokenizer, prompt, expected_keywords, device='cuda'):
    """Evaluate a single code completion"""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
        )
    generation_time = time.time() - start_time
    
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = output_text[len(prompt):]
    
    # Check for expected keywords
    keywords_found = sum(1 for kw in expected_keywords if kw.lower() in generated_text.lower())
    keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0
    
    return {
        'generated_text': generated_text,
        'keyword_score': keyword_score,
        'generation_time': generation_time,
        'tokens_per_second': 150 / generation_time,
    }


def run_benchmarks(model, tokenizer, device='cuda'):
    """Run all benchmarks"""
    print("\n" + "="*80)
    print("SmallCoder Evaluation Benchmarks")
    print("="*80 + "\n")
    
    results = []
    total_keyword_score = 0
    total_time = 0
    
    for i, benchmark in enumerate(CODE_BENCHMARKS, 1):
        print(f"[{i}/{len(CODE_BENCHMARKS)}] Testing: {benchmark['name']}")
        print(f"Prompt: {benchmark['prompt'][:50]}...")
        
        result = evaluate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=benchmark['prompt'],
            expected_keywords=benchmark['expected_keywords'],
            device=device
        )
        
        print(f"  Keyword score: {result['keyword_score']:.2%}")
        print(f"  Generation time: {result['generation_time']:.2f}s")
        print(f"  Speed: {result['tokens_per_second']:.1f} tokens/s")
        print(f"  Generated: {result['generated_text'][:100]}...")
        print()
        
        results.append({
            'benchmark': benchmark['name'],
            **result
        })
        
        total_keyword_score += result['keyword_score']
        total_time += result['generation_time']
    
    # Summary
    avg_keyword_score = total_keyword_score / len(CODE_BENCHMARKS)
    avg_time = total_time / len(CODE_BENCHMARKS)
    avg_speed = (150 * len(CODE_BENCHMARKS)) / total_time
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Average keyword score: {avg_keyword_score:.2%}")
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Average speed: {avg_speed:.1f} tokens/s")
    print("="*80 + "\n")
    
    return {
        'results': results,
        'summary': {
            'avg_keyword_score': avg_keyword_score,
            'avg_generation_time': avg_time,
            'avg_tokens_per_second': avg_speed,
        }
    }


def measure_memory_usage(model, device='cuda'):
    """Measure model memory usage"""
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Dummy forward pass
        dummy_input = torch.randint(0, 1000, (1, 512)).to(device)
        with torch.no_grad():
            _ = model(input_ids=dummy_input)
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'peak_gb': peak,
        }
    else:
        return {'message': 'Memory tracking only available on CUDA'}


def main():
    parser = argparse.ArgumentParser(description='Evaluate SmallCoder model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='codellama/CodeLlama-7b-hf',
                        help='Tokenizer to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model, config = load_model(args.checkpoint, device=args.device, quantize=False)
    
    # Measure memory
    print("\nMeasuring memory usage...")
    memory_stats = measure_memory_usage(model, device=args.device)
    print(f"Memory stats: {json.dumps(memory_stats, indent=2)}")
    
    # Run benchmarks
    results = run_benchmarks(model, tokenizer, device=args.device)
    
    # Add memory stats to results
    results['memory_stats'] = memory_stats
    results['model_config'] = {
        'hidden_size': config.hidden_size,
        'num_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads,
        'num_key_value_heads': config.num_key_value_heads,
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
