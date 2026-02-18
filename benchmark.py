"""
Comprehensive evaluation benchmarks for SmallCoder
Compares SmallCoder with other coding models on various benchmarks
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
from pathlib import Path
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from model import SmallCoderConfig, SmallCoderForCausalLM
from inference import load_model


# Code completion benchmarks
CODE_BENCHMARKS = [
    {
        "name": "fibonacci",
        "prompt": "def fibonacci(n):",
        "expected_keywords": ["if", "return", "fibonacci", "n"],
        "description": "Recursive function implementation",
    },
    {
        "name": "binary_search",
        "prompt": "def binary_search(arr, target):",
        "expected_keywords": ["while", "mid", "left", "right", "return"],
        "description": "Search algorithm implementation",
    },
    {
        "name": "quicksort",
        "prompt": "def quicksort(arr):",
        "expected_keywords": ["if", "len", "pivot", "return", "quicksort"],
        "description": "Sorting algorithm with recursion",
    },
    {
        "name": "class_definition",
        "prompt": "class Stack:",
        "expected_keywords": ["def", "__init__", "self", "push", "pop"],
        "description": "Object-oriented programming",
    },
    {
        "name": "list_comprehension",
        "prompt": "# Create a list of squares from 1 to 10\nsquares =",
        "expected_keywords": ["for", "in", "range"],
        "description": "Pythonic list comprehension",
    },
    {
        "name": "error_handling",
        "prompt": "def read_file(filename):\n    try:",
        "expected_keywords": ["open", "except", "return"],
        "description": "Exception handling",
    },
    {
        "name": "async_function",
        "prompt": "async def fetch_data(url):",
        "expected_keywords": ["await", "async", "return"],
        "description": "Asynchronous programming",
    },
    {
        "name": "decorator",
        "prompt": "def timer(func):",
        "expected_keywords": ["def", "wrapper", "return", "time"],
        "description": "Python decorators",
    },
]

# Additional coding challenges
CODING_CHALLENGES = [
    {
        "name": "palindrome_check",
        "prompt": "def is_palindrome(s):\n    \"\"\"Check if a string is a palindrome\"\"\"\n    ",
        "expected_keywords": ["return", "lower", "==", "reverse"],
        "description": "String manipulation",
    },
    {
        "name": "merge_sort",
        "prompt": "def merge_sort(arr):\n    \"\"\"Implement merge sort algorithm\"\"\"\n    ",
        "expected_keywords": ["if", "len", "mid", "merge", "return"],
        "description": "Divide and conquer algorithm",
    },
    {
        "name": "linked_list",
        "prompt": "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None\n\nclass LinkedList:\n    ",
        "expected_keywords": ["def", "__init__", "self", "head", "insert"],
        "description": "Data structure implementation",
    },
]

# Comparison models configuration
COMPARISON_MODELS = {
    "codellama-7b": {
        "model_id": "codellama/CodeLlama-7b-hf",
        "type": "huggingface",
        "description": "Meta's CodeLlama 7B model",
    },
    "codegen-350m": {
        "model_id": "Salesforce/codegen-350M-mono",
        "type": "huggingface",
        "description": "Salesforce CodeGen 350M",
    },
    "starcoder-1b": {
        "model_id": "bigcode/starcoderbase-1b",
        "type": "huggingface",
        "description": "BigCode StarCoder 1B",
    },
}


class ModelWrapper:
    """Wrapper to handle different model types uniformly"""
    def __init__(self, model, tokenizer, model_type='smallcoder'):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        
    def generate(self, prompt, max_new_tokens=150, temperature=0.7, top_p=0.9, top_k=50, device='cuda'):
        """Generate text from prompt"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            if self.model_type == 'smallcoder':
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                )
            else:  # HuggingFace model
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def load_comparison_model(model_config: Dict, device: str = 'cuda') -> Optional[ModelWrapper]:
    """Load a model for comparison"""
    try:
        print(f"Loading {model_config['description']}...")
        tokenizer = AutoTokenizer.from_pretrained(model_config['model_id'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
            trust_remote_code=True,
        )
        
        if device == 'cpu':
            model = model.to(device)
        
        model.eval()
        
        return ModelWrapper(model, tokenizer, model_type='huggingface')
    except Exception as e:
        print(f"Failed to load {model_config['description']}: {e}")
        return None


def evaluate_completion(model_wrapper: ModelWrapper, prompt: str, expected_keywords: List[str], 
                       device: str = 'cuda') -> Dict:
    """Evaluate a single code completion"""
    # Generate
    start_time = time.time()
    try:
        output_text = model_wrapper.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            device=device
        )
        generation_time = time.time() - start_time
        generated_text = output_text[len(prompt):]
        
        # Check for expected keywords
        keywords_found = sum(1 for kw in expected_keywords if kw.lower() in generated_text.lower())
        keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0
        
        return {
            'generated_text': generated_text,
            'keyword_score': keyword_score,
            'generation_time': generation_time,
            'tokens_per_second': 150 / generation_time if generation_time > 0 else 0,
            'success': True,
        }
    except Exception as e:
        return {
            'generated_text': '',
            'keyword_score': 0,
            'generation_time': 0,
            'tokens_per_second': 0,
            'success': False,
            'error': str(e),
        }


def run_benchmarks(model_wrapper: ModelWrapper, model_name: str, benchmarks: List[Dict], 
                  device: str = 'cuda') -> Dict:
    """Run all benchmarks for a single model"""
    print("\n" + "="*80)
    print(f"Running benchmarks for: {model_name}")
    print("="*80 + "\n")
    
    results = []
    total_keyword_score = 0
    total_time = 0
    successful_runs = 0
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"[{i}/{len(benchmarks)}] Testing: {benchmark['name']}")
        print(f"Description: {benchmark.get('description', 'N/A')}")
        print(f"Prompt: {benchmark['prompt'][:60]}...")
        
        result = evaluate_completion(
            model_wrapper=model_wrapper,
            prompt=benchmark['prompt'],
            expected_keywords=benchmark['expected_keywords'],
            device=device
        )
        
        if result['success']:
            print(f"  ✓ Keyword score: {result['keyword_score']:.2%}")
            print(f"  ✓ Generation time: {result['generation_time']:.2f}s")
            print(f"  ✓ Speed: {result['tokens_per_second']:.1f} tokens/s")
            print(f"  Generated: {result['generated_text'][:80]}...")
            total_keyword_score += result['keyword_score']
            total_time += result['generation_time']
            successful_runs += 1
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        print()
        
        results.append({
            'benchmark': benchmark['name'],
            'description': benchmark.get('description', ''),
            **result
        })
    
    # Summary
    avg_keyword_score = total_keyword_score / successful_runs if successful_runs > 0 else 0
    avg_time = total_time / successful_runs if successful_runs > 0 else 0
    avg_speed = (150 * successful_runs) / total_time if total_time > 0 else 0
    
    print("="*80)
    print(f"SUMMARY for {model_name}")
    print("="*80)
    print(f"Successful runs: {successful_runs}/{len(benchmarks)}")
    print(f"Average keyword score: {avg_keyword_score:.2%}")
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Average speed: {avg_speed:.1f} tokens/s")
    print("="*80 + "\n")
    
    return {
        'model_name': model_name,
        'results': results,
        'summary': {
            'total_benchmarks': len(benchmarks),
            'successful_runs': successful_runs,
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
        
        # Dummy forward pass to measure memory
        dummy_input = torch.randint(0, 1000, (1, 512)).to(device)
        with torch.no_grad():
            try:
                _ = model(input_ids=dummy_input)
            except Exception as e:
                print(f"Warning: Memory measurement failed: {e}")
                return {'error': str(e)}
        
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


def generate_comparison_report(all_results: List[Dict], output_dir: str):
    """Generate comparison report with tables and charts"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full JSON results
    json_path = output_path / f"benchmark_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Full results saved to {json_path}")
    
    # Generate markdown comparison table
    md_path = output_path / f"benchmark_comparison_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write("# SmallCoder Benchmark Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary comparison table
        f.write("## Model Performance Summary\n\n")
        f.write("| Model | Success Rate | Avg Keyword Score | Avg Speed (tok/s) | Avg Time (s) |\n")
        f.write("|-------|-------------|-------------------|-------------------|-------------|\n")
        
        for result in all_results:
            model_name = result['model_name']
            summary = result['summary']
            success_rate = summary['successful_runs'] / summary['total_benchmarks'] * 100
            f.write(f"| {model_name} | {success_rate:.1f}% | {summary['avg_keyword_score']:.2%} | "
                   f"{summary['avg_tokens_per_second']:.1f} | {summary['avg_generation_time']:.2f} |\n")
        
        # Memory usage comparison
        f.write("\n## Memory Usage Comparison\n\n")
        f.write("| Model | Allocated (GB) | Reserved (GB) | Peak (GB) |\n")
        f.write("|-------|----------------|---------------|----------|\n")
        
        for result in all_results:
            model_name = result['model_name']
            memory = result.get('memory_stats', {})
            if 'allocated_gb' in memory:
                f.write(f"| {model_name} | {memory['allocated_gb']:.2f} | "
                       f"{memory['reserved_gb']:.2f} | {memory['peak_gb']:.2f} |\n")
            else:
                f.write(f"| {model_name} | N/A | N/A | N/A |\n")
        
        # Per-benchmark detailed results
        f.write("\n## Detailed Benchmark Results\n\n")
        
        # Get all benchmark names
        if all_results and all_results[0]['results']:
            benchmark_names = [r['benchmark'] for r in all_results[0]['results']]
            
            for bench_name in benchmark_names:
                f.write(f"\n### {bench_name}\n\n")
                f.write("| Model | Keyword Score | Time (s) | Speed (tok/s) |\n")
                f.write("|-------|---------------|----------|---------------|\n")
                
                for result in all_results:
                    model_name = result['model_name']
                    bench_result = next((r for r in result['results'] if r['benchmark'] == bench_name), None)
                    if bench_result and bench_result['success']:
                        f.write(f"| {model_name} | {bench_result['keyword_score']:.2%} | "
                               f"{bench_result['generation_time']:.2f} | "
                               f"{bench_result['tokens_per_second']:.1f} |\n")
                    else:
                        f.write(f"| {model_name} | Failed | - | - |\n")
        
        # Key findings
        f.write("\n## Key Findings\n\n")
        
        # Find best model for each metric
        best_accuracy = max(all_results, key=lambda x: x['summary']['avg_keyword_score'])
        best_speed = max(all_results, key=lambda x: x['summary']['avg_tokens_per_second'])
        
        f.write(f"- **Best Accuracy**: {best_accuracy['model_name']} "
               f"({best_accuracy['summary']['avg_keyword_score']:.2%})\n")
        f.write(f"- **Best Speed**: {best_speed['model_name']} "
               f"({best_speed['summary']['avg_tokens_per_second']:.1f} tokens/s)\n")
        
        # SmallCoder specific insights
        smallcoder_results = next((r for r in all_results if 'smallcoder' in r['model_name'].lower()), None)
        if smallcoder_results:
            f.write(f"\n### SmallCoder Performance\n\n")
            f.write(f"- Success Rate: {smallcoder_results['summary']['successful_runs']}/{smallcoder_results['summary']['total_benchmarks']}\n")
            f.write(f"- Average Keyword Score: {smallcoder_results['summary']['avg_keyword_score']:.2%}\n")
            f.write(f"- Average Speed: {smallcoder_results['summary']['avg_tokens_per_second']:.1f} tokens/s\n")
            
            memory = smallcoder_results.get('memory_stats', {})
            if 'peak_gb' in memory:
                f.write(f"- Peak Memory Usage: {memory['peak_gb']:.2f} GB\n")
    
    print(f"✓ Comparison report saved to {md_path}")
    
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare SmallCoder model')
    parser.add_argument('--checkpoint', type=str, 
                        help='Path to SmallCoder model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='codellama/CodeLlama-7b-hf',
                        help='Tokenizer to use for SmallCoder')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with other models')
    parser.add_argument('--comparison-models', type=str, nargs='+',
                        choices=list(COMPARISON_MODELS.keys()),
                        help='Models to compare against (space-separated)')
    parser.add_argument('--include-challenges', action='store_true',
                        help='Include additional coding challenges')
    parser.add_argument('--skip-memory-test', action='store_true',
                        help='Skip memory usage measurement')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Prepare benchmarks
    benchmarks = CODE_BENCHMARKS.copy()
    if args.include_challenges:
        benchmarks.extend(CODING_CHALLENGES)
        print(f"Including {len(CODING_CHALLENGES)} additional coding challenges")
    
    print(f"Total benchmarks: {len(benchmarks)}")
    
    all_results = []
    
    # Evaluate SmallCoder if checkpoint provided
    if args.checkpoint:
        print("\n" + "="*80)
        print("Loading SmallCoder Model")
        print("="*80)
        
        # Load tokenizer
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model, config = load_model(args.checkpoint, device=args.device, quantize=False)
        model_wrapper = ModelWrapper(model, tokenizer, model_type='smallcoder')
        
        # Print model info
        print(f"\nSmallCoder Configuration:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        print(f"  Max sequence length: {config.max_position_embeddings}")
        
        # Measure memory
        if not args.skip_memory_test:
            print("\nMeasuring memory usage...")
            memory_stats = measure_memory_usage(model, device=args.device)
            print(f"Memory stats: {json.dumps(memory_stats, indent=2)}")
        else:
            memory_stats = {}
        
        # Run benchmarks
        results = run_benchmarks(model_wrapper, "SmallCoder", benchmarks, device=args.device)
        results['memory_stats'] = memory_stats
        results['model_config'] = {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_attention_heads': config.num_attention_heads,
            'num_key_value_heads': config.num_key_value_heads,
        }
        all_results.append(results)
        
        # Clean up
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    
    # Compare with other models if requested
    if args.compare:
        models_to_compare = args.comparison_models or list(COMPARISON_MODELS.keys())
        
        for model_key in models_to_compare:
            if model_key not in COMPARISON_MODELS:
                print(f"Warning: Unknown model '{model_key}', skipping...")
                continue
            
            model_config = COMPARISON_MODELS[model_key]
            
            print("\n" + "="*80)
            print(f"Loading {model_config['description']}")
            print("="*80)
            
            model_wrapper = load_comparison_model(model_config, device=args.device)
            
            if model_wrapper is None:
                print(f"Skipping {model_key} due to loading error")
                continue
            
            # Measure memory
            if not args.skip_memory_test:
                print("\nMeasuring memory usage...")
                memory_stats = measure_memory_usage(model_wrapper.model, device=args.device)
                print(f"Memory stats: {json.dumps(memory_stats, indent=2)}")
            else:
                memory_stats = {}
            
            # Run benchmarks
            results = run_benchmarks(model_wrapper, model_config['description'], 
                                   benchmarks, device=args.device)
            results['memory_stats'] = memory_stats
            results['model_id'] = model_config['model_id']
            all_results.append(results)
            
            # Clean up
            del model_wrapper
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    
    # Generate comparison report
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("Generating Comparison Report")
        print("="*80)
        
        json_path, md_path = generate_comparison_report(all_results, args.output_dir)
        
        print("\n" + "="*80)
        print("Benchmarking Complete!")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  - JSON: {json_path}")
        print(f"  - Markdown: {md_path}")
    else:
        print("\nNo models were benchmarked. Please provide --checkpoint or use --compare.")


if __name__ == "__main__":
    main()
