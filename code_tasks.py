"""
Comprehensive coding task test suite for SmallCoder
Tests the model on various programming challenges
"""

import torch
from transformers import AutoTokenizer
import time
import argparse
import json
from pathlib import Path

from model import SmallCoderConfig, SmallCoderForCausalLM
from inference import load_model


# Comprehensive coding task benchmarks
CODING_TASKS = [
    # Algorithm Implementation Tasks
    {
        "category": "algorithms",
        "name": "merge_sort",
        "prompt": "def merge_sort(arr):\n    '''Sort array using merge sort algorithm'''",
        "expected_keywords": ["if", "len", "mid", "merge", "left", "right", "return"],
        "difficulty": "medium",
    },
    {
        "category": "algorithms",
        "name": "dijkstra",
        "prompt": "def dijkstra(graph, start):\n    '''Find shortest path using Dijkstra's algorithm'''",
        "expected_keywords": ["distance", "visited", "min", "while", "for"],
        "difficulty": "hard",
    },
    {
        "category": "algorithms",
        "name": "binary_tree_traversal",
        "prompt": "class TreeNode:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None\n\ndef inorder_traversal(root):",
        "expected_keywords": ["if", "root", "left", "right", "return"],
        "difficulty": "medium",
    },
    
    # Data Structure Tasks
    {
        "category": "data_structures",
        "name": "linked_list",
        "prompt": "class LinkedList:\n    '''Implement a linked list with insert, delete, and search operations'''",
        "expected_keywords": ["def", "__init__", "self", "node", "next"],
        "difficulty": "medium",
    },
    {
        "category": "data_structures",
        "name": "hash_table",
        "prompt": "class HashTable:\n    '''Implement a hash table with collision handling'''",
        "expected_keywords": ["def", "__init__", "hash", "insert", "get"],
        "difficulty": "hard",
    },
    {
        "category": "data_structures",
        "name": "priority_queue",
        "prompt": "class PriorityQueue:\n    '''Implement a priority queue using a heap'''",
        "expected_keywords": ["def", "heapify", "push", "pop", "heap"],
        "difficulty": "medium",
    },
    
    # Debugging Tasks
    {
        "category": "debugging",
        "name": "fix_off_by_one",
        "prompt": "# Fix the bug in this binary search function\ndef binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1\n\n# Fixed version:",
        "expected_keywords": ["left", "right", "mid", "+", "1"],
        "difficulty": "easy",
    },
    
    # Code Refactoring Tasks
    {
        "category": "refactoring",
        "name": "extract_function",
        "prompt": "# Refactor this code by extracting repeated logic into functions\ndef process_data(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result\n\n# Refactored version:",
        "expected_keywords": ["def", "filter", "map", "lambda"],
        "difficulty": "easy",
    },
    
    # Code Explanation Tasks
    {
        "category": "explanation",
        "name": "explain_decorator",
        "prompt": "# Explain what this decorator does:\ndef memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper\n\n# Explanation:",
        "expected_keywords": ["cache", "store", "result", "performance", "avoid"],
        "difficulty": "medium",
    },
    
    # Web Development Tasks
    {
        "category": "web_dev",
        "name": "rest_api_endpoint",
        "prompt": "from flask import Flask, jsonify, request\napp = Flask(__name__)\n\n# Create a RESTful API endpoint to handle user registration\n@app.route('/api/register', methods=['POST'])\ndef register():",
        "expected_keywords": ["request", "json", "get", "return", "jsonify"],
        "difficulty": "medium",
    },
    
    # Async Programming Tasks
    {
        "category": "async",
        "name": "async_web_scraper",
        "prompt": "import asyncio\nimport aiohttp\n\nasync def fetch_multiple_urls(urls):\n    '''Fetch multiple URLs concurrently'''",
        "expected_keywords": ["async", "await", "session", "gather", "for"],
        "difficulty": "hard",
    },
    
    # Error Handling Tasks
    {
        "category": "error_handling",
        "name": "robust_file_processor",
        "prompt": "def process_file(filename):\n    '''Process a file with proper error handling for common issues'''",
        "expected_keywords": ["try", "except", "finally", "open", "FileNotFoundError"],
        "difficulty": "easy",
    },
    
    # Testing Tasks
    {
        "category": "testing",
        "name": "unit_test",
        "prompt": "import unittest\n\nclass TestCalculator(unittest.TestCase):\n    '''Write unit tests for a calculator class'''",
        "expected_keywords": ["def", "test", "self", "assertEqual", "assert"],
        "difficulty": "easy",
    },
    
    # Database Tasks
    {
        "category": "database",
        "name": "sql_query",
        "prompt": "# Write a SQL query to find the top 5 customers by total purchase amount\n# Tables: customers (id, name), orders (id, customer_id, amount)\nquery = '''",
        "expected_keywords": ["SELECT", "FROM", "JOIN", "GROUP BY", "ORDER BY", "LIMIT"],
        "difficulty": "medium",
    },
    
    # Design Pattern Tasks
    {
        "category": "design_patterns",
        "name": "singleton_pattern",
        "prompt": "class DatabaseConnection:\n    '''Implement the Singleton pattern to ensure only one database connection exists'''",
        "expected_keywords": ["__new__", "_instance", "if", "not", "return"],
        "difficulty": "medium",
    },
]


def evaluate_coding_task(model, tokenizer, task, device='cuda', max_tokens=300):
    """Evaluate a single coding task"""
    prompt = task['prompt']
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Check if prompt is too long
    if input_ids.shape[1] > 2000:
        print(f"  Warning: Prompt is {input_ids.shape[1]} tokens, truncating...")
        input_ids = input_ids[:, :2000]
    
    # Generate
    start_time = time.time()
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
            )
        generation_time = time.time() - start_time
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        
        # Check for expected keywords
        keywords_found = sum(1 for kw in task['expected_keywords'] 
                           if kw.lower() in generated_text.lower())
        keyword_score = keywords_found / len(task['expected_keywords']) if task['expected_keywords'] else 0
        
        # Calculate tokens generated
        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
        
        return {
            'success': True,
            'generated_text': generated_text,
            'keyword_score': keyword_score,
            'generation_time': generation_time,
            'tokens_generated': tokens_generated,
            'tokens_per_second': tokens_generated / generation_time if generation_time > 0 else 0,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'keyword_score': 0,
            'generation_time': 0,
            'tokens_generated': 0,
            'tokens_per_second': 0,
        }


def run_coding_tasks(model, tokenizer, device='cuda', filter_category=None, max_tokens=300):
    """Run all coding tasks or filter by category"""
    print("\n" + "="*80)
    print("SmallCoder Comprehensive Coding Task Evaluation")
    print("="*80 + "\n")
    
    # Filter tasks if category specified
    tasks_to_run = CODING_TASKS
    if filter_category:
        tasks_to_run = [t for t in CODING_TASKS if t['category'] == filter_category]
        print(f"Running tasks in category: {filter_category}")
    else:
        print(f"Running all {len(CODING_TASKS)} coding tasks")
    
    print(f"Max tokens per generation: {max_tokens}")
    print()
    
    results = []
    category_stats = {}
    
    for i, task in enumerate(tasks_to_run, 1):
        print(f"[{i}/{len(tasks_to_run)}] {task['category'].upper()} - {task['name']}")
        print(f"  Difficulty: {task['difficulty']}")
        print(f"  Prompt: {task['prompt'][:80]}...")
        
        result = evaluate_coding_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            device=device,
            max_tokens=max_tokens
        )
        
        if result['success']:
            print(f"  ✓ Generated {result['tokens_generated']} tokens in {result['generation_time']:.2f}s")
            print(f"  Keyword score: {result['keyword_score']:.2%}")
            print(f"  Speed: {result['tokens_per_second']:.1f} tokens/s")
        else:
            print(f"  ✗ Failed: {result['error']}")
        print()
        
        # Store result
        result_data = {
            'task': task['name'],
            'category': task['category'],
            'difficulty': task['difficulty'],
            **result
        }
        results.append(result_data)
        
        # Update category stats
        if task['category'] not in category_stats:
            category_stats[task['category']] = {
                'count': 0,
                'total_score': 0,
                'total_time': 0,
                'total_tokens': 0,
            }
        
        category_stats[task['category']]['count'] += 1
        category_stats[task['category']]['total_score'] += result['keyword_score']
        category_stats[task['category']]['total_time'] += result['generation_time']
        category_stats[task['category']]['total_tokens'] += result['tokens_generated']
    
    # Calculate overall statistics
    successful_tasks = [r for r in results if r['success']]
    total_score = sum(r['keyword_score'] for r in successful_tasks)
    total_time = sum(r['generation_time'] for r in successful_tasks)
    total_tokens = sum(r['tokens_generated'] for r in successful_tasks)
    
    avg_score = total_score / len(successful_tasks) if successful_tasks else 0
    avg_time = total_time / len(successful_tasks) if successful_tasks else 0
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    
    # Print category summaries
    print("="*80)
    print("CATEGORY SUMMARIES")
    print("="*80)
    for category, stats in sorted(category_stats.items()):
        avg_cat_score = stats['total_score'] / stats['count'] if stats['count'] > 0 else 0
        avg_cat_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
        avg_cat_speed = stats['total_tokens'] / stats['total_time'] if stats['total_time'] > 0 else 0
        
        print(f"\n{category.upper()}:")
        print(f"  Tasks: {stats['count']}")
        print(f"  Avg keyword score: {avg_cat_score:.2%}")
        print(f"  Avg generation time: {avg_cat_time:.2f}s")
        print(f"  Avg speed: {avg_cat_speed:.1f} tokens/s")
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(results) - len(successful_tasks)}")
    print(f"Average keyword score: {avg_score:.2%}")
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Average speed: {avg_speed:.1f} tokens/s")
    print(f"Total tokens generated: {total_tokens}")
    print("="*80 + "\n")
    
    return {
        'results': results,
        'category_stats': category_stats,
        'summary': {
            'total_tasks': len(results),
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(results) - len(successful_tasks),
            'avg_keyword_score': avg_score,
            'avg_generation_time': avg_time,
            'avg_tokens_per_second': avg_speed,
            'total_tokens_generated': total_tokens,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test SmallCoder on comprehensive coding tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='pretrained_smallcoder.pt',
        help='Path to model checkpoint'
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
        '--category',
        type=str,
        default=None,
        choices=['algorithms', 'data_structures', 'debugging', 'refactoring', 
                'explanation', 'web_dev', 'async', 'error_handling', 'testing', 
                'database', 'design_patterns'],
        help='Filter tasks by category'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=300,
        help='Maximum tokens to generate per task'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='coding_tasks_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Use INT8 quantization'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" "*25 + "SmallCoder Coding Tasks")
    print("="*80 + "\n")
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print()
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 1
    
    print()
    
    # Check if model exists, create if needed
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Model checkpoint not found at {args.checkpoint}")
        print("Creating pre-trained model...")
        from pretrained_model import create_pretrained_model
        create_pretrained_model(str(checkpoint_path))
        print()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device=args.device, quantize=args.quantize)
    print("✓ Model loaded successfully")
    print()
    
    # Run coding tasks
    results = run_coding_tasks(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        filter_category=args.category,
        max_tokens=args.max_tokens
    )
    
    # Add metadata
    results['metadata'] = {
        'checkpoint': args.checkpoint,
        'tokenizer': args.tokenizer,
        'device': str(device),
        'max_tokens': args.max_tokens,
        'quantize': args.quantize,
        'model_config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'context_length': config.max_position_embeddings,
        }
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
