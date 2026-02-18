#!/usr/bin/env python3
"""
Validation script for benchmark.py structure
Tests the code structure without requiring heavy dependencies
"""

import ast
import sys
from pathlib import Path

def validate_python_file(filepath):
    """Validate Python file syntax and structure"""
    print(f"Validating {filepath}...")
    
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source, filename=filepath)
        
        # Find all function and class definitions
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return True, functions, classes
    
    except SyntaxError as e:
        return False, str(e), []
    except Exception as e:
        return False, str(e), []

def main():
    print("="*80)
    print("SmallCoder Benchmark Validation")
    print("="*80)
    print()
    
    benchmark_file = Path("benchmark.py")
    
    if not benchmark_file.exists():
        print(f"✗ Error: {benchmark_file} not found")
        return False
    
    # Validate syntax
    valid, functions, classes = validate_python_file(benchmark_file)
    
    if not valid:
        print(f"✗ Syntax Error in {benchmark_file}:")
        print(f"  {functions}")
        return False
    
    print(f"✓ {benchmark_file} syntax is valid")
    print()
    
    # Check expected classes
    expected_classes = ["ModelWrapper"]
    print("Checking required classes...")
    for cls in expected_classes:
        if cls in classes:
            print(f"  ✓ {cls} found")
        else:
            print(f"  ✗ {cls} missing")
            return False
    print()
    
    # Check expected functions
    expected_functions = [
        "load_comparison_model",
        "evaluate_completion",
        "run_benchmarks",
        "measure_memory_usage",
        "generate_comparison_report",
        "main",
    ]
    
    print("Checking required functions...")
    for func in expected_functions:
        if func in functions:
            print(f"  ✓ {func} found")
        else:
            print(f"  ✗ {func} missing")
            return False
    print()
    
    # Check for required data structures
    print("Checking data structures...")
    with open(benchmark_file, 'r') as f:
        content = f.read()
    
    required_vars = ["CODE_BENCHMARKS", "CODING_CHALLENGES", "COMPARISON_MODELS"]
    for var in required_vars:
        if var in content:
            print(f"  ✓ {var} found")
        else:
            print(f"  ✗ {var} missing")
            return False
    print()
    
    # Validate documentation files
    print("Checking documentation...")
    doc_files = ["BENCHMARK_GUIDE.md", "benchmark_examples.py", "README.md"]
    for doc in doc_files:
        if Path(doc).exists():
            print(f"  ✓ {doc} exists")
        else:
            print(f"  ✗ {doc} missing")
    print()
    
    # Check benchmark definitions
    print("Validating benchmark definitions...")
    tree = ast.parse(content)
    
    # Find CODE_BENCHMARKS
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CODE_BENCHMARKS":
                    if isinstance(node.value, ast.List):
                        num_benchmarks = len(node.value.elts)
                        print(f"  ✓ Found {num_benchmarks} core benchmarks")
                        
                        # Validate benchmark structure
                        for i, benchmark in enumerate(node.value.elts):
                            if isinstance(benchmark, ast.Dict):
                                keys = [k.value if isinstance(k, ast.Constant) else k.s 
                                       for k in benchmark.keys if isinstance(k, (ast.Constant, ast.Str))]
                                required_keys = ["name", "prompt", "expected_keywords"]
                                if all(key in keys for key in required_keys):
                                    print(f"    ✓ Benchmark {i+1} has required keys")
                                else:
                                    print(f"    ✗ Benchmark {i+1} missing required keys")
                                    print(f"      Found: {keys}")
                                    print(f"      Required: {required_keys}")
    print()
    
    # Check comparison models
    print("Validating comparison models...")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "COMPARISON_MODELS":
                    if isinstance(node.value, ast.Dict):
                        num_models = len(node.value.keys)
                        print(f"  ✓ Found {num_models} comparison models configured")
    print()
    
    print("="*80)
    print("✓ All validations passed!")
    print("="*80)
    print()
    print("The benchmark system is properly structured and ready to use.")
    print("To run benchmarks, you'll need:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Have a trained model checkpoint")
    print("  3. Run: python benchmark.py --checkpoint model.pt")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
