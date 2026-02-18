#!/usr/bin/env python3
"""
Example script demonstrating how to use the benchmarking system
Run this after training or using the pre-trained model
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print results"""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found. Make sure dependencies are installed.")
        return False

def main():
    print("="*80)
    print("SmallCoder Benchmarking Examples")
    print("="*80)
    print("\nThis script demonstrates different benchmarking scenarios.")
    print("Note: You need a trained model checkpoint to run these examples.")
    print("\nMake sure you have:")
    print("  1. A trained model checkpoint (or use pretrained_model.py)")
    print("  2. Required dependencies installed (pip install -r requirements.txt)")
    
    checkpoint_path = input("\nEnter path to your model checkpoint (or press Enter to use 'model.pt'): ").strip()
    if not checkpoint_path:
        checkpoint_path = "model.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        print("\nTo create a pre-trained model, run:")
        print("  python pretrained_model.py")
        return
    
    # Example 1: Basic benchmark
    print("\n" + "="*80)
    print("Example 1: Basic SmallCoder Benchmark")
    print("="*80)
    print("This runs core coding benchmarks on SmallCoder only.")
    
    if input("\nRun Example 1? (y/n): ").lower() == 'y':
        run_command([
            sys.executable, "benchmark.py",
            "--checkpoint", checkpoint_path,
            "--device", "cuda",
            "--output-dir", "./examples/basic_benchmark"
        ], "Basic Benchmark")
    
    # Example 2: Compare with other models
    print("\n" + "="*80)
    print("Example 2: Compare with Other Models")
    print("="*80)
    print("This compares SmallCoder with CodeGen-350M and StarCoder-1B.")
    print("Note: This will download comparison models (~1-2GB).")
    
    if input("\nRun Example 2? (y/n): ").lower() == 'y':
        run_command([
            sys.executable, "benchmark.py",
            "--checkpoint", checkpoint_path,
            "--compare",
            "--comparison-models", "codegen-350m", "starcoder-1b",
            "--device", "cuda",
            "--output-dir", "./examples/comparison_benchmark"
        ], "Comparison Benchmark")
    
    # Example 3: Extended benchmarks
    print("\n" + "="*80)
    print("Example 3: Extended Coding Challenges")
    print("="*80)
    print("This includes additional coding challenges beyond core benchmarks.")
    
    if input("\nRun Example 3? (y/n): ").lower() == 'y':
        run_command([
            sys.executable, "benchmark.py",
            "--checkpoint", checkpoint_path,
            "--include-challenges",
            "--device", "cuda",
            "--output-dir", "./examples/extended_benchmark"
        ], "Extended Benchmark")
    
    # Example 4: CPU benchmark
    print("\n" + "="*80)
    print("Example 4: CPU-Only Benchmark")
    print("="*80)
    print("This runs benchmarks on CPU (useful if no GPU available).")
    
    if input("\nRun Example 4? (y/n): ").lower() == 'y':
        run_command([
            sys.executable, "benchmark.py",
            "--checkpoint", checkpoint_path,
            "--device", "cpu",
            "--skip-memory-test",
            "--output-dir", "./examples/cpu_benchmark"
        ], "CPU Benchmark")
    
    # Example 5: Full comparison suite
    print("\n" + "="*80)
    print("Example 5: Full Comparison Suite")
    print("="*80)
    print("This runs all benchmarks and compares with all available models.")
    print("Warning: This may take 10-20 minutes and requires significant GPU memory.")
    
    if input("\nRun Example 5? (y/n): ").lower() == 'y':
        run_command([
            sys.executable, "benchmark.py",
            "--checkpoint", checkpoint_path,
            "--compare",
            "--comparison-models", "codellama-7b", "codegen-350m", "starcoder-1b",
            "--include-challenges",
            "--device", "cuda",
            "--output-dir", "./examples/full_benchmark"
        ], "Full Comparison Suite")
    
    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80)
    print("\nCheck the ./examples/ directory for benchmark results.")
    print("Each run creates JSON and Markdown reports with detailed results.")

if __name__ == "__main__":
    main()
