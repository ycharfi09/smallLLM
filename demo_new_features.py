#!/usr/bin/env python3
"""
Demo script showing the new features of SmallCoder:
1. Extended context length (8192 tokens)
2. Comprehensive coding task testing
3. Lightweight coding agent capabilities
"""

import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def demo_extended_context():
    """Demonstrate extended context length"""
    print_header("✨ Feature 1: Extended Context Length")
    
    from model import SmallCoderConfig
    
    config = SmallCoderConfig()
    
    print("SmallCoder now supports an extended context length!")
    print(f"  • Previous context length: 4096 tokens")
    print(f"  • New context length: {config.max_position_embeddings} tokens")
    print(f"  • Improvement: {config.max_position_embeddings / 4096:.1f}x larger")
    print("\nThis means you can:")
    print("  ✓ Process longer code files")
    print("  ✓ Maintain more context in conversations")
    print("  ✓ Handle complex multi-file projects")
    print("  ✓ Generate longer code completions")


def demo_coding_tasks():
    """Demonstrate coding task testing"""
    print_header("🧪 Feature 2: Comprehensive Coding Task Testing")
    
    import code_tasks
    
    # Count tasks by category
    categories = {}
    for task in code_tasks.CODING_TASKS:
        cat = task['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"SmallCoder includes {len(code_tasks.CODING_TASKS)} comprehensive coding tasks:")
    print()
    
    for category, count in sorted(categories.items()):
        print(f"  • {category:20s} : {count} tasks")
    
    print("\nTask categories cover:")
    print("  ✓ Algorithm implementation (sorting, searching, graphs)")
    print("  ✓ Data structures (trees, hash tables, linked lists)")
    print("  ✓ Debugging and bug fixing")
    print("  ✓ Code refactoring")
    print("  ✓ Code explanation")
    print("  ✓ Web development (APIs, frameworks)")
    print("  ✓ Async programming")
    print("  ✓ Error handling")
    print("  ✓ Unit testing")
    print("  ✓ Database queries")
    print("  ✓ Design patterns")
    
    print("\nExample usage:")
    print("  # Run all tasks")
    print("  python code_tasks.py --checkpoint pretrained_smallcoder.pt")
    print()
    print("  # Run specific category")
    print("  python code_tasks.py --category algorithms")
    print()
    print("  # Save detailed results")
    print("  python code_tasks.py --output results.json")


def demo_coding_agent():
    """Demonstrate coding agent capabilities"""
    print_header("🤖 Feature 3: Lightweight Coding Agent")
    
    print("SmallCoder now includes an intelligent coding agent that can:")
    print()
    print("  • Solve Problems Iteratively")
    print("    - Automatically refine solutions based on test results")
    print("    - Learn from execution errors")
    print("    - Multiple attempts until success")
    print()
    print("  • Execute and Test Code")
    print("    - Safe code execution in isolated environment")
    print("    - Automatic test case validation")
    print("    - Detailed error reporting")
    print()
    print("  • Analyze Code Quality")
    print("    - Detect common issues")
    print("    - Suggest improvements")
    print("    - Check for best practices")
    print()
    print("  • Interactive Mode")
    print("    - Natural language problem descriptions")
    print("    - Real-time code generation")
    print("    - Multi-turn conversations")
    
    print("\nExample agent commands:")
    print("  # Interactive mode")
    print("  python coding_agent.py --interactive")
    print()
    print("  # Solve a specific problem")
    print('  python coding_agent.py --problem "Write a function to reverse a string"')
    print()
    print("  # Solve with custom test cases")
    print('  python coding_agent.py --problem "Implement bubble sort" --tests tests.json')
    print()
    print("  # Adjust complexity")
    print('  python coding_agent.py --problem "Complex task" --max_iterations 5')
    
    print("\nAgent workflow:")
    print("  1. Generate initial solution")
    print("  2. Analyze for issues")
    print("  3. Execute and test")
    print("  4. Refine based on results")
    print("  5. Repeat until success or max iterations")


def demo_usage_examples():
    """Show practical usage examples"""
    print_header("📚 Usage Examples")
    
    print("Quick Start (3 simple steps):")
    print()
    print("  1. Install dependencies")
    print("     $ pip install torch transformers")
    print()
    print("  2. Generate pre-trained model")
    print("     $ python pretrained_model.py")
    print()
    print("  3. Start using the agent")
    print("     $ python coding_agent.py --interactive")
    print()
    
    print("\nCommon Use Cases:")
    print()
    print("  📝 Code Generation")
    print("     $ python run_model.py --prompt 'def fibonacci(n):'")
    print()
    print("  🧪 Benchmark Testing")
    print("     $ python code_tasks.py --category algorithms")
    print()
    print("  🤖 Problem Solving")
    print("     $ python coding_agent.py --problem 'Sort array in-place'")
    print()
    print("  💻 Interactive Development")
    print("     $ python coding_agent.py --interactive")


def main():
    print("\n" + "="*80)
    print(" "*20 + "SmallCoder v2.0 - New Features Demo")
    print("="*80)
    print("\nSmallCoder has been enhanced with powerful new capabilities!")
    print("This demo will walk you through the new features.")
    
    # Run all demos
    demo_extended_context()
    demo_coding_tasks()
    demo_coding_agent()
    demo_usage_examples()
    
    # Summary
    print_header("🎉 Summary")
    print("SmallCoder is now a more powerful coding assistant with:")
    print()
    print("  ✅ 2x extended context length (8192 tokens)")
    print("  ✅ 15+ comprehensive coding task benchmarks")
    print("  ✅ Intelligent agent with iterative refinement")
    print("  ✅ Code execution and testing capabilities")
    print("  ✅ Multi-step problem solving")
    print()
    print("All while maintaining:")
    print("  • ~304M parameters (small footprint)")
    print("  • 2GB VRAM compatibility")
    print("  • Fast inference speed")
    print("  • Easy to use interface")
    print()
    print("Get started:")
    print("  $ python pretrained_model.py")
    print("  $ python coding_agent.py --interactive")
    print()
    print("="*80)
    print("\nFor more information, see the updated README.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
